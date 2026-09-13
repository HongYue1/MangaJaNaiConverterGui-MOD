"""Picks the compute device and builds the node context the vendored nodes want.

Enumeration, the FP16 decision and context creation are one group on purpose:
the context cannot be built without first knowing which device will be used and
whether it has a usable half-precision path, and both answers come from the same
device table.

Two things here are load-bearing:

* `device_objects()` is the single source of truth for the device list. The
  probe, the GUI's device picker and `make_context` all read it, so a device the
  GUI can offer is always a device the worker can select. It forces the heavy
  import because accelerator detection needs torch.
* `make_context` publishes the fresh ProgressController on `CTRL` and then
  re-applies an already-recorded cancel, so a stop that arrived before the
  context existed is not lost.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

from janai.worker import runtime
from janai.worker.control import CTRL
from janai.worker.events import log


def device_objects() -> list[dict]:
    """Available compute devices, CPU first, in backend order."""
    runtime.load_backend()
    out: list[dict] = []
    try:
        from accelerator_detection import get_accelerator_detector

        out.extend(
            {
                "value": d.device_string,
                "label": ("CPU" if d.type.value == "cpu" else f"{d.name} ({d.device_string})"),
                "kind": d.type.value,
                "index": d.index,
                "fp16": bool(d.supports_fp16),
                "bf16": bool(d.supports_bf16),
                "vram": int(d.memory_total or 0),
            }
            for d in get_accelerator_detector().available_devices
        )
    except Exception as exc:
        log(f"accelerator detection failed ({exc}); using torch directly", "warn")
        out.append(
            {
                "value": "cpu",
                "label": "CPU",
                "kind": "cpu",
                "index": 0,
                "fp16": False,
                "bf16": True,
                "vram": 0,
            }
        )
        try:
            if runtime.torch.cuda.is_available():
                for i in range(runtime.torch.cuda.device_count()):
                    props = runtime.torch.cuda.get_device_properties(i)
                    out.append(
                        {
                            "value": f"cuda:{i}",
                            "label": f"{props.name} (cuda:{i})",
                            "kind": "cuda",
                            "index": i,
                            "fp16": True,
                            "bf16": getattr(props, "major", 0) >= 8,
                            "vram": int(props.total_memory),
                        }
                    )
        except Exception:
            pass
    return out


def wants_fp16(value: Any) -> bool:
    """FP16 is the default: only an explicit false/off turns it off."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() not in ("0", "false", "no", "off", "fp32")
    return bool(value)


def device_supports_fp16(device: str) -> bool:
    """Whether this device has a half-precision path worth using."""
    if not device or device == "cpu":
        return False
    for d in device_objects():
        if str(d.get("value")) == device:
            return bool(d.get("fp16"))
    try:
        if device.startswith("cuda") and runtime.torch.cuda.is_available():
            index = int(device.split(":")[1]) if ":" in device else 0
            major, minor = runtime.torch.cuda.get_device_capability(index)
            # 5.3+ has real half math; everything since Pascal is fast at it
            return (major, minor) >= (5, 3)
    except Exception:
        pass
    return True


# --------------------------------------------------------------------------- #
# node context (copied from the original backend, minus the chain executor)
# --------------------------------------------------------------------------- #
def make_context(perf: dict):

    class ExecutorNodeContext(runtime.NodeContext):
        def __init__(self, progress, settings, storage_dir: Path) -> None:
            super().__init__()
            self.progress = progress
            self.__settings = settings
            self._storage_dir = storage_dir
            self.chain_cleanup_fns = set()
            self.node_cleanup_fns = set()

        @property
        def aborted(self) -> bool:
            return self.progress.aborted

        @property
        def paused(self) -> bool:
            time.sleep(0.001)
            return self.progress.paused

        def set_progress(self, progress: float) -> None:
            self.check_aborted()

        @property
        def settings(self):
            return self.__settings

        @property
        def storage_dir(self) -> Path:
            return self._storage_dir

        def add_cleanup(self, fn, after="chain") -> None:
            if after == "node":
                self.node_cleanup_fns.add(fn)
            else:
                self.chain_cleanup_fns.add(fn)

    device = str(perf.get("device") or "").strip()
    if not device:
        # no device asked for: use the best available one, like the probe's default_device
        device = next((d["value"] for d in device_objects() if d.get("value") != "cpu"), "cpu")
    use_cpu = device == "cpu"
    gpu_index = 0
    accel_index = 0
    if not use_cpu:
        gpus = [d for d in device_objects() if d.get("value") != "cpu"]
        match = [i for i, d in enumerate(gpus) if d.get("value") == device]
        if match:
            accel_index = match[0]
            gpu_index = int(gpus[accel_index].get("index") or 0)
        else:
            log(f"device {device} not present, falling back to the first GPU", "warn")

    want_fp16 = wants_fp16(perf.get("use_fp16", True))
    fp16 = want_fp16 and not use_cpu and device_supports_fp16(device)
    if want_fp16 and not fp16 and not use_cpu:
        log(f"{device} has no usable FP16 path, running in FP32", "warn")
    settings = runtime.SettingsParser(
        {
            "use_cpu": use_cpu,
            "use_fp16": fp16,
            "gpu_index": int(gpu_index),
            "accelerator_device_index": int(accel_index),
            "budget_limit": int(perf.get("budget_limit") or 0),
            "force_cache_wipe": bool(perf.get("force_cache_wipe", False)),
        }
    )
    storage = Path(tempfile.gettempdir()) / "janai-upscaler"
    storage.mkdir(parents=True, exist_ok=True)
    progress = runtime.ProgressController()
    CTRL.progress = progress
    if CTRL.cancelled:
        CTRL.cancel()
    return ExecutorNodeContext(progress, settings, storage), device, fp16
