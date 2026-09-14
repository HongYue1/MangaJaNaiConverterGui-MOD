"""Keeping an accelerator awake between jobs.

This is a subcommand of its own rather than part of a job because the GUI runs
it as a separate long-lived process: the whole point is to hold a context open
while *no* job is running, so the next upscale does not pay for context
creation and clock ramp.

torch is imported inside `do_hold` rather than at module level, and the model
backend is never imported at all, so importing this module stays free for the
CLI paths that hold nothing.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import Any

from janai.worker.events import emit


def resolve_hold_device(torch_mod: Any, device: str) -> str:
    """The accelerator to keep warm. Empty result means 'nothing to hold'."""
    want = (device or "").strip()
    if want == "cpu":
        return ""
    if want:
        return want
    try:
        if torch_mod.cuda.is_available() and torch_mod.cuda.device_count():
            return "cuda:0"
    except Exception:
        pass
    try:
        if hasattr(torch_mod, "xpu") and torch_mod.xpu.is_available():
            return "xpu:0"
    except Exception:
        pass
    return ""


def do_hold(device: str = "", interval: float = 15.0) -> int:
    """Keep a GPU context alive until stdin says stop.

    Creating a CUDA context is what actually wakes the card: without one the
    driver leaves it in a low power state (and on laptops the dGPU parks
    entirely), so the first upscale of a session pays several seconds of
    context creation and clock ramp before any real work starts. This holds a
    tiny tensor on the device and touches it every `interval` seconds, which
    costs a few MB of VRAM and effectively no power, but keeps the context
    resident. Only torch is imported, never the model backend.
    """
    try:
        import torch
    except Exception as exc:
        emit("hold", ok=False, device=device or "auto", error=f"torch unavailable: {exc}")
        return 1

    target = resolve_hold_device(torch, device)
    if not target:
        emit("hold", ok=False, device=device or "auto", error="no GPU available to hold")
        return 0

    try:
        dev = torch.device(target)
        pin = torch.zeros(256, 256, dtype=torch.float32, device=dev)
        name, reserved = target, 0
        if dev.type == "cuda":
            name = torch.cuda.get_device_name(dev)
            torch.cuda.synchronize(dev)
            reserved = int(torch.cuda.memory_reserved(dev))
        elif dev.type == "xpu":
            try:
                name = torch.xpu.get_device_name(dev)
            except Exception:
                pass
    except Exception as exc:
        emit("hold", ok=False, device=target, error=f"{type(exc).__name__}: {exc}")
        return 1

    emit("hold", ok=True, device=target, name=name, reserved=reserved, pid=os.getpid())

    stop = threading.Event()

    def watch() -> None:
        try:
            for line in sys.stdin:
                if line.strip().lower() in ("stop", "cancel", "quit", "exit"):
                    break
        except Exception:
            pass
        stop.set()

    threading.Thread(target=watch, name="hold-stdin", daemon=True).start()

    while not stop.wait(max(1.0, float(interval))):
        try:
            pin.add_(1.0)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
        except Exception as exc:
            emit("hold", ok=False, device=target, error=f"lost the device: {exc}")
            return 1

    try:
        del pin
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        elif dev.type == "xpu":
            torch.xpu.empty_cache()
    except Exception:
        pass
    emit("hold", ok=True, device=target, released=True)
    return 0
