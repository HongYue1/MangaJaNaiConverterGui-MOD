#!/usr/bin/env python
"""JaNaiUpscaler worker.

Runs the upscale pipeline out of process so the GUI stays responsive and never
imports torch. Three modes:

    worker.py --probe            print one JSON line describing this machine
    worker.py --job job.json     run a job, streaming JSONL events on stdout
    worker.py --hold [--device]  keep a GPU context awake until stdin says stop

Events (one JSON object per line, always with a "type" key):
    start     {total, out_dir, device, fp16, tile, format}
    progress  {i, total, path, sub_i, sub_n}
    file      {i, total, path, out, ms, bytes, w, h, gray, model, error}
    log       {level, message}
    done      {ok, processed, failed, skipped, cancelled, elapsed}
    probe     {...}
    hold      {ok, device, name, reserved, pid} / {ok, device, released}

Control commands arrive as lines on stdin: cancel, pause, resume (and stop,
which ends --hold).

The pipeline mirrors the original MangaJaNaiConverterGui backend
(read via libvips -> grayscale detection -> optional auto levels ->
spandrel/torch upscale -> dot-gain-aware final resize -> encode), minus the
chain/workflow machinery.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from queue import Queue
from typing import Any
from zipfile import ZIP_STORED, ZipFile

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]  # <app folder>/src, the import root
ROOT = SRC.parent  # the app folder itself

# Run as a script, sys.path[0] is this directory, so the package itself would
# not be importable. Put the import root in front before anything of ours.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from janai.core import paths as _paths, rules as _rules

# The runtime, the models, the backend source and the ICC profiles all live in
# backend/, unless janai.config.json points somewhere else. janai.core.paths
# works that out once, here.
PATHS = _paths.resolve(ROOT)
MODELS_DIR = PATHS.models_dir or (ROOT / "backend" / "models")

for _p in reversed(PATHS.import_paths()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Bundled command line tools (cjxl, djxl) win over anything already on PATH.
if PATHS.tools_dir:
    os.environ["PATH"] = f"{PATHS.tools_dir}{os.pathsep}{os.environ.get('PATH', '')}"

from janai.core.formats import (
    CONTAINERS,
    FORMATS,
    merged,
    packs_archive,
    save_kwargs,
)

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".jfif",
    ".webp",
    ".avif",
    ".jxl",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".heic",
    ".heif",
    ".ppm",
    ".pgm",
}
ARCHIVE_EXTS = {".zip", ".cbz", ".rar", ".cbr"}
MODEL_EXTS = {".pth", ".safetensors", ".pt", ".ckpt"}

_stdout_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
def emit(kind: str, **payload: Any) -> None:
    payload["type"] = kind
    try:
        line = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive
        line = json.dumps({"type": "log", "level": "warn", "message": "unserialisable event"})
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log(message: Any, level: str = "info") -> None:
    emit("log", level=level, message=str(message))


_warnings_installed = False


def install_warning_filters() -> None:
    """Silence the known-harmless library chatter; route the rest into the log.

    ``torch.meshgrid: in an upcoming release, it will be required to pass the
    indexing argument`` comes from inside the model architectures, which build
    their coordinate grids the old way. It says nothing about the job, cannot be
    fixed from here, and used to reach the GUI as a two line stderr dump with a
    site-packages path in it, once per model load. Everything else that warns is
    kept, but arrives as a single tidy debug line.
    """
    global _warnings_installed
    if _warnings_installed:
        return
    _warnings_installed = True
    warnings.filterwarnings("ignore", message=r".*torch\.meshgrid.*")
    warnings.filterwarnings("ignore", message=r".*indexing argument.*")
    warnings.filterwarnings("ignore", message=r".*__floordiv__ is deprecated.*")

    def show(message, category, filename, lineno, file=None, line=None) -> None:
        try:
            name = getattr(category, "__name__", str(category))
            log(f"{name}: {message} ({Path(str(filename)).name}:{lineno})", "debug")
        except Exception:
            pass

    warnings.showwarning = show


def hwc(image: Any) -> tuple[int, int, int]:
    """(height, width, channels) without needing the backend helpers imported."""
    if image.ndim == 2:
        return int(image.shape[0]), int(image.shape[1]), 1
    return int(image.shape[0]), int(image.shape[1]), int(image.shape[2])


class Control:
    """Cancel/pause flags fed by stdin, mirrored into the backend progress token."""

    def __init__(self) -> None:
        self._cancel = threading.Event()
        self._pause = threading.Event()
        self.progress = None  # backend ProgressController, attached later

    def start(self) -> None:
        threading.Thread(target=self._pump, name="stdin", daemon=True).start()

    def _pump(self) -> None:
        try:
            for raw in sys.stdin:
                cmd = raw.strip().lower()
                if not cmd:
                    continue
                if cmd in ("cancel", "abort", "stop"):
                    self.cancel()
                elif cmd == "pause":
                    self.pause()
                elif cmd == "resume":
                    self.resume()
        except Exception:
            pass

    def cancel(self) -> None:
        self._cancel.set()
        self._pause.clear()
        self._call("abort")
        self._call("resume")

    def pause(self) -> None:
        self._pause.set()
        self._call("pause")

    def resume(self) -> None:
        self._pause.clear()
        self._call("resume")

    def _call(self, name: str) -> None:
        fn = getattr(self.progress, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    def gate(self) -> None:
        """Block while paused; returns immediately when cancelled."""
        while self._pause.is_set() and not self._cancel.is_set():
            time.sleep(0.05)


CTRL = Control()


class Cancelled(Exception):
    pass


# --------------------------------------------------------------------------- #
# heavy backend, imported on demand
# --------------------------------------------------------------------------- #
np = None
cv2 = None
pyvips = None
torch = None
_PILImage = None
_ImageCms = None
_ImageFilter = None
_cx_resize = None
_ResizeFilter = None
_normalize = None
_to_uint8 = None
_get_h_w_c = None
_upscale_image_node = None
_load_model_node = None
_SettingsParser = None
_NodeContext = None
_ProgressController = None
TILE = {}
_heavy_loaded = False
_icc_pair = None
_icc_warned = False


def apply_perf_env(perf: dict) -> None:
    """Environment that must be set before libvips/torch are imported."""
    vc = int(perf.get("vips_concurrency") or 0)
    if vc > 0:
        os.environ["VIPS_CONCURRENCY"] = str(vc)
    tt = int(perf.get("torch_threads") or 0)
    if tt > 0:
        os.environ.setdefault("OMP_NUM_THREADS", str(tt))
        os.environ.setdefault("MKL_NUM_THREADS", str(tt))


def load_imaging(perf: dict | None = None) -> None:
    """Import only what pixels need: numpy, OpenCV, libvips, Pillow.

    A dry run and the encoder capability probe stop here, so neither pays for
    importing torch or for creating a device context.
    """
    global np, cv2, pyvips, _PILImage, _ImageCms, _ImageFilter
    if np is not None:
        return
    if perf:
        apply_perf_env(perf)
    install_warning_filters()

    import cv2 as cv2_mod
    import numpy
    import pyvips as pyvips_mod
    from PIL import Image as PILImage, ImageCms, ImageFilter

    np = numpy
    cv2 = cv2_mod
    pyvips = pyvips_mod
    _PILImage, _ImageCms, _ImageFilter = PILImage, ImageCms, ImageFilter


def load_backend(perf: dict | None = None) -> None:
    global torch, _cx_resize, _ResizeFilter, _normalize, _to_uint8, _get_h_w_c
    global _upscale_image_node, _load_model_node, _SettingsParser, _NodeContext
    global _ProgressController, TILE, _heavy_loaded
    load_imaging(perf)
    if _heavy_loaded:
        return

    import spandrel_custom
    import torch as torch_mod
    from api import NodeContext, SettingsParser
    from chainner_ext import ResizeFilter, resize as cx_resize
    from nodes.impl.image_utils import normalize, to_uint8
    from nodes.impl.upscale.auto_split_tiles import (
        ESTIMATE,
        MAX_TILE_SIZE,
        NO_TILING,
        TileSize,
    )
    from nodes.utils.utils import get_h_w_c
    from packages.chaiNNer_pytorch.pytorch.io.load_model import load_model_node
    from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import (
        upscale_image_node,
    )
    from progress_controller import ProgressController

    installer = getattr(spandrel_custom, "install", None)
    if callable(installer):
        try:
            installer()
        except Exception as exc:
            log(f"spandrel_custom.install() failed: {exc}", "warn")

    torch = torch_mod
    _cx_resize, _ResizeFilter = cx_resize, ResizeFilter
    _normalize, _to_uint8, _get_h_w_c = normalize, to_uint8, get_h_w_c
    _upscale_image_node, _load_model_node = upscale_image_node, load_model_node
    _SettingsParser, _NodeContext = SettingsParser, NodeContext
    _ProgressController = ProgressController
    TILE = {
        "estimate": ESTIMATE,
        "maximum": MAX_TILE_SIZE,
        "none": NO_TILING,
        "cls": TileSize,
    }
    _heavy_loaded = True


def apply_torch_perf(perf: dict) -> None:
    """Knobs that genuinely affect throughput. No placebo switches."""
    tt = int(perf.get("torch_threads") or 0)
    try:
        if tt > 0:
            torch.set_num_threads(tt)
    except Exception as exc:
        log(f"set_num_threads({tt}) ignored: {exc}", "warn")
    try:
        torch.backends.cudnn.benchmark = bool(perf.get("cudnn_benchmark", False))
        if bool(perf.get("allow_tf32", True)):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.set_grad_enabled(False)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# node context (copied from the original backend, minus the chain executor)
# --------------------------------------------------------------------------- #
def make_context(perf: dict):

    class ExecutorNodeContext(_NodeContext):
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
    settings = _SettingsParser(
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
    progress = _ProgressController()
    CTRL.progress = progress
    if CTRL.cancelled:
        CTRL.cancel()
    return ExecutorNodeContext(progress, settings, storage), device, fp16


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
        if device.startswith("cuda") and torch.cuda.is_available():
            index = int(device.split(":")[1]) if ":" in device else 0
            major, minor = torch.cuda.get_device_capability(index)
            # 5.3+ has real half math; everything since Pascal is fast at it
            return (major, minor) >= (5, 3)
    except Exception:
        pass
    return True


def parse_tile(value: Any) -> tuple[str, int]:
    """The tile setting as (mode, size): auto, maximum, none or a fixed size."""
    s = str(value if value is not None else "").strip().lower()
    if s in ("", "auto", "estimate", "auto (estimate)"):
        return "auto", 0
    if s in ("max", "maximum"):
        return "maximum", 0
    if s in ("none", "no tiling", "off", "0"):
        return "none", 0
    if s.isdecimal():
        return "fixed", max(16, int(s))
    return "auto", 0


class TilePlanner:
    """Works out the tile size handed to the upscaler for each image.

    The estimator inherited from chaiNNer budgets 60% of *total* VRAM, ignores
    what other processes already hold, and then rounds the tile down to a power
    of two, which throws away up to half of the usable area: a 700 px budget
    becomes 512 px, so a 1400 px page is cut into four tiles instead of two.
    Every extra tile is another forward pass plus blending, and a budget based
    on total memory is what makes a busy 6 GB laptop GPU fall into the
    out-of-memory retry path halfway through a chapter.

    This planner instead:

    * budgets memory that is actually free, minus the weights and a fixed
      headroom for the context and the allocator,
    * rounds to a multiple of 32 rather than a power of two,
    * never returns a tile bigger than the image, so a page that fits is done
      in one pass with no blending at all,
    * measures what the first image really allocated and calibrates the
      per-pixel cost from it, so later pages are sized for the model in use
      instead of for a constant,
    * only ever revises that cost upwards, and shrinks the tile for the rest
      of the run only on real memory pressure - an allocator retry, or a peak
      that came within a hair of the memory actually free. A tile that fitted
      is a tile worth keeping: on a 6 GB laptop card, stepping 1248px down to
      864px costs about 9% throughput on every page that follows.
    """

    ALIGN = 32
    MIN_TILE = 128
    OVERLAP = 16  # auto_split's default padding per tile edge
    HEADROOM = 256 * 1024**2
    SAFETY = 0.85
    MARGIN = 1.10  # applied to a measured cost before reusing it

    def __init__(
        self, mode: str, fixed: int, device: str, fp16: bool, budget_limit_gib: int = 0
    ) -> None:
        self.mode = mode
        self.fixed = int(fixed or 0)
        self.device = str(device or "")
        self.fp16 = bool(fp16)
        self.budget_limit = max(0, int(budget_limit_gib or 0)) * 1024**3
        self.elem = 2 if fp16 else 4
        self.last = 0
        self._model_bytes: dict[int, int] = {}
        self._per_px: dict[int, float] = {}
        self._cap: dict[int, int] = {}
        self._pending: tuple[int, int, int, int, int, int] | None = None
        self._retries = 0
        self._pressure: dict[int, int] = {}
        self._good: dict[int, int] = {}

    # -- memory ------------------------------------------------------------ #
    def _free_bytes(self) -> int:
        # Deliberately the driver's number, not the driver's number plus
        # torch's reusable cache. Counting the cache made the planner size a
        # 1280px tile on a 6 GB card, which then hit a cudaMalloc retry on
        # every page: 97s per page against 74s at 1248px. Retries cost far
        # more than the tile gains.
        if self.device.startswith("cuda"):
            try:
                index = int(self.device.split(":")[1]) if ":" in self.device else 0
                free, total = torch.cuda.mem_get_info(index)
                return int(min(free, total))
            except Exception:
                return 0
        if self.device.startswith("xpu"):
            try:
                free, _total = torch.xpu.mem_get_info(self.device)  # type: ignore[attr-defined]
                return int(free)
            except Exception:
                return 0
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:
            return 0

    def _budget(self, model_bytes: int) -> int:
        free = self._free_bytes()
        if free <= 0:
            return 0
        if self.budget_limit:
            free = min(free, self.budget_limit)
        return max(0, int((free - self.HEADROOM - model_bytes) * self.SAFETY))

    def _alloc_retries(self) -> int:
        """cudaMalloc retries so far: the allocator's own sign of pressure."""
        if not self.device.startswith("cuda"):
            return 0
        try:
            return int(torch.cuda.memory_stats(self.device).get("num_alloc_retries", 0))
        except Exception:
            return 0

    def model_bytes(self, model: Any) -> int:
        key = id(model)
        got = self._model_bytes.get(key)
        if got is None:
            try:
                params = sum(p.numel() for p in model.model.parameters())
            except Exception:
                params = 0
            got = int(params) * self.elem
            self._model_bytes[key] = got
        return got

    # -- planning ---------------------------------------------------------- #
    def _align(self, size: float) -> int:
        return max(self.MIN_TILE, (int(size) // self.ALIGN) * self.ALIGN)

    def _tile_pixels(self, w: int, h: int, tile: int, c: int) -> int:
        """Input pixels in the largest tile the splitter will actually cut."""
        count_x = max(1, math.ceil(w / tile))
        count_y = max(1, math.ceil(h / tile))
        size_x = min(w + 2 * self.OVERLAP, math.ceil(w / count_x) + 2 * self.OVERLAP)
        size_y = min(h + 2 * self.OVERLAP, math.ceil(h / count_y) + 2 * self.OVERLAP)
        return max(1, size_x * size_y * max(1, c))

    def choose(self, model: Any, image: Any):
        """A TileSize for this image, in the backend's own encoding."""
        h, w, c = hwc(image)
        self._pending = None
        if self.mode == "none":
            self.last = -1
            return TILE["none"]
        if self.mode == "maximum":
            self.last = -2
            return TILE["maximum"]
        if self.mode == "fixed":
            self.last = self.fixed
            return TILE["cls"](self.fixed)
        if model is None:
            self.last = 0
            return TILE["estimate"]

        key = id(model)
        model_bytes = self.model_bytes(model)
        budget = self._budget(model_bytes)
        if budget <= 0:
            # Page 2 and later usually land here: torch's caching allocator
            # still holds page 1's blocks, so the driver reports almost
            # nothing free even though that memory will be handed straight
            # back. Repeat the tile that already worked instead of falling
            # back to the backend's blind guess.
            proven = self._good.get(key, 0)
            if proven:
                limit = self._cap.get(key) or proven
                return self._arm(key, model, w, h, c, min(proven, limit), 0, 0)
            self.last = 0
            return TILE["estimate"]

        per_px = self._per_px.get(key)
        if per_px is None:
            # chaiNNer's calibration, used until a real measurement replaces it
            per_px = (model_bytes / (1024 * 52)) * max(1, c) * self.elem
        tile = self._align(math.sqrt(budget / max(per_px, 1e-6)))
        free = self._free_bytes()
        proven = self._good.get(key, 0)
        if proven > tile:
            # Page 2 and later: the driver still counts page 1's blocks as
            # used, because torch's allocator holds them to hand straight
            # back. Budgeting from that number alone quietly shrank a proven
            # 1952px tile to 864px, costing 23s a page against 17s. Repeat
            # the size that already finished instead - and only that size,
            # never larger: sizing from free memory plus the cache is what
            # made a 1280px tile retry on every page.
            tile = proven
            # Peak-against-free is meaningless once the cache is serving the
            # allocation, so leave real allocator retries as the only signal.
            free = 0
        cap = self._cap.get(key)
        if cap:
            tile = min(tile, cap)
        return self._arm(key, model, w, h, c, tile, budget, free)

    def _arm(
        self, key: int, model: Any, w: int, h: int, c: int, tile: int, budget: int, free: int
    ) -> Any:
        """Note what is about to be attempted, then hand the tile over."""
        # one tile that covers the page is the fastest case there is
        tile = max(self.MIN_TILE, min(tile, self._align(max(w, h) + self.ALIGN)))
        self.last = tile
        # The whole-page input and output tensors are allocated whatever the
        # tile size is. Keeping them out of the per-pixel cost stops a large
        # page from inflating the estimate and shrinking every later tile.
        try:
            model_scale = max(1, int(getattr(model, "scale", 1) or 1))
        except Exception:
            model_scale = 1
        page_bytes = int(w * h * max(1, c) * self.elem * (1 + model_scale**2))
        self._pending = (key, self._tile_pixels(w, h, tile, c), tile, budget, page_bytes, free)
        return TILE["cls"](tile)

    # -- measurement ------------------------------------------------------- #
    def before(self) -> None:
        self._retries = self._alloc_retries()
        if self._pending and self.device.startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

    def after(self) -> None:
        """Calibrate the per-pixel cost from what the last upscale allocated."""
        pending, self._pending = self._pending, None
        if pending is None or not self.device.startswith("cuda"):
            return
        key, pixels, tile, _budget, page_bytes, free = pending
        if pixels < 256 * 256:
            return  # too small to measure anything useful
        try:
            peak = int(torch.cuda.max_memory_allocated(self.device))
        except Exception:
            return
        model_bytes = self._model_bytes.get(key, 0)
        activations = peak - model_bytes - page_bytes
        if activations <= 0:
            activations = peak - model_bytes
        if activations <= 0:
            return
        measured = (activations / pixels) * self.MARGIN
        previous = self._per_px.get(key)
        self._per_px[key] = measured if previous is None else max(previous, measured)
        if previous is None:
            log(f"tile cost calibrated at {tile}px, peak {peak // 1024**2} MiB", "debug")
        # this tile finished, so it is the one to repeat when the driver stops
        # reporting usable free memory on later pages
        self._good[key] = max(self._good.get(key, 0), tile)
        # Shrink only when memory was genuinely tight. Going over the safety
        # budget does not qualify: that budget is deliberately pessimistic, and
        # a tile that finished without the allocator stalling is one worth
        # keeping, since every step down is paid on every later page.
        retried = max(0, self._alloc_retries() - self._retries)
        near_limit = bool(free) and peak > int(free * 0.92)
        if not (retried or near_limit):
            # A page that finished cleanly clears the slate: pressure has to be
            # consecutive to mean anything, or one fragmented page early in a
            # chapter ratchets the tile down for every page after it.
            self._pressure.pop(key, None)
            return
        strikes = self._pressure.get(key, 0) + max(1, retried)
        self._pressure[key] = strikes
        # Measured on a 6 GB card, 1920x1080 -> 7680x4320 through the x4 FDAT
        # model: the proven 1952px tile holds 18.1s a page even while the
        # allocator retries once per page, against 20.9s at 1632px, 21.0s at
        # 1376px and 20.8s at 1152px. One retry a page is far cheaper than a
        # permanently smaller tile, and a genuine miss is already caught and
        # re-tiled for that page by auto_split. So only pressure that repeats
        # inside a single page earns a lasting step down.
        if retried >= 2:
            why = "repeated allocator retries in one page"
        elif not retried and strikes >= 2:
            why = "peaks near free memory"
        else:
            log("memory was tight, keeping the proven tile", "debug")
            return
        capped = max(self.MIN_TILE, self._align(tile * 0.85))
        if capped < (self._cap.get(key) or tile):
            self._cap[key] = capped
            log(f"tile capped at {capped}px after {why}", "debug")


# --------------------------------------------------------------------------- #
# device + capability probe
# --------------------------------------------------------------------------- #
def device_objects() -> list[dict]:
    """Available compute devices, CPU first, in backend order."""
    load_backend()
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
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
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


def vips_has(op: str) -> bool:
    try:
        return bool(pyvips.type_find("VipsOperation", op))
    except Exception:
        return False


def find_tool(name: str) -> str:
    """A tool bundled in the tools folder, else whatever PATH offers."""
    if PATHS.tools_dir:
        for cand in (PATHS.tools_dir / f"{name}.exe", PATHS.tools_dir / name):
            if cand.is_file():
                return str(cand)
    return shutil.which(name) or ""


def find_cjxl() -> str:
    return find_tool("cjxl")


def find_djxl() -> str:
    return find_tool("djxl")


def pillow_jxl_available() -> bool:
    try:
        import pillow_jxl  # noqa: F401

        return True
    except Exception:
        return False


def encode_capabilities() -> dict:
    """What this install can really write, checked by encoding a 1x1 image."""
    caps: dict[str, dict] = {}
    probe = np.zeros((1, 1), dtype=np.uint8)
    for fid, spec in FORMATS.items():
        entry = {"ok": False, "via": "", "reason": ""}
        if vips_has(spec.probe):
            try:
                vips_from_array(probe).write_to_buffer(spec.suffix)
                entry.update(ok=True, via="libvips")
            except Exception as exc:
                entry["reason"] = f"libvips {spec.probe}: {exc}"
        else:
            entry["reason"] = f"libvips has no {spec.probe}"
        if not entry["ok"] and fid == "jxl":
            if pillow_jxl_available():
                entry.update(ok=True, via="pillow-jxl", reason="")
            elif find_cjxl():
                entry.update(ok=True, via="cjxl", reason="")
        caps[fid] = entry
    return caps


def list_models(models_dir: Path) -> list[dict]:
    if not models_dir.is_dir():
        return []
    return [
        model_info(p)
        for p in sorted(models_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in MODEL_EXTS
    ]


def model_info(p: Path) -> dict:
    name = p.stem
    low = name.lower()
    m = re.match(r"(\d+)x[_\-]", name)
    scale = int(m.group(1)) if m else 0
    hm = re.search(r"(\d{3,4})p", low)
    height = int(hm.group(1)) if hm else 0
    if "mangajanai" in low:
        family = "manga"
    elif "illustrationjanai" in low:
        family = "illustration"
    else:
        family = "other"
    return {
        "name": p.name,
        "path": str(p),
        "scale": scale,
        "height": height,
        "family": family,
        "denoise": "denoise" in low,
        "detail": "detail" in low,
        "fp16": "fp16" in low,
    }


def do_probe(models_dir: Path) -> int:
    info: dict[str, Any] = {
        "root": str(ROOT),
        "models_dir": str(models_dir),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "paths": PATHS.as_dict(),
        "ok": True,
        "errors": [],
    }
    try:
        load_backend()
    except Exception as exc:
        info["ok"] = False
        info["errors"].append(f"backend import failed: {exc}")
        emit("probe", **info)
        return 1

    try:
        info["libvips"] = ".".join(str(pyvips.version(i)) for i in range(3))
    except Exception:
        info["libvips"] = "?"
    info["pyvips"] = getattr(pyvips, "__version__", "?")
    info["torch"] = getattr(torch, "__version__", "?")
    try:
        info["cuda"] = torch.version.cuda or ""
    except Exception:
        info["cuda"] = ""
    try:
        info["numpy"] = np.__version__
        info["opencv"] = cv2.__version__
    except Exception:
        pass

    info["devices"] = device_objects()
    gpu = next((d for d in info["devices"] if d["value"] != "cpu"), None)
    info["default_device"] = gpu["value"] if gpu else "cpu"
    info["formats"] = encode_capabilities()
    info["read_jxl"] = vips_has("jxlload") or bool(find_djxl())
    info["read_heif"] = vips_has("heifload")
    info["models"] = list_models(models_dir)
    info["icc"] = PATHS.icc() is not None
    info["tools"] = {name: find_tool(name) for name in ("cjxl", "djxl")}
    try:
        import rarfile

        info["rar"] = bool(
            rarfile.tool_setup(sevenzip=True, sevenzip2=True, unrar=True, bsdtar=True)
        )
    except Exception:
        info["rar"] = False
    info["cpu_count"] = os.cpu_count() or 1
    emit("probe", **info)
    return 0


# --------------------------------------------------------------------------- #
# image pipeline
# --------------------------------------------------------------------------- #
def vips_from_array(arr):
    a = np.ascontiguousarray(arr)
    if a.ndim == 2:
        h, w = a.shape
        bands = 1
    else:
        h, w, bands = a.shape
    img = pyvips.Image.new_from_memory(a.tobytes(), w, h, bands, "uchar")
    try:
        img = img.copy(interpretation="b-w" if bands == 1 else "srgb")
    except Exception:
        pass
    return img


def read_image(path: Path):
    if path.suffix.lower() == ".jxl" and not vips_has("jxlload"):
        return read_jxl_djxl(path)
    return (
        pyvips.Image.new_from_file(str(path), access="sequential", fail=True)
        .icc_transform("srgb")
        .numpy()
    )


def read_image_bytes(data: bytes, name: str = ""):
    if name.lower().endswith(".jxl") and not vips_has("jxlload"):
        with tempfile.TemporaryDirectory(prefix="janai-jxl-") as td:
            src = Path(td) / "in.jxl"
            src.write_bytes(data)
            return read_jxl_djxl(src)
    return pyvips.Image.new_from_buffer(data, "", access="sequential").icc_transform("srgb").numpy()


def read_jxl_djxl(path: Path):
    """Decode JPEG XL through djxl, for a libvips built without jxlload."""
    exe = find_djxl()
    if not exe:
        raise RuntimeError(
            "this libvips cannot read JPEG XL; put djxl.exe in the tools folder "
            "or on PATH to read .jxl inputs"
        )
    with tempfile.TemporaryDirectory(prefix="janai-jxl-") as td:
        dst = Path(td) / "decoded.png"
        run = subprocess.run(
            [exe, str(path), str(dst)], check=False, capture_output=True, creationflags=no_window()
        )
        if run.returncode != 0 or not dst.exists():
            raise RuntimeError(f"djxl failed: {run.stderr.decode('utf-8', 'replace')[:300]}")
        return (
            pyvips.Image.new_from_file(str(dst), access="sequential", fail=True)
            .icc_transform("srgb")
            .numpy()
        )


def icc_transforms():
    """(dotgain20 -> gamma1, gamma1 -> dotgain20) or None when profiles are missing."""
    global _icc_pair, _icc_warned
    if _icc_pair is not None:
        return _icc_pair
    gamma = PATHS.icc("Custom Gray Gamma 1.0.icc")
    dot = PATHS.icc("Dot Gain 20%.icc")
    if not (gamma and dot):
        if not _icc_warned:
            log("grayscale ICC profiles missing, using Lanczos for the final resize", "warn")
            _icc_warned = True
        _icc_pair = ()
        return _icc_pair
    g = _ImageCms.getOpenProfile(str(gamma))
    d = _ImageCms.getOpenProfile(str(dot))
    _icc_pair = (
        _ImageCms.buildTransformFromOpenProfiles(d, g, "L", "L"),
        _ImageCms.buildTransformFromOpenProfiles(g, d, "L", "L"),
    )
    return _icc_pair


def standard_resize(image, new_size: tuple[int, int]):
    out = image.astype(np.float32) / 255.0
    out = _cx_resize(out, new_size, _ResizeFilter.Lanczos, False)
    out = (out * 255).round().astype(np.uint8)
    if _get_h_w_c(image)[2] == 1 and out.ndim == 3:
        out = np.squeeze(out, axis=-1)
    return out


def dotgain20_resize(image, new_size: tuple[int, int]):
    pair = icc_transforms()
    if not pair:
        return standard_resize(image, new_size)
    to_gamma, to_dotgain = pair
    h = _get_h_w_c(image)[0]
    size_ratio = h / max(1, new_size[1])
    blur = (1 / size_ratio - 1) / 3.5
    if blur >= 0.1:
        blur = min(blur, 250)
    pil = _PILImage.fromarray(image, mode="L")
    pil = pil.filter(_ImageFilter.GaussianBlur(radius=blur))
    pil = _ImageCms.applyTransform(pil, to_gamma, False)
    out = np.array(pil).astype(np.float32) / 255.0
    out = _cx_resize(out, new_size, _ResizeFilter.CubicCatrom, False)
    out = (out * 255).round().astype(np.uint8)
    pil = _PILImage.fromarray(out[:, :, 0] if out.ndim == 3 else out, mode="L")
    return np.array(_ImageCms.applyTransform(pil, to_dotgain, False))


def image_resize(image, new_size: tuple[int, int], is_gray: bool):
    if is_gray and image.ndim == 2:
        return dotgain20_resize(image, new_size)
    return standard_resize(image, new_size)


GRAY_SAMPLE = 768  # long edge of the copy the grayscale test looks at


def gray_stats(image, threshold: float, colour_percent: float = 0.25) -> tuple[bool, float, float]:
    """(is_grayscale, mean colour excess, percent of clearly coloured pixels).

    Three things were wrong with the inherited test:

    * it averaged over every pixel of a full size page, which is slow on an
      8000 px scan and, worse, blind to a small but unmistakably coloured area:
      a title logo or one colour panel averages away to nothing, the page is
      called gray, and the colour is then squashed out of it for good;
    * it summed the three channel differences in uint8, so a genuinely colourful
      pixel could wrap past 255 back down to a small number and count as gray;
    * scanner and JPEG chroma noise pushed clean gray pages over the threshold,
      which is what made the setting feel arbitrary.

    So: measure on an area-averaged sample (fast, and averaging is what removes
    the chroma noise), sum in int32, and refuse to call a page gray when a
    non-trivial share of its pixels are properly coloured. The mean-excess
    metric and its ``threshold / 12`` comparison are kept, so an existing
    threshold still means the same thing.
    """
    h, w, c = hwc(image)
    if c == 1:
        return True, 0.0, 0.0

    sample = image[:, :, :3]
    long_edge = max(h, w)
    if long_edge > GRAY_SAMPLE:
        factor = GRAY_SAMPLE / float(long_edge)
        sample = cv2.resize(
            sample, (max(1, int(w * factor)), max(1, int(h * factor))), interpolation=cv2.INTER_AREA
        )

    b, g, r = cv2.split(sample)
    t = int(max(0, min(255, round(threshold))))
    excess = (
        cv2.subtract(cv2.absdiff(r, g), t).astype(np.int32)
        + cv2.subtract(cv2.absdiff(r, b), t).astype(np.int32)
        + cv2.subtract(cv2.absdiff(g, b), t).astype(np.int32)
    )
    high = cv2.max(cv2.max(r, g), b)
    low = cv2.min(cv2.min(r, g), b)
    keep = ~np.logical_or(high == 0, low == 255)  # skip pure black / pure white
    kept = int(np.count_nonzero(keep))
    if kept == 0:
        return False, 0.0, 0.0

    mean_excess = float(excess[keep].sum()) / (kept * 3)
    spread = cv2.subtract(high, low)
    coloured = int(np.count_nonzero(np.logical_and(keep, spread > max(8, 2 * t))))
    percent = 100.0 * coloured / kept
    is_gray = mean_excess <= threshold / 12 and percent <= max(0.0, colour_percent)
    return is_gray, mean_excess, percent


def to_grayscale(image):
    c = _get_h_w_c(image)[2]
    if c == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if c == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return image


def auto_levels(image):
    """Black/white point stretch based on histogram peaks (grayscale only)."""
    pil = _PILImage.fromarray(image).convert("L")
    hist = pil.histogram()

    black, peak = 0, hist[0]
    for i in range(1, 31):
        if hist[i] > peak:
            peak, black = hist[i], i
    run = 0
    for i in range(31, 256):
        if hist[i] > peak:
            run, peak, black = 0, hist[i], i
        elif hist[i] < peak:
            run += 1
            if run > 1:
                break

    white, peak = 255, hist[255]
    for i in range(254, 224, -1):
        if hist[i] > peak:
            peak, white = hist[i], i
    run = 0
    for i in range(223, -1, -1):
        if hist[i] > peak:
            run, peak, white = 0, hist[i], i
        elif hist[i] < peak:
            run += 1
            if run > 1:
                break

    if white <= black:
        return _normalize(image)
    arr = np.array(pil).astype("float32")
    arr = np.maximum(arr - black, 0) / (white - black)
    return np.clip(arr, 0, 1)


def final_resize(image, scale: float, width: int, height: int, ow: int, oh: int, is_gray: bool):
    if height and width:
        if height / oh < width / ow:
            width = 0
        else:
            height = 0
    h, w = _get_h_w_c(image)[:2]
    if height:
        if h != height:
            return image_resize(image, (round(w * height / h), height), is_gray)
    elif width:
        if w != width:
            return image_resize(image, (width, round(h * width / w)), is_gray)
    else:
        target = round(oh * scale)
        if h != target:
            return image_resize(image, (round(w * target / h), target), is_gray)
    return image


def is_long_strip(w: int, h: int, max_side: int, min_aspect: float, min_pixels: int) -> bool:
    """True for webtoon-style mega strips that may be passed through untouched.

    Adopted from the other fork's SkipLargeLong* settings, with one deliberate
    change: the pixel clause also requires the long aspect, so an ordinary
    large page (a 4000x2400 spread, say) is never mistaken for a strip.
    """
    if w <= 0 or h <= 0:
        return False
    aspect = max(w, h) / max(1, min(w, h))
    if aspect < min_aspect:
        return False
    return max(w, h) >= max_side or (w * h) >= min_pixels


class ModelCache:
    def __init__(self, ctx, want_scale: float = 0.0) -> None:
        self.ctx = ctx
        # Target factor for plain scale runs, 0 when the target is a width,
        # height or display fit (there the factor depends on each page).
        self.want_scale = float(want_scale or 0.0)
        self._cache: dict[str, Any] = {}

    def get(self, path: str):
        got = self._cache.get(path)
        if got is None:
            loaded = _load_model_node(self.ctx, Path(path))
            got = loaded[0] if isinstance(loaded, tuple) else loaded
            self._cache[path] = got
            scale = getattr(got, "scale", None)
            log(f"loaded model {Path(path).name} (x{scale or '?'})")
            # Read off the weights rather than the filename: this is the
            # authoritative version of the warning the GUI shows from the name.
            if scale and self.want_scale > 0 and abs(float(scale) - self.want_scale) > 0.01:
                log(
                    f"{Path(path).name} is x{scale} but the target is "
                    f"{self.want_scale:g}x, so every page gets resampled to the "
                    f"target and loses detail",
                    "warn",
                )
        return got


# Upstream's height bands and the colour defaults live with the rules engine,
# so the shipped default working set and this fallback picker can never drift
# apart. See janai/core/rules.py.
GRAY_HEIGHT_BANDS = _rules.GRAY_HEIGHT_BANDS
GRAY_TOP_BUCKET = _rules.GRAY_TOP_BUCKET
COLOUR_DEFAULTS = _rules.COLOUR_DEFAULTS

_auto_pick_logged: set[tuple] = set()


def gray_bucket(src_h: int) -> int:
    """The MangaJaNai page-height bucket (1200p, 1300p, ...) for a source height."""
    for limit, bucket in GRAY_HEIGHT_BANDS:
        if src_h <= limit:
            return bucket
    return GRAY_TOP_BUCKET


def choose_model(models: list[dict], is_gray: bool, src_h: int, target_scale: float) -> dict | None:
    """Resolve "auto" to an installed model, the way the original fork did.

    Gray pages follow upstream's height bands: a 1920px page gets the 1920p
    model, and the 2x or 4x flavour is chosen from the target scale. Colour
    pages get the current IllustrationJaNai denoise default for that scale.
    Both fall back to the nearest installed match rather than failing.
    """
    if not models:
        return None
    want_scale = 2 if target_scale <= 2.0 else 4
    want_family = "manga" if is_gray else "illustration"
    pool = [m for m in models if m["family"] == want_family] or models
    scaled = [m for m in pool if m["scale"] == want_scale] or pool

    if is_gray:
        bucket = gray_bucket(src_h)
        tagged = [m for m in scaled if m["height"]]
        if tagged:
            pick = min(tagged, key=lambda m: (abs(m["height"] - bucket), m["height"], m["name"]))
            why = (
                f"{bucket}p band for a {src_h}px page"
                if pick["height"] == bucket
                else f"{bucket}p band for a {src_h}px page, nearest installed"
            )
        else:
            pick = min(scaled, key=lambda m: m["name"])
            why = "no height-tagged MangaJaNai model installed"
    else:
        wanted = COLOUR_DEFAULTS.get(want_scale, "")
        pick = next((m for m in scaled if m["name"] == wanted), None)
        if pick is not None:
            why = f"default x{want_scale} colour model"
        else:
            denoise = [m for m in scaled if m["denoise"]]
            pick = min(denoise or scaled, key=lambda m: m["name"])
            why = f"{wanted} not installed, closest match" if wanted else "closest installed match"

    key = (is_gray, want_scale, pick["name"], why)
    if key not in _auto_pick_logged:
        _auto_pick_logged.add(key)
        log(f"auto {'gray' if is_gray else 'colour'} model -> {pick['name']} ({why})")
    return pick


def upscale_array(ctx, image, model, tile):
    if model is None:
        return image
    result = _upscale_image_node(ctx, image, model, False, 0, tile, 256, False)
    if hwc(image)[2] == 1 and result.ndim == 3:
        result = np.squeeze(result, axis=-1)
    return result


# --------------------------------------------------------------------------- #
# encoding
# --------------------------------------------------------------------------- #
def encode(image, fid: str, opts: dict, caps: dict) -> bytes:
    spec = FORMATS[fid]
    cap = caps.get(fid, {})
    via = cap.get("via") or "libvips"
    if via == "libvips":
        return encode_vips(image, fid, opts)
    if fid == "jxl" and via == "pillow-jxl":
        return encode_jxl_pillow(image, opts)
    if fid == "jxl" and via == "cjxl":
        return encode_jxl_cjxl(image, opts)
    raise RuntimeError(f"no encoder available for {spec.label}")


def encode_vips(image, fid: str, opts: dict) -> bytes:
    spec = FORMATS[fid]
    img = vips_from_array(image)
    if img.bands == 4 and fid == "jpeg":
        img = img.flatten(background=255)
    kwargs = save_kwargs(fid, opts)
    try:
        return img.write_to_buffer(spec.suffix, **kwargs)
    except Exception as exc:
        keep = {
            k: v for k, v in kwargs.items() if k in ("Q", "lossless", "compression", "distance")
        }
        log(f"{spec.label}: {exc}; retrying with {keep or 'defaults'}", "warn")
        return img.write_to_buffer(spec.suffix, **keep)


def _pil_image(image):
    if image.ndim == 2:
        return _PILImage.fromarray(image, mode="L")
    if image.shape[2] == 4:
        return _PILImage.fromarray(image, mode="RGBA")
    return _PILImage.fromarray(image[:, :, :3], mode="RGB")


def encode_jxl_pillow(image, opts: dict) -> bytes:
    import pillow_jxl  # noqa: F401

    vals = merged("jxl", opts)
    kwargs: dict[str, Any] = {"effort": int(vals["effort"])}
    if vals.get("lossless"):
        kwargs["lossless"] = True
    else:
        if vals.get("rate_mode") == "distance":
            log("pillow-jxl has no distance control; using the quality value instead", "warn")
        kwargs["quality"] = int(vals["Q"])
    buf = BytesIO()
    _pil_image(image).save(buf, format="JXL", **kwargs)
    return buf.getvalue()


def encode_jxl_cjxl(image, opts: dict) -> bytes:
    exe = find_cjxl()
    if not exe:
        raise RuntimeError("cjxl not found")
    vals = merged("jxl", opts)
    with tempfile.TemporaryDirectory(prefix="janai-jxl-") as td:
        src = Path(td) / "in.png"
        dst = Path(td) / "out.jxl"
        vips_from_array(image).write_to_file(str(src))
        cmd = [exe, str(src), str(dst), "-e", str(int(vals["effort"]))]
        if vals.get("lossless"):
            cmd += ["-d", "0", "--lossless_jpeg=0"]
        elif vals.get("rate_mode") == "distance":
            cmd += ["-d", str(float(vals["distance"]))]
        else:
            cmd += ["-q", str(int(vals["Q"]))]
        run = subprocess.run(cmd, check=False, capture_output=True, creationflags=no_window())
        if run.returncode != 0 or not dst.exists():
            raise RuntimeError(f"cjxl failed: {run.stderr.decode('utf-8', 'replace')[:300]}")
        return dst.read_bytes()


def no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


# --------------------------------------------------------------------------- #
# job plumbing
# --------------------------------------------------------------------------- #
def natural_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def gather_units(inp: dict) -> list[dict]:
    raw = Path(str(inp.get("path") or "")).expanduser()
    mode = str(inp.get("mode") or ("single" if raw.is_file() else "bulk"))
    include_archives = bool(inp.get("include_archives", True))
    recursive = bool(inp.get("recursive", True))
    units: list[dict] = []

    def add(p: Path, base: Path) -> None:
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            units.append({"path": p, "base": base, "kind": "image"})
        elif include_archives and ext in ARCHIVE_EXTS:
            units.append({"path": p, "base": base, "kind": "archive"})

    if raw.is_file():
        add(raw, raw.parent)
    elif raw.is_dir():
        it = raw.rglob("*") if recursive else raw.glob("*")
        for p in sorted(
            (q for q in it if q.is_file()),
            key=lambda q: (str(q.parent).lower(), natural_key(q.name)),
        ):
            add(p, raw)
    else:
        raise FileNotFoundError(f"input not found: {raw}")
    if mode == "single" and len(units) > 1:
        units = units[:1]
    return units


def format_name(pattern: str, src: Path, index: int, total: int) -> str:
    width = max(3, len(str(total)))
    out = pattern or "{name}"
    repl = {
        "{name}": src.stem,
        "{parent}": src.parent.name,
        "{index}": str(index),
        "{index0}": str(index).zfill(width),
    }
    for key, value in repl.items():
        out = out.replace(key, value)
    return re.sub(r'[<>:"/\\|?*]', "_", out).strip() or src.stem


def resolve_out(
    unit: dict, out_dir: Path, pattern: str, ext: str, keep_structure: bool, index: int, total: int
) -> Path:
    src: Path = unit["path"]
    base: Path = unit["base"]
    sub = Path()
    if keep_structure:
        try:
            sub = src.parent.relative_to(base)
        except ValueError:
            sub = Path()
    return out_dir / sub / (format_name(pattern, src, index, total) + ext)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    for n in range(1, 10000):
        cand = path.with_name(f"{stem} ({n}){suffix}")
        if not cand.exists():
            return cand
    return path


class WritePool:
    """Encodes and writes in the background so the GPU is not waiting on disk."""

    def __init__(self, workers: int) -> None:
        self.workers = max(1, int(workers or 1))
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="write")
        self.slots = threading.Semaphore(self.workers * 2)
        self.pending: set = set()
        self.lock = threading.Lock()

    def submit(self, fn, *args) -> None:
        self.slots.acquire()

        def task():
            try:
                fn(*args)
            finally:
                self.slots.release()

        fut = self.pool.submit(task)
        with self.lock:
            self.pending.add(fut)
        fut.add_done_callback(self._finished)

    def _finished(self, fut) -> None:
        with self.lock:
            self.pending.discard(fut)

    def drain(self) -> None:
        while True:
            with self.lock:
                futs = list(self.pending)
            if not futs:
                return
            for f in futs:
                try:
                    f.result()
                except Exception as exc:
                    log(f"write failed: {exc}", "error")

    def close(self) -> None:
        self.drain()
        self.pool.shutdown(wait=True)


def prefetch(units: list[dict], workers: int, reader):
    """Yield (unit, array_or_exception) in order, decoding ahead of the pipeline."""
    workers = max(1, int(workers or 1))
    if workers == 1:
        for u in units:
            if CTRL.cancelled:
                return
            try:
                yield u, reader(u)
            except Exception as exc:
                yield u, exc
        return

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="read")
    queue: Queue = Queue(maxsize=workers + 1)
    stop = threading.Event()

    def pump():
        try:
            for u in units:
                if stop.is_set() or CTRL.cancelled:
                    break
                queue.put((u, pool.submit(reader, u)))
        finally:
            queue.put(None)

    threading.Thread(target=pump, name="prefetch", daemon=True).start()
    try:
        while True:
            item = queue.get()
            if item is None:
                return
            unit, fut = item
            if CTRL.cancelled:
                stop.set()
                fut.cancel()
                return
            try:
                yield unit, fut.result()
            except Exception as exc:
                yield unit, exc
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)


# --------------------------------------------------------------------------- #
# job execution
# --------------------------------------------------------------------------- #
def safe_name(name: Any) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", str(name)).strip() or "output"


def relative_dir(unit: dict) -> Path:
    """Where this file sits inside the input folder."""
    try:
        return unit["path"].parent.relative_to(unit["base"])
    except ValueError:
        return Path()


def chapter_dest(unit: dict, out_dir: Path, keep_structure: bool) -> Path:
    """The .cbz that this file's own folder becomes."""
    rel = relative_dir(unit)
    name = rel.name or Path(str(unit["base"])).name or unit["path"].stem
    parent = out_dir / (rel.parent if keep_structure else Path())
    return parent / f"{safe_name(name)}.cbz"


def build_tasks(
    units: list[dict], out_dir: Path, keep_structure: bool, container_id: str
) -> list[dict]:
    """Group the units into the things this run will actually produce.

    Loose images stay in one run of consecutive units so the decoder can read
    ahead across the whole batch. With a cbz container they are grouped by the
    folder they came from instead, which is what turns "one folder per chapter"
    into one archive per chapter.
    """
    pack = packs_archive(container_id)
    tasks: list[dict] = []
    groups: dict[str, dict] = {}
    for index, unit in enumerate(units, 1):
        unit["index"] = index
        if unit["kind"] == "archive":
            tasks.append({"kind": "archive", "unit": unit})
            continue
        if not pack:
            if tasks and tasks[-1]["kind"] == "images":
                tasks[-1]["units"].append(unit)
            else:
                tasks.append({"kind": "images", "key": "", "dest": None, "units": [unit]})
            continue
        key = "" if container_id == "cbz_single" else relative_dir(unit).as_posix()
        group = groups.get(key)
        if group is None:
            if container_id == "cbz_single":
                base = Path(str(unit["base"]))
                stem = base.name if base.is_dir() else unit["path"].stem
                dest = out_dir / f"{safe_name(stem)}.cbz"
            else:
                dest = chapter_dest(unit, out_dir, keep_structure)
            group = {"kind": "bundle", "key": key, "dest": dest, "units": []}
            groups[key] = group
            tasks.append(group)
        group["units"].append(unit)
    return tasks


class BundleWriter:
    """Packs finished pages into one .cbz, off the GPU thread but in order.

    Encoding a page costs real time (JPEG XL especially), so it happens on a
    worker like every other write. A single worker keeps the pages in the order
    they were produced, which is the order a reader expects, and the archive is
    only moved into place once it is complete: an interrupted run leaves a
    .cbz.part behind rather than a half written chapter.
    """

    def __init__(self, encode_fn, on_page, on_fail, on_done) -> None:
        self.encode = encode_fn
        self.on_page = on_page
        self.on_fail = on_fail
        self.on_done = on_done
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pack")
        self.slots = threading.Semaphore(4)
        self.key: str | None = None
        self.dest: Path | None = None
        self.tmp: Path | None = None
        self.zf: Any = None
        self.entries = 0
        self.failed = 0
        self.started = 0.0
        self.futures: list = []

    def open(self, key: str, dest: Path) -> None:
        self.close()
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.key, self.dest = key, dest
        self.tmp = dest.with_suffix(".cbz.part")
        self.tmp.unlink(missing_ok=True)
        self.zf = ZipFile(self.tmp, "w", ZIP_STORED)
        self.entries = 0
        self.failed = 0
        self.started = time.perf_counter()

    def add(self, name: str, image, meta: dict) -> None:
        if self.zf is None:
            return
        self.slots.acquire()
        zf = self.zf

        def task() -> None:
            try:
                data = self.encode(image)
                zf.writestr(name, data)
                self.entries += 1
                self.on_page(meta, name, len(data))
            except Exception as exc:
                self.failed += 1
                self.on_fail(meta, name, f"{type(exc).__name__}: {exc}")
            finally:
                self.slots.release()

        self.futures.append(self.pool.submit(task))

    def drain(self) -> None:
        for fut in self.futures:
            try:
                fut.result()
            except Exception as exc:
                log(f"pack failed: {exc}", "error")
        self.futures.clear()

    def close(self, keep: bool = True) -> None:
        if self.zf is None:
            return
        self.drain()
        try:
            self.zf.close()
        finally:
            self.zf = None
        key, dest, tmp = self.key, self.dest, self.tmp
        entries, failed = self.entries, self.failed
        elapsed = time.perf_counter() - self.started
        self.key = self.dest = self.tmp = None
        if tmp is None or dest is None:
            return
        if not keep or entries == 0:
            tmp.unlink(missing_ok=True)
            return
        tmp.replace(dest)
        self.on_done(key or "", dest, entries, failed, elapsed)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)


def predict_size(ow: int, oh: int, t_scale: float, t_w: int, t_h: int) -> tuple[int, int]:
    """The size final_resize() lands on, without touching a pixel."""
    if t_w and t_h:  # fit: whichever side runs out first wins
        if t_h / max(1, oh) < t_w / max(1, ow):
            t_w = 0
        else:
            t_h = 0
    if t_h:
        return max(1, round(ow * t_h / max(1, oh))), t_h
    if t_w:
        return t_w, max(1, round(oh * t_w / max(1, ow)))
    return max(1, round(ow * t_scale)), max(1, round(oh * t_scale))


def probe_image(path: Path, threshold: float, colour_percent: float):
    """(width, height, is_gray, score, coloured percent) without a full decode.

    The size comes from the header and the colour verdict from a thumbnail,
    which libvips produces with shrink-on-load, so a dry run over a folder of
    8000 px scans costs a fraction of a second per page.
    """
    header = pyvips.Image.new_from_file(str(path), access="sequential")
    w, h = int(header.width), int(header.height)
    gray, score, coloured = None, 0.0, 0.0
    try:
        sample = pyvips.Image.thumbnail(str(path), GRAY_SAMPLE).numpy()
        if sample.ndim == 3 and sample.shape[2] >= 3:
            gray, score, coloured = gray_stats(sample[:, :, :3], threshold, colour_percent)
        else:
            gray = True
    except Exception:
        pass
    return w, h, gray, score, coloured


def dry_run(tasks: list[dict], total: int, cfg: dict) -> int:
    """Report exactly what a real run would produce, writing nothing at all."""
    out_dir: Path = cfg["out_dir"]
    ext: str = cfg["ext"]
    pattern: str = cfg["pattern"]
    overwrite: bool = cfg["overwrite"]
    counters = {"processed": 0, "failed": 0, "skipped": 0}
    clock = time.perf_counter()
    cancelled = False

    emit(
        "start",
        total=total,
        out_dir=str(out_dir),
        device=cfg["device"] or "auto",
        fp16=cfg["fp16"],
        tile=cfg["tile_label"],
        format=cfg["fid"],
        container=cfg["container"],
        models=cfg["model_count"],
        dry=True,
        bundles=sum(1 for t in tasks if t["kind"] == "bundle"),
    )

    for task in tasks:
        if CTRL.cancelled:
            cancelled = True
            break
        if task["kind"] == "archive":
            unit = task["unit"]
            src: Path = unit["path"]
            index = int(unit.get("index") or 0)
            dest = resolve_out(unit, out_dir, pattern, ".cbz", cfg["keep_structure"], index, total)
            entries = 0
            try:
                opener = open_archive(src)
                entries = len(opener[0]) if opener else 0
            except Exception as exc:
                log(f"{src.name}: {exc}", "warn")
            exists = dest.exists() and not overwrite
            counters["skipped" if exists else "processed"] += 1
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                dry=True,
                entries=entries,
                error="exists, would skip" if exists else None,
            )
            continue

        bundle = task["kind"] == "bundle"
        units: list[dict] = task["units"]
        dest_bundle: Path | None = task.get("dest")
        if bundle and dest_bundle is not None and dest_bundle.exists() and not overwrite:
            counters["skipped"] += len(units)
            emit(
                "file",
                i=int(units[0].get("index") or 0),
                total=total,
                path=str(units[0]["path"].parent),
                out=str(dest_bundle),
                dry=True,
                entries=len(units),
                error="exists, would skip",
            )
            continue
        if bundle and dest_bundle is not None:
            emit("bundle", out=str(dest_bundle), entries=len(units), planned=True, dry=True)

        seen: set[str] = set()
        for position, unit in enumerate(units, 1):
            if CTRL.cancelled:
                cancelled = True
                break
            CTRL.gate()
            src = unit["path"]
            index = int(unit.get("index") or position)
            emit(
                "progress",
                i=index,
                total=total,
                path=str(src),
                sub_i=position if bundle else 0,
                sub_n=len(units) if bundle else 0,
            )
            try:
                w, h, gray, score, coloured = probe_image(
                    src, cfg["threshold"], cfg["colour_percent"]
                )
            except Exception as exc:
                counters["failed"] += 1
                emit(
                    "file",
                    i=index,
                    total=total,
                    path=str(src),
                    dry=True,
                    error=f"cannot read: {type(exc).__name__}: {exc}",
                )
                continue
            pick = cfg["pick_model"](bool(gray), h, w)
            pw, ph = predict_size(w, h, cfg["t_scale"], cfg["t_w"], cfg["t_h"])
            entry = ""
            if bundle and dest_bundle is not None:
                entry = format_name(pattern, src, position, len(units)) + ext
                while entry.lower() in seen:
                    entry = f"{entry[: -len(ext)]}_{position}{ext}"
                seen.add(entry.lower())
                dest = dest_bundle
                exists = False
            else:
                dest = resolve_out(unit, out_dir, pattern, ext, cfg["keep_structure"], index, total)
                exists = dest.exists() and not overwrite
            counters["skipped" if exists else "processed"] += 1
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                entry=entry,
                w=pw,
                h=ph,
                src_w=w,
                src_h=h,
                gray=bool(gray),
                score=round(score, 2),
                colour=round(coloured, 2),
                model=(pick["name"] if pick else ""),
                dry=True,
                error="exists, would skip" if exists else None,
            )
        if cancelled:
            break

    emit(
        "done",
        ok=counters["failed"] == 0 and not cancelled,
        cancelled=cancelled,
        elapsed=round(time.perf_counter() - clock, 2),
        dry=True,
        **counters,
    )
    return 0 if counters["failed"] == 0 else 1


def run_job(job: dict) -> int:
    inp = job.get("input", {}) or {}
    outp = job.get("output", {}) or {}
    fmt_cfg = job.get("format", {}) or {}
    ups = job.get("upscale", {}) or {}
    perf = job.get("perf", {}) or {}
    dry = bool(job.get("dry_run"))

    fid = str(fmt_cfg.get("id") or "png")
    if fid not in FORMATS:
        log(f"unknown format {fid}, using png", "warn")
        fid = "png"
    opts = merged(fid, fmt_cfg.get("options"))
    ext = FORMATS[fid].ext
    container_id = str(outp.get("container") or "files")
    if container_id not in CONTAINERS:
        log(f"unknown output container {container_id}, writing loose files", "warn")
        container_id = "files"

    if dry:
        load_imaging(perf)  # a dry run never imports torch
    else:
        load_backend(perf)
        apply_torch_perf(perf)
    CTRL.start()  # only safe after the imports; see the note in main()

    models_dir = Path(str(ups.get("models_dir") or MODELS_DIR))
    caps = encode_capabilities()
    if not caps.get(fid, {}).get("ok"):
        emit(
            "done",
            ok=False,
            processed=0,
            failed=0,
            skipped=0,
            cancelled=False,
            elapsed=0,
            error=(
                f"{FORMATS[fid].label} cannot be written here: "
                f"{caps.get(fid, {}).get('reason', '')}"
            ),
        )
        return 2

    units = gather_units(inp)
    total = len(units)
    out_dir = Path(str(outp.get("dir") or "")).expanduser()
    if not out_dir:
        raise ValueError("output.dir is required")

    models = list_models(models_dir)
    mode = str(ups.get("mode") or "scale")
    scale = float(ups.get("scale") or 2.0)
    width = int(ups.get("width") or 0)
    height = int(ups.get("height") or 0)
    threshold = float(ups.get("grayscale_threshold", 12))
    colour_percent = float(ups.get("grayscale_colour_percent", 0.25))
    do_gray = bool(ups.get("grayscale_convert", True))
    do_levels = bool(ups.get("auto_levels", True))
    pre_h = int(ups.get("pre_downscale_height") or 0)
    # The rules table is the only chooser the interface exposes. These two
    # names remain as an internal fallback for a page no rule claims, and for
    # job files written before the table existed; "auto" hands the page to the
    # built-in height-band picker.
    model_colour = str(ups.get("model") or "auto")
    model_gray = str(ups.get("model_gray") or "auto") if do_gray else model_colour
    rule_set = _rules.RuleSet.from_dicts(ups.get("rules"))
    rules_logged: set[str] = set()
    overwrite = bool(outp.get("overwrite", False))
    pattern = str(outp.get("pattern") or "{name}")
    keep_structure = bool(outp.get("keep_structure", True))
    io_workers = int(perf.get("io_workers") or 2)
    tile_mode, tile_fixed = parse_tile(perf.get("tile"))
    tile_label = str(perf.get("tile", "auto"))
    skip_long = bool(ups.get("skip_long_strips", False))
    long_max_side = int(ups.get("long_strip_max_side") or 3000)
    long_aspect = float(ups.get("long_strip_min_aspect") or 2.8)
    long_pixels = int(ups.get("long_strip_min_pixels") or 9_000_000)

    if mode == "scale":
        t_scale, t_w, t_h = scale, 0, 0
    elif mode == "width":
        t_scale, t_w, t_h = 1.0, width, 0
    elif mode == "height":
        t_scale, t_w, t_h = 1.0, 0, height
    else:  # fit
        t_scale, t_w, t_h = 1.0, width, height

    tasks = build_tasks(units, out_dir, keep_structure, container_id)

    def page_factor(oh: int, ow: int) -> float:
        """The factor this page will actually be upscaled by."""
        if mode == "height" and t_h:
            return t_h / max(1, oh)
        if mode == "width" and t_w:
            return t_w / max(1, ow)
        if mode == "fit" and t_w and t_h:
            return min(t_w / max(1, ow), t_h / max(1, oh))
        return t_scale

    def resolve_model(wanted: str, gray: bool, oh: int, factor: float) -> dict | None:
        """Turn a model name (or "auto") into an installed model."""
        name = (wanted or "").strip()
        if name.lower() in ("", "auto"):
            return choose_model(models, gray, oh, factor)
        found = next((m for m in models if m["name"] == name or m["path"] == name), None)
        if found is None:
            found = choose_model(models, gray, oh, factor)
            if found:
                log(f"model {name} not found, using {found['name']}", "warn")
        return found

    def note_rule(hit) -> None:
        """Say which rule fired, once per distinct rule, not once per page."""
        text = hit.describe()
        if text not in rules_logged:
            rules_logged.add(text)
            log(f"rule: {text}")

    def page_plan(gray: bool, oh: int, ow: int) -> tuple[dict | None, bool]:
        """What happens to one page: which model, and whether to auto-level.

        A matching rule decides. A page no rule claims falls back to the
        built-in picker, which is exactly what the shipped table's catch-all
        rows do explicitly.
        """
        is_gray = gray and do_gray
        levels = do_levels
        if not models:
            return None, levels
        factor = page_factor(oh, ow)
        hit = rule_set.match(gray=is_gray, width=ow, height=oh, scale=factor)
        if hit is None:
            wanted = model_gray if is_gray else model_colour
        else:
            note_rule(hit)
            wanted = hit.model
            if hit.auto_levels is not None:
                levels = bool(hit.auto_levels)
        return resolve_model(wanted, is_gray, oh, factor), levels

    def pick_model(gray: bool, oh: int, ow: int) -> dict | None:
        """Model only; the dry run reports models without touching levels."""
        return page_plan(gray, oh, ow)[0]

    if dry:
        return dry_run(
            tasks,
            total,
            {
                "out_dir": out_dir,
                "ext": ext,
                "pattern": pattern,
                "overwrite": overwrite,
                "keep_structure": keep_structure,
                "threshold": threshold,
                "colour_percent": colour_percent,
                "pick_model": pick_model,
                "t_scale": t_scale,
                "t_w": t_w,
                "t_h": t_h,
                "fid": fid,
                "container": container_id,
                "model_count": len(models),
                "device": str(perf.get("device") or ""),
                "fp16": wants_fp16(perf.get("use_fp16", True)),
                "tile_label": tile_label,
            },
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx, device, fp16 = make_context(perf)
    planner = TilePlanner(tile_mode, tile_fixed, device, fp16, int(perf.get("budget_limit") or 0))
    # Only a plain scale target has one fixed factor to compare models against;
    # width/height/fit factors depend on each page, so no warning there.
    cache = ModelCache(ctx, t_scale if mode == "scale" else 0.0)

    emit(
        "start",
        total=total,
        out_dir=str(out_dir),
        device=device,
        fp16=fp16,
        tile=tile_label,
        format=fid,
        models=len(models),
        container=container_id,
        bundles=sum(1 for t in tasks if t["kind"] == "bundle"),
    )
    if not models:
        log(f"no models found in {models_dir}; images will only be resized", "warn")

    writer = WritePool(io_workers)
    counters = {"processed": 0, "failed": 0, "skipped": 0}
    clock = time.perf_counter()

    def process_array(image, src_name: str):
        """Full single-image pipeline: (uint8 array, is_gray, model name, info)."""
        oh, ow = hwc(image)[:2]
        gray, score, coloured = gray_stats(image, threshold, colour_percent)
        if skip_long and is_long_strip(ow, oh, long_max_side, long_aspect, long_pixels):
            log(f"{src_name}: {ow}x{oh} long strip, passed through without upscaling", "warn")
            return (
                image,
                gray,
                "",
                {
                    "w": ow,
                    "h": oh,
                    "src_w": ow,
                    "src_h": oh,
                    "score": round(score, 2),
                    "colour": round(coloured, 2),
                    "tile": 0,
                },
            )
        if gray and do_gray:
            image = to_grayscale(image)
        if pre_h and oh > pre_h:
            image = standard_resize(image, (round(ow * pre_h / oh), pre_h))

        pick, want_levels = page_plan(gray, oh, ow)
        model = cache.get(pick["path"]) if pick else None

        image = auto_levels(image) if want_levels and image.ndim == 2 else _normalize(image)
        CTRL.gate()
        tile = planner.choose(model, image)
        planner.before()
        image = upscale_array(ctx, image, model, tile)
        planner.after()
        image = _to_uint8(image, normalized=True)
        image = final_resize(image, t_scale, t_w, t_h, ow, oh, gray and do_gray)
        out_h, out_w = hwc(image)[:2]
        info = {
            "w": out_w,
            "h": out_h,
            "src_w": ow,
            "src_h": oh,
            "score": round(score, 2),
            "colour": round(coloured, 2),
            "tile": planner.last,
        }
        return image, gray, (pick["name"] if pick else ""), info

    def encode_now(image) -> bytes:
        return encode(image, fid, opts, caps)

    def write_result(
        index: int,
        src: Path,
        dest: Path,
        image,
        gray: bool,
        model_name: str,
        info: dict,
        started: float,
    ):
        try:
            data = encode_now(image)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            counters["processed"] += 1
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                ms=int((time.perf_counter() - started) * 1000),
                bytes=len(data),
                gray=bool(gray),
                model=model_name,
                **info,
            )
        except Exception as exc:
            counters["failed"] += 1
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                error=f"{type(exc).__name__}: {exc}",
            )

    def on_bundle_page(meta: dict, name: str, size: int) -> None:
        counters["processed"] += 1
        emit(
            "file",
            i=meta.get("i"),
            total=total,
            path=meta.get("src"),
            out=meta.get("bundle"),
            entry=name,
            bytes=size,
            ms=int((time.perf_counter() - float(meta.get("started") or 0)) * 1000),
            gray=bool(meta.get("gray")),
            model=meta.get("model"),
            **(meta.get("info") or {}),
        )

    def on_bundle_fail(meta: dict, name: str, error: str) -> None:
        counters["failed"] += 1
        emit(
            "file",
            i=meta.get("i"),
            total=total,
            path=meta.get("src"),
            out=meta.get("bundle"),
            entry=name,
            error=error,
        )

    def on_bundle_done(key: str, dest: Path, entries: int, failed: int, elapsed: float) -> None:
        emit(
            "bundle",
            key=key,
            out=str(dest),
            entries=entries,
            failed=failed,
            bytes=(dest.stat().st_size if dest.exists() else 0),
            ms=int(elapsed * 1000),
        )

    bundle = BundleWriter(encode_now, on_bundle_page, on_bundle_fail, on_bundle_done)

    def handle_archive(index: int, unit: dict) -> None:
        src: Path = unit["path"]
        dest = resolve_out(unit, out_dir, pattern, ".cbz", keep_structure, index, total)
        if dest.exists() and not overwrite:
            counters["skipped"] += 1
            emit(
                "file", i=index, total=total, path=str(src), out=str(dest), error="exists, skipped"
            )
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        opener = open_archive(src)
        if opener is None:
            counters["failed"] += 1
            emit("file", i=index, total=total, path=str(src), error="unsupported archive")
            return
        names, reader = opener
        tmp = dest.with_suffix(".cbz.part")
        written = 0
        try:
            with ZipFile(tmp, "w", ZIP_STORED) as zf:
                for k, name in enumerate(names, 1):
                    if CTRL.cancelled:
                        raise Cancelled
                    CTRL.gate()
                    emit("progress", i=index, total=total, path=str(src), sub_i=k, sub_n=len(names))
                    try:
                        raw = reader(name)
                        image, _gray, _model, _info = process_array(
                            read_image_bytes(raw, name), name
                        )
                        data = encode_now(image)
                        zf.writestr(str(Path(name).with_suffix(FORMATS[fid].ext).as_posix()), data)
                        written += 1
                    except Cancelled:
                        raise
                    except Exception as exc:
                        log(f"{src.name}:{name}: {exc}", "warn")
            tmp.replace(dest)
            counters["processed"] += 1
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                ms=int((time.perf_counter() - started) * 1000),
                bytes=dest.stat().st_size,
                entries=written,
            )
        except Cancelled:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            counters["failed"] += 1
            emit("file", i=index, total=total, path=str(src), error=f"{type(exc).__name__}: {exc}")

    def reader(unit: dict):
        if unit["kind"] == "image":
            return read_image(unit["path"])
        return None

    def run_images(items: list[dict], into: dict | None) -> None:
        """Upscale a run of images, either to loose files or into one archive."""
        count = len(items)
        seen: set[str] = set()
        position = 0
        for unit, payload in prefetch(items, io_workers, reader):
            position += 1
            if CTRL.cancelled:
                raise Cancelled
            CTRL.gate()
            index = int(unit.get("index") or position)
            src: Path = unit["path"]
            emit(
                "progress",
                i=index,
                total=total,
                path=str(src),
                sub_i=position if into else 0,
                sub_n=count if into else 0,
            )
            if isinstance(payload, Exception):
                counters["failed"] += 1
                emit(
                    "file",
                    i=index,
                    total=total,
                    path=str(src),
                    error=f"read failed: {type(payload).__name__}: {payload}",
                )
                continue
            dest: Path | None = None
            if into is None:
                dest = resolve_out(unit, out_dir, pattern, ext, keep_structure, index, total)
                if dest.exists() and not overwrite:
                    counters["skipped"] += 1
                    emit(
                        "file",
                        i=index,
                        total=total,
                        path=str(src),
                        out=str(dest),
                        error="exists, skipped",
                    )
                    continue
                if dest.resolve() == src.resolve():
                    dest = unique_path(dest)
            started = time.perf_counter()
            try:
                image, gray, model_name, info = process_array(payload, src.name)
            except Exception as exc:
                if CTRL.cancelled:
                    raise Cancelled from exc
                counters["failed"] += 1
                emit(
                    "file",
                    i=index,
                    total=total,
                    path=str(src),
                    error=f"{type(exc).__name__}: {exc}",
                )
                log(traceback.format_exc(limit=4), "debug")
                continue
            if into is None and dest is not None:
                writer.submit(
                    write_result, index, src, dest, image, gray, model_name, info, started
                )
                continue
            entry = format_name(pattern, src, position, count) + ext
            while entry.lower() in seen:
                entry = f"{entry[: -len(ext)]}_{position}{ext}"
            seen.add(entry.lower())
            bundle.add(
                entry,
                image,
                {
                    "i": index,
                    "src": str(src),
                    "gray": gray,
                    "model": model_name,
                    "info": info,
                    "started": started,
                    "bundle": str(into["dest"]),
                },
            )

    cancelled = False
    try:
        for task in tasks:
            if CTRL.cancelled:
                cancelled = True
                break
            if task["kind"] == "archive":
                unit = task["unit"]
                emit("progress", i=int(unit.get("index") or 0), total=total, path=str(unit["path"]))
                handle_archive(int(unit.get("index") or 0), unit)
                continue
            if task["kind"] == "images":
                run_images(task["units"], None)
                continue
            dest_bundle: Path = task["dest"]
            units_here: list[dict] = task["units"]
            if dest_bundle.exists() and not overwrite:
                counters["skipped"] += len(units_here)
                emit(
                    "file",
                    i=int(units_here[0].get("index") or 0),
                    total=total,
                    path=str(units_here[0]["path"].parent),
                    out=str(dest_bundle),
                    entries=len(units_here),
                    error="exists, skipped",
                )
                continue
            bundle.open(task["key"], dest_bundle)
            run_images(units_here, task)
            bundle.close()
    except Cancelled:
        cancelled = True
    except KeyboardInterrupt:
        cancelled = True
    finally:
        try:
            bundle.close(keep=not CTRL.cancelled)
        except Exception as exc:
            log(f"could not finish the archive: {exc}", "error")
        bundle.shutdown()
        writer.close()
        if CTRL.cancelled:
            cancelled = True
        try:
            for fn in list(getattr(ctx, "chain_cleanup_fns", ())):
                fn()
        except Exception:
            pass
        try:
            if device != "cpu" and torch is not None:
                torch.cuda.empty_cache()
        except Exception:
            pass

    emit(
        "done",
        ok=counters["failed"] == 0 and not cancelled,
        cancelled=cancelled,
        elapsed=round(time.perf_counter() - clock, 2),
        **counters,
    )
    return 0 if counters["failed"] == 0 else 1


def open_archive(path: Path):
    """Return (sorted entry names, read(name) -> bytes) or None."""
    ext = path.suffix.lower()
    if ext in (".zip", ".cbz"):
        zf = ZipFile(path)
        names = sorted(
            (
                n
                for n in zf.namelist()
                if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS
            ),
            key=natural_key,
        )
        return names, zf.read
    if ext in (".rar", ".cbr"):
        try:
            import rarfile

            rf = rarfile.RarFile(str(path))
            names = sorted(
                (
                    n
                    for n in rf.namelist()
                    if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS
                ),
                key=natural_key,
            )
            return names, rf.read
        except Exception as exc:
            log(f"cannot open {path.name}: {exc} (RAR needs unrar/7z on PATH)", "warn")
            return None
    return None


# --------------------------------------------------------------------------- #
# GPU wake lock
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="worker", description="JaNaiUpscaler worker")
    ap.add_argument("--job", help="path to a job JSON file")
    ap.add_argument("--probe", action="store_true", help="report devices, encoders and models")
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument(
        "--hold", action="store_true", help="hold a GPU context awake until stdin says stop"
    )
    ap.add_argument("--device", default="", help="device for --hold, e.g. cuda:0")
    ap.add_argument(
        "--hold-interval",
        type=float,
        default=15.0,
        help="seconds between keep-alive touches (default 15)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="with --job: report what would happen, write nothing"
    )
    args = ap.parse_args(argv)

    install_warning_filters()  # before torch is imported anywhere in this process

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", newline="\n")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.probe:
        return do_probe(Path(args.models_dir))
    if args.hold:
        return do_hold(args.device, args.hold_interval)
    if not args.job:
        ap.error("one of --job, --probe or --hold is required")

    # The stdin control reader is deliberately NOT started here. On Windows a
    # thread parked in a blocking read on a *piped* stdin while numpy's
    # extension modules are being loaded deadlocks the loader: "import numpy"
    # never returns, so a GUI-launched job hung forever before printing a
    # single event. run_job() starts the reader once the backend is imported;
    # anything the GUI sends in the meantime waits in the pipe and is handled
    # as soon as the reader comes up, so no cancel/pause is lost.
    try:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    except Exception as exc:
        emit(
            "done",
            ok=False,
            processed=0,
            failed=0,
            skipped=0,
            cancelled=False,
            elapsed=0,
            error=f"bad job file: {exc}",
        )
        return 2
    if args.dry_run:
        job["dry_run"] = True
    try:
        return run_job(job)
    except Exception as exc:
        emit("log", level="error", message=traceback.format_exc(limit=8))
        emit(
            "done",
            ok=False,
            processed=0,
            failed=0,
            skipped=0,
            cancelled=CTRL.cancelled,
            elapsed=0,
            error=f"{type(exc).__name__}: {exc}",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
