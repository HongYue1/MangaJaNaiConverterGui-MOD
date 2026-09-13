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
from io import BytesIO
from pathlib import Path
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
from janai.worker.control import CTRL, Cancelled
from janai.worker.events import emit, log
from janai.worker.pipeline import BundleWriter, WritePool, prefetch

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
      of the run only on real memory pressure - an allocator retry once the
      allocator has settled, or a peak that came within a hair of the memory
      actually free - and then only as far as a cut that is really cheaper.
      A tile that fitted is a tile worth keeping: on a 6 GB laptop card,
      stepping 1248px down to 864px costs about 9% throughput on every page
      that follows.
    * prices a tile through the cut the splitter will really make - the tile
      grid, the per-edge overlap and the channel count - rather than pricing
      tile x tile, which overstated the affordable tile by about sqrt(3) on a
      colour page: that is what planned a 1930px whole-page pass on a 6 GB
      card and left auto_split to clean the miss up,
    * distrusts the inherited constant until it has measured the model once.
      For FDAT it is optimistic by roughly 9x, so the first page of an unseen
      model is clamped to UNVERIFIED_TILE instead of being sized from a guess.
      One cautious page is nearly free, because what costs memory is the grid
      the page is cut into and not the number asked for: a 1920x1080 page is
      cut into the same 2x2 grid of 960x540 tiles by anything from 960px to
      1079px, so 965px and 1024px are the very same work. The next coarser
      grid is 2x1, one 960x1080 tile, which needs about twice the activations
      and does not fit in 6 GB. A miss, by contrast, costs a whole wasted
      pass.
    """

    ALIGN = 32
    MIN_TILE = 128
    OVERLAP = 16  # auto_split's default padding per tile edge
    HEADROOM = 256 * 1024**2
    SAFETY = 0.85
    MARGIN = 1.10  # applied to a measured cost before reusing it
    UNVERIFIED_TILE = 1024  # ceiling until this model has been measured once

    def __init__(
        self,
        mode: str,
        fixed: int,
        device: str,
        fp16: bool,
        budget_limit_gib: int = 0,
        profile: dict | None = None,
    ) -> None:
        self.mode = mode
        self.fixed = int(fixed or 0)
        self.device = str(device or "")
        self.fp16 = bool(fp16)
        self.budget_limit = max(0, int(budget_limit_gib or 0)) * 1024**3
        self.elem = 2 if fp16 else 4
        self.last = 0
        self._said_clamp = False
        # What the one-off hardware profile measured, keyed by model file name.
        # A profile carries the two terms a single page can never separate:
        # the cost that exists before a tile is cut, and the cost each
        # channel-pixel of tile input adds. See load_profile().
        self._profile: dict[str, dict] = {}
        self._names: dict[int, str] = {}
        self._said_profile: set[str] = set()
        self.load_profile(profile)
        self._model_bytes: dict[int, int] = {}
        self._per_px: dict[int, float] = {}
        self._cap: dict[int, int] = {}
        self._pending: tuple[int, int, int, int, int, int] | None = None
        self._retries = 0
        self._pressure: dict[int, int] = {}
        # Pages this model has measured. The first one runs against a cold
        # allocator with the weights just uploaded, so its cudaMalloc retries
        # describe a cache that has not settled rather than a tile that is too
        # big. See after().
        self._pages: dict[int, int] = {}
        self._good: dict[int, int] = {}
        self._failed: dict[int, int] = {}
        self._last_page: tuple[int, int, int, int] | None = None
        self._contaminated = False

    # -- the hardware profile ---------------------------------------------- #
    def load_profile(self, profile: dict | None) -> None:
        """Take the measurements the app recorded for this machine.

        Only entries carrying a positive per-pixel cost are kept: a model
        whose profiling run failed has to fall back to the cautious
        first-page path, not to a cost of zero.
        """
        self._profile = {}
        for name, entry in (profile or {}).items():
            if isinstance(entry, dict) and float(entry.get("per_px") or 0.0) > 0:
                self._profile[str(name)] = entry

    def note_model(self, model: Any, name: str) -> None:
        """Tie a loaded model to the file name its measurements are keyed by."""
        if model is not None and name:
            self._names[id(model)] = str(name)

    def _measured(self, key: int) -> dict | None:
        """The profile entry for a loaded model, if this machine has one."""
        name = self._names.get(key)
        return self._profile.get(name) if name else None

    def tile_input_pixels(self, w: int, h: int, tile: int, c: int) -> int:
        """The cut-aware price of a tile, in the units a profile records.

        Public because the profiler has to measure in exactly the units the
        planner spends in, or the two would disagree about what a tile costs.
        """
        return self._tile_pixels(w, h, tile, c)

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

    def _fit(self, w: int, h: int, c: int, budget: int, per_px: float, ceiling_px: int = 0) -> int:
        """The largest aligned tile whose real cut is predicted to fit.

        What costs memory is the grid the splitter cuts, not the number it was
        asked for, and _tile_pixels is a step function of that number: every
        tile from 672px to 960px cuts a 1920x1080 page into the same 3x2 grid.
        So walk the aligned sizes down from a whole-page pass and take the
        first that fits. Inverting tile x tile algebraically instead is what
        dropped the channel count and the overlap from the price.
        """
        affordable = budget / max(per_px, 1e-6)
        if ceiling_px > 0:
            # A tile input size that ran out of memory while profiling is a
            # hard ceiling, and unlike a tile *number* it carries across page
            # sizes: it is counted in the same channel-pixels _tile_pixels
            # returns, so a smaller page cutting fewer tiles cannot talk the
            # planner past what the card already refused.
            affordable = min(affordable, float(ceiling_px))
        tile = self._align(max(w, h) + self.ALIGN)
        while tile > self.MIN_TILE:
            if self._tile_pixels(w, h, tile, c) <= affordable:
                return tile
            tile -= self.ALIGN
        return self.MIN_TILE

    def _ceiling(self, key: int) -> int:
        """The largest aligned tile still below a size that missed, if any."""
        failed = self._failed.get(key, 0)
        if not failed:
            return 0
        return max(self.MIN_TILE, self._align(failed - self.ALIGN))

    def _cheaper(self, w: int, h: int, c: int, tile: int) -> int:
        """The largest aligned tile that cuts this page into cheaper passes.

        A step down is only worth taking if it reaches a different cut, and
        stepping by a ratio need not: 0.85 x 1312px aligns to 1088px, and on a
        720x4000 colour page both cut the same four passes of 752x1032. So the
        old cap shrank the number in the log and changed no work at all, while
        reading like a response to memory pressure. Walk the aligned sizes down
        until the price per pass really drops - the mirror of the blind halving
        that turned a 1152px request into 576px tiles.
        """
        pixels = self._tile_pixels(w, h, tile, c)
        step = self._align(tile) - self.ALIGN
        while step > self.MIN_TILE:
            if self._tile_pixels(w, h, step, c) < pixels:
                return step
            step -= self.ALIGN
        return self.MIN_TILE

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
                limit = min(self._cap.get(key) or proven, self._ceiling(key) or proven)
                return self._arm(key, model, w, h, c, min(proven, limit), 0, 0)
            self.last = 0
            return TILE["estimate"]

        per_px = self._per_px.get(key)
        calibrated = per_px is not None
        measured = self._measured(key) if per_px is None else None
        ceiling_px = 0
        if measured is not None:
            # This machine has already been measured for this model, so the
            # first page is not a guess. Two things come from the profile that
            # a single page can never supply: the per-pixel slope, and the
            # fixed cost - weights, context, workspace - which is charged to
            # the budget once instead of being priced per pixel. Conflating
            # the two is what made a 1152px tile look like it needed ~4.8 GB
            # when the entire measured peak was 2586 MiB.
            per_px = float(measured.get("per_px") or 0.0) * self.MARGIN
            overhead = max(0, int(measured.get("fixed_bytes") or 0) - model_bytes)
            budget = max(0, budget - int(overhead * self.MARGIN))
            ceiling_px = int(measured.get("max_pixels") or 0)
            calibrated = per_px > 0
            name = self._names.get(key, "")
            if calibrated and name and name not in self._said_profile:
                self._said_profile.add(name)
                fixed_mib = int(measured.get("fixed_bytes") or 0) // 1024**2
                log(
                    f"sizing tiles from this machine's profile of {name}:"
                    f" {fixed_mib} MiB fixed plus"
                    f" {per_px / self.MARGIN:.1f} bytes per pixel",
                    "debug",
                )
        if per_px is None or per_px <= 0:
            # chaiNNer's calibration, put in the same units as a measurement:
            # bytes per channel-pixel of tile input. The `* c` it used to carry
            # is already inside _tile_pixels, so keeping both counted colour
            # twice while the algebraic inversion dropped it again.
            per_px = (model_bytes / (1024 * 52)) * self.elem
            calibrated = False
        tile = self._fit(w, h, c, budget, per_px, ceiling_px)
        if not calibrated:
            # An unmeasured model is a guess, and this is the guess that
            # planned 1930px and missed. Clamp the first page, measure it, then
            # trust the measurement for every page after it.
            if tile > self.UNVERIFIED_TILE and not self._said_clamp:
                # Say it once, so a single-image job that never gets to use
                # the measurement does not look like an unexplained ceiling.
                self._said_clamp = True
                log(
                    "first page of an unmeasured model: holding the tile at"
                    f" {self.UNVERIFIED_TILE}px rather than the estimated"
                    f" {tile}px until the cost has been measured"
                )
            tile = min(tile, self.UNVERIFIED_TILE)
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
        ceiling = self._ceiling(key)
        if ceiling:
            # a size that missed is a ceiling, not a target
            tile = min(tile, ceiling)
        return self._arm(key, model, w, h, c, tile, budget, free)

    def _arm(
        self, key: int, model: Any, w: int, h: int, c: int, tile: int, budget: int, free: int
    ) -> Any:
        """Note what is about to be attempted, then hand the tile over."""
        # one tile that covers the page is the fastest case there is
        tile = max(self.MIN_TILE, min(tile, self._align(max(w, h) + self.ALIGN)))
        self.last = tile
        self._last_page = (key, w, h, c)
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
        self._contaminated = False
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
        if self._contaminated:
            # This page's peak belongs partly to an attempt that ran out of
            # memory, so dividing it by the tile that did finish would inflate
            # the model's cost and shrink every page after it. Keep what the
            # page proved about what fits, discard its measurement.
            log("re-tiled page: not calibrating from its peak", "debug")
        else:
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
        pages = self._pages.get(key, 0) + 1
        self._pages[key] = pages
        if not (retried or near_limit):
            # A page that finished cleanly clears the slate: pressure has to be
            # consecutive to mean anything, or one fragmented page early in a
            # chapter ratchets the tile down for every page after it.
            self._pressure.pop(key, None)
            return
        if pages == 1 and retried and not near_limit:
            # The first page of a model pays for a cold allocator: the weights
            # have only just been uploaded and the cache has nothing to reuse,
            # so cudaMalloc retries here are warm-up and not a tile that is too
            # big. Proven on a 6 GB card with a 720x4000 colour page that
            # retried at 1312px: forcing 1088px across the batch produced the
            # very same 17.9s first page, and later pages 0.2s slower. Shrink
            # only if a settled allocator repeats the retries.
            log("first page of this model: allocator warming up, keeping the tile", "debug")
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
        # Step to the next cut that is genuinely cheaper rather than stepping
        # by a ratio, which can land inside the same cut and cost a whole run
        # a smaller number for no change in the work done.
        last = self._last_page
        if last is not None and last[0] == key:
            capped = self._cheaper(last[1], last[2], last[3], tile)
        else:
            capped = max(self.MIN_TILE, self._align(tile * 0.85))
        if capped < (self._cap.get(key) or tile):
            self._cap[key] = capped
            log(f"tile capped at {capped}px after {why}", "debug")

    def retiled(self, tile: int) -> None:
        """Correct the plan with the tile auto_split actually finished with.

        `choose` hands out a plan. A page that turns out not to fit is re-tiled
        inside auto_split, which the planner never hears about: the result line
        then reported a tile that never ran, contradicting the splitter's own
        warning in the same log. Called before `after()` so the calibration
        measures the tiles that did run, and so the next page starts at the
        size that worked instead of repeating the failed full-page attempt.

        What a miss proves is that the *planned* size was too big, not that the
        size auto_split fell back to is the ceiling: halving 1930px to 965px
        says nothing against 1024px, which measures faster than both. So the
        planned size becomes a ceiling, the fallback becomes a proven floor,
        and the measured cost chooses between them.
        """
        attempted = self.last
        if tile <= 0 or attempted <= 0 or tile >= attempted:
            return
        self.last = tile
        self._contaminated = True
        if self._last_page is None:
            return
        key, w, h, c = self._last_page
        if self._pending is not None:
            pending_key, _pixels, _planned, budget, page_bytes, free = self._pending
            self._pending = (
                pending_key,
                self._tile_pixels(w, h, tile, c),
                tile,
                budget,
                page_bytes,
                free,
            )
        self._good[key] = max(self._good.get(key, 0), tile)
        known = self._failed.get(key, 0)
        if not known or attempted < known:
            self._failed[key] = attempted
            log(f"{attempted}px missed, {tile}px fitted: planning under {attempted}px", "debug")


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


def _split_report() -> dict | None:
    """auto_split's note of a page it had to re-tile, when it is reachable."""
    try:
        from nodes.impl.upscale.auto_split import retile_report
    except Exception:
        return None
    return retile_report


def tile_actually_used() -> int:
    """The tile the last upscale finished with, or 0 if nothing was lowered."""
    report = _split_report()
    return int((report or {}).get("tile", 0) or 0)


def upscale_array(ctx, image, model, tile):
    if model is None:
        return image
    report = _split_report()
    if report is not None:
        report["tile"] = 0
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
            # Ask for the model first, because that is what logs the matched
            # rule, then let an exclusion overrule the answer. The plan used
            # to name a model the real run would never load, and predict a
            # size the page was never going to reach.
            pick = cfg["pick_model"](bool(gray), h, w)
            excluded = bool(cfg.get("excluded") and cfg["excluded"](bool(gray), h, w))
            if excluded:
                pick = None
                pw, ph = w, h
                log(
                    f"{src.name}: {w}x{h} excluded by a rule,"
                    " would be re-encoded without upscaling",
                    "warn",
                )
            else:
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
                passthrough=excluded,
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
    # Three exclusive cases, with the old flag as the fallback so a settings
    # file written by an earlier build still means what it meant.
    page_kind = str(ups.get("page_kind") or "").strip().lower()
    if page_kind not in {"detect", "grayscale", "colour"}:
        page_kind = "detect" if bool(ups.get("grayscale_convert", True)) else "colour"
    do_gray = page_kind != "colour"
    force_gray = page_kind == "grayscale"
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
        is_gray = force_gray or (gray and do_gray)
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

    def excluded_by_rule(gray: bool, oh: int, ow: int) -> bool:
        """A rule can exclude a page from the model instead of choosing one.

        The row's page-size condition decides which pages skip upscaling and
        are only re-encoded - the same outcome as the old long-strip switch,
        with the sizes visible and editable instead of hardcoded.
        """
        hit = rule_set.match(
            gray=force_gray or (gray and do_gray),
            width=ow,
            height=oh,
            scale=page_factor(oh, ow),
        )
        return hit is not None and str(getattr(hit, "action", "upscale")) == "passthrough"

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
                "excluded": excluded_by_rule,
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
    stored_profile = perf.get("profile")
    planner = TilePlanner(
        tile_mode,
        tile_fixed,
        device,
        fp16,
        int(perf.get("budget_limit") or 0),
        # The app only forwards measurements taken on this machine in this
        # precision; anything stale is dropped there rather than here.
        profile=stored_profile if isinstance(stored_profile, dict) else None,
    )
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
        if force_gray:
            gray = True
        by_rule = excluded_by_rule(gray, oh, ow)
        by_size = skip_long and is_long_strip(ow, oh, long_max_side, long_aspect, long_pixels)
        if by_rule or by_size:
            why = "excluded by a rule" if by_rule else "long strip"
            log(f"{src_name}: {ow}x{oh} {why}, passed through without upscaling", "warn")
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
                    "passthrough": True,
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
        planner.note_model(model, pick["name"] if pick else "")
        tile = planner.choose(model, image)
        planner.before()
        image = upscale_array(ctx, image, model, tile)
        planner.retiled(tile_actually_used())
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
# --------------------------------------------------------------------------- #
# hardware profile
# --------------------------------------------------------------------------- #
#: The ladder the profiler walks. Each page is square and exactly one tile
#: wide, so every measurement is one clean (pixels, peak) point with no grid
#: geometry folded into it.
PROFILE_TILES: tuple[int, ...] = (512, 768, 1024, 1280, 1536, 1792, 2048)


def _is_oom(exc: BaseException) -> bool:
    """Whether a failure was the card running out of memory, not a bug."""
    if "outofmemory" in type(exc).__name__.lower():
        return True
    text = str(exc).lower()
    return "out of memory" in text or ("alloc" in text and "fail" in text)


def _fit_cost(points: list[tuple[int, int]]) -> tuple[int, float]:
    """Split measured peaks into a fixed cost and a per-pixel cost.

    `points` is [(tile input channel-pixels, peak bytes)]. A real job can only
    ever produce one point per model, and dividing a single peak by its pixel
    count prices the resident weights, the CUDA context and the workspace as
    though they grew with the tile. That is exactly why a 1152px tile looked
    like it needed ~4.8 GB when the whole measured peak was 2586 MiB. Two or
    more points separate the constant from the slope by least squares.
    """
    usable = [(float(px), float(peak)) for px, peak in points if px > 0 and peak > 0]
    if not usable:
        return 0, 0.0
    if len(usable) == 1:
        px, peak = usable[0]
        return 0, peak / px
    n = float(len(usable))
    sx = sum(p for p, _ in usable)
    sy = sum(q for _, q in usable)
    sxx = sum(p * p for p, _ in usable)
    sxy = sum(p * q for p, q in usable)
    denom = n * sxx - sx * sx
    if denom <= 0:
        return 0, sy / max(sx, 1.0)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    if slope <= 0:
        # Too flat to read a slope from. Charge it all per pixel rather than
        # invent a negative cost that would size tiles without limit.
        return 0, sy / max(sx, 1.0)
    return max(0, int(intercept)), slope


def _device_hardware(device: str) -> dict:
    """Name and total VRAM of the device being profiled, for the record."""
    out: dict[str, Any] = {"device": device, "name": "", "vram": 0}
    try:
        if device.startswith("cuda"):
            index = int(device.split(":")[1]) if ":" in device else 0
            props = torch.cuda.get_device_properties(index)
            out["name"] = str(getattr(props, "name", "") or "")
            out["vram"] = int(getattr(props, "total_memory", 0) or 0)
    except Exception:
        pass
    return out


def _profile_picks(job: dict, models_dir: Path) -> list[dict]:
    """Which models to measure: the ones asked for, else one per scale."""
    try:
        installed = list_models(models_dir)
    except Exception:
        installed = []
    by_path = {str(m.get("path")): m for m in installed if isinstance(m, dict)}
    wanted = [by_path[str(p)] for p in (job.get("models") or []) if str(p) in by_path]
    if wanted:
        return wanted
    # Nothing named: one model per distinct scale is enough, since what this
    # measures is per model and the shipped set is 2x and 4x.
    by_scale: dict[Any, dict] = {}
    for info in by_path.values():
        by_scale.setdefault(info.get("scale"), info)
    return list(by_scale.values())[:2]


def _profile_one(ctx, planner, model: Any, name: str, offset: int, total: int) -> dict | None:
    """Walk the tile ladder for one model and return what it measured."""
    import numpy as _np

    device = planner.device
    model_cost = planner.model_bytes(model)
    card = _device_hardware(device).get("vram") or 0
    steps: list[dict] = []
    points: list[tuple[int, int]] = []
    max_pixels = 0
    failed_pixels = 0
    step = offset
    for tile in PROFILE_TILES:
        if CTRL.cancelled:
            break
        pixels = planner.tile_input_pixels(tile, tile, tile, 3)
        if len(points) >= 2 and card:
            fixed_so_far, slope_so_far = _fit_cost(points)
            if fixed_so_far + slope_so_far * pixels > card:
                # Hopeless on this card by its own numbers, so stop rather
                # than spend a whole pass proving it.
                log(
                    f"{name}: {tile}px would need more than this card has,"
                    " so the ladder stops here",
                    "debug",
                )
                break
        step += 1
        emit("profile_progress", model=name, tile=tile, index=step, total=total)
        page = _np.random.default_rng(0x1A11).random((tile, tile, 3), dtype=_np.float32)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
        result = None
        ok = True
        oom = False
        begin = time.time()
        try:
            result = upscale_array(ctx, page, model, TILE["cls"](tile))
        except Exception as exc:
            ok = False
            oom = _is_oom(exc)
            if not oom:
                log(f"{name}: {tile}px failed, {type(exc).__name__}: {exc}", "warn")
        seconds = time.time() - begin
        # A tile that does not fit does not always raise: auto_split catches
        # the miss itself and finishes the page at a smaller tile. Counting
        # that as a success would divide a peak measured at the lowered tile
        # by this tile's pixel count, understating the cost, and would claim a
        # largest single-pass tile that never ran in one pass. Measured here:
        # a 4x model asked for 768px was quietly finished at 736px.
        lowered = tile_actually_used()
        if ok and lowered and lowered < tile:
            log(
                f"{name}: {tile}px did not fit in one pass (auto_split finished it at {lowered}px)",
                "debug",
            )
            ok = False
            oom = True
        peak = 0
        try:
            peak = int(torch.cuda.max_memory_allocated(device))
        except Exception:
            peak = 0
        entry: dict[str, Any] = {
            "tile": tile,
            "pixels": pixels,
            "peak": peak,
            "seconds": round(seconds, 3),
            "ok": ok,
        }
        if ok and result is not None and seconds > 0:
            try:
                oh, ow, _oc = hwc(result)
                entry["mpx_per_s"] = round(int(ow) * int(oh) / seconds / 1e6, 2)
            except Exception:
                pass
        result = None
        page = None
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        steps.append(entry)
        emit(
            "profile_progress",
            model=name,
            tile=tile,
            index=step,
            total=total,
            ok=ok,
            peak=peak,
            seconds=entry["seconds"],
        )
        if ok:
            points.append((pixels, peak))
            max_pixels = max(max_pixels, pixels)
            continue
        failed_pixels = pixels
        if not oom:
            # A failure that is not memory pressure says nothing about
            # capacity, so the rest of the ladder would measure noise.
            return None
        break
    fixed_bytes, per_px = _fit_cost(points)
    if per_px <= 0:
        return None
    ran = [s for s in steps if s.get("ok")]
    fastest = max(
        (s for s in ran if s.get("mpx_per_s")),
        key=lambda s: float(s["mpx_per_s"]),
        default=None,
    )
    largest = max((int(s["tile"]) for s in ran), default=0)
    tail = f", fastest {fastest['tile']}px at {fastest['mpx_per_s']:.1f} MPx/s" if fastest else ""
    log(
        f"{name}: {int(fixed_bytes) // 1024**2} MiB fixed plus {per_px:.1f} bytes"
        f" per pixel, largest single-pass tile {largest}px{tail}"
    )
    return {
        "fixed_bytes": int(fixed_bytes),
        "per_px": round(float(per_px), 4),
        "max_pixels": int(max_pixels),
        "failed_pixels": int(failed_pixels),
        "model_bytes": int(model_cost),
        "best_tile": int(fastest["tile"]) if fastest else 0,
        "largest_tile": largest,
        "steps": steps,
    }


def do_profile(job: dict) -> int:
    """Measure what each model costs and how fast it runs on this machine.

    A real job measures one page at one tile size, so it can only ever produce
    a single (pixels, peak) point per model. That is why the planner has to
    hold the first page of an unseen model at a cautious 1024px, and why it
    cannot tell a model's fixed cost apart from its per-pixel cost afterwards.

    This walks a ladder of tile sizes on synthetic square pages - one pass
    each, so the cut is not part of the measurement - and records both terms,
    the largest tile input that fits, and how fast each size actually ran.
    Everything it learns is stored against the hardware fingerprint it was
    given, so it is re-measured when the machine changes rather than trusted
    forever. It writes no images.
    """
    from janai.core import hardware as _hardware

    perf = dict(job.get("perf") or {})
    ups = dict(job.get("upscale") or {})
    models_dir = Path(str(ups.get("models_dir") or MODELS_DIR))
    started = time.time()
    try:
        load_backend(perf)
        apply_torch_perf(perf)
    except Exception as exc:
        emit("profile", ok=False, error=f"backend import failed: {type(exc).__name__}: {exc}")
        return 1
    CTRL.start()

    ctx, device, fp16 = make_context(perf)
    if not device.startswith(("cuda", "xpu")):
        emit(
            "profile",
            ok=False,
            error="profiling measures VRAM, and this run is on the CPU",
        )
        return 1
    planner = TilePlanner("auto", 0, device, fp16, int(perf.get("budget_limit") or 0))
    cache = ModelCache(ctx)
    picks = _profile_picks(job, models_dir)
    if not picks:
        emit("profile", ok=False, error=f"no models found in {models_dir}")
        return 1

    total = len(picks) * len(PROFILE_TILES)
    log(
        f"profiling {len(picks)} model(s) on {device} in"
        f" {'FP16' if fp16 else 'FP32'} - this writes no images"
    )
    measured: dict[str, dict] = {}
    done = 0
    for pick in picks:
        if CTRL.cancelled:
            break
        name = str(pick.get("name") or Path(str(pick.get("path") or "")).name)
        try:
            model = cache.get(str(pick["path"]))
        except Exception as exc:
            log(f"{name}: will not load, skipping ({type(exc).__name__}: {exc})", "warn")
            done += len(PROFILE_TILES)
            continue
        got = _profile_one(ctx, planner, model, name, done, total)
        done += len(PROFILE_TILES)
        if got:
            measured[name] = got
        else:
            log(f"{name}: could not be measured on this device", "warn")

    profile = {
        "version": _hardware.PROFILE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fingerprint": str(job.get("fingerprint") or ""),
        "device": device,
        "fp16": bool(fp16),
        "hardware": _device_hardware(device),
        "models": measured,
    }
    emit(
        "profile",
        ok=bool(measured),
        cancelled=CTRL.cancelled,
        elapsed=round(time.time() - started, 2),
        profile=profile,
        error="" if measured else "no model could be measured",
    )
    return 0 if measured else 1


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
    ap.add_argument(
        "--profile",
        action="store_true",
        help="with --job: measure this machine's tile cost and speed, write nothing",
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
    if args.profile:
        try:
            return do_profile(job)
        except Exception as exc:
            emit("log", level="error", message=traceback.format_exc(limit=8))
            emit("profile", ok=False, error=f"{type(exc).__name__}: {exc}")
            return 1
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
