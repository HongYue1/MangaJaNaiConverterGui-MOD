"""Owns the heavy stack: numpy, OpenCV, libvips, Pillow, torch and the vendored
chaiNNer nodes. Loading them is all this module does.

Two rules are the whole reason it exists:

* **Nothing heavy is imported at module import time.** A dry run, the encoder
  capability probe and the model list all finish without paying for torch, which
  costs seconds and hundreds of MB. `scripts/selftest.py` asserts this by
  reporting the dry run as "no torch import" - if that line ever disappears, an
  eager import crept in.
* **Perf environment has to be set before the first import.**
  `VIPS_CONCURRENCY`, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are read by
  libvips and OpenMP as their libraries load, so `apply_perf_env` must run ahead
  of `load_imaging`. Setting them afterwards is silently ignored, which looks
  exactly like the setting having no effect.

The handles below start as None and are bound by the loaders. Reference them
through the module (`runtime.torch`, `runtime.np`) rather than importing them by
name: `from janai.worker.runtime import torch` copies None at import time and
never sees the loaded module.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

from janai.worker.events import log

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


# --------------------------------------------------------------------------- #
# handles, bound by the loaders below
# --------------------------------------------------------------------------- #
np = None
cv2 = None
pyvips = None
torch = None
PILImage = None
ImageCms = None
ImageFilter = None
cx_resize = None
ResizeFilter = None
normalize = None
to_uint8 = None
get_h_w_c = None
upscale_image_node = None
load_model_node = None
SettingsParser = None
NodeContext = None
ProgressController = None
TILE: dict[str, Any] = {}
_heavy_loaded = False


def hwc(image: Any) -> tuple[int, int, int]:
    """(height, width, channels): the stdlib-only twin of `get_h_w_c` above.

    Kept beside that handle because shapes are needed before, or entirely
    without, the vendored backend being imported - a dry run, the tile planner
    under test and the encoder probe all ask for an image's shape while
    `get_h_w_c` is still None.
    """
    if image.ndim == 2:
        return int(image.shape[0]), int(image.shape[1]), 1
    return int(image.shape[0]), int(image.shape[1]), int(image.shape[2])


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
    global np, cv2, pyvips, PILImage, ImageCms, ImageFilter
    if np is not None:
        return
    if perf:
        apply_perf_env(perf)
    install_warning_filters()

    import cv2 as cv2_mod
    import numpy
    import pyvips as pyvips_mod
    from PIL import Image as PILImage_mod, ImageCms as ImageCms_mod, ImageFilter as ImageFilter_mod

    np = numpy
    cv2 = cv2_mod
    pyvips = pyvips_mod
    PILImage, ImageCms, ImageFilter = PILImage_mod, ImageCms_mod, ImageFilter_mod


def load_backend(perf: dict | None = None) -> None:
    global torch, cx_resize, ResizeFilter, normalize, to_uint8, get_h_w_c
    global upscale_image_node, load_model_node, SettingsParser, NodeContext
    global ProgressController, TILE, _heavy_loaded
    load_imaging(perf)
    if _heavy_loaded:
        return

    import spandrel_custom
    import torch as torch_mod
    from api import NodeContext as NodeContext_cls, SettingsParser as SettingsParser_cls
    from chainner_ext import ResizeFilter as ResizeFilter_cls, resize as cx_resize_fn
    from nodes.impl.image_utils import normalize as normalize_fn, to_uint8 as to_uint8_fn
    from nodes.impl.upscale.auto_split_tiles import (
        ESTIMATE,
        MAX_TILE_SIZE,
        NO_TILING,
        TileSize,
    )
    from nodes.utils.utils import get_h_w_c as get_h_w_c_fn
    from packages.chaiNNer_pytorch.pytorch.io.load_model import (
        load_model_node as load_model_node_fn,
    )
    from packages.chaiNNer_pytorch.pytorch.processing.upscale_image import (
        upscale_image_node as upscale_image_node_fn,
    )
    from progress_controller import ProgressController as ProgressController_cls

    installer = getattr(spandrel_custom, "install", None)
    if callable(installer):
        try:
            installer()
        except Exception as exc:
            log(f"spandrel_custom.install() failed: {exc}", "warn")

    torch = torch_mod
    cx_resize, ResizeFilter = cx_resize_fn, ResizeFilter_cls
    normalize, to_uint8, get_h_w_c = normalize_fn, to_uint8_fn, get_h_w_c_fn
    upscale_image_node, load_model_node = upscale_image_node_fn, load_model_node_fn
    SettingsParser, NodeContext = SettingsParser_cls, NodeContext_cls
    ProgressController = ProgressController_cls
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
