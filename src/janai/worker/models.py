"""Which model to use, and the loaded network behind it.

These belong in one module because they are three views of a single decision.
`list_models()`/`model_info()` read what is installed straight off the
filenames; `choose_model()` resolves the user's "auto" against that list; and
`ModelCache` turns the chosen path into a network exactly once per process.
Splitting them would leave the filename-parsing conventions (the `4x_`, `1920p`,
`denoise` tokens) restated in two places, free to drift apart.

`upscale_array()` lives here rather than with the pixel transforms because it is
the one call that consumes a loaded model; everything else in the pixel path
works on arrays alone.

The heavy handles are read as `runtime.<name>` at call time and never imported
by value, so this module stays importable before the heavy stack is loaded.
That matters for `list_models()` in particular: `--probe` reports the installed
models and must not pull in torch to do it.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from janai.core import rules as _rules
from janai.worker import runtime, tiling
from janai.worker.events import log

MODEL_EXTS = {".pth", ".safetensors", ".pt", ".ckpt"}


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
            loaded = runtime.load_model_node(self.ctx, Path(path))
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

    # Declared before the branch because the two arms start from different
    # shapes: the gray arm from min(), which always yields a model, and the
    # colour arm from next(..., None), which may not.
    pick: dict | None
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
    report = tiling.split_report()
    if report is not None:
        report["tile"] = 0
    result = runtime.upscale_image_node(ctx, image, model, False, 0, tile, 256, False)
    if runtime.hwc(image)[2] == 1 and result.ndim == 3:
        result = runtime.np.squeeze(result, axis=-1)
    return result
