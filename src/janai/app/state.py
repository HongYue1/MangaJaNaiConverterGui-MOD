"""Persisted settings. One JSON file next to the app, written atomically."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from janai.core.formats import all_defaults


def defaults() -> dict[str, Any]:
    return {
        "theme": "dark",
        "input": {
            "path": "",
            "recursive": True,
            "include_archives": True,
        },
        "output": {
            "dir": "",
            "same_as_input": True,
            "subfolder": "upscaled",
            "pattern": "{name}",
            "overwrite": False,
            "keep_structure": True,
            # files | cbz (one archive per folder) | cbz_single (one archive total)
            "container": "files",
        },
        "format": {
            "id": "png",
            "options": all_defaults(),
        },
        "upscale": {
            "mode": "scale",
            "scale": 2.0,
            "width": 2048,
            "height": 2160,
            # "Fit" target presets, mirroring the original fork's display list
            "display": "custom",
            "display_portrait": True,
            # Rules-based selection (janai.core.rules) is the only thing that
            # picks a model. Empty means "not set up yet": the interface seeds
            # the shipped table from the installed models the first time it
            # runs, so the choice is always visible and editable instead of
            # hidden behind an "auto" switch.
            "rules": [],
            "auto_levels": True,
            "grayscale_convert": True,
            "grayscale_threshold": 12,
            # percent of sampled pixels that may be clearly coloured before a
            # page is treated as colour, even when the average says gray
            "grayscale_colour_percent": 0.25,
            "pre_downscale_height": 0,
            # webtoon-style mega strips: off by default because the adaptive
            # tiler copes with them; when on they are copied straight through
            # in the chosen output format instead of being upscaled
            "skip_long_strips": False,
            "long_strip_max_side": 3000,
            "long_strip_min_aspect": 2.8,
            "long_strip_min_pixels": 9000000,
        },
        "perf": {
            "device": "",
            # True means "FP16 wherever the device supports it"; the worker
            # downgrades to FP32 on its own when it does not.
            "use_fp16": True,
            "tile": "auto",
            "budget_limit": 0,
            "force_cache_wipe": False,
            "torch_threads": 0,
            "io_workers": 2,
            "vips_concurrency": 0,
            # cuDNN autotune re-benchmarks each new tile shape, and the tile
            # planner varies tile size per page, so the cost never amortises
            # (measured 21-23s/page on, ~17s off). Opt-in only.
            "cudnn_benchmark": False,
            "allow_tf32": True,
            "gpu_wake_lock": True,
        },
        "log": {
            "auto_save": True,
            "dir": "logs",
            "keep": 30,
            "wrap": False,
            "show_debug": False,
        },
        "ui": {
            "advanced_format": False,
            "perf_open": False,
            "log_open": False,
            "log_weight": 2,
            "geometry": "",
        },
        "probe": None,
    }


def _merge(base: dict, patch: Any) -> dict:
    out = copy.deepcopy(base)
    if not isinstance(patch, dict):
        return out
    for key, value in patch.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _merge(out[key], value)
        elif key in out or key == "probe":
            out[key] = value
    return out


#: Keys older builds wrote that no longer exist. ``model``/``model_gray`` were
#: the two pickers the rules table replaced; ``rules_enabled`` switched the
#: table off, which left nothing driving the choice.
RETIRED_UPSCALE_KEYS = ("model", "model_gray", "rules_enabled")


def _migrate(raw: Any) -> Any:
    """Carry older settings files forward.

    Two changes need translating rather than dropping:

    * the colour-pixel guard used to be stored in per-mille and labelled with a
      per-mille sign, which read as a stray glyph in the UI;
    * the model pickers are gone, so their values are handed to the rules table
      as a starting point when that table has not been written yet.
    """
    if not isinstance(raw, dict):
        return raw
    ups = raw.get("upscale")
    if not isinstance(ups, dict):
        return raw
    if "grayscale_colour_permille" in ups:
        old = ups.pop("grayscale_colour_permille")
        if "grayscale_colour_percent" not in ups:
            try:
                ups["grayscale_colour_percent"] = round(float(old) / 10.0, 4)
            except (TypeError, ValueError):
                pass
    if not ups.get("rules"):
        ups["rules"] = _rules_from_pickers(ups)
    for key in RETIRED_UPSCALE_KEYS:
        ups.pop(key, None)
    return raw


def _rules_from_pickers(ups: dict) -> list[dict]:
    """Two named models -> two catch-all rules, so an upgrade keeps working.

    Anything left on "auto" is skipped: the interface reseeds the shipped table
    from the installed models, which is a better answer than a placeholder.
    """
    out: list[dict] = []
    colour = str(ups.get("model") or "").strip()
    gray = str(ups.get("model_gray") or "").strip()
    if colour and colour.lower() != "auto":
        out.append({"kind": "colour", "model": colour, "note": "carried over from the old picker"})
    if gray and gray.lower() != "auto" and bool(ups.get("grayscale_convert", True)):
        out.append(
            {
                "kind": "grayscale",
                "model": gray,
                "auto_levels": bool(ups.get("auto_levels", True)),
                "note": "carried over from the old picker",
            }
        )
    return out


class Settings:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = defaults()

    def load(self) -> Settings:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.data = _merge(defaults(), _migrate(raw))
        except FileNotFoundError:
            pass
        except Exception:
            self.data = defaults()
        return self

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            pass

    # convenience accessors ------------------------------------------------ #
    def section(self, name: str) -> dict:
        return self.data.setdefault(name, {})

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        return self.section(section).get(key, fallback)

    def set(self, section: str, key: str, value: Any) -> None:
        self.section(section)[key] = value

    def format_options(self, fid: str) -> dict:
        opts = self.data.setdefault("format", {}).setdefault("options", all_defaults())
        return opts.setdefault(fid, {})
