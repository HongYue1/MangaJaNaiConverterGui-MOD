"""Persisted settings. One JSON file next to the app, written atomically."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

from common.formats import all_defaults


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
            # model used for colour pages (and for every page when grayscale
            # detection is off); model_gray is used for detected gray pages.
            "model": "auto",
            "model_gray": "auto",
            "auto_levels": True,
            "grayscale_convert": True,
            "grayscale_threshold": 12,
            # per-mille of sampled pixels that may be clearly coloured before a
            # page is treated as colour, even when the average says gray
            "grayscale_colour_permille": 2.5,
            "pre_downscale_height": 0,
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
            "cudnn_benchmark": True,
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


class Settings:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = defaults()

    def load(self) -> "Settings":
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self.data = _merge(defaults(), raw)
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
            os.replace(tmp, self.path)
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
