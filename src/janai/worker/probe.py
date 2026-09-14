"""The one-shot capability report the GUI reads before it can draw anything.

The app runs `--probe` at startup: until it answers, the GUI does not know
which devices exist, which encoders are usable, or which models are installed,
so it cannot populate its controls.

**The emitted `probe` payload is a wire format.** Renaming or dropping a key
here silently empties a control on the other side, so treat the key names as
public API even though nothing enforces them.

Every lookup is individually guarded on purpose. A probe that raises tells the
GUI nothing at all, so each optional capability degrades to a placeholder
("?", "", False) and the report still arrives complete.
"""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

from janai.worker import capabilities, devices, imageio, runtime
from janai.worker.environment import PATHS, ROOT
from janai.worker.events import emit
from janai.worker.models import list_models


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
        runtime.load_backend()
    except Exception as exc:
        info["ok"] = False
        info["errors"].append(f"backend import failed: {exc}")
        emit("probe", **info)
        return 1

    try:
        info["libvips"] = ".".join(str(runtime.pyvips.version(i)) for i in range(3))
    except Exception:
        info["libvips"] = "?"
    info["pyvips"] = getattr(runtime.pyvips, "__version__", "?")
    info["torch"] = getattr(runtime.torch, "__version__", "?")
    try:
        info["cuda"] = runtime.torch.version.cuda or ""
    except Exception:
        info["cuda"] = ""
    try:
        info["numpy"] = runtime.np.__version__
        info["opencv"] = runtime.cv2.__version__
    except Exception:
        pass

    info["devices"] = devices.device_objects()
    gpu = next((d for d in info["devices"] if d["value"] != "cpu"), None)
    info["default_device"] = gpu["value"] if gpu else "cpu"
    info["formats"] = imageio.encode_capabilities()
    info["read_jxl"] = capabilities.vips_has("jxlload") or bool(capabilities.find_djxl())
    info["read_heif"] = capabilities.vips_has("heifload")
    info["models"] = list_models(models_dir)
    info["icc"] = PATHS.icc() is not None
    info["tools"] = {name: capabilities.find_tool(name) for name in ("cjxl", "djxl")}
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
