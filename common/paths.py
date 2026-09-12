"""Where JaNai Upscaler finds its runtime, models and support files.

Everything the app needs lives in this one folder:

    backend/python        the interpreter and its packages (a uv venv)
    backend/models        the model weights
    backend/src           the chaiNNer-derived upscaling backend
    backend/ImageMagick   the ICC profiles used by the dot-gain resize
    tools/                cjxl.exe, djxl.exe, uv.exe

Each location is resolved at run time, in this order:

1. an explicit path in ``janai.config.json`` next to the app
2. ``<app>/backend/...`` - the normal layout, whether a real folder or a
   junction someone pointed elsewhere
3. the MangaJaNaiConverterGui checkout this folder sits in, for ``src``,
   ``ImageMagick`` and ``resources`` only, which is handy while developing

Standard library only: the GUI imports this module, and the GUI never imports
torch. For a report of what this copy of the app resolves to::

    python common/paths.py
    python common/paths.py --json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

CONFIG_NAME = "janai.config.json"
REPO_NAME = "MangaJaNaiConverterGui"
MODEL_EXTS = {".pth", ".safetensors", ".pt", ".ckpt"}
ICC_MARKER = "Dot Gain 20%.icc"
SRC_MARKER = "progress_controller.py"

#: Every configurable location, in the order the report prints them.
LOCATIONS = (
    "python_dir",
    "models_dir",
    "src_dir",
    "icc_dir",
    "resources_dir",
    "extras_dir",
    "tools_dir",
)


def app_root() -> Path:
    """The portable folder: the one holding app/, worker/ and common/."""
    return Path(__file__).resolve().parent.parent


def config_file(root: Path | None = None) -> Path:
    return (root or app_root()) / CONFIG_NAME


def read_config(root: Path | None = None) -> dict:
    """The saved configuration, or an empty dict when there is none.

    setup.ps1 writes this file. It only needs entries for locations that sit
    outside the app folder, so on a normal install it is pure provenance.
    """
    try:
        # utf-8-sig: tolerate a BOM, which Windows editors and PowerShell like to add
        data = json.loads(config_file(root).read_text(encoding="utf-8-sig"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------- #
# small path helpers
# --------------------------------------------------------------------------- #
def _expand(value: object, root: Path) -> Path | None:
    """A configured value as an absolute path, or None if it is not usable."""
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(os.path.expandvars(value.strip())).expanduser()
    return path if path.is_absolute() else root / path


def _key(path: Path) -> str:
    try:
        text = str(path.resolve())
    except OSError:
        text = str(path)
    return text.casefold() if os.name == "nt" else text


def _unique(paths: Iterable[Path | None]) -> list[Path]:
    out: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        if path is None:
            continue
        key = _key(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def interpreter(python_dir: Path | None, windowless: bool = False) -> Path | None:
    """The interpreter inside a runtime folder, whichever layout it uses.

    Handles a virtual environment (``Scripts/python.exe``, ``bin/python3``), a
    standalone CPython (``python.exe``) and the nested ``python/python.exe``
    layout that a junction to an installed app can expose.
    """
    if python_dir is None:
        return None
    names = ["python.exe", "pythonw.exe", "python3", "python"]
    if windowless:
        names.insert(0, "pythonw.exe")
    for base in (
        python_dir / "Scripts",
        python_dir / "bin",
        python_dir,
        python_dir / "python",
    ):
        for name in names:
            cand = base / name
            if cand.is_file():
                return cand
    return None


def _has_models(d: Path) -> bool:
    """True when a folder holds model weights, at the top or one level down."""
    for p in d.iterdir():
        if p.is_file() and p.suffix.lower() in MODEL_EXTS:
            return True
        if p.is_dir():
            try:
                if any(q.is_file() and q.suffix.lower() in MODEL_EXTS for q in p.iterdir()):
                    return True
            except OSError:
                continue
    return False


# --------------------------------------------------------------------------- #
# resolution
# --------------------------------------------------------------------------- #
@dataclass
class Paths:
    """Resolved locations, plus a note of where each one came from."""

    root: Path
    python_dir: Path | None = None
    models_dir: Path | None = None
    src_dir: Path | None = None
    icc_dir: Path | None = None
    resources_dir: Path | None = None
    extras_dir: Path | None = None
    tools_dir: Path | None = None
    origins: dict = field(default_factory=dict)
    config: dict = field(default_factory=dict)

    @property
    def backend(self) -> Path:
        return self.root / "backend"

    def interpreter(self, windowless: bool = False) -> Path | None:
        return interpreter(self.python_dir, windowless)

    def import_paths(self) -> list[Path]:
        """Directories the worker adds to sys.path, in priority order."""
        return _unique([self.root, self.extras_dir, self.src_dir])

    def icc(self, name: str = ICC_MARKER) -> Path | None:
        if self.icc_dir:
            cand = self.icc_dir / name
            if cand.is_file():
                return cand
        return None

    def as_dict(self) -> dict:
        out: dict = {"root": str(self.root)}
        for name in LOCATIONS:
            value = getattr(self, name)
            out[name] = str(value) if value else ""
        exe = self.interpreter()
        out["interpreter"] = str(exe) if exe else ""
        out["origins"] = dict(self.origins)
        return out


def resolve(root: Path | None = None) -> Paths:
    """Work out where everything lives for this copy of the app."""
    root = (root or app_root()).resolve()
    cfg = read_config(root)
    backend = root / "backend"
    paths = Paths(root=root, config=cfg)

    # The app normally sits inside the MangaJaNaiConverterGui checkout, whose
    # backend folder carries the upscaling source and the ICC profiles. setup
    # copies them into backend/, so this is only a fallback for a folder that
    # has not been set up yet.
    repos = _unique(
        [
            root.parent / REPO_NAME / "backend",
            root.parent.parent / REPO_NAME / "backend",
            root.parent / "backend",
        ]
    )

    def pick(
        name: str,
        tagged: list[tuple[str, Path]],
        check: Callable[[Path], bool] | None = None,
        strict: bool = False,
    ) -> None:
        configured = _expand(cfg.get(name), root)
        options = ([("config", configured)] if configured else []) + tagged
        best: tuple[str, Path] | None = None
        first: tuple[str, Path] | None = None
        seen: set[str] = set()
        for tag, path in options:
            key = _key(path)
            if key in seen:
                continue
            seen.add(key)
            if not path.is_dir():
                continue
            if first is None:
                first = (tag, path)
            if check is None:
                best = (tag, path)
                break
            try:
                if check(path):
                    best = (tag, path)
                    break
            except OSError:
                continue
        chosen = best if best else (None if strict else first)
        if chosen:
            setattr(paths, name, chosen[1])
            paths.origins[name] = chosen[0]

    pick(
        "python_dir",
        [("app", backend / "python")],
        check=lambda d: interpreter(d) is not None,
        strict=True,
    )
    pick("models_dir", [("app", backend / "models")], check=_has_models)
    pick(
        "src_dir",
        [("app", backend / "src")] + [("repo", r / "src") for r in repos],
        check=lambda d: (d / SRC_MARKER).is_file(),
        strict=True,
    )
    pick(
        "icc_dir",
        [("app", backend / "ImageMagick")] + [("repo", r / "ImageMagick") for r in repos],
        check=lambda d: (d / ICC_MARKER).is_file(),
        strict=True,
    )
    pick(
        "resources_dir",
        [("app", backend / "resources")] + [("repo", r / "resources") for r in repos],
    )
    pick("extras_dir", [("app", backend / "extras")])
    pick("tools_dir", [("app", root / "tools"), ("app", backend / "tools")])
    return paths


def report(paths: Paths | None = None) -> str:
    """A human-readable summary of one resolution."""
    p = paths or resolve()
    exe = p.interpreter()
    lines = [
        f"{'root':<15} {p.root}",
        f"{'interpreter':<15} {exe if exe else '(not found)'}",
    ]
    for name in LOCATIONS:
        value = getattr(p, name)
        origin = p.origins.get(name, "")
        shown = str(value) if value else "(not found)"
        lines.append(f"{name:<15} {shown}" + (f"  [{origin}]" if origin else ""))
    if p.config:
        lines.append(f"{'config':<15} {config_file(p.root)}")
    return "\n".join(lines)


if __name__ == "__main__":
    resolved = resolve()
    if "--json" in sys.argv[1:]:
        print(json.dumps(resolved.as_dict(), indent=2))
    else:
        print(report(resolved))
