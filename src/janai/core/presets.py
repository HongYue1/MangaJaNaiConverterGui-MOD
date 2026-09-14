"""Settings presets: export what is on screen, import it again later.

A preset is a small JSON file holding the parts of the settings that describe
*how to convert*, and nothing that describes *this machine*. Window geometry,
the last input folder, the probed encoder list and the output directory all
stay behind, so a preset can be shared, checked into a repo, or kept next to a
library without dragging someone else's paths along.

    {
      "format": "janai.preset",
      "version": 1,
      "name": "Manga 2x -> CBZ",
      "created": "2026-09-13T05:40:00",
      "settings": {"output": {...}, "format": {...}, "upscale": {...}, ...}
    }

The interface saves the working settings on every change, so "remember what I
had" needs no preset at all; presets are for keeping *several* working sets.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

FORMAT = "janai.preset"
VERSION = 1
SUFFIX = ".janai.json"
FOLDER = "presets"

#: The sections a preset carries. Everything else is per-machine state.
SECTIONS: tuple[str, ...] = ("output", "format", "upscale", "perf", "log")

#: Keys inside those sections that are still per-machine and never travel.
VOLATILE: dict[str, tuple[str, ...]] = {
    "output": ("dir",),
    "perf": ("device",),
    "log": ("dir",),
}


class PresetError(Exception):
    """Raised when a file is not a preset we can use."""


# --------------------------------------------------------------------------- #
# building and applying
# --------------------------------------------------------------------------- #
def snapshot(data: dict[str, Any], sections: Sequence[str] = SECTIONS) -> dict[str, Any]:
    """Copy the portable half of a settings dict."""
    out: dict[str, Any] = {}
    for name in sections:
        section = data.get(name)
        if not isinstance(section, dict):
            continue
        skip = set(VOLATILE.get(name, ()))
        out[name] = {k: v for k, v in section.items() if k not in skip}
    return out


def build(name: str, data: dict[str, Any], note: str = "", app_version: str = "") -> dict[str, Any]:
    """Wrap a settings snapshot in the preset envelope."""
    return {
        "format": FORMAT,
        "version": VERSION,
        "name": (name or "Preset").strip(),
        "note": note.strip(),
        "app": app_version,
        "created": datetime.now().isoformat(timespec="seconds"),
        "settings": snapshot(data),
    }


def apply(
    data: dict[str, Any], preset: dict[str, Any], sections: Sequence[str] = SECTIONS
) -> list[str]:
    """Merge a preset into a settings dict in place; returns what changed.

    Merging is per key, so a preset written by an older build cannot delete
    settings it never knew about.
    """
    payload = preset.get("settings")
    if not isinstance(payload, dict):
        raise PresetError("this preset has no settings in it")
    applied: list[str] = []
    for name in sections:
        incoming = payload.get(name)
        if not isinstance(incoming, dict):
            continue
        skip = set(VOLATILE.get(name, ()))
        target = data.setdefault(name, {})
        if not isinstance(target, dict):
            continue
        changed = False
        for key, value in incoming.items():
            if key in skip:
                continue
            if target.get(key) != value:
                target[key] = value
                changed = True
        if changed:
            applied.append(name)
    return applied


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def validate(raw: object) -> dict[str, Any]:
    """Check a parsed file really is a preset, and return it."""
    if not isinstance(raw, dict):
        raise PresetError("that file does not contain a preset")
    if str(raw.get("format") or "") != FORMAT:
        raise PresetError("that file is not a JaNai preset")
    version = raw.get("version")
    if isinstance(version, (int, float)) and int(version) > VERSION:
        raise PresetError(f"that preset was written by a newer version (v{int(version)})")
    if not isinstance(raw.get("settings"), dict):
        raise PresetError("that preset has no settings in it")
    return raw


def read(path: str | Path) -> dict[str, Any]:
    """Load and validate a preset file."""
    target = Path(path)
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PresetError(f"could not read {target.name}: {exc.strerror or exc}") from exc
    except ValueError as exc:
        raise PresetError(f"{target.name} is not valid JSON: {exc}") from exc
    return validate(raw)


def write(path: str | Path, preset: dict[str, Any]) -> Path:
    """Write a preset, creating the folder and replacing atomically."""
    target = Path(path)
    if not target.name.endswith(SUFFIX) and target.suffix.lower() != ".json":
        target = target.with_name(target.name + SUFFIX)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".part")
    temp.write_text(json.dumps(preset, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(target)
    return target


def folder(root: str | Path) -> Path:
    """Where presets live inside the app folder."""
    return Path(root) / FOLDER


def available(root: str | Path) -> list[tuple[str, Path]]:
    """Every readable preset in the presets folder, sorted by name."""
    out: list[tuple[str, Path]] = []
    base = folder(root)
    if not base.is_dir():
        return out
    for path in sorted(base.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        try:
            preset = read(path)
        except PresetError:
            continue
        out.append((str(preset.get("name") or path.stem), path))
    out.sort(key=lambda item: item[0].lower())
    return out


_UNSAFE = re.compile(r"[^A-Za-z0-9 ._-]+")


def filename(name: str) -> str:
    """A safe filename for a preset called `name`."""
    clean = _UNSAFE.sub("-", (name or "preset").strip()).strip(" .-") or "preset"
    return clean[:60] + SUFFIX


def summary(preset: dict[str, Any]) -> str:
    """One line describing what a preset will do, for a confirmation."""
    settings = preset.get("settings") or {}
    upscale = settings.get("upscale") or {}
    output = settings.get("output") or {}
    fmt = settings.get("format") or {}
    mode = str(upscale.get("mode") or "scale")
    scale = upscale.get("scale")
    target = f"{float(scale):g}x" if mode == "scale" and scale else mode
    bits = [target, str(fmt.get("id") or "").upper(), str(output.get("container") or "")]
    rules = upscale.get("rules")
    if isinstance(rules, list) and rules:
        bits.append(f"{len(rules)} rules")
    return "  ·  ".join(b for b in bits if b)
