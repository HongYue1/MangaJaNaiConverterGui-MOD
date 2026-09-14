"""Works out what a run will produce, before a single pixel is decoded.

Discovery (`gather_units`), output naming (`format_name`, `resolve_out`,
`unique_path`) and grouping (`build_tasks`) all answer the same question: which
inputs become which files. Keeping them together means the whole "one folder per
chapter becomes one .cbz" decision can be read in one place.

Two things here are user-visible and must not drift:

* `natural_key` is why page 2 sorts before page 10. It decides page order inside
  an archive, so the reader sees any change to it.
* `format_name` and `safe_name` strip the characters Windows forbids in a file
  name. Loosening either turns a valid job into an OSError halfway through.

This module is deliberately free of numpy, torch and pyvips: a dry run and the
GUI's job preview walk these functions and must not pay to import the imaging
stack.
"""

from __future__ import annotations

import re
from collections.abc import Container
from pathlib import Path
from typing import Any

from janai.core.formats import packs_archive

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
"""Extensions treated as a page. Also filters entries when listing an archive,
so a chapter's cover.txt or ComicInfo.xml is never fed to the model."""

ARCHIVE_EXTS = {".zip", ".cbz", ".rar", ".cbr"}
"""Extensions treated as a container of pages rather than a page."""

RESERVED_CHARS = re.compile(r'[<>:"/\\|?*]')
"""Characters Windows rejects in a file name; replaced with an underscore."""

MIN_INDEX_WIDTH = 3
"""Zero-padding floor for `{index0}`, so a 12-page chapter still yields 001."""

MAX_DEDUPE_ATTEMPTS = 10000
"""Give up suffixing " (n)" after this many tries and overwrite instead of
looping forever on a pathological directory."""


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
    width = max(MIN_INDEX_WIDTH, len(str(total)))
    out = pattern or "{name}"
    repl = {
        "{name}": src.stem,
        "{parent}": src.parent.name,
        "{index}": str(index),
        "{index0}": str(index).zfill(width),
    }
    for key, value in repl.items():
        out = out.replace(key, value)
    return RESERVED_CHARS.sub("_", out).strip() or src.stem


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


def path_key(path: Path) -> str:
    """Identity of an output path for de-dup purposes.

    Case-folded because Windows treats A.png and a.png as the same file, so
    comparing raw strings would let one page quietly overwrite another on the
    platform this app actually ships on.
    """
    return str(path).casefold()


def unique_path(path: Path, taken: Container[str] = frozenset()) -> Path:
    """A path that neither exists on disk nor has been claimed by this run.

    `taken` holds the `path_key` values already handed out by the current job.
    It is required rather than decorative: writes are queued, so `exists()`
    cannot see a sibling page whose write is still sitting in the write pool,
    and two sources that resolve to one name would collapse into one file.
    """
    if not path.exists() and path_key(path) not in taken:
        return path
    stem, suffix = path.stem, path.suffix
    for n in range(1, MAX_DEDUPE_ATTEMPTS):
        cand = path.with_name(f"{stem} ({n}){suffix}")
        if not cand.exists() and path_key(cand) not in taken:
            return cand
    return path


def safe_name(name: Any) -> str:
    return RESERVED_CHARS.sub("_", str(name)).strip() or "output"


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
