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

from janai.core.filetypes import ARCHIVE_EXTS, IMAGE_EXTS
from janai.core.formats import packs_archive

MACOS_METADATA_DIR = "__MACOSX"
"""Folder macOS adds when it writes a zip; everything under it is metadata."""

APPLE_DOUBLE_PREFIX = "._"
"""Prefix of an AppleDouble sidecar. It carries the page's own extension but
holds a resource fork, so an extension test alone cannot tell them apart."""

UTF8_NAME_FLAG = 0x800
"""Zip general-purpose bit that declares an entry name to be UTF-8. Older comic
tooling omits it, and the zip spec then says cp437, which is what `zipfile`
decodes - turning a Japanese page name into mojibake."""

RESERVED_CHARS = re.compile(r'[<>:"/\\|?*]')
"""Characters Windows rejects in a file name; replaced with an underscore."""

MIN_INDEX_WIDTH = 3
"""Zero-padding floor for `{index0}`, so a 12-page chapter still yields 001."""

MAX_DEDUPE_ATTEMPTS = 10000
"""Give up suffixing " (n)" after this many tries and overwrite instead of
looping forever on a pathological directory."""


def natural_key(name: str) -> list[int | str]:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def is_page_entry(name: str) -> bool:
    """Whether an archive entry is a page rather than something alongside one.

    An extension test alone is not enough: a zip written on macOS carries an
    AppleDouble sidecar per file, with the page's own extension, so the
    decoder is handed resource forks. A healthy chapter then reports failed
    pages, and the per-chapter progress total counts entries that were never
    images.

    Only the exact AppleDouble shapes are rejected - a name that merely
    contains "._", or a folder that merely starts with "__MACOSX", is a real
    page and must survive.
    """
    if name.endswith("/"):
        return False
    # Zip stores posix separators; some writers emit backslashes anyway.
    parts = name.replace("\\", "/").split("/")
    if MACOS_METADATA_DIR in parts:
        return False
    base = parts[-1]
    if base.startswith(APPLE_DOUBLE_PREFIX):
        return False
    return Path(base).suffix.lower() in IMAGE_EXTS


def decode_entry_name(name: str, *, utf8_flag: bool) -> str:
    """Undo the cp437 decode `zipfile` applies to an unflagged entry name.

    Self-proving, which is why it is safe to do unasked: cp437 maps all 256 byte
    values, so re-encoding recovers the bytes exactly as stored, and those bytes
    only decode as UTF-8 if they really were UTF-8. A wrong guess raises instead
    of renaming a page.

    Names in some other codepage (Shift-JIS, GBK) are therefore left exactly as
    they are - ugly but readable. Recovering those needs a detector or a
    user-set codepage, and guessing would rename pages silently.
    """
    if utf8_flag:
        return name
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def gather_units(inp: dict[str, Any]) -> list[dict[str, Any]]:
    raw = Path(str(inp.get("path") or "")).expanduser()
    mode = str(inp.get("mode") or ("single" if raw.is_file() else "bulk"))
    include_archives = bool(inp.get("include_archives", True))
    recursive = bool(inp.get("recursive", True))
    units: list[dict[str, Any]] = []

    def add(p: Path, base: Path) -> None:
        ext = p.suffix.lower()
        if ext in IMAGE_EXTS:
            units.append({"path": p, "base": base, "kind": "image"})
        elif include_archives and ext in ARCHIVE_EXTS:
            units.append({"path": p, "base": base, "kind": "archive"})

    if raw.is_file():
        add(raw, raw.parent)
        # `base` has to stay the parent folder -- `keep_structure` and the
        # resume key are both relative to it -- but that folder is not a
        # chapter. Without this flag a one-file run packed itself as
        # "Downloads.cbz", named after whatever folder the file happened to
        # sit in. See bundle_stem().
        for unit in units:
            unit["solo"] = True
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
    unit: dict[str, Any],
    out_dir: Path,
    pattern: str,
    ext: str,
    keep_structure: bool,
    index: int,
    total: int,
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


def relative_dir(unit: dict[str, Any]) -> Path:
    """Where this file sits inside the input folder."""
    src: Path = unit["path"]
    base: Path = unit["base"]
    try:
        return src.parent.relative_to(base)
    except ValueError:
        return Path()


def bundle_stem(unit: dict[str, Any]) -> str:
    """What the archive holding this page is named after.

    A chosen file names itself; a chosen folder names the folder. `base`
    cannot answer this alone: for a single-file run it is the parent folder,
    so both "one .cbz" and "a .cbz per folder" used to name the archive after
    a folder the user never picked.
    """
    stem = Path(str(unit["path"])).stem
    if unit.get("solo"):
        return stem
    base = Path(str(unit["base"]))
    return base.name if base.is_dir() else stem


def chapter_dest(unit: dict[str, Any], out_dir: Path, keep_structure: bool) -> Path:
    """The .cbz that this file's own folder becomes."""
    rel = relative_dir(unit)
    name = rel.name or bundle_stem(unit)
    parent = out_dir / (rel.parent if keep_structure else Path())
    return parent / f"{safe_name(name)}.cbz"


def build_tasks(
    units: list[dict[str, Any]], out_dir: Path, keep_structure: bool, container_id: str
) -> list[dict[str, Any]]:
    """Group the units into the things this run will actually produce.

    Loose images stay in one run of consecutive units so the decoder can read
    ahead across the whole batch. With a cbz container they are grouped by the
    folder they came from instead, which is what turns "one folder per chapter"
    into one archive per chapter.
    """
    pack = packs_archive(container_id)
    tasks: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}
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
                dest = out_dir / f"{safe_name(bundle_stem(unit))}.cbz"
            else:
                dest = chapter_dest(unit, out_dir, keep_structure)
            group = {"kind": "bundle", "key": key, "dest": dest, "units": []}
            groups[key] = group
            tasks.append(group)
        group["units"].append(unit)
    return tasks
