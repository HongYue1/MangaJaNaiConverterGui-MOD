"""Proof for the Windows long-path boundary (F25).

Why this gate is shaped the way it is: on the machine this was written on,
``LongPathsEnabled`` is 1, so a 311-character write already SUCCEEDS without
any prefix. An end-to-end "deep tree fails" test would therefore pass against
the unfixed tree and prove nothing. What can be proved everywhere is the
boundary helper's own behaviour, plus the rule that keeps the prefix from
leaking into anything the user sees.

Asserted:
  * an ordinary path is returned untouched -- the property that makes this a
    low-risk change, since every normal run must behave exactly as before
  * a path at or past the threshold gets the extended-length prefix, is
    normalised first, is not prefixed twice, and a UNC path gets the UNC form
  * a prefixed 300+ character path really is writable: mkdir, write, read back,
    and a .cbz through zipfile, which is the exact call handle_archive makes
  * ``path_too_long`` stays quiet for a long-but-legal path and names the
    culprit for a component over 255 -- the residue no prefix can lift
  * the prefix never leaks: no ``str(io_path(...))`` and no
    ``path_key(io_path(...))`` anywhere in the worker, because an event payload
    or a de-dup key carrying ``\\\\?\\`` breaks the interface and F13/F20

Windows-only assertions are skipped on POSIX, where the helper is identity by
design; the file-system assertions below still run there.

Run: backend/python/python.exe scripts/longpath_check.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# The embedded interpreter ships a python313._pth, so it runs isolated and
# ignores PYTHONPATH entirely. Every gate has to put src on the path itself.
sys.path.insert(0, str(ROOT / "src"))

from janai.core.fspath import (
    LONG_PATH_PREFIX,
    MAX_COMPONENT,
    PREFIX_ABOVE,
    io_path,
    path_too_long,
)

WINDOWS = os.name == "nt"
failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def skip(name: str, why: str) -> None:
    print(f"  [SKIP] {name} - {why}")


def long_dir(root: Path, levels: int = 4) -> Path:
    """A directory under *root* comfortably past MAX_PATH."""
    out = root
    for i in range(levels):
        out = out / ("d" * 60 + str(i))
    return out


def check_identity() -> None:
    print("short paths are untouched:")
    for sample in (Path("out.png"), Path("C:/library/out.png"), ROOT / "scripts"):
        got = io_path(sample)
        check(f"unchanged: {str(sample)[:40]}", got == sample, str(got)[:60])
    short = "x" * (PREFIX_ABOVE - 20)
    below = Path(f"C:\\{short}") if WINDOWS else Path(f"/{short}")
    check("a path just under the threshold is not prefixed", io_path(below) == below)


def check_prefixing(root: Path) -> None:
    print("\nlong paths get the extended-length prefix:")
    target = long_dir(root) / "page.png"
    got = str(io_path(target))
    if not WINDOWS:
        skip("prefix applied", "POSIX: io_path is identity by design")
        check("POSIX leaves the long path alone", io_path(target) == target)
        return
    check("prefix applied", got.startswith(LONG_PATH_PREFIX), got[:60])
    check("the original path is still inside it", got.endswith("page.png"))
    check("not prefixed twice", str(io_path(io_path(target))) == got)
    messy = Path(str(long_dir(root)).replace("\\", "/") + "/../" + "e" * 60 + "/p.png")
    fixed = str(io_path(messy))
    check("normalised before prefixing: no forward slashes", "/" not in fixed[4:], fixed[:60])
    check("normalised before prefixing: no '..' left", ".." not in fixed)
    unc = Path("\\\\server\\share\\" + "u" * 240 + "\\p.png")
    check("UNC gets the UNC form", str(io_path(unc)).startswith("\\\\?\\UNC\\server"))


def check_writes(root: Path) -> None:
    print("\na >260 character path is actually writable through io_path:")
    page = long_dir(root) / "page.png"
    target = io_path(page)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x" * 64)
        check("write and read back", target.read_bytes() == b"x" * 64, f"{len(str(page))} chars")
        check("exists() agrees", target.exists())
        target.unlink()
        check("unlink works", not target.exists())
    except OSError as exc:
        check("write and read back", False, f"{type(exc).__name__}: {exc}")

    book = long_dir(root) / "chapter.cbz"
    tmp, dest = io_path(book.with_suffix(".cbz.part")), io_path(book)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as zf:
            zf.writestr("p1.png", b"x" * 64)
        tmp.replace(dest)
        with zipfile.ZipFile(dest) as zf:
            names = zf.namelist()
        check("the .part -> .cbz publish works", names == ["p1.png"], f"{len(str(book))} chars")
        check("stat() on the published archive works", dest.stat().st_size > 0)
    except OSError as exc:
        check("the .part -> .cbz publish works", False, f"{type(exc).__name__}: {exc}")


def check_residue(root: Path) -> None:
    print("\nthe residue no prefix can lift:")
    check("a long but legal path is not reported", path_too_long(long_dir(root) / "p.png") is None)
    name = "n" * 300 + ".png"
    message = path_too_long(root / name)
    if not WINDOWS:
        skip("an over-long component is reported", "POSIX: the limit is the file system's")
        return
    check("an over-long component is reported", bool(message), str(message)[:70])
    check(
        "the message names the length and the limit",
        bool(message)
        and f"{len(name)} characters" in str(message)
        and str(MAX_COMPONENT) in str(message),
        str(message)[:90],
    )


def check_no_leak() -> None:
    """The prefix must never reach a payload, the log or a de-dup key."""
    print("\nthe prefix does not leak out of the boundary:")
    for name in ("job.py", "pipeline.py"):
        text = (ROOT / "src" / "janai" / "worker" / name).read_text(encoding="utf-8")
        check(f"{name}: no str(io_path(...)) into a payload", "str(io_path(" not in text)
        check(f"{name}: no path_key(io_path(...))", "path_key(io_path(" not in text)
        check(f"{name}: still uses the boundary helper", "io_path(" in text)


def main() -> int:
    print(f"os.name={os.name} threshold={PREFIX_ABOVE}")
    with tempfile.TemporaryDirectory(prefix="janai-lp-") as raw:
        root = Path(raw)
        check_identity()
        check_prefixing(root)
        check_writes(root)
        check_residue(root)
        check_no_leak()
        # Clean up through the prefix as well: rmtree on the plain path would
        # be the very failure this gate exists for.
        for path in sorted(io_path(root).rglob("*"), key=lambda p: len(str(p)), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
    print("\nALL PASS" if not failures else f"\nFAILURES: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
