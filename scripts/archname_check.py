"""Live proof for what `open_archive` calls a page (F22), what it calls it (F21),
and that it closes what it opens (F19).

`archive_check.py` covers what happens to pages once they are found; this gate
covers the step before that - deciding which zip entries ARE pages. It needs no
GPU and decodes nothing, but it imports `janai.worker.job`, which pulls in the
imaging stack, so it stays local-only. The pure-name half of the same policy is
asserted dependency-free in `smoke.py`.

Why it exists: a zip written on macOS carries an AppleDouble sidecar for every
file (`__MACOSX/._page-001.jpg`). Those are resource forks with an image
extension, never images. Feeding them to the decoder makes a healthy chapter
report failed pages, and inflates both the per-chapter progress total and the
dry run's `entries` preview.

The `[names]` section covers the other half. A zip entry name stored without the
UTF-8 flag (bit 0x800) is decoded as cp437, so a Japanese page name arrives as
mojibake - and `handle_archive` writes the name it is given into the output CBZ,
which makes the corruption permanent. Recovery is self-proving: cp437 maps all
256 byte values, so re-encoding recovers the bytes exactly, and those bytes only
decode as UTF-8 if they really were UTF-8. Names in some other codepage
(Shift-JIS) cannot be recovered without guessing, so they must be left exactly as
they are - ugly but readable - and a recovery must never collapse two entries
onto one name, which would drop a page.

Run: backend/python/python.exe scripts/archname_check.py
"""

from __future__ import annotations

import io
import sys
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The Windows console is cp1252 and these fixtures carry non-ASCII names; the
# worker reconfigures its own stdout the same way (see worker.main(), named
# rather than cited by line so the reference cannot go stale). Only the
# concrete TextIOWrapper has reconfigure(); typeshed types sys.stdout as
# TextIO, so narrow instead of suppressing the error.
if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from janai.worker.archives import open_archive

PAGE = b"\x89PNG\r\n\x1a\nnot decoded by this gate"
APPLE_DOUBLE = b"\x00\x05\x16\x07resource fork, not an image"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def build(path: Path, entries: dict[str, bytes]) -> None:
    with ZipFile(path, "w", ZIP_STORED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)


def legacy_zip(path: Path, raw_name: bytes, extra: dict[str, bytes] | None = None) -> None:
    """Write a zip whose entry name is `raw_name` with NO UTF-8 flag.

    zipfile always sets bit 0x800 for a non-ASCII name, so the only way to
    produce the legacy shape is to write an ASCII placeholder of exactly the
    same byte length and patch the bytes afterwards. The name is not covered by
    the CRC and its length is unchanged, so every stored offset stays valid.
    """
    placeholder = "a" * (len(raw_name) - 4) + ".jpg"
    assert len(placeholder.encode("ascii")) == len(raw_name), "placeholder length must match"
    with ZipFile(path, "w", ZIP_STORED) as zf:
        zf.writestr(placeholder, PAGE)
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    blob = path.read_bytes()
    assert blob.count(placeholder.encode("ascii")) == 2, "expected local + central copies"
    path.write_bytes(blob.replace(placeholder.encode("ascii"), raw_name))


@contextmanager
def opened(path: Path) -> Iterator[tuple[list[str], Callable[[str], bytes]]]:
    """Yield (names, reader) as `pack_archive` receives them.

    A context manager because `open_archive` is one: the reader is only valid
    while the archive is open, so every assertion about it belongs inside.
    """
    with open_archive(path) as pair:
        assert pair is not None, f"open_archive refused {path.name}"
        yield list(pair[0]), pair[1]


def reads(reader, name: str) -> bool:
    """Whether the caller can read a page back using the name it was given."""
    try:
        return bool(reader(name))
    except Exception:
        return False


def listed(path: Path) -> list[str]:
    with open_archive(path) as pair:
        return list(pair[0]) if pair else []


def main() -> int:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        base = Path(tmp)

        print("[control] what a page is must not change")
        plain = base / "Plain.cbz"
        build(
            plain,
            {
                "page-010.jpg": PAGE,
                "page-002.jpg": PAGE,
                "ComicInfo.xml": b"<ComicInfo/>",
                "sub/page-003.png": PAGE,
            },
        )
        names = listed(plain)
        check("non-image entries are excluded", "ComicInfo.xml" not in names, repr(names))
        check("nested pages are included", "sub/page-003.png" in names)
        check(
            "natural order: page-002 before page-010",
            names.index("page-002.jpg") < names.index("page-010.jpg")
            if {"page-002.jpg", "page-010.jpg"} <= set(names)
            else False,
            repr(names),
        )
        check("exactly the three real pages", len(names) == 3, f"{len(names)} entries")

        print("\n[junk] macOS sidecars are not pages")
        mac = base / "Mac.cbz"
        build(
            mac,
            {
                "page-001.jpg": PAGE,
                "__MACOSX/._page-001.jpg": APPLE_DOUBLE,
                "__MACOSX/sub/._page-002.jpg": APPLE_DOUBLE,
                "sub/._page-002.jpg": APPLE_DOUBLE,
                "._cover.png": APPLE_DOUBLE,
                "sub/page-002.jpg": PAGE,
            },
        )
        names = listed(mac)
        check(
            "the real pages survive",
            {"page-001.jpg", "sub/page-002.jpg"} <= set(names),
            repr(names),
        )
        check("__MACOSX sidecar dropped", "__MACOSX/._page-001.jpg" not in names)
        check("nested __MACOSX sidecar dropped", "__MACOSX/sub/._page-002.jpg" not in names)
        check("AppleDouble beside the page dropped", "sub/._page-002.jpg" not in names)
        check("AppleDouble at the root dropped", "._cover.png" not in names)
        check("two pages for two pages", len(names) == 2, f"{len(names)} entries")

        print("\n[junk] but a legitimate name is never over-filtered")
        odd = base / "Odd.cbz"
        build(
            odd,
            {
                "page._final.jpg": PAGE,  # contains '._' but does not start with it
                "__MACOSX_fanbook/page-001.jpg": PAGE,  # merely starts with the same letters
                ".hidden/page-001.jpg": PAGE,  # a dot-directory is not AppleDouble
            },
        )
        names = listed(odd)
        check("'._' inside a name is kept", "page._final.jpg" in names, repr(names))
        check("a folder merely prefixed __MACOSX is kept", "__MACOSX_fanbook/page-001.jpg" in names)
        check("a dot-directory is kept", ".hidden/page-001.jpg" in names)

        true_name = "\u7b2c01\u8a71.jpg"  # 第01話.jpg

        print("\n[names] a flagged non-ASCII name is passed through untouched")
        flagged = base / "Flagged.cbz"
        build(flagged, {true_name: PAGE})
        with opened(flagged) as (names, reader):
            check("the flagged name is unchanged", names == [true_name], repr(names))
            check("and it reads back", reads(reader, true_name))

        print("\n[names] UTF-8 bytes with the flag unset are recovered")
        noflag = base / "NoFlag.cbz"
        legacy_zip(noflag, true_name.encode("utf-8"))
        with opened(noflag) as (names, reader):
            check("the true name is recovered, not mojibake", names == [true_name], repr(names))
            check("the recovered name is what the reader accepts", reads(reader, true_name))

        print("\n[names] bytes that are not UTF-8 are left alone")
        sjis = base / "Sjis.cbz"
        legacy_zip(sjis, true_name.encode("shift_jis"))
        with opened(sjis) as (names, reader):
            check("the entry is still listed", len(names) == 1, repr(names))
            check("it is still readable", reads(reader, names[0]) if names else False)
            check(
                "it is not mangled with replacement characters",
                all("\ufffd" not in n for n in names),
                repr(names),
            )

        print("\n[names] a recovery that would collide must not drop a page")
        clash = base / "Clash.cbz"
        legacy_zip(clash, true_name.encode("utf-8"), extra={true_name: PAGE})
        with opened(clash) as (names, reader):
            check("both entries survive", len(names) == 2, repr(names))
            check("their names stay distinct", len(set(names)) == 2, repr(names))
            check("both are readable", all(reads(reader, n) for n in names))

        print("\n[close] the archive is released when the caller is done (F19)")
        held = base / "Held.cbz"
        build(held, {"page-001.jpg": PAGE})
        cm = open_archive(held)
        check(
            "open_archive is a context manager",
            hasattr(cm, "__enter__") and hasattr(cm, "__exit__"),
            f"got {type(cm).__name__}",
        )
        if hasattr(cm, "__enter__"):
            with cm as pair:
                check("it yields the (names, reader) pair", pair is not None and len(pair) == 2)
                held_reader = pair[1] if pair else None
                check("a page reads inside the block", reads(held_reader, "page-001.jpg"))
            check(
                "the reader is dead once the block exits",
                not reads(held_reader, "page-001.jpg"),
            )
            # The user-visible half: Windows refuses to move or delete a file
            # while a handle on it is open, so a leaked archive locks the very
            # source the user just converted.
            moved = base / "Held-moved.cbz"
            try:
                held.rename(moved)
                lock = ""
            except OSError as exc:
                lock = f"{type(exc).__name__}: {exc}"
            check("the source file can be renamed afterwards", moved.exists(), lock)
            with open_archive(base / "Nope.7z") as unsupported:
                check("an unsupported container yields None", unsupported is None)

    print(f"\n{len(failures)} FAILED: {failures}" if failures else "\nALL PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
