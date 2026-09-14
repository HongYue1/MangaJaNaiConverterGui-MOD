"""Reading the inside of a CBZ/CBR source.

``open_archive`` is the read side of an archive: it lists the pages and hands
back a reader bound to the still-open container, which is why it is a context
manager rather than a function returning a pair.

It lives in a module of its own, *below* the orchestrators, because both the
real run and the dry run open sources and the two must agree on what counts as
a page and on what a page is called. Keeping it here is what lets the run
orchestrators and the prediction path share it without importing each other.
"""

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from janai.worker.events import log
from janai.worker.planning import (
    UTF8_NAME_FLAG,
    decode_entry_name,
    is_page_entry,
    natural_key,
)

ArchiveReader = Callable[[str], bytes]
"""Reads one entry of an open archive, by the display name it was listed under."""

OpenedArchive = tuple[list[str], ArchiveReader]


def _open_rar(path: Path) -> tuple[Any, list[str]] | None:
    """Open a RAR source and list its pages, or report why it cannot be read.

    Typed `Any` because rarfile ships no stubs and is an optional, lazily
    imported dependency -- the import itself is the usual failure here.
    """
    try:
        import rarfile

        # rarfile decodes entry names itself, so there is no cp437 decode to
        # undo on this path.
        rf = rarfile.RarFile(str(path))
        return rf, sorted((n for n in rf.namelist() if is_page_entry(n)), key=natural_key)
    except Exception as exc:
        log(f"cannot open {path.name}: {exc} (RAR needs unrar/7z on PATH)", "warn")
        return None


@contextmanager
def open_archive(path: Path) -> Iterator[OpenedArchive | None]:
    """Yield (sorted page names, read(name) -> bytes) for a CBZ/CBR, or None.

    A context manager because the caller reads entries *through* the archive:
    it has to stay open for the whole page loop and be closed exactly once
    afterwards. This used to hand back a reader bound to an open ZipFile and
    leave closing to refcounting, which is a CPython implementation detail
    rather than a guarantee -- and until it closes, Windows holds a lock on the
    source the user just converted, so it cannot be moved or deleted.

    The names are *display* names: an entry stored without the UTF-8 flag is
    recovered here, and the reader maps that name back to the raw key it is
    stored under. That split matters because `pack_archive` writes the name it
    is handed into the output CBZ, so a mojibake name would become permanent.

    An unsupported or unopenable container yields None; both callers report
    that themselves. No `yield` may sit inside the RAR branch's `except
    Exception`: a generator that caught the consumer's exception there would
    swallow it, turning a failed chapter into a silent success.
    """
    ext = path.suffix.lower()
    if ext in (".zip", ".cbz"):
        with ZipFile(path) as zf:
            pages = [info for info in zf.infolist() if is_page_entry(info.filename)]
            stored = {info.filename for info in pages}
            raw_by_name: dict[str, str] = {}
            names: list[str] = []
            for info in pages:
                name = decode_entry_name(
                    info.filename, utf8_flag=bool(info.flag_bits & UTF8_NAME_FLAG)
                )
                # A recovery must never hide another entry: if the recovered form is
                # already spoken for, keep the raw name so no page is lost.
                if name != info.filename and (name in stored or name in raw_by_name):
                    name = info.filename
                names.append(name)
                raw_by_name[name] = info.filename
            names.sort(key=natural_key)

            def read_zip(name: str) -> bytes:
                # Maps the display name back to the raw key the entry is stored
                # under; unrecovered names map to themselves.
                return zf.read(raw_by_name.get(name, name))

            yield names, read_zip
        return
    if ext in (".rar", ".cbr"):
        rar = _open_rar(path)
        if rar is None:
            yield None
            return
        rf, rar_names = rar
        # closing() rather than `with rf`: close() is RarFile's documented API
        # in every version, while the context-manager protocol is not.
        with closing(rf):
            yield rar_names, rf.read
        return
    yield None
