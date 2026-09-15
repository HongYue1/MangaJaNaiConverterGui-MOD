"""Per-unit orchestration: one source archive, or one run of loose images.

Lifted out of `run_job` because everything in here answers one question --
what happens to a single planned unit -- while `run_job` answers a different
one: what the job means, and which unit comes next. Separating them leaves
`run_job` as setup plus a dispatch loop.

The dependency runs one way: `job` -> `orchestrate` -> the primitives below
(`archives`, `pipeline`, `page`, `reporting`). The archive readers were moved
into :mod:`janai.worker.archives` first precisely so this module and the dry
run in :mod:`janai.worker.job` could both use them without importing each
other.

Invariants that live here:

* One ``file`` event per unit of work. Every exit from `handle_archive`, and
  every page in `run_images`, emits exactly one, because the interface counts
  units by counting those events.
* A page that fails inside an archive is counted twice -- once locally, for
  that archive's own ``file`` line, and once as job-level ``pages_failed`` --
  but never as ``failed``, which alone decides ``done.ok`` and the exit code.
* Entry names are chosen on the producing thread, so output order inside a CBZ
  follows read order rather than whichever encode finished first. Order is
  user-visible in a reader.
"""

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, BadZipFile, ZipFile

from janai.core.fspath import io_path, path_too_long
from janai.worker import imageio
from janai.worker.archives import ArchiveReader, open_archive
from janai.worker.control import CTRL, Cancelled
from janai.worker.events import emit, log
from janai.worker.imagetypes import ImageArray
from janai.worker.page import PageEncoder, PageWorker
from janai.worker.pipeline import BundleWriter, Counters, PagePacker, Unit, WritePool, prefetch
from janai.worker.planning import format_name, path_key, resolve_out, unique_path
from janai.worker.reporting import JobReporter
from janai.worker.resume import ResumeLog, planned_entries, unit_key


class NoPagesError(Exception):
    """An archive finished with nothing worth publishing.

    Raised rather than handled inline so the abandoned ``.part`` is discarded,
    the unit is counted failed and the ``file`` event is emitted by the *one*
    existing failure path in ``handle_archive``. Duplicating that cleanup risks
    a second ``file`` event for the same archive, and the interface counts one
    unit of work per ``file`` event.
    """


@dataclass(frozen=True, slots=True)
class UnitRunner:
    """Everything one job needs in order to convert one unit.

    Frozen, and constructed once per job, because none of these may change
    while the run is in flight: the settings are what the plan was built from,
    and the collaborators are shared by reference on purpose. `counters` above
    all -- it is locked, it is the job's single tally, and rebinding it would
    let the progress the user watches disagree with the final ``done`` payload.

    It exists as an object rather than a pile of parameters because these
    thirteen names were previously captured by four closures inside `run_job`;
    passing them positionally would be a thirteen-argument call at every site.
    """

    pager: PageWorker
    encoder: PageEncoder
    reporter: JobReporter
    writer: WritePool
    bundle: BundleWriter
    counters: Counters
    out_dir: Path
    pattern: str
    ext: str
    keep_structure: bool
    overwrite: bool
    total: int
    io_workers: int
    #: Resume state for this output folder. Added after the thirteen fields
    #: above and shared by reference like the other collaborators, because it
    #: is this job's single record of which sources are already finished.
    resume: ResumeLog

    def handle_archive(self, index: int, unit: Unit) -> None:
        src: Path = unit["path"]
        key = unit_key(src, Path(str(unit["base"])))
        dest = resolve_out(
            unit, self.out_dir, self.pattern, ".cbz", self.keep_structure, index, self.total
        )
        # Asked before `exists()` and regardless of `overwrite`, because it is
        # a different question: the manifest records what *this* job finished
        # under *these* settings, which is exactly what a resumed run needs to
        # know. `exists()` cannot tell a chapter this job published from an
        # unrelated file sitting at that path, and says nothing at all when
        # the user has overwrite on.
        if self.resume.is_done(key):
            self.counters.bump("skipped")
            emit(
                "file",
                i=index,
                total=self.total,
                path=str(src),
                out=str(dest),
                error="already done, skipped",
            )
            return
        if io_path(dest).exists() and not self.overwrite:
            self.counters.bump("skipped")
            emit(
                "file",
                i=index,
                total=self.total,
                path=str(src),
                out=str(dest),
                error="exists, skipped",
            )
            return
        too_long = path_too_long(dest)
        if too_long:
            # Fail this chapter, not the run, and name the culprit -- the same
            # shape as the unsupported-archive branch just below.
            self.counters.bump("failed")
            emit("file", i=index, total=self.total, path=str(src), error=f"OSError: {too_long}")
            return
        io_path(dest).parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        # Scoped to exactly the span that reads from it: `pack_archive` pulls
        # every page through `reader` and nothing touches the source after it
        # returns, so the handle is released the moment packing is done.
        with open_archive(src) as opened:
            if opened is None:
                self.counters.bump("failed")
                emit("file", i=index, total=self.total, path=str(src), error="unsupported archive")
                return
            names, reader = opened
            self.pack_archive(index, src, dest, names, reader, started, key)

    def pack_archive(
        self,
        index: int,
        src: Path,
        dest: Path,
        names: list[str],
        reader: ArchiveReader,
        started: float,
        key: str = "",
    ) -> None:
        """Encode every page of one already-open archive into its output CBZ.

        Split from `handle_archive` so the source can be held by a `with` for
        precisely as long as it is read: this half is the only code that calls
        `reader`. The failure accounting lives here because the `.part` file it
        has to clean up is created here too.
        """
        tmp = dest.with_suffix(".cbz.part")
        # `dest` and `tmp` stay plain: they are what the `file` event and the
        # log carry. Only these two handles cross into the file system, so only
        # they may wear the extended-length prefix.
        tmp_io, dest_io = io_path(tmp), io_path(dest)
        written = 0
        failed_entries = 0
        seen: set[str] = set()
        # What this run *would* name each page, worked out without decoding
        # anything, purely so a surviving `.part` can be checked against it.
        planned = planned_entries(names, self.ext)
        resumed = self.resume_point(key, tmp, planned)
        if resumed:
            # Re-claim the names already in the archive, so the de-dup inside
            # the loop cannot hand one of them out a second time.
            seen.update(entry.lower() for entry in planned[:resumed])
            log(f"{src.name}: resuming after {resumed} of {len(names)} pages", "info")

        def on_page_fail(entry: str, exc: BaseException) -> None:
            """Report a page the packer could not encode. Runs on the pack thread.

            The same two tallies the decode half below keeps, for the same
            reason. Both are safe off-thread: ``counters`` is locked and
            ``emit`` holds the stdout lock.
            """
            self.counters.bump("pages_failed")
            log(f"{src.name}:{entry}: {exc}", "warn")
            log(traceback.format_exc(limit=4), "debug")

        try:
            # Append when resuming so the pages already packed keep their
            # place: "w" would truncate them, and page order is user-visible
            # in a reader.
            with ZipFile(tmp_io, "a" if resumed else "w", ZIP_STORED) as zf:
                # Encoding is not cheap beside the upscale it follows -- ~9% of a
                # page for PNG, ~72% for AVIF -- and it used to run inline, so
                # the GPU idled through all of it. The packer overlaps it with
                # the next page's upscale, with one worker so pages still land
                # in the order they were read.
                packer = PagePacker(zf, self.encoder.encode, on_page_fail)
                try:
                    # `k` stays the absolute page number so progress still
                    # reads 1..N and the de-dup suffix keeps its meaning.
                    for k, name in enumerate(names[resumed:], resumed + 1):
                        if CTRL.cancelled:
                            raise Cancelled
                        CTRL.gate()
                        emit(
                            "progress",
                            i=index,
                            total=self.total,
                            path=str(src),
                            sub_i=k,
                            sub_n=len(names),
                        )
                        try:
                            raw = reader(name)
                            image, _gray, _model, _info = self.pager.run(
                                imageio.read_image_bytes(raw, name), name
                            )
                            # Re-encoding collapses distinct source names onto
                            # one output name - a.jpg and a.png both become
                            # a.png - and a zip stores two entries under the
                            # identical name without complaint, so readers show
                            # one page twice or drop one and nothing in the log
                            # says which. Same de-dup the loose-file bundle path
                            # already applies. Named here, on the producer
                            # thread, so the suffix follows page order rather
                            # than whichever encode happened to finish first.
                            entry = str(Path(name).with_suffix(self.ext).as_posix())
                            while entry.lower() in seen:
                                entry = f"{entry[: -len(self.ext)]}_{k}{self.ext}"
                            seen.add(entry.lower())
                            packer.add(entry, image)
                        except Cancelled:
                            raise
                        except Exception as exc:
                            # A page that fails here is dropped and the CBZ is
                            # silently short: the entry count was the only
                            # trace, and a short chapter looks like a short
                            # chapter. Count it so the summary line can say so.
                            #
                            # Two tallies on purpose: this archive's own count
                            # feeds its `file` line, and the job-level
                            # `pages_failed` puts the loss in the `done`
                            # summary, so the run as a whole admits it. NEITHER
                            # is counters["failed"], which alone sets done.ok
                            # and the exit code -- a chapter that lost one
                            # unreadable page still converted, so the process
                            # still exits 0. That split is the chosen policy,
                            # not an oversight. Losing *every* page is a
                            # different case, and is caught below.
                            failed_entries += 1
                            self.counters.bump("pages_failed")
                            log(f"{src.name}:{name}: {exc}", "warn")
                            log(traceback.format_exc(limit=4), "debug")
                finally:
                    # The zip must not close under an in-flight writestr, and
                    # the packer's tallies are only readable once every task has
                    # joined. A cancelled run still drops its queued pages
                    # rather than encoding pages nobody is waiting for; what
                    # already reached the .part is kept for the next run.
                    packer.close(drain=not CTRL.cancelled)
                    # Pages carried over from an interrupted run count as well.
                    # Without them a fully-resumed archive would look empty and
                    # be thrown away by the guard below.
                    written = resumed + packer.written
                    failed_entries += packer.failed
            if written == 0:
                # Publishing now would put an EMPTY .cbz where a chapter
                # belongs - and with overwrite on, over a good one - while the
                # run still reported success. Nothing was converted, so this is
                # a failed unit: the far end of the policy above. Same answer
                # when the archive held no page to begin with, because an empty
                # output is never the right one.
                raise NoPagesError(
                    f"{failed_entries} of {len(names)} pages failed"
                    if failed_entries
                    else "no pages in archive"
                )
            tmp_io.replace(dest_io)
            # Only now. The manifest's one promise is that a recorded unit is
            # published, and this rename is the moment that becomes true.
            self.resume.mark_done(key, dest, written)
            self.counters.bump("processed")
            emit(
                "file",
                i=index,
                total=self.total,
                path=str(src),
                out=str(dest),
                ms=int((time.perf_counter() - started) * 1000),
                bytes=dest_io.stat().st_size,
                entries=written,
                failed=failed_entries,
            )
        except Cancelled:
            # Keep what was packed, so the next run carries on inside this
            # archive instead of starting it over. Nothing is published by
            # doing so: `replace()` above is still the only way `dest` is ever
            # created, so a partial chapter cannot pose as a finished one.
            self.keep_partial(key, tmp)
            raise
        except Exception as exc:
            tmp_io.unlink(missing_ok=True)
            # The .part is gone, so the record pointing at it has to go too,
            # or the next run would try to resume from a file that is no
            # longer there.
            self.resume.drop(key)
            self.counters.bump("failed")
            emit(
                "file",
                i=index,
                total=self.total,
                path=str(src),
                error=f"{type(exc).__name__}: {exc}",
            )

    def resume_point(self, key: str, tmp: Path, planned: list[str]) -> int:
        """How many pages of this archive a surviving ``.part`` already holds.

        Zero means start over. The survivor must hold exactly the entry names
        of a *prefix* of this run's plan: a gap -- a page that failed last time
        -- or an unexpected name means it cannot be appended to without
        changing page order, which is user-visible in a reader, so it is
        discarded instead of being repaired on a guess.
        """
        recorded = self.resume.partial_entries(key)
        if not recorded or not io_path(tmp).exists():
            return 0
        try:
            with ZipFile(io_path(tmp)) as zf:
                existing = zf.namelist()
        except (OSError, BadZipFile) as exc:
            log(f"ignoring an unreadable partial archive: {exc}", "debug")
            return 0
        if existing != recorded or planned[: len(existing)] != existing:
            return 0
        return len(existing)

    def keep_partial(self, key: str, tmp: Path) -> None:
        """Record how far an interrupted archive got, and keep its ``.part``.

        The entry names are read back from the closed zip rather than tracked
        in memory, because only the file itself knows which encodes actually
        landed before the cancel arrived. An unreadable or empty survivor is
        deleted: it could only mislead the next run.
        """
        if not self.resume.enabled:
            io_path(tmp).unlink(missing_ok=True)
            return
        try:
            with ZipFile(io_path(tmp)) as zf:
                entries = zf.namelist()
        except (OSError, BadZipFile) as exc:
            log(f"discarding an unusable partial archive: {exc}", "debug")
            io_path(tmp).unlink(missing_ok=True)
            self.resume.drop(key)
            return
        if not entries:
            io_path(tmp).unlink(missing_ok=True)
            self.resume.drop(key)
            return
        self.resume.mark_partial(key, tmp, entries)

    def read_unit(self, unit: Unit) -> ImageArray | None:
        """Decode one planned unit ahead of the pipeline, on a prefetch thread.

        Named for the thing it reads -- a *unit* -- because `pack_archive`
        takes a `reader` parameter that reads *archive entries*. As sibling
        closures in one scope the two names shadowed each other.

        ``None`` means "nothing to decode ahead of time": an archive unit is
        opened by `handle_archive` on the consuming side instead.
        """
        if unit["kind"] == "image":
            return imageio.read_image(unit["path"])
        return None

    def run_images(self, items: list[Unit], into: dict[str, Any] | None) -> None:
        """Upscale a run of images, either to loose files or into one archive."""
        count = len(items)
        # `seen` claims entry names inside a bundle; `taken` claims paths on
        # disk. Both exist because re-encoding is not injective, and the two
        # namespaces de-dupe differently.
        seen: set[str] = set()
        taken: set[str] = set()
        position = 0
        for unit, payload in prefetch(items, self.io_workers, self.read_unit):
            position += 1
            if CTRL.cancelled:
                raise Cancelled
            CTRL.gate()
            index = int(unit.get("index") or position)
            src: Path = unit["path"]
            emit(
                "progress",
                i=index,
                total=self.total,
                path=str(src),
                sub_i=position if into else 0,
                sub_n=count if into else 0,
            )
            if isinstance(payload, Exception):
                self.counters.bump("failed")
                emit(
                    "file",
                    i=index,
                    total=self.total,
                    path=str(src),
                    error=f"read failed: {type(payload).__name__}: {payload}",
                )
                continue
            dest: Path | None = None
            # Stays empty for a bundle member: those are recorded once, by the
            # job, after the archive's atomic publish. A per-page record here
            # would claim pages that so far exist only inside a `.part`.
            resume_key = ""
            if into is None:
                dest = resolve_out(
                    unit,
                    self.out_dir,
                    self.pattern,
                    self.ext,
                    self.keep_structure,
                    index,
                    self.total,
                )
                # The manifest is asked before exists(), and regardless of
                # `overwrite`, exactly as the archive path asks it: it records
                # what THIS job finished, which exists() cannot tell apart from
                # an unrelated file that happens to sit at the destination.
                resume_key = unit_key(src, Path(str(unit["base"])))
                if self.resume.is_done(resume_key):
                    self.counters.bump("skipped")
                    emit(
                        "file",
                        i=index,
                        total=self.total,
                        path=str(src),
                        out=str(dest),
                        error="already done, skipped",
                    )
                    continue
                # a.jpg and a.png both resolve to a.png, and a {parent} pattern
                # collapses a whole folder onto one name. A name this run has
                # already handed out must NOT take the skip branch: it belongs
                # to a sibling page whose write may still be queued, so exists()
                # cannot tell it apart from output left by an earlier run.
                # path_key stays on the PLAIN path: a reservation keyed on a
                # prefixed string would never match the same name again.
                claimed = path_key(dest) in taken
                if not claimed and io_path(dest).exists() and not self.overwrite:
                    self.counters.bump("skipped")
                    emit(
                        "file",
                        i=index,
                        total=self.total,
                        path=str(src),
                        out=str(dest),
                        error="exists, skipped",
                    )
                    continue
                if claimed or dest.resolve() == src.resolve():
                    dest = unique_path(dest, taken)
                taken.add(path_key(dest))
            started = time.perf_counter()
            try:
                image, gray, model_name, info = self.pager.run(payload, src.name)
            except Exception as exc:
                if CTRL.cancelled:
                    raise Cancelled from exc
                self.counters.bump("failed")
                emit(
                    "file",
                    i=index,
                    total=self.total,
                    path=str(src),
                    error=f"{type(exc).__name__}: {exc}",
                )
                log(traceback.format_exc(limit=4), "debug")
                continue
            if into is None and dest is not None:
                self.writer.submit(
                    self.reporter.write_page,
                    index,
                    src,
                    dest,
                    image,
                    gray,
                    model_name,
                    info,
                    started,
                    resume_key,
                )
                continue
            if into is None:
                # A page with neither a bundle nor a destination has nowhere to
                # go. This used to surface two lines below as "NoneType is not
                # subscriptable", which named the wrong cause entirely.
                raise RuntimeError(f"no output destination for {src}")
            entry = format_name(self.pattern, src, position, count) + self.ext
            while entry.lower() in seen:
                entry = f"{entry[: -len(self.ext)]}_{position}{self.ext}"
            seen.add(entry.lower())
            self.bundle.add(
                entry,
                image,
                {
                    "i": index,
                    "src": str(src),
                    "gray": gray,
                    "model": model_name,
                    "info": info,
                    "started": started,
                    "bundle": str(into["dest"]),
                },
            )
