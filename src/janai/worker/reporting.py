"""Job-level reporting: page writes, page failures, bundle completion.

Lifted out of `run_job` because all four callbacks here do the same two things
-- move a counter and emit exactly one JSONL event -- and none of them decides
anything about pixels or output paths. Keeping them in one module makes the
wire format reviewable in one place: the GUI parses these `file` and `bundle`
events, so their keys may be added to but never renamed or dropped.

Every method can run off the main thread: `write_page` is submitted to the
`WritePool`, and the bundle callbacks are invoked by `BundleWriter`. They may
therefore only touch `Counters`, which is locked, and `emit`, which serialises
its writes.
"""

import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from janai.core.fspath import io_path, path_too_long
from janai.worker.events import emit, log
from janai.worker.page import PageEncoder
from janai.worker.pipeline import Counters


@dataclass(frozen=True, slots=True)
class JobReporter:
    """Counts one job's pages and emits the events the GUI renders.

    Frozen because `total` is fixed when the job starts and `counters` is the
    single tally for the run: rebinding either mid-run would let the progress
    the user watches disagree with the final `done` payload. The `Counters`
    object itself is mutable and locked -- that is the intended shared state.
    """

    counters: Counters
    total: int
    encoder: PageEncoder

    def write_page(
        self,
        index: int,
        src: Path,
        dest: Path,
        image: Any,
        gray: bool,
        model_name: str,
        info: dict,
        started: float,
    ) -> None:
        try:
            # A name longer than the file system allows is not a MAX_PATH
            # problem and no prefix lifts it, so say which name and how long
            # rather than letting a bare "[Errno 22] Invalid argument" be the
            # whole explanation the user gets from the handler below.
            too_long = path_too_long(dest)
            if too_long:
                raise OSError(too_long)
            data = self.encoder.encode(image)
            # `dest` stays plain: it is what the `file` event below reports.
            target = io_path(dest)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            self.counters.bump("processed")
            emit(
                "file",
                i=index,
                total=self.total,
                path=str(src),
                out=str(dest),
                ms=int((time.perf_counter() - started) * 1000),
                bytes=len(data),
                gray=bool(gray),
                model=model_name,
                **info,
            )
        except Exception as exc:
            self.counters.bump("failed")
            emit(
                "file",
                i=index,
                total=self.total,
                path=str(src),
                out=str(dest),
                error=f"{type(exc).__name__}: {exc}",
            )
            # The event carries only "OSError: ...", and encode/write failures
            # land in library frames, so the message alone rarely says where.
            # Same limit as the read/upscale path so both read alike.
            log(traceback.format_exc(limit=4), "debug")

    def bundle_page(self, meta: dict, name: str, size: int) -> None:
        self.counters.bump("processed")
        emit(
            "file",
            i=meta.get("i"),
            total=self.total,
            path=meta.get("src"),
            out=meta.get("bundle"),
            entry=name,
            bytes=size,
            ms=int((time.perf_counter() - float(meta.get("started") or 0)) * 1000),
            gray=bool(meta.get("gray")),
            model=meta.get("model"),
            **(meta.get("info") or {}),
        )

    def bundle_failed(self, meta: dict, name: str, error: str) -> None:
        self.counters.bump("failed")
        emit(
            "file",
            i=meta.get("i"),
            total=self.total,
            path=meta.get("src"),
            out=meta.get("bundle"),
            entry=name,
            error=error,
        )

    def bundle_done(self, key: str, dest: Path, entries: int, failed: int, elapsed: float) -> None:
        written_to = io_path(dest)
        emit(
            "bundle",
            key=key,
            out=str(dest),
            entries=entries,
            failed=failed,
            bytes=(written_to.stat().st_size if written_to.exists() else 0),
            ms=int(elapsed * 1000),
        )
