"""The concurrency primitives the pipeline stages are built from.

These rules hold this together and each one paid for itself in a bug:

* **Every queue is bounded.** ``WritePool``'s slot semaphore, ``prefetch``'s
  queue and ``BundleWriter``'s pack slots all cap how far a fast stage may run
  ahead of a slow one. Decoding a 200-page chapter is far cheaper than upscaling
  it, so an unbounded read-ahead holds every decoded page in memory at once.
* **Writes stay ordered.** ``BundleWriter`` and ``PagePacker`` each pack with
  exactly one worker thread. Page order inside a .cbz is what the reader sees,
  so neither pool may be widened for throughput.
* **Nothing blocks without an abort path.** ``prefetch`` re-checks ``CTRL``
  between items, its pump puts the terminating ``None`` from a ``finally`` so a
  consumer can never wait on a producer that died, and cancelling tears the pool
  down without waiting for queued reads. A consumer that leaves early also
  drains the queue on its way out: the pump can be parked inside ``put``, where
  setting ``stop`` cannot reach it. ``WritePool.submit`` and ``BundleWriter.add``
  wait for their slot in bounded steps for the same reason: a slot is only freed
  by an in-flight encode, so an unbounded wait would make Cancel arrive a page
  late.
* **A slot taken is a slot returned.** Both pools take a slot *before* handing
  the task to their executor, and the matching release lives in that task's
  ``finally`` -- so a submit that raises has to release the slot itself. A slot
  lost that way is retired permanently, and enough of them leave an acquire
  that can never succeed: a hang with no abort path.
* **Shared tallies are locked.** ``Counters`` guards the job's page totals,
  because the write pool, the pack thread and the main loop all bump them and
  ``d[key] += 1`` is a load, an add and a store rather than one step.

The drain paths log and continue instead of raising: one failed page must not
abandon the other 199.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Generator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Queue
from typing import Any, TypeAlias
from zipfile import ZIP_STORED, ZipFile

from janai.core.fspath import io_path
from janai.worker.control import CTRL, Cancelled
from janai.worker.events import log
from janai.worker.imagetypes import ImageArray

WRITE_SLOTS_PER_WORKER = 2
"""Writes allowed to queue per write thread: enough that the GPU never waits on
disk, few enough that finished pages cannot pile up in memory."""

SLOT_WAIT_SECONDS = 0.05
"""How long ``submit`` and ``add`` wait for a slot before re-checking the abort
flag. Matches the pause poll: short enough that Cancel is not made to wait out
an encode, long enough not to spin a core."""

PACK_SLOTS = 4
"""Pages allowed to queue for the single packing thread, for the same reason."""


class Counters:
    """The job's page tallies, written by several threads at once.

    ``processed`` and ``failed`` are bumped by the write pool, by
    ``BundleWriter``'s pack thread and by the main loop. ``d[key] += 1`` is a
    load, an add and a store, so two threads can read the same value and store
    the same result, silently dropping a page from the total. These numbers are
    the run summary the user reads, and ``failed == 0`` decides the process exit
    code, so they are worth a lock: it costs well under a microsecond per page
    against milliseconds of encoding.

    ``pages_failed`` counts pages *inside* an archive that could not be decoded
    or encoded. It is kept out of ``failed`` deliberately: ``failed`` alone sets
    ``done.ok`` and the exit code, and a chapter that lost one unreadable page
    still converted. Folding these into ``failed`` would report every such run
    as a failed job.
    """

    __slots__ = ("_counts", "_lock")

    def __init__(self) -> None:
        # Seeded rather than created on first bump: ``snapshot`` *is* the
        # ``done`` payload, so the GUI must be able to tell "nothing was lost"
        # from "this worker is too old to say".
        self._counts: dict[str, int] = {
            "processed": 0,
            "failed": 0,
            "skipped": 0,
            "pages_failed": 0,
        }
        self._lock = threading.Lock()

    def bump(self, key: str, n: int = 1) -> None:
        """Add ``n`` to ``key`` as one atomic step."""
        with self._lock:
            self._counts[key] = self._counts.get(key, 0) + n

    def __getitem__(self, key: str) -> int:
        with self._lock:
            return self._counts[key]

    def snapshot(self) -> dict[str, int]:
        """A consistent copy, for building the ``done`` payload."""
        with self._lock:
            return dict(self._counts)


class WritePool:
    """Encodes and writes in the background so the GPU is not waiting on disk."""

    def __init__(self, workers: int) -> None:
        self.workers = max(1, int(workers or 1))
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="write")
        self.slots = threading.Semaphore(self.workers * WRITE_SLOTS_PER_WORKER)
        self.pending: set[Future[None]] = set()
        self.lock = threading.Lock()

    def submit(self, fn: Callable[..., object], *args: object) -> None:
        """Queue a write, waiting for a slot but never waiting past a cancel.

        A free slot is taken even when cancelled, so a cancel never abandons a
        page that was about to be written anyway; only the *blocking* path
        aborts. It raises ``Cancelled`` instead of dropping the page quietly so
        there is still exactly one abort path: the producer unwinds into
        ``run_job``'s teardown, as it does at every other gate.
        """
        while not self.slots.acquire(timeout=SLOT_WAIT_SECONDS):
            if CTRL.cancelled:
                raise Cancelled

        def task() -> None:
            try:
                fn(*args)
            finally:
                self.slots.release()

        try:
            fut = self.pool.submit(task)
        except BaseException:
            self.slots.release()  # the task never ran, so its finally never will
            raise
        with self.lock:
            self.pending.add(fut)
        fut.add_done_callback(self._finished)

    def _finished(self, fut: Future[None]) -> None:
        with self.lock:
            self.pending.discard(fut)

    def drain(self) -> None:
        while True:
            with self.lock:
                futs = list(self.pending)
            if not futs:
                return
            for f in futs:
                try:
                    f.result()
                except Exception as exc:
                    log(f"write failed: {exc}", "error")

    def close(self) -> None:
        self.drain()
        self.pool.shutdown(wait=True)


PUMP_RELEASE_SECONDS = 2.0
"""How long teardown may spend freeing a pump parked in ``queue.put``. Bounded
so an abort can never hang on the very thread it is trying to release."""

PUMP_RELEASE_POLL_SECONDS = 0.05
"""How often that drain re-checks, matching the pause poll: fast enough to feel
instant, slow enough not to spin a core."""

Unit: TypeAlias = dict[str, Any]
"""One thing to convert: a loose image, or one entry inside an archive. It is
built from the JSON job payload, so its keys are the wire's rather than ours."""

PrefetchItem: TypeAlias = tuple[Unit, Future[ImageArray]] | None
"""What the pump hands the consumer: a unit with the future decoding it, or the
``None`` sentinel that proves the pump reached its ``finally``."""


def _release_pump(pending: Queue[PrefetchItem]) -> None:
    """Drain until the pump's sentinel arrives, so the pump thread can exit.

    ``stop`` is only tested between items, so setting it cannot wake a thread
    already blocked inside ``put``. Freeing a slot lets that put complete; the
    pump then sees ``stop``, breaks, and emits its sentinel from its ``finally``.
    Without this, a read-ahead abandoned by a cancel leaks the thread and every
    decoded page it is still holding -- the largest objects in the process.
    """
    deadline = time.monotonic() + PUMP_RELEASE_SECONDS
    while time.monotonic() < deadline:
        try:
            if pending.get(timeout=PUMP_RELEASE_POLL_SECONDS) is None:
                return
        except Empty:
            continue


def prefetch(
    units: list[Unit],
    workers: int,
    reader: Callable[[Unit], ImageArray],
) -> Generator[tuple[Unit, ImageArray | Exception], None, None]:
    """Yield (unit, array_or_exception) in order, decoding ahead of the pipeline.

    Declared as a Generator rather than an Iterator because *closing it is part
    of the contract*: a consumer that walks away early -- a cancel raises
    ``Cancelled`` straight out of the ``for`` loop in ``run_images`` -- must be
    able to call ``close()`` to release the pump thread parked in
    ``queue.put``. ``prefetch_check`` exercises exactly that, and an
    ``Iterator`` annotation hides ``close()`` from both the checker and the
    next reader.
    """
    workers = max(1, int(workers or 1))
    if workers == 1:
        for u in units:
            if CTRL.cancelled:
                return
            try:
                yield u, reader(u)
            except Exception as exc:
                yield u, exc
        return

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="read")
    # One slot per reader plus one, so the pump stays exactly one item ahead of
    # the readers rather than racing the whole unit list into memory.
    queue: Queue[PrefetchItem] = Queue(maxsize=workers + 1)
    stop = threading.Event()

    def pump() -> None:
        try:
            for u in units:
                if stop.is_set() or CTRL.cancelled:
                    break
                queue.put((u, pool.submit(reader, u)))
        finally:
            queue.put(None)

    threading.Thread(target=pump, name="prefetch", daemon=True).start()
    ended = False  # the sentinel proves the pump already reached its finally
    try:
        while True:
            item = queue.get()
            if item is None:
                ended = True
                return
            unit, fut = item
            if CTRL.cancelled:
                stop.set()
                fut.cancel()
                return
            try:
                yield unit, fut.result()
            except Exception as exc:
                yield unit, exc
    finally:
        stop.set()
        pool.shutdown(wait=False, cancel_futures=True)
        if not ended:
            _release_pump(queue)


PageMeta: TypeAlias = dict[str, Any]
"""The per-page record the reporting callbacks pass straight through on its way
to the `file` and `bundle` events. It comes off the job wire too, so its keys
are the wire's rather than ours to narrow."""

EncodeFn: TypeAlias = Callable[[ImageArray], bytes]
"""Encodes one finished page into the bytes stored in the archive."""


class BundleWriter:
    """Packs finished pages into one .cbz, off the GPU thread but in order.

    Encoding a page costs real time (JPEG XL especially), so it happens on a
    worker like every other write. A single worker keeps the pages in the order
    they were produced, which is the order a reader expects, and the archive is
    only moved into place once it is complete: an interrupted run leaves a
    .cbz.part behind rather than a half written chapter.
    """

    def __init__(
        self,
        encode_fn: EncodeFn,
        on_page: Callable[[PageMeta, str, int], None],
        on_fail: Callable[[PageMeta, str, str], None],
        on_done: Callable[[str, Path, int, int, float], None],
    ) -> None:
        self.encode = encode_fn
        self.on_page = on_page
        self.on_fail = on_fail
        self.on_done = on_done
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pack")
        self.slots = threading.Semaphore(PACK_SLOTS)
        self.key: str | None = None
        self.dest: Path | None = None
        self.tmp: Path | None = None
        self.zf: ZipFile | None = None
        self.entries = 0
        self.failed = 0
        self.started = 0.0
        self.futures: list[Future[None]] = []

    def open(self, key: str, dest: Path) -> None:
        self.close()
        # `dest` and `tmp` are kept plain because the `bundle` event reports
        # them; only the handles used for the syscalls carry the long-path
        # prefix. See janai.core.fspath for why that separation matters.
        io_path(dest).parent.mkdir(parents=True, exist_ok=True)
        self.key, self.dest = key, dest
        self.tmp = dest.with_suffix(".cbz.part")
        tmp_io = io_path(self.tmp)
        tmp_io.unlink(missing_ok=True)
        self.zf = ZipFile(tmp_io, "w", ZIP_STORED)
        self.entries = 0
        self.failed = 0
        self.started = time.perf_counter()

    def add(self, name: str, image: ImageArray, meta: PageMeta) -> None:
        """Queue a page for packing, waiting for a slot but never past a cancel.

        The same bounded wait as ``WritePool.submit``, for the same reason: only
        an in-flight encode frees a pack slot, so an unbounded wait would make
        Cancel arrive a page late. Aborting here costs nothing the user can see,
        because ``run_job`` closes the bundle with ``keep=not CTRL.cancelled``:
        the .part of a cancelled job is deleted, so any page packed after the
        cancel was pure waste. It raises rather than dropping the page silently
        so the producer unwinds through the one abort path, as at every gate.
        """
        if self.zf is None:
            return
        while not self.slots.acquire(timeout=SLOT_WAIT_SECONDS):
            if CTRL.cancelled:
                raise Cancelled
        zf = self.zf

        def task() -> None:
            try:
                data = self.encode(image)
                zf.writestr(name, data)
                self.entries += 1
                self.on_page(meta, name, len(data))
            except Exception as exc:
                self.failed += 1
                self.on_fail(meta, name, f"{type(exc).__name__}: {exc}")
            finally:
                self.slots.release()

        try:
            self.futures.append(self.pool.submit(task))
        except BaseException:
            self.slots.release()  # the task never ran, so its finally never will
            raise

    def drain(self) -> None:
        for fut in self.futures:
            try:
                fut.result()
            except Exception as exc:
                log(f"pack failed: {exc}", "error")
        self.futures.clear()

    def close(self, keep: bool = True) -> None:
        if self.zf is None:
            return
        self.drain()
        try:
            self.zf.close()
        finally:
            self.zf = None
        key, dest, tmp = self.key, self.dest, self.tmp
        entries, failed = self.entries, self.failed
        elapsed = time.perf_counter() - self.started
        self.key = self.dest = self.tmp = None
        if tmp is None or dest is None:
            return
        if not keep or entries == 0:
            io_path(tmp).unlink(missing_ok=True)
            return
        io_path(tmp).replace(io_path(dest))
        self.on_done(key or "", dest, entries, failed, elapsed)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)


class PagePacker:
    """Encodes and packs the pages of ONE already-open archive, off the GPU thread.

    The CBZ *input* path used to encode inline, so the GPU sat idle for the whole
    encode of every page. Measured on this path: encoding is ~9% of a page for
    PNG and ~72% for AVIF, because the upscale costs the same whatever the
    encoder, and an A/B over the same six pages ran 12.95s as loose files
    against 13.88s inside a .cbz. This is the trade ``BundleWriter`` already
    makes for the loose-file path. It is a separate class rather than a reuse
    because ``BundleWriter`` owns the archive it publishes, while this one
    borrows a zip its caller opened and leaves publishing, the unit tallies and
    the single ``file`` event to that caller.

    One worker, for ``BundleWriter``'s reason: it makes the queue FIFO, so page
    order survives by construction rather than by care. That is also what makes
    the plain ``int`` tallies safe -- only the pack thread writes them, and the
    caller may only read them once ``close`` has joined every task.
    """

    def __init__(
        self,
        zf: ZipFile,
        encode_fn: EncodeFn,
        on_fail: Callable[[str, Exception], None],
    ) -> None:
        self.zf = zf
        self.encode = encode_fn
        self.on_fail = on_fail
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pack")
        self.slots = threading.Semaphore(PACK_SLOTS)
        self.written = 0
        self.failed = 0
        self.futures: list[Future[None]] = []

    def add(self, name: str, image: ImageArray) -> None:
        """Queue one page, waiting for a slot but never waiting past a cancel.

        The bounded wait of ``WritePool.submit``, for its reason: only an
        in-flight encode frees a slot, so an unbounded wait would make Cancel
        arrive a page late. A queued page also pins its decoded image in memory,
        which is why the bound is small.
        """
        while not self.slots.acquire(timeout=SLOT_WAIT_SECONDS):
            if CTRL.cancelled:
                raise Cancelled

        def task() -> None:
            try:
                self.zf.writestr(name, self.encode(image))
                self.written += 1
            except Exception as exc:
                self.failed += 1
                self.on_fail(name, exc)
            finally:
                self.slots.release()

        try:
            self.futures.append(self.pool.submit(task))
        except BaseException:
            self.slots.release()  # the task never ran, so its finally never will
            raise

    def drain(self) -> None:
        for fut in self.futures:
            try:
                fut.result()
            except Exception as exc:
                log(f"pack failed: {exc}", "error")
        self.futures.clear()

    def close(self, drain: bool = True) -> None:
        """Join the pack thread, so the caller may close its zip and read the tallies.

        ``drain=False`` is the cancel path: queued pages are dropped instead of
        encoded, because the caller discards the .part anyway. The pool is still
        waited on, because the page already being written has to finish before
        the zip can close -- ``cancel_futures`` drops the queue, not the task
        that is running.
        """
        try:
            if drain:
                self.drain()
            else:
                self.futures.clear()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=not drain)
