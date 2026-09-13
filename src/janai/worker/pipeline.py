"""The concurrency primitives the pipeline stages are built from.

Three rules hold this together and each one paid for itself in a bug:

* **Every queue is bounded.** ``WritePool``'s slot semaphore, ``prefetch``'s
  queue and ``BundleWriter``'s pack slots all cap how far a fast stage may run
  ahead of a slow one. Decoding a 200-page chapter is far cheaper than upscaling
  it, so an unbounded read-ahead holds every decoded page in memory at once.
* **Writes stay ordered.** ``BundleWriter`` packs with exactly one worker
  thread. Page order inside a .cbz is what the reader sees, so this pool must
  not be widened for throughput.
* **Nothing blocks without an abort path.** ``prefetch`` re-checks ``CTRL``
  between items, its pump puts the terminating ``None`` from a ``finally`` so a
  consumer can never wait on a producer that died, and cancelling tears the pool
  down without waiting for queued reads.

The drain paths log and continue instead of raising: one failed page must not
abandon the other 199.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import Any
from zipfile import ZIP_STORED, ZipFile

from janai.worker.control import CTRL
from janai.worker.events import log

WRITE_SLOTS_PER_WORKER = 2
"""Writes allowed to queue per write thread: enough that the GPU never waits on
disk, few enough that finished pages cannot pile up in memory."""

PACK_SLOTS = 4
"""Pages allowed to queue for the single packing thread, for the same reason."""


class WritePool:
    """Encodes and writes in the background so the GPU is not waiting on disk."""

    def __init__(self, workers: int) -> None:
        self.workers = max(1, int(workers or 1))
        self.pool = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="write")
        self.slots = threading.Semaphore(self.workers * WRITE_SLOTS_PER_WORKER)
        self.pending: set = set()
        self.lock = threading.Lock()

    def submit(self, fn, *args) -> None:
        self.slots.acquire()

        def task():
            try:
                fn(*args)
            finally:
                self.slots.release()

        fut = self.pool.submit(task)
        with self.lock:
            self.pending.add(fut)
        fut.add_done_callback(self._finished)

    def _finished(self, fut) -> None:
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


def prefetch(units: list[dict], workers: int, reader):
    """Yield (unit, array_or_exception) in order, decoding ahead of the pipeline."""
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
    queue: Queue = Queue(maxsize=workers + 1)
    stop = threading.Event()

    def pump():
        try:
            for u in units:
                if stop.is_set() or CTRL.cancelled:
                    break
                queue.put((u, pool.submit(reader, u)))
        finally:
            queue.put(None)

    threading.Thread(target=pump, name="prefetch", daemon=True).start()
    try:
        while True:
            item = queue.get()
            if item is None:
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


class BundleWriter:
    """Packs finished pages into one .cbz, off the GPU thread but in order.

    Encoding a page costs real time (JPEG XL especially), so it happens on a
    worker like every other write. A single worker keeps the pages in the order
    they were produced, which is the order a reader expects, and the archive is
    only moved into place once it is complete: an interrupted run leaves a
    .cbz.part behind rather than a half written chapter.
    """

    def __init__(self, encode_fn, on_page, on_fail, on_done) -> None:
        self.encode = encode_fn
        self.on_page = on_page
        self.on_fail = on_fail
        self.on_done = on_done
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pack")
        self.slots = threading.Semaphore(PACK_SLOTS)
        self.key: str | None = None
        self.dest: Path | None = None
        self.tmp: Path | None = None
        self.zf: Any = None
        self.entries = 0
        self.failed = 0
        self.started = 0.0
        self.futures: list = []

    def open(self, key: str, dest: Path) -> None:
        self.close()
        dest.parent.mkdir(parents=True, exist_ok=True)
        self.key, self.dest = key, dest
        self.tmp = dest.with_suffix(".cbz.part")
        self.tmp.unlink(missing_ok=True)
        self.zf = ZipFile(self.tmp, "w", ZIP_STORED)
        self.entries = 0
        self.failed = 0
        self.started = time.perf_counter()

    def add(self, name: str, image, meta: dict) -> None:
        if self.zf is None:
            return
        self.slots.acquire()
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

        self.futures.append(self.pool.submit(task))

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
            tmp.unlink(missing_ok=True)
            return
        tmp.replace(dest)
        self.on_done(key or "", dest, entries, failed, elapsed)

    def shutdown(self) -> None:
        self.pool.shutdown(wait=True)
