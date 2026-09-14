"""Gate for BundleWriter.add's abort path and both pools' slot accounting (F18a/F18b).

Run: backend/python/python.exe scripts/bundle_check.py

No GPU, no models, no fixtures beyond a temp dir. Four scenarios, and they run
in this order on purpose: CTRL is a process-wide singleton, so once [cancel]
has set the flag it cannot be unset without reaching into its private event.
Every scenario that needs a cancel-free pipeline therefore runs first.

* [control] must pass before *and* after any change: 12 pages queue through 4
  pack slots, land in the archive in the order they were added, and leave no
  .part behind and no slot held.
* [leak] F18b: when the executor refuses the task, the slot that was taken for
  it must come back. The release lives in the task's own finally, so a submit
  that never ran the task retires that slot permanently; after PACK_SLOTS (or
  workers * WRITE_SLOTS_PER_WORKER) such failures no page can ever be admitted
  again -- a hang with no abort path. Checked on both pools, because both take
  their slot before handing the task over.
* [cancel] F18a: with every pack slot held and a cancel already requested, add
  must not park for a whole encode. Against the unfixed tree it blocks for the
  full task, returns normally, and packs the page the user just cancelled.
"""

from __future__ import annotations

import sys
import tempfile
import threading
import time
from pathlib import Path
from zipfile import ZipFile

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from janai.worker.control import CTRL, Cancelled
from janai.worker.pipeline import (
    PACK_SLOTS,
    WRITE_SLOTS_PER_WORKER,
    BundleWriter,
    WritePool,
)

SLOW = 1.0  # one "encode", long enough that parking on it is unmistakable
ABORT_BUDGET = 0.5  # a cancel must be noticed well inside one encode

results: list[bool] = []


def check(ok: bool, label: str) -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {label}")


def free_slots(sem: threading.Semaphore, cap: int) -> int:
    """How many of ``cap`` slots are currently free, measured without blocking."""
    taken = 0
    while taken < cap and sem.acquire(blocking=False):
        taken += 1
    for _ in range(taken):
        sem.release()
    return taken


class RefusingPool:
    """A pool whose submit always fails, i.e. thread-start failure or MemoryError.

    RuntimeError-after-shutdown is NOT reachable in the live worker (close() and
    shutdown() run only in run_job's finally, after the last add), so this
    stands in for the triggers that are.
    """

    def __init__(self) -> None:
        self.calls = 0

    def submit(self, *_args, **_kwargs):
        self.calls += 1
        raise RuntimeError("submit refused")

    def shutdown(self, wait: bool = True) -> None:
        return None


def make_writer(encode, done: list) -> BundleWriter:
    return BundleWriter(
        encode,
        lambda *_a: None,
        lambda *_a: None,
        lambda *args: done.append(args),
    )


def scenario_control(tmp: Path) -> None:
    done: list = []
    writer = make_writer(lambda image: image, done)
    dest = tmp / "Control.cbz"
    writer.open("control", dest)

    names = [f"p{n:02d}.png" for n in range(12)]
    for name in names:
        writer.add(name, name.encode(), {"path": name})
    writer.close(keep=True)
    writer.shutdown()

    packed = ZipFile(dest).namelist() if dest.exists() else []
    check(packed == names, f"[control] all 12 pages packed in order (got {len(packed)})")
    check(
        bool(done) and done[0][2] == 12 and done[0][3] == 0,
        f"[control] on_done reports entries=12 failed=0 (got {done[0][2:4] if done else None})",
    )
    check(
        not list(tmp.glob("*.part")),
        f"[control] no .part left behind ({[p.name for p in tmp.glob('*.part')]})",
    )
    got = free_slots(writer.slots, PACK_SLOTS)
    check(got == PACK_SLOTS, f"[control] every pack slot released ({got}/{PACK_SLOTS} free)")


def scenario_leak_bundle(tmp: Path) -> None:
    writer = make_writer(lambda image: image, [])
    writer.open("leak", tmp / "Leak.cbz")
    writer.pool = RefusingPool()

    raised: BaseException | None = None
    try:
        writer.add("p00.png", b"x", {"path": "p00.png"})
    except RuntimeError as exc:  # pre-existing behaviour: the failure propagates
        raised = exc

    name = type(raised).__name__ if raised is not None else "nothing"
    check(raised is not None, f"[leak] add propagates a refused submit (raised {name})")
    got = free_slots(writer.slots, PACK_SLOTS)
    check(
        got == PACK_SLOTS,
        f"[leak] BundleWriter.add released its slot after the failure ({got}/{PACK_SLOTS} free)",
    )
    writer.close(keep=False)


def scenario_leak_writepool() -> None:
    pool = WritePool(1)
    cap = pool.workers * WRITE_SLOTS_PER_WORKER
    pool.pool = RefusingPool()

    raised: BaseException | None = None
    try:
        pool.submit(lambda: None)
    except RuntimeError as exc:
        raised = exc

    name = type(raised).__name__ if raised is not None else "nothing"
    check(raised is not None, f"[leak] submit propagates a refused submit (raised {name})")
    got = free_slots(pool.slots, cap)
    check(
        got == cap,
        f"[leak] WritePool.submit released its slot after the failure ({got}/{cap} free)",
    )


def scenario_cancel_while_full(tmp: Path) -> None:
    running = threading.Event()

    def slow_encode(image):
        running.set()
        time.sleep(SLOW)
        return image

    writer = make_writer(slow_encode, [])
    dest = tmp / "Cancel.cbz"
    writer.open("cancel", dest)

    held = [f"p{n:02d}.png" for n in range(PACK_SLOTS)]
    for name in held:
        writer.add(name, name.encode(), {"path": name})
    check(running.wait(2.0), f"[cancel] all {PACK_SLOTS} pack slots are held by queued pages")

    CTRL.cancel()  # the real flag, exactly as the stdin pump sets it
    start = time.perf_counter()
    raised: BaseException | None = None
    try:
        writer.add("p99.png", b"aborted", {"path": "p99.png"})
    except Cancelled as exc:
        raised = exc
    waited = time.perf_counter() - start

    name = type(raised).__name__ if raised is not None else "nothing"
    check(raised is not None, f"[cancel] add aborts instead of parking (raised {name})")
    check(
        waited < ABORT_BUDGET,
        f"[cancel] cancel noticed in {waited:.2f}s (budget {ABORT_BUDGET}s, one encode is {SLOW}s)",
    )

    # keep=True only so the archive can be inspected; the live job discards it
    # (close(keep=not CTRL.cancelled)), which is why aborting early loses nothing.
    writer.close(keep=True)
    writer.shutdown()
    packed = ZipFile(dest).namelist() if dest.exists() else []
    check("p99.png" not in packed, f"[cancel] the aborted page was not packed ({packed})")
    check(
        packed == held,
        f"[cancel] the pages queued before the cancel still packed in order ({packed})",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="bundle_check_") as raw:
        tmp = Path(raw)
        scenario_control(tmp)
        scenario_leak_bundle(tmp)
        scenario_leak_writepool()
        scenario_cancel_while_full(tmp)
    passed = sum(results)
    print(f"\n{passed}/{len(results)} PASS")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
