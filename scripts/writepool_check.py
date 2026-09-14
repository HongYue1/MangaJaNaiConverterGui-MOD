"""Gate for WritePool.submit's abort path (F17).

Run: backend/python/python.exe scripts/writepool_check.py

No GPU, no models, no fixtures. Two scenarios:

* [control] must pass before and after any change: every submitted write runs,
  drain/close return, and nothing is left pending.
* [cancel] the finding: with every write slot held and a cancel already
  requested, submit must not park for a whole encode. Against the unfixed tree
  it blocks for the full task duration and then returns normally, so exactly
  two assertions fail; the fix raises Cancelled within a poll interval and does
  not queue the page.

The scenarios run in this order on purpose: CTRL is a process-wide singleton,
so once [cancel] has set the flag it cannot be unset without reaching into its
private event.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from janai.worker.control import CTRL, Cancelled
from janai.worker.pipeline import WRITE_SLOTS_PER_WORKER, WritePool

SLOW = 1.0  # one "encode", long enough that parking on it is unmistakable
ABORT_BUDGET = 0.5  # a cancel must be noticed well inside one encode

results: list[bool] = []


def check(ok: bool, label: str) -> None:
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {label}")


def scenario_control() -> None:
    pool = WritePool(2)
    ran: list[int] = []
    lock = threading.Lock()

    def task(n: int) -> None:
        time.sleep(0.02)
        with lock:
            ran.append(n)

    for n in range(12):
        pool.submit(task, n)
    pool.close()
    check(len(ran) == 12, f"[control] every queued write ran (ran={len(ran)}/12)")
    check(not pool.pending, f"[control] nothing left pending after close ({len(pool.pending)})")


def scenario_cancel_while_full() -> None:
    pool = WritePool(1)
    slots = pool.workers * WRITE_SLOTS_PER_WORKER
    running = threading.Event()
    written: list[int] = []

    def slow(n: int) -> None:
        running.set()
        time.sleep(SLOW)
        written.append(n)

    for n in range(slots):
        pool.submit(slow, n)
    check(running.wait(2.0), f"[cancel] all {slots} slots are held by in-flight writes")

    CTRL.cancel()  # the real flag, exactly as the stdin pump sets it
    start = time.perf_counter()
    raised: BaseException | None = None
    try:
        pool.submit(slow, 99)
    except Cancelled as exc:
        raised = exc
    waited = time.perf_counter() - start

    name = type(raised).__name__ if raised is not None else "nothing"
    check(raised is not None, f"[cancel] submit aborts instead of parking (raised {name})")
    check(
        waited < ABORT_BUDGET,
        f"[cancel] cancel noticed in {waited:.2f}s (budget {ABORT_BUDGET}s, one encode is {SLOW}s)",
    )

    pool.close()
    check(
        99 not in written, f"[cancel] the aborted page was not written (written={sorted(written)})"
    )


def main() -> int:
    scenario_control()
    scenario_cancel_while_full()
    passed = sum(results)
    print(f"\n{passed}/{len(results)} PASS")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
