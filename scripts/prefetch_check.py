"""Does leaving prefetch() early strand its pump thread?

Run with the embedded interpreter (no GPU, no models needed):

    backend/python/python.exe scripts/prefetch_check.py

prefetch() runs a daemon pump thread that submits reads and hands futures to a
bounded queue. The pump only tests ``stop`` at the top of its loop, so once the
queue is full it parks in ``queue.put`` -- and a consumer that leaves early
(cancel raises ``Cancelled`` straight out of the for loop in ``run_images``)
never drains it again. Setting ``stop`` cannot free a thread that is already
blocked inside ``put``.

What that costs: the thread never exits, and it keeps a reference to up to
workers+1 decoded pages, which are the largest objects in the process.

Each scenario asserts the pump is gone shortly after the consumer stops. The
full-consumption case is the control: it must pass both before and after any
fix, otherwise the fix broke the normal path.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janai.worker.control import CTRL
from janai.worker.pipeline import prefetch

UNITS = 200
WORKERS = 4
READ_SECONDS = 0.02  # slow enough that the bounded queue is full almost at once
JOIN_DEADLINE = 3.0  # generous: a released pump exits in microseconds
SETTLE = 0.4

checks: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    checks.append((bool(ok), label))
    print(f"{'PASS' if ok else 'FAIL'}  {label}")


class Reads:
    """Counts reads so we can tell a released pump from a runaway one."""

    def __init__(self) -> None:
        self.n = 0
        self.lock = threading.Lock()

    def __call__(self, unit: dict) -> int:
        time.sleep(READ_SECONDS)
        with self.lock:
            self.n += 1
        return int(unit["n"])

    @property
    def count(self) -> int:
        with self.lock:
            return self.n


def pumps() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t.name == "prefetch" and t.is_alive()]


def wait_for_pump_exit() -> bool:
    deadline = time.monotonic() + JOIN_DEADLINE
    while time.monotonic() < deadline:
        if not pumps():
            return True
        time.sleep(0.05)
    return False


def units() -> list[dict]:
    return [{"kind": "image", "n": i} for i in range(UNITS)]


def scenario_full() -> None:
    """Control case: consume everything. Must pass before and after the fix."""
    reads = Reads()
    got = [payload for _, payload in prefetch(units()[:12], WORKERS, reads)]
    check(got == list(range(12)), "[full] every unit is yielded once, in order")
    check(wait_for_pump_exit(), "[full] the pump thread exits when the list is done")


def scenario_early_exit() -> None:
    """The real shape of a cancel: the consumer walks away mid-stream."""
    reads = Reads()
    gen = prefetch(units(), WORKERS, reads)
    first = next(gen)
    check(first[1] == 0, "[early] the first unit is yielded before we leave")
    time.sleep(SETTLE)  # let the pump fill the bounded queue and park in put()
    gen.close()  # runs the generator's finally: stop.set() + pool shutdown
    exited = wait_for_pump_exit()
    check(exited, "[early] the pump thread is gone once the consumer has left")
    settled = reads.count
    time.sleep(SETTLE)
    check(reads.count == settled, "[early] no reads continue after the consumer has left")
    check(settled < UNITS, f"[early] the read-ahead stayed bounded ({settled} of {UNITS})")


def scenario_cancel() -> None:
    """Cancel for real, the way the stdin reader does. Sticky, so run it last."""
    reads = Reads()
    seen = 0
    for _unit, _payload in prefetch(units(), WORKERS, reads):
        seen += 1
        if seen == 2:
            CTRL.cancel()
    check(seen < UNITS, f"[cancel] the stream stopped early ({seen} of {UNITS} yielded)")
    check(wait_for_pump_exit(), "[cancel] the pump thread is gone after a cancel")


def main() -> int:
    scenario_full()
    scenario_early_exit()
    scenario_cancel()
    failed = [label for ok, label in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} PASS")
    for label in failed:
        print(f"  failed: {label}")
    if pumps():
        print(f"  live pump threads at exit: {len(pumps())}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
