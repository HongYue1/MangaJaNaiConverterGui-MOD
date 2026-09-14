"""Are the job's page tallies safe to bump from several threads?

    backend/python/python.exe scripts/counters_check.py

run_job bumps ONE tallies object from the write pool (io_workers threads), from
BundleWriter's single pack thread and from the main loop. The obvious worry is
that ``d[key] += 1`` is a load, an add and a store. MEASURED ON THIS BUILD it is
nevertheless safe: since CPython 3.10 the eval breaker is polled at calls and
backward jumps, not between arbitrary bytecodes, so no thread switch can land
inside a bare read-modify-write.

That safety is an implementation detail, not a contract, and it is shape
sensitive -- put a call inside the expression (``d[key] += len(units)``, which
is what job.py does for a skipped folder) and the poll happens between the load
and the store, so updates ARE lost. Hence Counters takes a lock.

[bare]   canary: must NOT lose updates. If it ever does, this is a free-threaded
         interpreter and Counters' lock is the only thing holding the tallies
         together -- it does not mean the app regressed.
[call]   premise: the racy shape must lose updates, which is why the lock exists.
[locked] the real assertion: Counters must be exact under both shapes.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from janai.worker.pipeline import Counters

THREADS = 8
"""Six write threads plus the pack thread plus the main loop: what a real job
looks like at the default io_workers."""

BUMPS = 10_000

SWITCH_INTERVAL = 1e-6
"""Preempt hard so a race shows up in seconds instead of by luck. This changes
how often the interpreter switches threads, not whether the race exists."""

COST_BUMPS = 200_000
"""Uncontended cost sample. Production does a handful of bumps per page, so the
number that matters is the single-threaded one, not the contended one."""

results: list[bool] = []


def check(ok: bool, text: str) -> None:
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + text)


def one() -> int:
    """A call inside the read-modify-write, standing in for ``len(units_here)``."""
    return 1


def hammer(bump) -> None:
    def loop() -> None:
        for _ in range(BUMPS):
            bump()

    threads = [threading.Thread(target=loop) for _ in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def main() -> int:
    expected = THREADS * BUMPS
    previous = sys.getswitchinterval()
    sys.setswitchinterval(SWITCH_INTERVAL)
    try:
        bare = {"n": 0}
        hammer(lambda: _bare_bump(bare))
        lost_bare = expected - bare["n"]
        check(
            lost_bare == 0,
            f"[bare] a plain += 1 keeps every update under the GIL ({lost_bare}/{expected} lost)",
        )

        called = {"n": 0}
        hammer(lambda: _call_bump(called))
        lost_call = expected - called["n"]
        check(
            lost_call > 0,
            f"[call] += with a call inside it DOES lose updates ({lost_call} lost of {expected})",
        )

        tallies = Counters()
        hammer(lambda: tallies.bump("processed"))
        got = tallies["processed"]
        check(got == expected, f"[locked] Counters is exact for a plain bump ({got} of {expected})")

        tallies_n = Counters()
        hammer(lambda: tallies_n.bump("skipped", one()))
        got_n = tallies_n["skipped"]
        check(
            got_n == expected,
            f"[locked] Counters is exact for the shape that races ({got_n} of {expected})",
        )
    finally:
        sys.setswitchinterval(previous)

    single = Counters()
    single.bump("skipped", 7)
    check(single["skipped"] == 7, "[locked] bump(key, n) adds n, for the whole-folder skip")

    snap = single.snapshot()
    keys = sorted(snap)
    # Locked on purpose, and it may only ever grow: snapshot() *is* the done
    # payload the GUI parses, so a key disappearing here is a silently broken
    # status line rather than an error anyone would see. pages_failed joined the
    # set when per-page archive losses started being reported.
    check(
        keys == ["failed", "pages_failed", "processed", "skipped"],
        f"[locked] snapshot() carries the keys the done event needs ({keys})",
    )

    snap["skipped"] = 99
    check(
        single["skipped"] == 7, "[locked] snapshot() is a copy, so done cannot rewrite the tallies"
    )

    cost = Counters()
    start = time.perf_counter()
    for _ in range(COST_BUMPS):
        cost.bump("processed")
    locked_ns = (time.perf_counter() - start) / COST_BUMPS * 1e9

    plain = {"n": 0}
    start = time.perf_counter()
    for _ in range(COST_BUMPS):
        plain["n"] += 1
    plain_ns = (time.perf_counter() - start) / COST_BUMPS * 1e9

    print(
        f"\ncost (uncontended): {locked_ns:.0f} ns per locked bump vs {plain_ns:.0f} ns unlocked, "
        f"about {(locked_ns - plain_ns) * 4 / 1000:.2f} us per page at four bumps"
    )

    passed = sum(1 for ok in results if ok)
    print(f"\n{passed}/{len(results)} PASS")
    return 0 if passed == len(results) else 1


def _bare_bump(d: dict) -> None:
    d["n"] += 1


def _call_bump(d: dict) -> None:
    d["n"] += one()


if __name__ == "__main__":
    sys.exit(main())
