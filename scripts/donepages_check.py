"""Does a CBZ that lost pages say so in the job summary? (F10, counting half)

    backend/python/python.exe scripts/donepages_check.py

`handle_archive` drops a page it cannot decode or encode, counts it locally, and
reports it on that archive's own `file` event. Nothing carried the loss up to
the job-level `done` event, so a run that quietly shortened three chapters still
finished with "12 done" and said nothing about the pages it lost.

The policy the user chose: report the loss in the `done` payload, leave the exit
code alone. That is why the tally is NOT `counters["failed"]` - `ok`
(job.py:644) and the return code (job.py:649) are computed from `failed` alone,
so a chapter with one unreadable page must still exit 0.

[control] what must not change: the original keys, and `failed` remaining the
          only thing that decides success.
[report]  the finding: every `done` payload carries `pages_failed`, and the run
          log line says how many pages went missing.
[policy]  the decision: `pages_failed` must not leak into `failed`, so it can
          neither clear `ok` nor change the exit code.

Dependency-free on purpose - `janai.worker.pipeline` and `janai.app.runlog` pull
in no torch, pyvips or Qt - so this runs on a bare CI runner.
`scripts/archive_check.py` proves the same behaviour end to end, against a real
worker process and a real corrupt page.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from janai.app.runlog import format_done
from janai.worker.pipeline import Counters

PAGES_KEY = "pages_failed"
"""The `done` field this gate is about. The payload is a wire format shared with
the GUI, so renaming it here without renaming it in job.py and runlog.py breaks
the summary line silently."""

LOST_PAGES = 3
"""More than one, so the plural is exercised here; archive_check covers the
singular against a real worker."""

results: list[bool] = []


def check(ok: bool, text: str) -> None:
    results.append(ok)
    print(("PASS  " if ok else "FAIL  ") + text)


def done_payload(counters: Counters, *, cancelled: bool = False) -> dict:
    """The `done` event as run_job assembles it (job.py:642-648)."""
    return {
        "type": "done",
        "ok": counters["failed"] == 0 and not cancelled,
        "cancelled": cancelled,
        "elapsed": 1.0,
        **counters.snapshot(),
    }


def exit_code(counters: Counters) -> int:
    """run_job's own return expression (job.py:649)."""
    return 0 if counters["failed"] == 0 else 1


def main() -> int:
    fresh = Counters().snapshot()
    check(
        {"processed", "failed", "skipped"} <= set(fresh),
        f"[control] the original tallies are still present ({sorted(fresh)})",
    )
    check(
        all(value == 0 for value in fresh.values()),
        f"[control] a new Counters starts at zero ({fresh})",
    )
    check(
        fresh.get(PAGES_KEY) == 0,
        f"[report] every done payload carries {PAGES_KEY}, even on a clean run",
    )

    unit_failed = Counters()
    unit_failed.bump("failed")
    check(
        done_payload(unit_failed)["ok"] is False,
        "[control] a failed unit still clears done.ok",
    )
    check(exit_code(unit_failed) == 1, "[control] a failed unit still exits 1")

    # A real job: two chapters converted, one of them three pages short.
    job = Counters()
    job.bump("processed", 2)
    job.bump(PAGES_KEY, LOST_PAGES)
    payload = done_payload(job)

    check(
        payload.get(PAGES_KEY) == LOST_PAGES,
        f"[report] the done payload reports the lost pages ({payload.get(PAGES_KEY)!r})",
    )
    text, level = format_done(payload)
    check(
        f"{LOST_PAGES} pages failed" in text,
        f"[report] the summary line says how many pages went missing ({text!r})",
    )
    check(
        level == "warn",
        f"[report] a run that lost pages is not a clean green line ({level})",
    )

    check(job["failed"] == 0, "[policy] lost pages do not enter the failed tally")
    check(payload["ok"] is True, "[policy] done.ok stays true: the run is still a success")
    check(exit_code(job) == 0, "[policy] the process still exits 0")

    clean = Counters()
    clean.bump("processed", 2)
    clean_text, clean_level = format_done(done_payload(clean))
    check(
        "failed" not in clean_text and clean_level == "ok",
        f"[control] a clean run reads exactly as before ({clean_text!r}, {clean_level})",
    )

    dry = Counters()
    dry.bump("processed", 1)
    dry_level = format_done(done_payload(dry) | {"dry": True})[1]
    check(
        dry_level == "dry",
        f"[control] a dry run is still reported as a dry run ({dry_level})",
    )

    passed = sum(1 for ok in results if ok)
    print(f"\n{passed}/{len(results)} PASS")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
