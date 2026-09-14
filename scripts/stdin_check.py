r"""Gate: one undecodable stdin byte must not disable Cancel (F3).

``Control._pump`` reads bare control words off stdin. A byte the stdin codec
cannot decode raises inside its ``for raw in sys.stdin`` loop, and the handler
deliberately does not retry -- a text stream that raised mid-line reports EOF
on every later read -- so the reader thread ends and Cancel and Pause are dead
for the rest of the job with only a log line to say so. ``worker.main()``
prevents that by reconfiguring stdin with ``errors="replace"`` before the job
starts.

Run it from the repo root, feeding an undecodable byte followed by "cancel":

    printf '\x81\ncancel\n' | backend/python/python.exe scripts/stdin_check.py
    printf '\x81\ncancel\n' | backend/python/python.exe scripts/stdin_check.py --reconfigure

``--reconfigure`` reproduces the production configuration, and is the only mode
whose cancel outcome is **asserted**. Without the flag the script reports what
the platform's default stdin codec happens to do and asserts nothing about it,
because that is not a property of this code: Windows hands the worker
cp1252/surrogateescape, which never raises, while a UTF-8/strict stdin does.
Failing on that would make a platform difference look like a regression.

Required for any change to ``Control._pump``, its exception handler, or the
stdin reconfiguration in ``worker.main()``.
"""

from __future__ import annotations

import io
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from janai.worker.control import CTRL

CANCEL_DEADLINE_SECONDS = 5.0
"""Long enough that a loaded machine still sees the piped word, short enough
that a real regression fails the gate quickly."""

PUMP_EXIT_SECONDS = 2.0
"""The reader leaves its loop as soon as the pipe closes, so this wait is not
budgeting for work -- it only removes the race between that exit and the
check below."""

POLL_SECONDS = 0.05

results: list[bool] = []


def check(ok: bool, label: str) -> None:
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {label}")


def pump_alive() -> bool:
    """Is the reader thread still running? ``Control.start`` names it "stdin"."""
    return any(t.name == "stdin" and t.is_alive() for t in threading.enumerate())


def main() -> int:
    production = "--reconfigure" in sys.argv
    if production:
        # Only the concrete TextIOWrapper has reconfigure(); typeshed types
        # sys.stdin as TextIO. If stdin is anything else this gate cannot
        # reproduce worker.main()'s configuration, and the cancel assertion
        # below would pass for the wrong reason - so fail loudly instead of
        # silently skipping the thing under test.
        if not isinstance(sys.stdin, io.TextIOWrapper):
            print("FAIL  stdin is not a TextIOWrapper; cannot reproduce production config")
            return 1
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    print(f"stdin encoding={sys.stdin.encoding} errors={sys.stdin.errors}", flush=True)

    CTRL.start()
    deadline = time.monotonic() + CANCEL_DEADLINE_SECONDS
    while time.monotonic() < deadline and not CTRL.cancelled:
        time.sleep(POLL_SECONDS)

    if production:
        check(CTRL.cancelled, "cancel is still seen after an undecodable byte")
    else:
        print(f"  report only, platform default codec: CANCELLED={CTRL.cancelled}")

    exit_by = time.monotonic() + PUMP_EXIT_SECONDS
    while time.monotonic() < exit_by and pump_alive():
        time.sleep(POLL_SECONDS)
    check(not pump_alive(), "the reader thread does not outlive the closed pipe")

    print(f"\n{sum(results)}/{len(results)} PASS")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
