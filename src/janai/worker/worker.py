"""JaNaiUpscaler worker.

Runs the upscale pipeline out of process so the GUI stays responsive and never
imports torch. Modes:

    worker.py --probe              print one JSON line describing this machine
    worker.py --job job.json       run a job, streaming JSONL events on stdout
    worker.py --job j --dry-run    report what a job would do, write nothing
    worker.py --job j --profile    measure tile cost and speed, write nothing
    worker.py --hold [--device]    keep a GPU context awake until stdin says stop

Events (one JSON object per line, always with a "type" key):
    start             {total, out_dir, device, fp16, tile, format}
    progress          {i, total, path, sub_i, sub_n}
    file              {i, total, path, out, ms, bytes, w, h, gray, model, error}
    bundle            {...}
    log               {level, message}
    done              {ok, processed, failed, skipped, cancelled, elapsed}
    probe             {...}
    profile           {ok, cancelled, elapsed, profile, error}
    profile_progress  {model, tile, index, total, ok, peak, seconds}
    hold              {ok, device, name, reserved, pid} / {ok, device, released}

Control commands arrive as lines on stdin: cancel, pause, resume (and stop,
which ends --hold).

This file is the command line entry point and nothing else: it parses
arguments and dispatches. The work lives in sibling modules - `job.py` runs a
job and assembles the pipeline (read via libvips -> grayscale detection ->
optional auto levels -> spandrel/torch upscale -> dot-gain-aware final resize
-> encode), `probe.py` answers --probe, `hold.py` answers --hold, and
`profiling.py` answers --profile.

The file must stay at this path: runner.py, setup.ps1, setup.sh, bench.py and
selftest.py all hardcode it, as does the janai-worker console script.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1]  # <app folder>/src, the import root

# Run as a script, sys.path[0] is this directory, so the package itself would
# not be importable. Put the import root in front before anything of ours.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from janai.worker import runtime
from janai.worker.control import CTRL

# Importing environment resolves the install layout, puts the vendored backend
# on sys.path and the bundled tools on PATH. It must happen before any heavy
# import, which is why nothing below may be reordered above it.
from janai.worker.environment import MODELS_DIR
from janai.worker.events import emit
from janai.worker.hold import do_hold
from janai.worker.job import run_job
from janai.worker.probe import do_probe
from janai.worker.profiling import do_profile


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="worker", description="JaNaiUpscaler worker")
    ap.add_argument("--job", help="path to a job JSON file")
    ap.add_argument("--probe", action="store_true", help="report devices, encoders and models")
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument(
        "--hold", action="store_true", help="hold a GPU context awake until stdin says stop"
    )
    ap.add_argument("--device", default="", help="device for --hold, e.g. cuda:0")
    ap.add_argument(
        "--hold-interval",
        type=float,
        default=15.0,
        help="seconds between keep-alive touches (default 15)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="with --job: report what would happen, write nothing"
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="with --job: measure this machine's tile cost and speed, write nothing",
    )
    args = ap.parse_args(argv)

    runtime.install_warning_filters()  # before torch is imported anywhere in this process

    # UTF-8 on stdout is part of the wire format - the GUI decodes these lines
    # as JSONL - and errors="replace" on stdin is what keeps one undecodable
    # byte from killing the control reader (F3, asserted by stdin_check.py).
    # typeshed types sys.stdout/sys.stdin as TextIO, which has no
    # reconfigure(); only the concrete TextIOWrapper does. Narrowing is
    # behaviour-identical to the bare call it replaces: a stream that is not a
    # TextIOWrapper raised AttributeError straight into the except below, so it
    # was already a silent skip - but now it is checkable instead of hidden.
    try:
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace", newline="\n")
        if isinstance(sys.stdin, io.TextIOWrapper):
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.probe:
        return do_probe(Path(args.models_dir))
    if args.hold:
        return do_hold(args.device, args.hold_interval)
    if not args.job:
        ap.error("one of --job, --probe or --hold is required")

    # The stdin control reader is deliberately NOT started here. On Windows a
    # thread parked in a blocking read on a *piped* stdin while numpy's
    # extension modules are being loaded deadlocks the loader: "import numpy"
    # never returns, so a GUI-launched job hung forever before printing a
    # single event. run_job() starts the reader once the backend is imported;
    # anything the GUI sends in the meantime waits in the pipe and is handled
    # as soon as the reader comes up, so no cancel/pause is lost.
    try:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    except Exception as exc:
        emit(
            "done",
            ok=False,
            processed=0,
            failed=0,
            skipped=0,
            cancelled=False,
            elapsed=0,
            error=f"bad job file: {exc}",
        )
        return 2
    if args.dry_run:
        job["dry_run"] = True
    if args.profile:
        try:
            return do_profile(job)
        except Exception as exc:
            emit("log", level="error", message=traceback.format_exc(limit=8))
            emit("profile", ok=False, error=f"{type(exc).__name__}: {exc}")
            return 1
    try:
        return run_job(job)
    except Exception as exc:
        emit("log", level="error", message=traceback.format_exc(limit=8))
        emit(
            "done",
            ok=False,
            processed=0,
            failed=0,
            skipped=0,
            cancelled=CTRL.cancelled,
            elapsed=0,
            error=f"{type(exc).__name__}: {exc}",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
