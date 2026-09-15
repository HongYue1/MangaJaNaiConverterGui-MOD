"""Headless driver: run and script the upscaler without the GUI.

Why this exists
---------------
Every conversion is one JSON job payload handed to ``janai.worker.worker``.
The window builds that payload from its widgets; before this module, anything
else -- a shell script, a regression test, an agent verifying a fix -- had to
hand-write a ``job.json`` and re-derive the schema, which is precisely how a
driver drifts from the interface it is meant to mirror.

So this module reuses rather than reimplements:

* the payload is built from the same ``settings.json`` the GUI loads
  (:mod:`janai.app.state`), so a headless run reproduces what the window does;
* the worker is spawned and steered through the same process manager the GUI
  uses (:class:`janai.app.runner.Runner`), so there stays exactly one owner of
  the spawn/cancel contract and one reader of the JSON-Lines event stream.

Both of those modules are standard-library only by design -- no Qt -- and
``scripts/smoke.py`` asserts it, because a stray Qt import there would make
this driver unusable on a headless machine.

Commands
--------
    janai run   SRC [-o OUT] [options]   convert
    janai plan  SRC [...]                dry run: report every page, write nothing
    janai probe                          devices, encoders, installed models
    janai where                          the locations this install resolves to

``--json`` streams the worker's own event lines untouched, which is what a test
or another program should read. ``--cancel-after SECONDS`` requests a stop
mid-job without a terminal, so cancellation is testable instead of manual.

Exit codes: 0 finished, 1 the job reported a failure, 2 bad usage, 130 cancelled.
"""

from __future__ import annotations

import argparse
import json
import queue
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_SRC = Path(__file__).resolve().parents[1]  # <app folder>/src, the import root

# Run as a file path, sys.path[0] is this file's own folder, so the `janai`
# package itself would not be importable. Same bootstrap, and same reason, as
# worker.py; both are entry files rather than ordinary modules.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from janai.app.runner import Runner
from janai.app.state import Settings, defaults
from janai.core import hardware, paths as core_paths, rules as core_rules
from janai.core.formats import CONTAINERS, FORMAT_IDS
from janai.core.paths import MODEL_EXTS, app_root

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
#: 128 + SIGINT, the convention a shell reports, so `echo $?` tells "you
#: stopped it" apart from "it broke".
EXIT_CANCELLED = 130

#: The page kinds job.py accepts. Anything else silently falls back to "detect"
#: there, so a typo has to be rejected here instead of quietly changing the run.
PAGE_KINDS = ("detect", "grayscale", "colour")
#: The target modes job.py branches on.
TARGET_MODES = ("scale", "width", "height", "fit")
#: The GUI's settings file, shared deliberately: see the module docstring.
SETTINGS_NAME = "settings.json"
#: How long the event pump blocks before re-checking the clock, so
#: --cancel-after still fires while the worker is quiet.
POLL_SECONDS = 0.2
#: Matches the timeout Runner._probe allows the worker process.
PROBE_TIMEOUT_SECONDS = 600.0
#: Fallback output subfolder, the same one the output panel defaults to.
DEFAULT_SUBFOLDER = "upscaled"


def _number(value: object) -> float:
    """A numeric event field as a float, or 0 when absent or not a number."""
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return 0.0


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


# --------------------------------------------------------------------------- #
# settings -> job payload
# --------------------------------------------------------------------------- #
def installed_model_names(models_dir: Path | None) -> list[str]:
    """Model file names as the worker will see them.

    Mirrors ``janai.worker.models.list_models`` (recursive, bare file names)
    rather than calling it: that module imports torch, which a driver must not
    pay for merely to seed a default rule table.
    """
    if models_dir is None or not models_dir.is_dir():
        return []
    return [
        p.name
        for p in sorted(models_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in MODEL_EXTS
    ]


def ensure_rules(data: dict[str, Any], models_dir: Path | None) -> int:
    """Fill an empty rule table from the installed models; return how many are on.

    An empty table means "not set up yet" (see ``state.defaults``); the GUI
    seeds it on first launch once the probe reports what is installed. A
    headless run has neither a probe nor a window, so without this an untouched
    install would select no model, convert nothing, and still report success.
    """
    ups = data["upscale"]
    if not ups.get("rules"):
        ups["rules"] = core_rules.default_dicts(installed_model_names(models_dir))
    return sum(1 for rule in ups["rules"] if rule.get("enabled", True))


def resolve_out_dir(src: Path, out: str, data: dict[str, Any]) -> Path:
    """Where output goes: an explicit ``-o``, else the saved choice, else a
    subfolder beside the input.

    Mirrors ``output_panel.resolved_out_dir``. That copy reads Qt widgets so it
    cannot be reused, but the rule has to stay identical or the same settings
    would write to two different places depending on which front end ran.
    """
    if out.strip():
        return Path(out).expanduser()
    outp = data.get("output") or {}
    if not bool(outp.get("same_as_input", True)):
        saved = str(outp.get("dir") or "").strip()
        if saved:
            return Path(saved).expanduser()
    # Beside the input whether it is a file or a folder: a subfolder created
    # inside a folder input is walked by the next scan, so the run would read
    # its own output back in. A drive root has no "beside", so it keeps itself.
    base = src.parent if src.parent != src else src
    return base / (str(outp.get("subfolder") or "").strip() or DEFAULT_SUBFOLDER)


def build_job(
    data: dict[str, Any],
    src: Path,
    out_dir: Path,
    models_dir: Path | None,
    *,
    dry: bool = False,
    resume: bool = True,
) -> dict[str, Any]:
    """The worker payload for these settings.

    Mirror of ``run_panel.build_job``: same keys, same shapes, same source of
    truth. Annotated as ``dict[str, Any]`` because the payload is JSON for
    another process, so heterogeneous values are the point.
    """
    ups = dict(data["upscale"])
    if models_dir is not None:
        ups["models_dir"] = str(models_dir)
    perf = dict(data["perf"])
    fid = str(data["format"]["id"])
    options = data["format"].get("options") or {}
    job: dict[str, Any] = {
        "input": {
            "path": str(src),
            "mode": "single" if src.is_file() else "bulk",
            "recursive": bool(data["input"]["recursive"]),
            "include_archives": bool(data["input"]["include_archives"]),
        },
        "output": {
            "dir": str(out_dir),
            "container": str(data["output"]["container"]),
            "pattern": str(data["output"]["pattern"]),
            "overwrite": bool(data["output"]["overwrite"]),
            "keep_structure": bool(data["output"]["keep_structure"]),
        },
        "format": {"id": fid, "options": dict(options.get(fid) or {})},
        "upscale": ups,
        # Measurements belong to the machine and precision they were taken on;
        # hardware.profile_for_run enforces that, and an empty dict leaves the
        # worker on its cautious first-page path.
        "perf": dict(
            perf,
            profile=hardware.profile_for_run(
                data.get("profile"), data.get("probe"), bool(perf.get("use_fp16", True))
            ),
        ),
    }
    if dry:
        job["dry_run"] = True
    # Same shape as ``dry_run``: the key is present only when it deviates from
    # the worker's default, so the payload the GUI builds stays byte-identical
    # and the mirror claimed above stays true. Resume is on unless asked.
    if not resume:
        job["resume"] = False
    return job


def load_settings(root: Path, settings: str, *, ignore_saved: bool) -> dict[str, Any]:
    """Settings for this run: the app's own file unless told otherwise.

    Sharing ``settings.json`` with the GUI is the feature - a headless run
    should reproduce what the window would do, including format options.
    """
    if ignore_saved:
        return defaults()
    path = Path(settings).expanduser() if settings else root / SETTINGS_NAME
    return Settings(path).load().data


def load_rules(path: Path) -> list[dict[str, Any]]:
    """A rule table from JSON: the list the GUI stores under ``upscale.rules``."""
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, list):
        raise ValueError(f"{path} must hold a JSON list of rules")
    return [dict(item) for item in parsed]


def target_mode(args: argparse.Namespace, current: str) -> str:
    """The target mode this invocation asks for.

    An explicit ``--mode`` wins; otherwise the size flag given implies it,
    because ``--width 2048`` against a saved mode of "scale" would otherwise be
    accepted and then ignored.
    """
    if args.mode is not None:
        return str(args.mode)
    if args.scale is not None:
        return "scale"
    if args.width is not None:
        return "width"
    if args.height is not None:
        return "height"
    return current


def apply_overrides(data: dict[str, Any], args: argparse.Namespace) -> None:
    """Fold command-line options into the settings bag.

    ``None`` means "not asked for", so an omitted option keeps whatever the
    settings file - or the shipped default - already says.
    """
    inp, outp, ups, perf = data["input"], data["output"], data["upscale"], data["perf"]
    if args.recursive is not None:
        inp["recursive"] = bool(args.recursive)
    if args.archives is not None:
        inp["include_archives"] = bool(args.archives)
    if args.subfolder:
        outp["subfolder"] = str(args.subfolder)
    if args.container is not None:
        outp["container"] = str(args.container)
    if args.pattern is not None:
        outp["pattern"] = str(args.pattern)
    if args.overwrite is not None:
        outp["overwrite"] = bool(args.overwrite)
    if args.keep_structure is not None:
        outp["keep_structure"] = bool(args.keep_structure)
    if args.format_id is not None:
        data["format"]["id"] = str(args.format_id)
    if args.scale is not None:
        ups["scale"] = float(args.scale)
    if args.width is not None:
        ups["width"] = int(args.width)
    if args.height is not None:
        ups["height"] = int(args.height)
    ups["mode"] = target_mode(args, str(ups.get("mode") or "scale"))
    if args.page_kind is not None:
        ups["page_kind"] = str(args.page_kind)
        # grayscale_convert is the older spelling job.py falls back to. Keeping
        # the two in step means one payload cannot state two different things.
        ups["grayscale_convert"] = args.page_kind != "colour"
    if args.auto_levels is not None:
        ups["auto_levels"] = bool(args.auto_levels)
    if args.rules is not None:
        ups["rules"] = load_rules(Path(args.rules).expanduser())
    if args.device is not None:
        perf["device"] = str(args.device)
    if args.fp16 is not None:
        perf["use_fp16"] = bool(args.fp16)
    if args.tile is not None:
        perf["tile"] = str(args.tile)
    if args.io_workers is not None:
        perf["io_workers"] = int(args.io_workers)


# --------------------------------------------------------------------------- #
# event rendering
# --------------------------------------------------------------------------- #
class Console:
    """Renders worker events for a terminal, or passes the JSON Lines through.

    ``raw`` matters: the event stream is this app's contract with its worker, so
    a script or a test must be able to read exactly what the GUI reads rather
    than a prettier, weaker second format invented here.
    """

    def __init__(self, *, raw: bool = False, quiet: bool = False) -> None:
        self.raw = raw
        self.quiet = quiet
        self.done: dict[str, Any] | None = None
        self.exit_code: int | None = None

    def handle(self, event: dict[str, Any]) -> None:
        kind = str(event.get("type") or "")
        if kind == "done":
            self.done = dict(event)
        elif kind == "exit":
            self.exit_code = int(_number(event.get("code")))
        if self.raw:
            print(json.dumps(event, ensure_ascii=False), flush=True)
            return
        line = self.render(kind, event)
        if not line:
            return
        loud = kind == "log" and str(event.get("level") or "") in {"warn", "error"}
        print(line, file=sys.stderr if loud else sys.stdout, flush=True)

    def render(self, kind: str, event: dict[str, Any]) -> str:
        if self.quiet and kind in {"start", "file", "bundle", "progress"}:
            return ""
        if kind == "start":
            return self._render_start(event)
        if kind == "file":
            return self._render_file(event)
        if kind == "bundle":
            out = str(event.get("out") or event.get("path") or "")
            return f"packed {out}" if out else ""
        if kind == "log":
            level = str(event.get("level") or "info")
            if level == "debug" or (self.quiet and level == "info"):
                return ""
            return f"{level}: {event.get('message') or ''}"
        if kind == "done":
            return self._render_done(event)
        return ""

    @staticmethod
    def _render_start(event: dict[str, Any]) -> str:
        bits = [f"{int(_number(event.get('total')))} item(s)"]
        out_dir = str(event.get("out_dir") or "")
        if out_dir:
            bits.append(f"-> {out_dir}")
        detail = [str(event.get(key)) for key in ("device", "tile", "format") if event.get(key)]
        if event.get("fp16"):
            detail.append("fp16")
        if detail:
            bits.append(f"[{', '.join(detail)}]")
        return " ".join(bits)

    @staticmethod
    def _render_file(event: dict[str, Any]) -> str:
        index = int(_number(event.get("i")))
        total = int(_number(event.get("total")))
        head = f"[{index}/{total}]" if index and total else "[file]"
        name = Path(str(event.get("path") or "")).name
        error = str(event.get("error") or "")
        if error:
            # Lower case deliberately: the gate chains grep their logs for the
            # upper-case token, and one failed page is reported by `done`.
            return f"{head} {name}  failed: {error}"
        tail: list[str] = []
        width, height = int(_number(event.get("w"))), int(_number(event.get("h")))
        if width and height:
            tail.append(f"{width}x{height}")
        size = _number(event.get("bytes"))
        if size:
            tail.append(human_bytes(size))
        millis = _number(event.get("ms"))
        if millis:
            tail.append(f"{millis / 1000:.1f}s")
        model = str(event.get("model") or "")
        if model:
            tail.append(model)
        return f"{head} {name}" + (f"  {'  '.join(tail)}" if tail else "")

    @staticmethod
    def _render_done(event: dict[str, Any]) -> str:
        failed = int(_number(event.get("failed")))
        if event.get("cancelled"):
            state = "cancelled"
        elif event.get("ok") and not failed:
            state = "done"
        else:
            state = "finished with errors"
        counts = (
            f"{int(_number(event.get('processed')))} processed, "
            f"{failed} failed, "
            f"{int(_number(event.get('skipped')))} skipped"
        )
        line = f"{state}: {counts} in {_number(event.get('elapsed')):.1f}s"
        error = str(event.get("error") or "")
        return f"{line}\n{error}" if error else line


# --------------------------------------------------------------------------- #
# driving one worker process
# --------------------------------------------------------------------------- #
def install_interrupt(runner: Runner) -> None:
    """First Ctrl-C asks the worker to stop between pages; a second one kills it.

    Cancel travels over stdin, the worker's only control channel, so the default
    SIGINT behaviour - raise here and leave the child holding the GPU - is
    exactly the wrong thing for this process.
    """
    asked = False

    def handler(signum: int, frame: object) -> None:
        nonlocal asked
        if asked:
            print("killing the worker", file=sys.stderr, flush=True)
            runner.kill()
            return
        asked = True
        print(
            "stopping after the current page (Ctrl-C again to kill)",
            file=sys.stderr,
            flush=True,
        )
        runner.cancel()

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:
        # Not the main thread: whoever embedded this owns signal handling.
        pass


def exit_code_for(console: Console) -> int:
    """The process exit code implied by the events we saw."""
    done = console.done
    if done is None:
        # No `done` event means the contract was broken, whatever the code said.
        return EXIT_FAILED
    if done.get("cancelled"):
        return EXIT_CANCELLED
    if not done.get("ok") or int(_number(done.get("failed"))):
        return EXIT_FAILED
    return EXIT_OK


def drive(runner: Runner, console: Console, cancel_after: float) -> int:
    """Pump events until the worker exits, then report its outcome."""
    deadline = time.monotonic() + cancel_after if cancel_after > 0 else None
    asked_to_stop = False
    while console.exit_code is None:
        try:
            console.handle(runner.events.get(timeout=POLL_SECONDS))
        except queue.Empty:
            pass
        if deadline is not None and not asked_to_stop and time.monotonic() >= deadline:
            asked_to_stop = True
            print(f"cancelling after {cancel_after:g}s", file=sys.stderr, flush=True)
            runner.cancel()
    # The stderr reader can still be draining; those lines were emitted before
    # the process went away, so they belong in the output.
    runner.drain(console.handle)
    return exit_code_for(console)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace, *, dry: bool) -> int:
    root = app_root()
    src = Path(args.src).expanduser()
    if not src.exists():
        print(f"input not found: {src}", file=sys.stderr)
        return EXIT_USAGE
    data = load_settings(root, str(args.settings or ""), ignore_saved=bool(args.no_settings))
    apply_overrides(data, args)
    runner = Runner(root)
    models_dir = (
        Path(args.models_dir).expanduser() if args.models_dir else runner.paths().models_dir
    )
    if not ensure_rules(data, models_dir):
        print(
            "no rules are enabled, so no model would run; pass --rules or install models",
            file=sys.stderr,
        )
        return EXIT_USAGE
    job = build_job(
        data,
        src,
        resolve_out_dir(src, str(args.out or ""), data),
        models_dir,
        dry=dry,
        resume=bool(args.resume),
    )
    if args.print_job:
        print(json.dumps(job, indent=2, ensure_ascii=False))
        return EXIT_OK
    console = Console(raw=bool(args.json), quiet=bool(args.quiet))
    install_interrupt(runner)
    if not runner.start(job):
        runner.drain(console.handle)
        return EXIT_FAILED
    try:
        return drive(runner, console, float(args.cancel_after or 0.0))
    finally:
        # Never leave a worker holding the GPU because this process is leaving.
        runner.shutdown()


def cmd_probe(args: argparse.Namespace) -> int:
    runner = Runner(app_root())
    runner.probe()
    deadline = time.monotonic() + PROBE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            event = runner.events.get(timeout=POLL_SECONDS)
        except queue.Empty:
            continue
        kind = str(event.get("type") or "")
        if kind == "probe":
            print(
                json.dumps(event, ensure_ascii=False) if args.json else json.dumps(event, indent=2)
            )
            return EXIT_OK
        if kind == "probe_error":
            print(str(event.get("message") or "probe failed"), file=sys.stderr)
            return EXIT_FAILED
    print("the probe did not answer", file=sys.stderr)
    return EXIT_FAILED


def cmd_where(args: argparse.Namespace) -> int:
    resolved = core_paths.resolve(app_root())
    if args.json:
        print(json.dumps(resolved.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(core_paths.report(resolved))
    return EXIT_OK


# --------------------------------------------------------------------------- #
# argument surface
# --------------------------------------------------------------------------- #
def job_options() -> argparse.ArgumentParser:
    """The options `run` and `plan` share, so the two can never drift apart."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("src", help="image, folder or archive to convert")
    ap.add_argument("-o", "--out", metavar="DIR", help="output folder")
    ap.add_argument("--subfolder", metavar="NAME", help="subfolder used when -o is omitted")
    ap.add_argument("--container", choices=tuple(CONTAINERS), help="how output is packaged")
    ap.add_argument("--format", dest="format_id", choices=FORMAT_IDS, help="output format")
    ap.add_argument("--pattern", help='output name pattern, e.g. "{name}_JaNai"')
    ap.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip sources the output folder's manifest already records as done",
    )
    ap.add_argument("--keep-structure", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--archives",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="walk into CBZ/ZIP/CBR/RAR inputs",
    )
    ap.add_argument("--mode", choices=TARGET_MODES, help="how the target size is decided")
    ap.add_argument("--scale", type=float, help="factor for --mode scale")
    ap.add_argument("--width", type=int, help="target width for --mode width or fit")
    ap.add_argument("--height", type=int, help="target height for --mode height or fit")
    ap.add_argument("--page-kind", choices=PAGE_KINDS, help="treat pages as gray, colour or detect")
    ap.add_argument("--auto-levels", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--rules", metavar="FILE", help="JSON rule table to use instead of the saved one"
    )
    ap.add_argument("--device", help='device id, e.g. cuda:0 ("" picks the best)')
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--tile", help="auto, none, maximum or a pixel size")
    ap.add_argument("--io-workers", type=int, help="page reader threads")
    ap.add_argument("--models-dir", metavar="DIR", help="where the weights live")
    ap.add_argument("--settings", metavar="FILE", help="settings file to start from")
    ap.add_argument(
        "--no-settings",
        action="store_true",
        help="ignore the saved settings and start from the shipped defaults",
    )
    ap.add_argument("--json", action="store_true", help="stream the worker's event lines verbatim")
    ap.add_argument(
        "-q", "--quiet", action="store_true", help="only report problems and the result"
    )
    ap.add_argument(
        "--cancel-after",
        type=float,
        metavar="SECONDS",
        help="request a stop after this long, for testing cancellation",
    )
    ap.add_argument(
        "--print-job",
        action="store_true",
        help="print the resolved job payload and exit without running it",
    )
    return ap


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="janai", description="Run the JaNai upscaler without the GUI."
    )
    sub = ap.add_subparsers(dest="command", required=True)
    shared = job_options()
    run = sub.add_parser("run", parents=[shared], help="convert an image, folder or archive")
    run.set_defaults(action="run")
    plan = sub.add_parser(
        "plan", parents=[shared], help="dry run: report every page and write nothing"
    )
    plan.set_defaults(action="plan")
    probe = sub.add_parser("probe", help="report devices, encoders and installed models")
    probe.add_argument("--json", action="store_true", help="one line instead of indented JSON")
    probe.set_defaults(action="probe")
    where = sub.add_parser("where", help="report the locations this install resolves to")
    where.add_argument("--json", action="store_true", help="machine-readable output")
    where.set_defaults(action="where")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    action = str(getattr(args, "action", ""))
    try:
        if action == "run":
            return cmd_run(args, dry=False)
        if action == "plan":
            return cmd_run(args, dry=True)
        if action == "probe":
            return cmd_probe(args)
        return cmd_where(args)
    except (OSError, ValueError) as exc:
        # Bad paths and bad rule files are usage errors, not crashes: a driver
        # that prints a traceback for a typo is hostile to script it.
        print(str(exc), file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
