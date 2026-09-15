# AGENTS.md — JaNai Upscaler (MangaJaNaiConverterGui-MOD)

Read this before changing anything. It records what the app is, where the code
lives, the commands that actually verify a change, and the contracts that look
like cruft but are load-bearing.

**This file cites modules and symbols, not line numbers.** An earlier version
cited four (`window.py:2383`, `worker.py:1874`, `worker.py:1765`,
`worker.py:1720`); all four rotted in a single refactor and pointed at the wrong
code. Grep for the symbol instead.

## What the app is

A Windows-first (Linux-capable) desktop app that upscales manga and comic pages
with super-resolution models. You point it at images, folders or `.cbz`/`.zip`/
`.cbr`/`.rar` archives; it picks a model per page from a rules table, upscales,
re-encodes, and writes images, folders or a single archive back out.

It runs as **two processes**:

| Process | Code | Imports |
| --- | --- | --- |
| Interface | `src/janai/app/` | standard library + PySide6 only |
| Worker | `src/janai/worker/` (entry `worker.py`) | torch, spandrel, pyvips, numpy, the vendored `backend/src` backend |

The split is deliberate and worth preserving: the window opens in well under a
second because nothing in the GUI process can pull in torch, and a crash in the
upscaler cannot take the interface down with it. The GUI spawns one worker per
job, streams its events, and can cancel it.

## Stale-brief warning (read before planning a refactor)

Older task briefs describe a design this repo no longer uses. Verified against
the tree:

- **`PROGRESS=…` and `TOTALZIP=…` are not the wire format.** They appear nowhere
  in the live app. The protocol is JSON Lines (below).
- **`PipelineQueue`, `put_sentinel()`, `consumer_exited()` and
  `PipelineConsumerGone` do not exist here.** The live worker expresses the same
  "never hang a job" requirement with different primitives — `prefetch()`,
  `WritePool`, `BundleWriter`, `Control` — documented under Invariants.
- Both belonged to `backend/src/run_upscale.py`, the chaiNNer-era CLI that had no
  caller left: no entry point, launcher, setup script or workflow invoked it, and
  CI only byte-compiled it. **It was deleted** rather than kept as a misleading
  second copy of the pipeline; recover it from git history if ever needed.
- **There is no monolith left to split.** `worker/worker.py` went 3264 → 168
  lines and `app/window.py` 2764 → 811; the package is now 54 files. Do not
  "restore" the old protocol or the old pipeline classes into `src/janai/`.

## Architecture map

Every description below is the module's own docstring summary — if you change
what a module does, change its docstring and this map together.

```
JaNaiUpscaler.cmd          Windows launcher: finds an interpreter, runs src/janai/app/main.py
janai-upscaler.sh          the same for Linux
setup.cmd -> setup.ps1     one-shot setup: creates backend/python with uv, installs requirements.txt, fetches models
setup.sh                   the Linux equivalent

src/janai/
  __main__.py              `python -m janai` opens the interface
  cli.py                   headless driver: run/plan/probe/where, no Qt (see Headless driver)
  app/                     the Qt interface (stdlib + PySide6 only)
    main.py                GUI entry point; inserts src/ on sys.path when run as a file
    window.py              the main window: four cards, a footer, and the run log
    surface.py             the composed-window surface every panel mixin may assume (Protocols)
    widgets.py             the small widget kit the window is assembled from
    fields.py              the option ladders the controls offer, and label/value conversions
    theme.py               palette, type ramp and stylesheet
    input_panel.py         the Input card: choosing what to convert, and counting it
    output_panel.py        the Output card: encoder, packaging, destination and file names
    perf_panel.py          the Performance card: device, precision, tiling, threads, and the probe
    rules_panel.py         both rule tables: what runs on a page, what skips the model
    rules_table.py         a real Qt model over janai.core.rules.Rule rows
    run_panel.py           the footer, the worker's event stream, everything Start touches
    log_panel.py           the run-log panel: its widgets and everything that writes to them
    runlog.py              one aligned, readable line per event, mirrored to logs/Run_<ts>.log
    runner.py              worker supervision: launch, stream JSONL, send control lines
    state.py               persisted settings; one JSON file, written atomically
  core/                    imported by BOTH processes, so it must stay dependency-free
    paths.py               where the runtime, models and support files are found
    rules.py               rules-based model selection (the only thing that chooses a model)
    formats.py             output format catalogue, shared by GUI and worker
    displays.py            display presets for the "Fit" target mode
    hardware.py            hardware identity for the tile profile
    presets.py             settings presets: export what is on screen, import it later
    fspath.py              handing Windows a path it will actually accept
  worker/
    worker.py              CLI entry: argument parsing, mode dispatch, top-level failure reporting
    job.py                 job execution: reading a planned job's sources, predicting its output
    orchestrate.py         per-unit orchestration: one source archive, or one run of loose images
    archives.py            reading the inside of a CBZ/CBR source
    page.py                per-page pixel work: detect, transform, upscale, encode
    reporting.py           job-level reporting: page writes, page failures, bundle completion
    selection.py           which model a page gets, whether it is levelled, whether it is skipped
    pipeline.py            the concurrency primitives the pipeline stages are built from
    planning.py            what a run will produce, before a single pixel is decoded
    resume.py              resume state: which sources an output folder has already finished
    control.py             cancel and pause, fed by the interface as bare words on stdin
    events.py              the worker's half of the wire protocol: one JSON object per stdout line
    imageio.py             pixels in, bytes out: decoding, ICC transforms and encoding
    imagetypes.py          one name for a decoded page, so every stage spells it the same way
    transforms.py          per-page pixel transforms: resize, grayscale detection, levels
    models.py              which model to use, and the loaded network behind it
    devices.py             picks the compute device, builds the node context the vendored nodes want
    tiling.py              tile planning: how large a tile each page is upscaled in
    profiling.py           measuring what this machine can do, ahead of any real job
    capabilities.py        what this install can actually do: libvips operations and bundled tools
    environment.py         where this install keeps its runtime, models, backend source and tools
    runtime.py             owns the heavy stack: numpy, OpenCV, libvips, Pillow, torch, vendored nodes
    probe.py               the one-shot capability report the GUI reads before it can draw anything
    hold.py                keeping an accelerator awake between jobs

scripts/                   the project's own harnesses (see Verification)
backend/
  python/                  the interpreter and its packages (uv venv) — gitignored, per machine
  models/                  model weights — gitignored
  src/                     vendored chaiNNer-derived backend the worker imports
  ImageMagick/, tools/, resources/
```

### Dependency direction

Package level, one-way: `app -> core`, `worker -> core`, `worker ->
backend/src`. Nothing in `core` may import `app` or `worker`, and nothing in
`app` may import `worker` except `runner.py`, which only resolves the worker's
**file path** (`Path(worker_pkg.__file__).parent / "worker.py"`) to hand to
`subprocess`.

Inside the worker, the call chain is also one-way and must stay that way:

```
worker.py -> job.py -> orchestrate.py -> {archives, page, reporting, pipeline,
                                          planning, control, events, imageio,
                                          fspath}
```

`archives.py` exists as its own module for exactly one reason: `open_archive` is
needed by both `orchestrate.handle_archive` (the real run) and `job.dry_run`
(prediction). Leaving it in `job.py` made `orchestrate` and `job` import each
other. Do not move it back up. `imagetypes.py` is a leaf for the same kind of
reason — see Typing.

## The GUI ↔ worker contract

Both directions are a wire format between two processes. Changing a name on one
side without the other silently breaks the interface — no exception, just a UI
that stops updating.

### worker → GUI: JSON Lines on stdout

One JSON object per line. `emit(kind, **payload)` in `worker/events.py` sets
`payload["type"] = kind`, serialises with `ensure_ascii=False, default=str`, and
writes under `_stdout_lock`. The kinds the **worker** puts on the wire:

| `type` | Meaning | Notable fields |
| --- | --- | --- |
| `log` | human-readable line for the run log | `level`, `message` |
| `start` | a run has begun; the GUI shows device, precision and tile | `device`, `fp16`, `tile` |
| `progress` | a file (or archive entry) has been *started* | `i`, `total`, `path`, `sub_i`, `sub_n` |
| `file` | a file finished, was skipped, or failed | `i`, `total`, `path`, `error` |
| `bundle` | a single-archive output was planned or written | `out`, `entries`, `planned`, `dry` |
| `done` | the job ended; carries the final counters | `ok`, counters, `error` |
| `probe` | devices, encoders, models, library versions | whole `info` dict |
| `profile` | tile-profile measurement result | `ok`, `error`, measurements |
| `profile_progress` | one step of a measurement | `model`, `tile`, `index`, `total` |
| `hold` | GPU keep-alive process state | `ok`, `device`, `name`, `reserved`, `pid`, `released`, `error` |

**Three more kinds reach the GUI's dispatcher but are never on the wire** — the
GUI synthesises them into the same event queue, so grepping the worker for them
finds nothing. Do not "implement" them in the worker:

| `type` | Injected by | Why |
| --- | --- | --- |
| `probe_error` | `app/runner.py` | the interpreter was missing, or the worker produced no probe line |
| `done` (`ok=False`) | `app/runner.py` | the process died *without* emitting its own `done`, so the GUI must still finish the run |
| `exit` | `app/runner.py` | the process exit code, emitted after `wait()` |
| `scan` | `app/input_panel.py` | the GUI's own input-scan thread counting sources (`single`/`bulk`/`missing`) |

`app/run_panel.py` dispatches all of them in one `kind == …` chain; that chain is
the authoritative consumer list.

Rules that must hold:

- **`emit()` is the only writer of the worker's stdout, and it holds a lock.**
  Stdout is the wire; two threads writing it directly would interleave into
  unparseable lines. Anything diagnostic goes through `log()` (which is an
  `emit`) — never `print`.
- **Non-JSON stdout lines are tolerated, not parsed.** `runner.py` gates on
  `line.startswith("{")` in all three reader loops and turns everything else
  into a `log` event, so a stray library banner degrades to a log line instead
  of killing the run.
- Stdout is reconfigured to `utf-8`/`newline="\n"` in the worker's `main()` so
  Windows does not turn the protocol into CRLF or mangle non-ASCII filenames.
  `scripts/stdin_check.py --reconfigure` asserts this.

### GUI → worker: bare control words on stdin

One lowercase word per line, written by `runner._send()`:

| Word | Effect |
| --- | --- |
| `cancel` / `abort` / `stop` | `Control.cancel()`: sets the cancel event, clears pause, calls `abort()` and `resume()` on the backend progress token |
| `pause` | sets the pause event; `Control.gate()` blocks callers |
| `resume` | clears the pause event |
| `stop` / `cancel` / `quit` / `exit` (to a `--hold` process) | releases the held GPU context and exits |

`Control` (in `worker/control.py`) runs a daemon `stdin` pump thread; `Cancelled`
is the in-process abort exception. `gate()` returns immediately when cancelled,
so a paused job can still be cancelled — do not turn it into a plain blocking
wait.

### Worker CLI

```
worker.py --probe [--models-dir DIR]        report devices, encoders, models (one JSON line)
worker.py --job job.json                    run a job, streaming JSONL events
worker.py --job job.json --dry-run          report what would happen, write nothing
worker.py --job job.json --profile          measure this machine's tile cost, write nothing
worker.py --hold [--device cuda:0] [--hold-interval 15.0]
                                            keep a GPU context awake until stdin says stop
```

### Headless driver (`janai`)

`src/janai/cli.py` is the supported way to convert without the GUI, and the
thing to drive from a test or a script instead of hand-writing a `job.json`.

```
janai run   SRC [-o OUT] [options]   convert
janai plan  SRC [...]                dry run: report every page, write nothing
janai probe                          devices, encoders and installed models
janai where                          the locations this install resolves to
```

It **reuses** the GUI's machinery rather than copying it: the payload is built
from the same `settings.json` through `app/state.py`, and the worker is spawned,
cancelled and drained through the same `app/runner.py`. *Why:* a second spawn
path, or a second author of the job schema, is exactly how a driver drifts from
the app it is meant to mirror. The direction is `cli → app → core`, no cycle.

Flags worth knowing: `--print-job` prints the resolved payload and runs nothing;
`--json` passes the worker's event lines through untouched (that stream is the
contract — do not reformat it into a weaker second one); `--cancel-after
SECONDS` requests a stop mid-job, which is how cancellation is tested without a
human at a terminal; `--no-resume` converts everything again instead of trusting
the output folder's resume manifest. Exit codes: `0` finished, `1` the job
reported a failure, `2` bad usage, `130` cancelled.

The payload carries `resume` **only** when it differs from the worker's default,
so `janai run --print-job` still prints byte-for-byte what the GUI would build.
*Why:* the driver's whole value is being the same job the app runs, and a key
only the driver ever sets is how that stops being true.

Use the console script after an editable install, or run it straight from a
checkout — it bootstraps `sys.path` exactly as `worker.py` does:

```bash
$PY src/janai/cli.py where
$PY src/janai/cli.py plan DIR --print-job
```

## Invariants an agent must not "clean up"

Each line is a real bug that was already paid for once. The gate that proves it
is named where one exists.

1. **One status-label writer.** `RunPanel.render_status()` in `app/run_panel.py`
   is the only code that sets the status label; it shows `min(finished + 1,
   started_index)`, clamped to `>= finished` and `<= total` — the oldest file
   still in flight. *Why:* the worker reads, upscales and writes on separate
   threads, so the "started" index runs ahead of the finished count; two writers
   made the number jump forward and then fall back. The contract is enforced in
   code, not just here: the `app/surface.py` Protocol declares it as *"Write the
   status label. The ONLY writer of it."*
2. **Ordered archive writes.** `BundleWriter` (`worker/pipeline.py`) packs pages
   through a **single**-thread `ThreadPoolExecutor` plus a bounded semaphore, and
   `orchestrate.py` chooses each entry name **on the producing thread**, so CBZ
   order follows read order. *Why:* page order inside a `.cbz` is what a reader
   sees; a second packer, or naming on the writer thread, reorders it.
   (`scripts/bundle_check.py`, `scripts/archname_check.py`)
3. **Ordered, bounded prefetch.** `prefetch()` (`worker/pipeline.py`) yields
   `(unit, array_or_exception)` **in order** through a `Queue(maxsize=workers+1)`
   with a `stop` event, and callers `close()` the generator to release a pump
   parked in `put`. *Why:* the bound is the memory ceiling for decoded pages, and
   passing the exception along instead of raising in the reader thread is how a
   failed decode is reported rather than lost. (`scripts/prefetch_check.py`)
4. **Bounded write pool with tracked futures.** `WritePool`
   (`worker/pipeline.py`) admits work through a semaphore sized
   `workers * WRITE_SLOTS_PER_WORKER` and keeps outstanding futures in a
   `pending` set under a lock so it can drain. *Why:* unbounded submission
   balloons memory; untracked futures swallow write errors.
   (`scripts/writepool_check.py`)
5. **Cancellation is cooperative and always has an exit.** `Control` +
   `Cancelled` + `gate()`. *Why:* a blocking call with no abort path used to make
   "Cancel" mean "wait for the whole job". (`scripts/stdin_check.py`,
   `scripts/close_check.py`)
6. **`emit()` owns stdout under a lock** (see the protocol section).
7. **`janai.core` stays import-light and `janai.app` never imports torch,
   numpy, pyvips or spandrel.** *Why:* `core` is loaded by both processes and the
   interface's startup time depends on it. This is asserted, not assumed: gate
   legs import `janai.core.*`, `worker.pipeline` and `worker.imageio` and fail if
   any heavy module lands in `sys.modules`.
8. **`open_archive()` is a context manager and the reader dies with the block.**
   *Why:* a live `ZipFile`/`RarFile` handle keeps the source file locked on
   Windows, so the archive could not be renamed or replaced afterwards.
   (`scripts/archname_check.py` asserts both the type and the post-block rename)
9. **Tile profiles are bound to hardware identity** (`core/hardware.py`). *Why:*
   a measurement from another GPU is worse than no measurement.
10. **The final `done` event is the run's only source of truth for counters.**
    The GUI fabricates one with `ok=False` if the process dies without it, so
    never drop or conditionalise the real one. (`scripts/donepages_check.py`,
    `scripts/counters_check.py`)
11. **The headless driver's import chain stays Qt-free.** `janai.cli` →
    `app.runner` → `app.state` must pull in no PySide6. *Why:* the driver exists
    for machines with no display, so importing Qt for a job that never opens a
    window would defeat it. (`scripts/smoke.py`'s `headless driver` leg imports
    the driver and fails if `PySide6` lands in `sys.modules`)
12. **A source is recorded as finished only *after* whatever publishes it — the
    rename for an archive or a bundle, the write itself for a loose page — and
    `worker/resume.py` is the manifest's only writer.** *Why:* the
    manifest may forget a finished chapter — that costs one redundant
    re-convert — but if it can claim an unfinished one, a resumed run silently
    skips a chapter the user never got. The same asymmetry decides the rest of
    the contract: a cancelled archive keeps its `.cbz.part` and records the
    entry names inside it, and the next run appends only when those names are a
    *prefix* of this run's plan, because appending out of order changes page
    order in the reader. A manifest whose fingerprint does not match the current
    job is ignored, never merged. A bundle is recorded against its member
    *sources*, never against the task key, which planning leaves empty for a
    single-archive run — a record under that key would claim "the bundle for
    this folder is done" and skip a different input converted into the same
    folder. (`scripts/smoke.py`'s `resume manifest`, `resume wiring` and
    `resume records every path` legs)
13. **Every call into a vendored node translates `api.node_context.Aborted`
    back into `Cancelled`.** `ModelCache.get` and `upscale_array` (both in
    `worker/models.py`) are the only two such call sites, and both do it.
    *Why:* the vendored progress token raises its *own* `Aborted` when it
    notices our cancel, and the page loop's broad handler counted that as a lost
    page — a clean stop reported `pages_failed: 1` and logged a blank warning,
    because `Aborted` carries no message. Translate, never swallow: a genuine
    upscale failure must stay a failure, so the translation is keyed on that one
    class and on nothing wider. (`scripts/smoke.py`'s `cancel is not a lost
    page` leg)

## Verification

The embedded interpreter is the one the app ships with — use it for anything
import- or dependency-related. `PYTHONPATH` does **not** reliably reach it
through a POSIX shell on Windows; use `sys.path.insert(0, "src")` inside a
`-c` snippet instead.

```bash
PY=backend/python/python.exe            # Windows; Linux: backend/python/bin/python

$PY -m compileall -q src scripts        # byte-compile the app, worker and scripts
$PY -m compileall -q backend/src        # byte-compile the vendored backend
$PY scripts/smoke.py                    # dependency-free checks, ~1s
$PY scripts/plannercheck.py             # tile-planner regressions, no GPU needed
QT_QPA_PLATFORM=offscreen $PY scripts/uicheck.py   # builds the real window and measures it
$PY scripts/selftest.py                 # end-to-end; needs torch, pyvips and a model
$PY scripts/bench.py                    # per-file timing across tile settings (GPU + an image)

$PY src/janai/cli.py where                  # headless driver: what this install resolves to
$PY src/janai/cli.py plan DIR --print-job   # the exact payload the GUI would build
```

**Run the gates by glob, not from a list** — new ones get added and a
hand-written list goes stale:

```bash
for g in scripts/*_check.py; do
  case "$g" in
    */stdin_check.py) printf '\x81\ncancel\n' | $PY "$g" --reconfigure ;;
    */close_check.py) QT_QPA_PLATFORM=offscreen $PY "$g" ;;
    *) $PY "$g" ;;
  esac || echo "FAIL $g"
done
```

Each `*_check.py` prints `[PASS]`/`[FAIL]` lines and exits non-zero on failure.
`scripts/stubborn_child.py` is a fixture for `close_check`, not a gate.

Lint, format and type-check (none of these is installed in `backend/python`;
`uv` is available locally and CI pins the same versions):

```bash
uvx ruff@0.16.7 format --check .        # CI uses: ruff format --check --diff .
uvx ruff@0.16.7 check .                 # CI uses: ruff check --output-format=github .
uvx ruff@0.16.7 format .                # to apply formatting
uvx mypy@2.3.1                          # files/mypy_path come from pyproject.toml
```

mypy is **green with no config-level suppressions**: there is no
`disable_error_code` anywhere, and no blanket `# type: ignore`. Two narrow,
error-code-scoped ignores survive, both in `core/rules.py` and both at the
JSON-settings boundary — `_as_float` (`arg-type`), which hands an arbitrary
settings value to `float()` and catches the failure, and `RuleSet.match`
(`return-value`), where the `_MISS` cache sentinel widens the value type past
what `is not _MISS` narrows. Neither hides a real defect. If you touch either
line, delete the ignore rather than widen it.

`pyproject.toml` enables four extra strict flags
(`disallow_any_generics`, `disallow_untyped_defs`, `disallow_incomplete_defs`,
`warn_return_any`) for the Tier 1 modules that hold the contracts: `janai.core.*`,
`worker.pipeline`, `worker.planning`, `worker.control`, `worker.events`,
`app.runlog`. To see what those flags would cost everywhere else:

```bash
uvx mypy@2.3.1 --disallow-any-generics --disallow-untyped-defs \
               --disallow-incomplete-defs --warn-return-any
```

That reports **163 findings, all outside Tier 1, and they are deliberately
deferred**: most are unparameterised generics whose real parameters are torch,
numpy, pyvips or PySide6 types, which mypy cannot resolve because those packages
exist only in `backend/python`. Annotating them would mean writing `list[Any]` —
noise, not proof. Promote a file only when a real type exists (as
`worker/imageio.py` and `worker/orchestrate.py` did).

### What CI actually runs

`.github/workflows/ci.yml`, Python 3.13:

- **ruff job** (Ubuntu): `pip install ruff==0.16.7`, `ruff format --check --diff .`,
  `ruff check --output-format=github .`
- **matrix job** (windows-latest **and** ubuntu-latest): `compileall src scripts`,
  `compileall backend/src`, `smoke.py`, `plannercheck.py`, then the
  dependency-free gates `prefetch_check`, `counters_check`, `writepool_check`,
  `bundle_check`, `donepages_check` and `stdin_check --reconfigure`, then
  `uicheck.py` with `PySide6-Essentials==6.11.2` (Linux also installs `libegl1`
  and `libxkbcommon-x11-0`, which Qt links against).

**mypy is not in CI** — it is a local gate only. If you touch typing, run it
yourself; nothing else will.

## Conventions

- **Structure:** `src/` layout, package `janai`, three packages with the one-way
  dependency direction above. New shared code goes in `core` only if both
  processes need it and it stays dependency-free. No `utils` module: put a
  function with the thing it serves.
- **Typing:** annotate public functions. A decoded page is `ImageArray` from
  `worker/imagetypes.py`, which is `np.ndarray` under `TYPE_CHECKING` and `Any`
  at runtime. It lives in its own leaf module so `imageio`, `transforms`, `page`
  and `orchestrate` can name a page without importing the concurrency module,
  and numpy stays out of module scope (invariant 7). `-> Any` survives only where
  the value is a genuinely unresolvable library handle (a `pyvips.Image`, a
  `PIL.Image`); those sites say so in a comment. Config bags that cross a
  boundary are dataclasses (`DryRunPlan`, `UnitRunner`, `PagePolicy`), not dicts.
  Payloads that arrive off the JSON wire are still `dict[str, Any]`, which is
  honest about what a wire payload is.
- **Errors:** the worker reports failure as an event (`file` with `error=`,
  `profile`/`hold` with `ok=False`, a final `done` with `ok=False`) and keeps
  going where a job can continue. Never let an exception escape a thread
  silently; pass it along like `prefetch()` does. `Cancelled` is control flow —
  never swallow it in a broad `except`.
- **Paths:** `pathlib` everywhere. `core/paths.py` is the single source of truth
  for locations; do not re-derive `backend/…` paths elsewhere. `core/fspath.py`
  owns Windows path acceptability (`io_path`, `path_too_long`).
- **No magic numbers:** name them next to what they describe
  (`WRITE_SLOTS_PER_WORKER` in `pipeline.py` is the pattern).
- **Commits:** one concern per commit; the message says **why**. Behaviour-
  preserving moves and behaviour-changing fixes go in separate commits so a
  regression bisects to a real change. Verify before committing; never
  `git add -A` (verification runs can dirty the tree — stage reviewed paths).

## Gotchas and sharp edges

- **Line endings.** `.editorconfig` is authoritative: LF everywhere, **CRLF for
  `*.cmd`, `*.bat`, `*.ps1`** because `cmd.exe` mis-parses LF-only batch files.
  Git will say `LF will be replaced by CRLF` on almost every commit here; that is
  normal, not a problem to fix.
- **`Path("")` is truthy.** It normalises to `WindowsPath('.')`, so
  `if not out_dir:` on a `Path` never fires — a job with no output directory
  silently wrote pages into the worker's CWD. Validate the *string* before
  constructing the `Path`. (`scripts/outdir_check.py` locks this down.)
- **Windows paths.** Archive entry names are not filesystem-safe
  (`safe_name()`), output paths are de-duplicated by `unique_path()`, and
  `core/fspath.py` handles the 260-character limit. Long paths and unicode entry
  names in `.cbz`/`.rar` are the historical source of bugs here;
  `archname_check` round-trips `第01話.jpg` on purpose.
- **`backend/python`, `backend/models`, `settings.json`, `presets/`, `logs/`,
  `janai.config.json`, `janai.runtime.txt` are gitignored per-machine state.**
  Never commit them, never assume a fresh clone has them.
- **`setup.sh` refuses to run if `backend/src` is missing** — the worker imports
  the vendored backend from there, so it is not optional.
- **`sanic==24.6.0` is in `requirements.txt` only because vendored backend
  modules import `sanic.log`.** It is not a web server here.
- **ruff excludes `backend/`, `logs/` and `.tmp/`** (`pyproject.toml`); vendored
  code is not held to this project's style. Line length is **100**.
- **Orphan bytecode is rot, not cache.** A `__pycache__/*.pyc` whose `.py` is
  gone cannot be imported — a sourceless import must sit in the source location,
  not in `__pycache__` — but it keeps `grep` reporting modules that no longer
  exist. Two survived the Tk→Qt rewrite and the `window.py` split, and this
  document used to name one of the two. `smoke.py` now fails on any of them:
  delete the file, not the check.
- **`.janai-resume.json` in the output root is the resume manifest**
  (`worker/resume.py`), not output the user asked for. Deleting it is safe — it
  only costs the next run its progress — and the scanners ignore it because
  `.json` is in neither extension set.
- **A cancelled archive now leaves a `.cbz.part` behind on purpose.** It used to
  be deleted. That file is what lets "cancelled inside chapter 11" resume inside
  chapter 11, and keeping it publishes nothing: the atomic `replace()` is still
  the only way the real `.cbz` is ever created.

## Working protocol for agents

Two untracked files support agent work and must stay untracked (`.gitignore`
covers `MEMORY.md`, `.skills/` and `skills/`):

- **`MEMORY.md`** — the sole memory for a long task. Chat context is unreliable,
  so it is updated after every completed step, before and after every edit batch,
  on every decision, and on every finding. It must let a fresh agent resume
  without re-deriving anything, and it is **pruned** rather than appended
  forever: a finished item becomes a one-line record, not a story.
- **`.skills/`** — user-maintained guidance (note: dotted directory; briefs
  sometimes call it `./skills/`). Read it at the start of a session. It is
  guidance, not gospel: where it conflicts with the code, the code wins and the
  conflict gets recorded in `MEMORY.md`.

The method that worked for every commit in the refactor, in order:

1. **Prove the blast radius with a scoped grep** (always pass a path *and* an
   include filter — a repo-wide grep hits `.mypy_cache/` and blows the context
   budget).
2. **Write the gate first** where behaviour is at stake, and prove it *fails*
   against the unfixed tree. A gate that has never failed proves nothing.
3. Make the change as one atomic, sha-guarded edit batch.
4. **Run the chain with the commit chained behind `FAIL=0`**, so nothing can land
   on a failure. Order the legs **fast first** — compile, `ruff format --check`,
   `ruff check`, mypy — and exit before the GPU legs; a 101-character line once
   wasted a full real-GPU run.
5. **Re-measure line numbers and counts from the code** before recording them.
   Derived-by-arithmetic counts were wrong five times in this refactor.
