# AGENTS.md — JaNai Upscaler (MangaJaNaiConverterGui-MOD)

Read this before changing anything. It records what the app is, where the code
lives, the commands that actually verify a change, and the contracts that look
like cruft but are load-bearing.

## What the app is

A Windows-first (Linux-capable) desktop app that upscales manga and comic pages
with super-resolution models. You point it at images, folders or `.cbz`/`.zip`/
`.cbr`/`.rar` archives; it picks a model per page from a rules table, upscales,
re-encodes, and writes images, folders or a single archive back out.

It runs as **two processes**:

| Process | Code | Imports |
| --- | --- | --- |
| Interface | `src/janai/app/` | standard library + PySide6 only |
| Worker | `src/janai/worker/worker.py` | torch, spandrel, pyvips, numpy, the vendored `backend/src` backend |

The split is deliberate and worth preserving: the window opens in well under a
second because nothing in the GUI process can pull in torch, and a crash in the
upscaler cannot take the interface down with it. The GUI spawns one worker per
job, streams its events, and can cancel it.

## Stale-brief warning (read before planning a refactor)

Older task briefs and `backend/src/README.md` describe a design this repo no
longer uses. Verified as of this commit:

- **`backend/src/run_upscale.py` (1943 lines) is unreachable.** No entry point,
  launcher, setup script or workflow invokes it; the only thing CI does with it
  is `python -m compileall -q backend/src`. It is chaiNNer-era legacy kept next
  to the vendored backend it was written against.
- **`PROGRESS=…` and `TOTALZIP=…` are not the wire format.** Those strings exist
  only inside `run_upscale.py`. The live protocol is JSON Lines (below).
- **`PipelineQueue`, `put_sentinel()`, `consumer_exited()` and
  `PipelineConsumerGone` exist only in `run_upscale.py`.** The live worker
  expresses the same "never hang a job" requirement with different primitives
  (`prefetch()`, `WritePool`, `BundleWriter`, `Control`) — documented below.

So: do not "restore" the old protocol or the old pipeline classes into
`src/janai/`, and do not treat `run_upscale.py` as the monolith to split. The
real large files are `src/janai/worker/worker.py` (3264 lines) and
`src/janai/app/window.py` (2764 lines).

## Architecture map

```
JaNaiUpscaler.cmd          Windows launcher: finds an interpreter, runs src/janai/app/main.py
janai-upscaler.sh          the same for Linux
setup.cmd -> setup.ps1     one-shot setup: creates backend/python with uv, installs requirements.txt, fetches models
setup.sh                   the Linux equivalent

src/janai/
  app/                     the Qt interface (stdlib + PySide6 only)
    main.py                entry point; inserts src/ on sys.path when run as a file
    window.py              the main window: four cards, footer, run log; owns render_status()
    widgets.py             thin wrappers over real Qt widgets (cards, collapsible panel, drop target, field factories)
    rules_table.py         Qt model/view over janai.core.rules.Rule rows
    theme.py               palette, type ramp, one stylesheet for the whole window
    runner.py              worker supervision: Popen, stream JSONL, send control words, hold process
    runlog.py              one aligned line per event, mirrored to logs/Run_<timestamp>.log
    state.py               persisted settings: one JSON file written atomically
  core/                    imported by BOTH processes, so it must stay dependency-free
    paths.py               where the runtime, models, tools and support files live
    rules.py               rules-based model selection (the only thing that chooses a model)
    formats.py             output format catalogue; every option maps to a real libvips save argument
    displays.py            display presets for the "Fit" target mode
    hardware.py            hardware identity, so a tile profile is only trusted on the machine it was measured on
    presets.py             export/import the "how to convert" settings, never the per-machine ones
  worker/
    worker.py              everything heavy: probe, tile planner, model cache, decode, transform, upscale, encode, write, profile

scripts/                   the project's own harnesses (see Verification)
  smoke.py                 dependency-free checks (no torch, pyvips or Qt)
  plannercheck.py          tile-planner regression checks, no GPU and no torch
  uicheck.py               builds the real window offscreen and measures its geometry
  selftest.py              end-to-end: generates a fake library, drives the worker three times
  bench.py                 per-file timing across tile settings

backend/
  python/                  the interpreter and its packages (uv venv) — gitignored, per machine
  models/                  model weights — gitignored
  src/                     vendored chaiNNer-derived backend the worker imports (plus legacy run_upscale.py)
  ImageMagick/, tools/, resources/
```

Dependency direction is one-way: `app -> core`, `worker -> core`, `worker ->
backend/src`. Nothing in `core` may import `app` or `worker`, and nothing in
`app` may import `worker` except `runner.py`, which only resolves the worker's
**file path** (`Path(worker_pkg.__file__).parent / "worker.py"`) to hand to
`subprocess`.

## The GUI ↔ worker contract

Both directions are a wire format between two processes. Changing a name on one
side without the other silently breaks the interface — no exception, just a UI
that stops updating.

### worker → GUI: JSON Lines on stdout

One JSON object per line. `emit(kind, **payload)` in `worker.py` sets
`payload["type"] = kind`, serialises with `ensure_ascii=False, default=str`, and
writes under `_stdout_lock`. Event types in use:

| `type` | Meaning | Notable fields |
| --- | --- | --- |
| `log` | human-readable line for the run log | `level`, `message` |
| `progress` | a file (or archive entry) has been *started* | `i`, `total`, `path`, `sub_i`, `sub_n` |
| `file` | a file finished, was skipped, or failed | `i`, `total`, `path`, `error` |
| `bundle` | a single-archive output was planned or written | `out`, `entries`, `planned`, `dry` |
| `probe` | devices, encoders, models, library versions | whole `info` dict |
| `profile` | tile-profile measurement result | `ok`, `error`, measurements |
| `profile_progress` | one step of a measurement | `model`, `tile`, `index`, `total` |
| `hold` | GPU keep-alive process state | `ok`, `device`, `name`, `reserved`, `pid`, `released`, `error` |

Rules that must hold:

- **`emit()` is the only writer of the worker's stdout, and it holds a lock.**
  Stdout is the wire; two threads writing it directly would interleave into
  unparseable lines. Anything diagnostic goes through `log()` (which is an
  `emit`) — never `print`.
- **Non-JSON stdout lines are tolerated, not parsed.** `runner.py` gates on
  `line.startswith("{")` (lines 111, 179, 279) and treats everything else as
  plain log text, so a stray library banner degrades to a log line instead of
  killing the run.
- Stdout is reconfigured to `utf-8`/`newline="\n"` in `main()` so Windows does
  not turn the protocol into CRLF or mangle non-ASCII filenames.

### GUI → worker: bare control words on stdin

One lowercase word per line, written by `runner._send()`:

| Word | Effect |
| --- | --- |
| `cancel` / `abort` / `stop` | `Control.cancel()`: sets the cancel event, clears pause, calls `abort()` and `resume()` on the backend progress token |
| `pause` | sets the pause event; `Control.gate()` blocks callers |
| `resume` | clears the pause event |
| `stop` (to a `--hold` process) | releases the held GPU context and exits |

`Control` runs a daemon `stdin` pump thread; `Cancelled` is the in-process abort
exception. `gate()` returns immediately when cancelled, so a paused job can
still be cancelled — do not turn it into a plain blocking wait.

### Worker CLI

```
worker.py --probe [--models-dir DIR]        report devices, encoders, models (one JSON line)
worker.py --job job.json                    run a job, streaming JSONL events
worker.py --job job.json --dry-run          report what would happen, write nothing
worker.py --job job.json --profile          measure this machine's tile cost, write nothing
worker.py --hold [--device cuda:0] [--hold-interval 15.0]
                                            keep a GPU context awake until stdin says stop
```

## Invariants an agent must not "clean up"

Each line is a real bug that was already paid for once.

1. **One status-label writer.** `MainWindow.render_status()` (`window.py:2383`)
   is the only code that sets `lbl_status`. It shows `min(finished + 1,
   started_index)` clamped to `>= finished` and `<= total` — the oldest file
   still in flight. *Why:* the worker reads, upscales and writes on separate
   threads, so the "started" index runs ahead of the finished count; two writers
   made the number jump forward and then fall back.
2. **Ordered archive writes.** `BundleWriter` (`worker.py:1874`) packs pages with
   a **single**-thread `ThreadPoolExecutor` plus a bounded semaphore. *Why:* page
   order inside a `.cbz` is what a reader sees; more than one packer reorders it.
3. **Ordered, bounded prefetch.** `prefetch()` (`worker.py:1765`) yields
   `(unit, array_or_exception)` **in order** through a `Queue(maxsize=workers+1)`
   with a `stop` event. *Why:* the bound is the memory ceiling for decoded pages,
   and passing the exception along instead of raising in the reader thread is how
   a failed decode is reported rather than lost.
4. **Bounded write pool with tracked futures.** `WritePool` (`worker.py:1720`)
   admits work through `Semaphore(workers * 2)` and keeps outstanding futures in
   a `pending` set under a lock so it can drain. *Why:* unbounded submission
   balloons memory; untracked futures swallow write errors.
5. **Cancellation is cooperative and always has an exit.** `Control` +
   `Cancelled` + `gate()`. *Why:* a blocking call with no abort path used to make
   "Cancel" mean "wait for the whole job".
6. **`emit()` owns stdout under a lock** (see the protocol section).
7. **`janai.core` stays import-light and `janai.app` never imports torch,
   numpy, pyvips or spandrel.** *Why:* `core` is loaded by both processes and the
   interface's startup time depends on it.
8. **Tile profiles are bound to hardware identity** (`core/hardware.py`). *Why:*
   a measurement from another GPU is worse than no measurement.

## Verification

The embedded interpreter is the one the app ships with — use it for anything
import- or dependency-related:

```bash
PY=backend/python/python.exe            # Windows; Linux: backend/python/bin/python

$PY -m compileall -q src scripts        # byte-compile the app, worker and scripts
$PY -m compileall -q backend/src        # byte-compile the vendored backend
$PY scripts/smoke.py                    # dependency-free checks, ~1s
$PY scripts/plannercheck.py             # tile-planner regressions, no GPU needed
QT_QPA_PLATFORM=offscreen $PY scripts/uicheck.py   # builds the real window and measures it
$PY scripts/selftest.py                 # end-to-end; needs torch, pyvips and a model
```

Lint and format (ruff is **not** installed in `backend/python`; CI pins it, and
`uv` is available locally, so run the pinned version):

```bash
uvx ruff@0.16.7 format --check .        # CI uses: ruff format --check --diff .
uvx ruff@0.16.7 check .                 # CI uses: ruff check --output-format=github .
uvx ruff@0.16.7 format .                # to apply formatting
```

CI (`.github/workflows/ci.yml`) runs ruff on Ubuntu, then compile + `smoke.py` +
`plannercheck.py` + `uicheck.py` on **both** windows-latest and ubuntu-latest
with Python 3.13. Linux runners need `libegl1` and `libxkbcommon-x11-0` for Qt.

There is no type checker configured yet.

## Conventions

- **Structure:** `src/` layout, package `janai`, three packages with the one-way
  dependency direction above. New shared code goes in `core` only if both
  processes need it and it stays dependency-free.
- **Typing:** annotate public functions. Untyped `image` parameters in
  `worker.py` are deliberate — they are `pyvips.Image` or `numpy.ndarray` and
  annotating them would import those libraries at module scope, which invariant 7
  forbids in shared code paths; use string annotations or `TYPE_CHECKING` if you
  want them typed.
- **Errors:** the worker reports failure as an event (`file` with `error=`,
  `profile`/`hold` with `ok=False`) and keeps going where a job can continue.
  Never let an exception escape a thread silently; pass it along like
  `prefetch()` does.
- **Paths:** `pathlib` everywhere. `core/paths.py` is the single source of truth
  for locations; do not re-derive `backend/…` paths elsewhere.
- **Commits:** one concern per commit; the message says **why**. Behaviour-
  preserving moves and behaviour-changing fixes go in separate commits so a
  regression bisects to a real change. Verify before committing; never
  `git add -A` (verification runs can dirty the tree — stage reviewed paths).

## Gotchas and sharp edges

- **Line endings.** `.editorconfig` is authoritative: LF everywhere, **CRLF for
  `*.cmd`, `*.bat`, `*.ps1`** because `cmd.exe` mis-parses LF-only batch files.
  Git will say `LF will be replaced by CRLF` when touching `.gitignore` on
  Windows; that is normal here, not a problem to fix.
- **Windows paths.** Archive entry names are not filesystem-safe (`safe_name()`),
  and output paths are de-duplicated by `unique_path()`. Long paths and unicode
  entry names in `.cbz`/`.rar` are the historical source of bugs here.
- **`backend/python`, `backend/models`, `settings.json`, `presets/`, `logs/`,
  `janai.config.json`, `janai.runtime.txt` are gitignored per-machine state.**
  They are not repo content; never commit them, never assume a fresh clone has
  them.
- **`setup.sh` refuses to run if `backend/src` is missing** — the worker imports
  the vendored backend from there, so it is not optional even though
  `run_upscale.py` inside it is dead.
- **`sanic==24.6.0` is in `requirements.txt` only because vendored backend
  modules import `sanic.log`.** It is not a web server here.
- **ruff excludes `backend/`, `logs/` and `.tmp/`** (`pyproject.toml`); vendored
  code is not held to this project's style.
- **Stale artefact:** `src/janai/app/__pycache__/dnd.cpython-313.pyc` has no
  corresponding `dnd.py`. Ignore it; it is a leftover.

## Working protocol for agents

Two untracked files support agent work and must stay untracked (both are in
`.gitignore`):

- **`MEMORY.md`** — the sole memory for a long task. Chat context is unreliable,
  so it is updated after every completed step, before and after every edit batch,
  on every decision, and on every finding. It must let a fresh agent resume
  without re-deriving anything, and it is pruned rather than appended forever.
- **`.skills/`** — user-maintained guidance (note: dotted directory; briefs
  sometimes call it `./skills/`). Read it at the start of a session. It is
  guidance, not gospel: where it conflicts with the code, the code wins and the
  conflict gets recorded in `MEMORY.md`.
