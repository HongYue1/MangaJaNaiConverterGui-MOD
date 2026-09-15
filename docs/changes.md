# Changes since the fork point

This document consolidates the whole divergence between this repository and the
upstream project it was forked from. It is written for someone who knows the
upstream app and wants to know exactly what is different here, and why.

## 1. Fork point

| | |
| --- | --- |
| Upstream | [the-database/MangaJaNaiConverterGui](https://github.com/the-database/MangaJaNaiConverterGui) |
| Fork base commit | `e63e784` — *"fix defaulting to cpu upscaling, fix gpu list"* (2025-11-26) |
| Commits since | 112 (`git log e63e784..HEAD`) |
| Layout then | everything under a `MangaJaNaiConverterGui/` subdirectory, next to `MangaJaNaiConverterGui.sln` and `pack.bat` |
| Layout now | repository root is the app: `src/janai/`, `backend/`, `scripts/`, `setup.*`, launchers |

At the fork base the shipped product was an Avalonia/C# desktop application that
drove a Python worker (`backend/src/run_upscale.py`, 1628 lines) over a
line-oriented stdout protocol (`PROGRESS=…`, `TOTALZIP=…`). That application no
longer exists here. What remains of upstream is the vendored chaiNNer-derived
inference backend under `backend/src/`, plus the models, ICC profiles and
bundled tools it needs.

## 2. At a glance

| | Fork base | Now |
| --- | --- | --- |
| UI | Avalonia / C# / XAML (`App.axaml`, `Views/MainWindow.axaml`) | Python + PySide6 (`src/janai/app/`) |
| Worker entry | `backend/src/run_upscale.py` (1628 lines, mixed concerns) | `src/janai/worker/` (~23 modules, `worker.py` is 168 lines) |
| GUI ↔ worker wire | `PROGRESS=` / `TOTALZIP=` prefixed lines | JSON Lines on stdout, one object per event |
| UI state | `appstate2.json` | `settings.json` + named presets (`presets/<name>.janai.json`) |
| Headless use | none | `janai` CLI (`run` / `plan` / `probe` / `where`) |
| Resume after cancel | none | `.janai-resume.json` manifest, archives included |
| Platforms | Windows | Windows + Linux |
| Lint / types / CI | none | ruff 0.16.7, mypy 2.3.1, GitHub Actions on Windows + Ubuntu |
| Python package files | — | 54 `.py` files in `src/janai/`, 19 in `scripts/` |

## 3. Removed

| Removed | Commit | Why |
| --- | --- | --- |
| The Avalonia/C# application, its solution, XAML views, view models, `appstate2.json` and `pack.bat` (134 files, 26,138 lines) | `ec092f7` | Replaced wholesale by the Python UI; keeping a second UI would mean two settings formats and two wire protocols. |
| The upstream release/deploy workflow | `c2a42fc` | It built the C# app and published installers that no longer exist. |
| The "reuse an existing runtime" mechanism | `9edcbef` | The runtime is now installed inside the app directory, so there is exactly one interpreter to reason about. |
| `backend/src/run_upscale.py` (the chaiNNer-era upscaler CLI) | `7e726c2` | Proven unimported: nothing in the tree could reach it. It was a stale mirror of logic that had moved into `src/janai/worker/`, so it was a trap for anyone reading the repository. |

The deleted `run_upscale.py` is also the only place `PROGRESS=`, `TOTALZIP=`,
`PipelineQueue`, `put_sentinel()`, `consumer_exited()` and
`PipelineConsumerGone` ever appeared in this tree. Documentation that described
those as the live contract was corrected (see §9c).

## 4. The application that replaced it

`a0994ae` landed the Python rewrite (127 files); `3b8fcc6` restructured it into
a `src/` layout package. The package is split three ways with a one-way
dependency direction — `app → core`, `worker → core`, and nothing imports `app`:

```
src/janai/
  __main__.py        python -m janai
  cli.py             headless driver: run / plan / probe / where
  app/               PySide6 UI
    main, window, surface, widgets, fields, theme,
    input_panel, output_panel, perf_panel, rules_panel, rules_table,
    run_panel, log_panel, runlog, runner, state
  core/              shared, GUI-free, dependency-light
    paths, rules, formats, displays, hardware, presets, fspath, filetypes
  worker/            the subprocess that does the work
    worker, job, orchestrate, archives, page, reporting, selection,
    pipeline, planning, resume, control, events, imageio, imagetypes,
    transforms, models, devices, tiling, profiling, capabilities,
    environment, runtime, probe, hold
```

Only `app/runner.py` refers to the worker at all, and only to resolve its file
path before spawning it — the GUI never imports worker code into its own
process, so a heavy import (torch, pyvips) can never block the event loop.

## 5. The GUI ↔ worker contract

The worker is a separate process. Everything it reports is one JSON object per
line on stdout; `app/runner.py` accepts a line only if it starts with `{`, so
stray prints from any library cannot corrupt the stream. `events.emit()` holds a
lock around the write so concurrent stages cannot interleave half-lines.

* Event discriminator is the key `type`, and payloads are flat.
* Worker-emitted kinds: `log`, `start`, `progress`, `file`, `bundle`, `done`,
  `probe`, `profile`, `profile_progress`, `hold`.
* The GUI synthesises four more for its own state machine: `probe_error`,
  `done(ok=False)`, `exit`, `scan`.
* Control flows the other way as bare words on stdin: `cancel`, `abort`, `stop`,
  `pause`, `resume`, `quit`, `exit`.

The no-hang contract is `prefetch()` + `WritePool` + `BundleWriter` + `Control`,
not the sentinel/queue scheme described in older notes. Each of those has a
recorded invariant in `AGENTS.md`.

## 6. Features added

**Model selection.** A rules table is the only thing that chooses a model
(`ef87be5`, `c638f90`): each row matches on page traits and names the model to
use, so the same folder of mixed grayscale/colour pages is handled without
manual switching. Grayscale detection is measured per page, and a page whose
colour verdict could not be measured now says so instead of silently defaulting
(`a4955df`).

**Upscale modes.** Scale, Width, Height and Fit, with device presets. Long
strips bypass the scaler rather than being distorted.

**Output.** Loose files, one CBZ per folder, or a single CBZ for the whole run.
Encoders are discovered at startup by a real capability probe (`probe` event)
rather than assumed, so the format list offered in the UI reflects what the
bundled libvips/`cjxl` build can actually write on that machine.

**Adaptive tiling.** The machine is profiled once, then tiles are sized from
that measurement (`b1a2a50`) instead of being ratcheted down per image. A proven
tile is held across a chapter (`89f460a`), and the planner no longer penalises a
cold allocator (`807b0e1`).

**Keep GPU awake** (`hold`), **dry run** (reports exactly what a run would do,
including exclusions), **named presets**, and a **run log** (`logs/Run_*.log`,
40 kept).

**Linux support** (`6b72a0e`): `setup.sh` and `./janai-upscaler.sh` alongside
`setup.cmd` / `JaNaiUpscaler.cmd`.

**Headless driver** (`0f01da1`): `janai run|plan|probe|where`, with
`--print-job`, `--json`, `--cancel-after`, `--resume/--no-resume`, `-o`, and exit
codes 0 ok / 1 job failed / 2 bad usage / 130 cancelled. This exists so that
neither a user nor an automated test has to hand-write a `job.json` or scrape the
GUI to exercise the pipeline.

**Resume** (`7807581`, `af4dd0e`): a `.janai-resume.json` manifest is written
beside the output. It keys work by path relative to the input root and is
fingerprinted over the format/upscale/output settings — not over `perf`, and not
over the output directory, so moving the output or changing thread counts does
not invalidate completed work, while changing what the output *is* does. Three
sites record completion: archives, loose pages, and closed bundles. Records are
flushed every 64 entries or 10 seconds, because a naive flush-per-page would
have rewritten ~1.4 GiB of manifest over a 4,000-page run. A cancelled archive
keeps its `.part` file and is resumed by prefix, so cancelling inside file 11 of
100 resumes inside file 11.

## 7. Phase 1: the restructure (behaviour-preserving)

The monolith the original brief pointed at was dead code. The live monolith was
`src/janai/worker/worker.py` at 3,264 lines, and `src/janai/app/window.py` at
2,764.

| Target | Before | After | Commits |
| --- | --- | --- | --- |
| `worker/worker.py` | 3,264 lines | 168 lines | `034921e`, `3d2d532`, `7e40554`, `820a48f`, `c271b87`, `b5d9e78`, `e7902ae`, `22fcf93`, `55e10eb`, `c93da2d`, `d3d9f42`, `7f6a9d4`, `bb3d70f`, `cb121fb`, `1c4bfc0`, `a3c6cb6`, `9303c67` |
| `run_job()` | 726-line function, 16 nested closures | 591 lines across `PagePolicy`, `PageWorker`, `JobReporter`, `archives`, `orchestrate` | `f851104`, `1bfd51d`, `f14bbb1`, `59d5f13`, `d427caf` |
| `app/window.py` | 2,764 lines | 811 lines + 7 panel modules | `ad839c5`, `298b259`, `91abd3a`, `6317501`, `bf6079b`, `264bb3e`, `736df31` |

Every commit in this phase was a move: no renamed user-visible strings, no logic
changes, so a later regression bisects to an actual behaviour change rather than
to a move.

## 8. Phase 2/3: bugs fixed

### 8a. Concurrency, cancellation and shutdown

| Fix | Commit |
| --- | --- |
| A dead pipeline worker used to hang the job with no error; a stalled stage is now detected and reported | `ca5936e` |
| The prefetch pump is released when a consumer leaves early, instead of blocking forever on a queue nobody drains | `768a4f7` |
| A cancel leaves `WritePool.submit` immediately instead of waiting out an in-flight encode | `23ba72a` |
| A pool slot is never lost, and a cancel can leave the pack queue | `e934932` |
| The worker is stopped synchronously on window close, so it cannot be orphaned | `2876c0f` |
| The job's shared page tallies are guarded by a lock | `bfadb9a` |
| Teardown finishes even when a cleanup hook raises, and releases the *right* device's cache | `198f080` |
| A dying control reader says so instead of silently ending cancellation support | `6540548` |
| Page encoding overlaps the GPU on the archive path | `1bbf650` |

### 8b. Correctness

| Fix | Commit |
| --- | --- |
| A job with no output directory wrote upscaled pages into the worker's current directory while reporting success — `Path("")` is `WindowsPath('.')`, which is truthy, so the existing `if not out_dir` guard never fired. Now refused. | `1d41b15` |
| Two different sources could collapse onto one output file | `98e2183` |
| Re-encoded pages could collide inside a CBZ | `51fa031` |
| An archive that kept no page is never published | `944733c` |
| Pages lost inside an archive are counted and reported, with the reason | `3f9a05f`, `cebb1de` |
| Zip entry names stored without the UTF-8 flag are recovered instead of mangled | `83078c9` |
| macOS sidecar files (`._*`) are no longer counted as pages | `97df7eb` |
| The `MAX_PATH` ceiling is lifted at the filesystem boundary (long Windows paths) | `8e0adad` |
| A user cancel is reported as a cancel, not as a lost page | `5467446` |
| The source archive is closed deterministically (`open_archive` is a context manager) instead of relying on refcounts | `acf995b` |
| One definition of the page/archive extension sets, replacing three copies | `ce44ab9` |
| The Open dialog's file filter is derived from those sets. It had drifted to 13 of 15 image suffixes, so the picker hid `.ppm`/`.pgm` pages that the pre-run scan counted and the worker converts. | `e77a2f1` |
| Status label: one monotonic file counter, one writer | `50411f2` |
| The dry run reads the resume manifest, so previewing a half-finished job reports finished work as skipped instead of listing it as work still to do. `fingerprint()` excludes `dry_run`, so the preview and the real run share one manifest. | `0fdec01` |
| A bundle consults the manifest, not only whether the destination `.cbz` exists. With overwrite on the existence test never fired, so a run announced `resuming: 1 of 1 already done` and then re-converted and re-packed the page anyway. | `0fdec01` |
| A single chosen file names its own CBZ in both packaging modes, instead of being named after whichever folder it sat in (`planning.bundle_stem()`; a lone file used to come out as `Downloads.cbz`) | `19e95b8` |
| A fixed tile size holds the size that fitted after a step-down, instead of re-asking for the size that failed on every page — the vendored `auto_split` absorbs the OOM inside the page, so the cost was one failed pass per page and a log that looked like the setting was ignored | `fe1e239` |

### 8c. The vendored inference backend

`backend/src/` is chaiNNer-derived code inherited from upstream. It was reviewed
in this pass, and two defects present at the fork base were fixed. Both are
stated here only because the upstream bytes were read directly
(`git show e63e784:MangaJaNaiConverterGui/backend/src/…`), and both are now
asserted by a tracked gate in `scripts/smoke.py`.

1. **Tile budget was computed from total VRAM, not free VRAM** (`772111a`).
   In `packages/chaiNNer_pytorch/pytorch/processing/upscale_image.py`,
   `torch.cuda.mem_get_info()` returns `(free, total)`; the code bound `_free`
   and discarded it, then sized tiles from `total * 0.75 * 0.8`. On a card
   already holding another process's allocation, that plans tiles the device
   cannot fit. Now the budget is `min(total, free)`. The same shape existed on
   the XPU branch and was fixed with it.

2. **The OOM recovery path copied the failing tile to host RAM, inside an
   exception handler that swallowed everything** (`5e938ce`). In
   `nodes/impl/pytorch/auto_split.py` the recovery branch ran
   `input_tensor.detach().cpu()` — discarding the result — wrapped in
   `try/except Exception: pass`. That asks for host memory for an entire tile at
   the moment an allocation has just failed, and hides any failure while
   cleaning up. Recovery is now `del` + `gc.collect()` + cache-empty, which is
   what actually frees the accelerator.

   The same commit fixed the adjacent pause branch, which called
   `safe_cuda_cache_empty()` — a name that module neither defines nor imports
   (it imports `safe_accelerator_cache_empty`). It is now the imported,
   device-aware call.

Otherwise the vendored backend is deliberately unmodified: `backend/resources`
and the ICC profiles are byte-identical to upstream, spandrel is stock 0.4.1 with
FDAT support kept out-of-tree in `backend/src/spandrel_custom/`.

### 8d. Documentation that had gone stale

These were caught by their own gates and are recorded because a document that
lies is a defect: the UI described as Tk in three present-tense places after the
Qt port (`9177ec5`), `AGENTS.md`'s module count drifting because the gate quoted
the number instead of measuring it — now it measures it (`2526d50`), orphan
bytecode purged and a claim naming one file of two corrected (`706ac43`),
performance numbers replaced with measured ones (`5a930a7`).

## 9. Performance work

Measured, not assumed. Where measurement said "no change needed", nothing was
changed.

| Measurement | Result |
| --- | --- |
| Tile size sweep (real GPU, same pages) | 1952 px → 36.27 s / 547 ms·MP (best); 1632 → 41.83 / 630; 1376 → 41.91 / 632; 1152 → 41.64 / 628. Auto, four pages: 67.50 s / 509 ms·MP |
| cuDNN benchmark A/B | 34.35 s vs 39.78 s (518 vs 600 ms·MP) — enabled by default |
| Pipeline occupancy (24 pages, 872 thread sweeps over 27.5 s) | GPU thread 98.4 % working / 1.6 % blocked; read pool 99.2 % idle; pack pool 13.1 % busy; prefetch pump blocked by design. GPU is the bottleneck by ~6× over decode and ~7× over encode. |
| Resume manifest flush cost | naive per-page write ≈ 1,412 MiB rewritten over ~4,000 pages → deferred flush (64 records / 10 s) |

The occupancy measurement is the answer to "is any stage blocking another": no.
The only serialisation found is per-archive `drain()`, which puts encoding behind
the single pack thread at archive close — measured at 1.0 % of wall time, so it
was left alone rather than traded for more complexity.

Structural wins taken: bounded queue depths (`workers + 1`), a single typed
decode/encode boundary, single-owner extension sets, archive handles released at
scope exit, encode overlapped with the GPU on the archive path. Micro-
optimisations were rejected on measurement — the run is dominated by model
inference, so shaving Python around it buys nothing observable.

## 10. Typing, lint and CI

* **ruff 0.16.7**, config committed in `pyproject.toml`: 100 columns, `py311`
  target, double quotes, LF. `PLC0415`, `SIM105`, `PLW0603` and `BLE001` are
  disabled with reasons; `N802` is off for three UI modules that must match Qt's
  camelCase overrides.
* **mypy 2.3.1**, config committed: clean across 73 files. Tier 1 (`core/*`,
  `pipeline`, `planning`, `control`, `events`, `runlog`) additionally enforces
  `--disallow-any-generics`, `--disallow-untyped-defs`,
  `--disallow-incomplete-defs`, `--warn-return-any` and is at zero.
* Both run via `uvx`, so neither tool is ever installed into the shipped
  `backend/python` environment — zero packaging impact.
* **CI** (`.github/workflows/ci.yml`): ruff and the dependency-free gates on
  Windows and Ubuntu (`0611aff`, `11ca36b`), plus a `types` job running
  `mypy --platform win32` (`f4b84f7`). The platform flag is load-bearing:
  targeting Linux, mypy reports `os.startfile` as missing at `app/runner.py:28`
  on a line `sys.platform == "win32"` already guards, so an unpinned platform
  would fail CI on correct code.
* Typing was fixed rather than suppressed: `89ef378`, `34ed5be`, `60e772a`,
  `2723bd0`, `82bf2a2`, `4143884`, `2d28e67`, `4140997`, `01cc7b4`, `6ecbb3b`,
  `17ad60f`. `82bf2a2` and `4143884` each exposed a real error that a blanket
  override had been hiding.

## 11. Dependencies

No new runtime dependency was added. The work here was making the existing set
correct and current, with each verdict measured on the shipped interpreter rather
than read off a changelog.

| Candidate | Measured | Verdict |
| --- | --- | --- |
| `pillow>=11.0.0` | the only unpinned line: a fresh `setup.cmd` resolved 12.3.0 while this machine ran 11.3.0 | **pinned to 11.3.0** (`f401d5c`) |
| `pyvips` binding 3.0.0 → 3.2.0 | every format re-encoded byte-identical against the shipped libvips | accepted (`77c7049`) |
| `psutil` 7.2.2, `pynvml` 13.0.1, `packaging` 26.3, `rarfile` 4.5, `sanic` 25.12.1 | API surface identical at every call site actually used, including a live NVML query | accepted (`77c7049`) |
| `mypy` 1.18.2 → 2.3.1 | clean at 73 files, identical 163-finding strict residue | accepted (`4d40f8a`) |
| **`pyvips-binary` 8.18.6** (i.e. libvips itself) | PNG sha changes at identical length; AVIF changes size (gray 736 → 762 B, rgb 1218 → 1091 B); JPEG/WebP identical; still no `jxlsave` | **rejected, held at 8.16.1** — it would silently change output bytes for every user on identical models and settings, and gains nothing, since `.jxl` goes out through the bundled `cjxl` either way |
| torch 2.14.0, torchvision 0.29.0, numpy 2.5.3, opencv 5.0.0.93, spandrel 0.4.2 | pixel path; not validatable without a full real-GPU output comparison | deferred, reasons recorded |

`f401d5c` also added a `dependency pins` gate that forbids an unpinned line and
forbids `requirements.txt` and `backend/src/pyproject.toml` from disagreeing —
those two lists had no guard against drift before. Note that the pins are
*declared*, not installed: an existing install keeps its current versions until
the next `setup` run.

## 12. Invariants now written down

`AGENTS.md` records 13 invariants, each with a one-line reason so nobody
"cleans it up". The load-bearing ones:

* stdout is JSON Lines, discriminated on `type`, emitted under one lock;
  `runner.py` gates on a leading `{`.
* Encoded pages are written in sequence order via a sequence number and a
  pending dict — output order is user-visible inside a CBZ.
* The status label has exactly one writer (`run_panel.py`), declared as such in
  the window `Protocol`; a second writer reintroduces flicker and a
  non-monotonic counter.
* A stage that leaves must release whoever feeds it, or the job hangs with no
  error.
* The resume fingerprint covers what the output *is*, never how fast it was
  produced.
* Never bump libvips without re-measuring encoder output bytes.

## 13. Verification

Everything below runs against the embedded interpreter
(`backend/python/python.exe`, Python 3.13.9; `backend/python/bin/python` on
Linux). Exact commands live in `AGENTS.md`; the short version:

```
$PY -m compileall -q src scripts backend/src
$PY scripts/smoke.py                      # dependency-free claims, incl. dependency pins
$PY scripts/plannercheck.py
QT_QPA_PLATFORM=offscreen $PY scripts/uicheck.py
$PY scripts/selftest.py                   # real GPU, end to end
$PY scripts/bench.py
$PY scripts/<name>_check.py               # 15 focused gates: archives, names,
                                          # output dirs, counters, write pool,
                                          # bundles, solo bundles, resume
                                          # plans, long paths, stdin, ...
uvx ruff@0.16.7 check . && uvx ruff@0.16.7 format --check .
uvx mypy@2.3.1
```

Every commit in this history landed behind that chain, ordered fast-legs-first
with an early exit, with the commit itself chained behind a zero failure count.
New guards were proven to fail against the unfixed tree before the fix landed, so
the same run is both the proof the guard has teeth and the measurement of the
harm.

## 14. Deliberately unchanged

| Item | Reason |
| --- | --- |
| Peak-VRAM reporting is CUDA-only behind a gate that admits XPU | no XPU hardware here to verify a fix against |
| Two narrow, error-code-scoped type ignores at the JSON settings boundary (`core/rules.py`) | the JSON wire genuinely has no static type; a blanket override would hide real errors |
| 163 strict-mode findings in 36 files, all outside Tier 1 | almost all are generics whose real parameters are library types mypy cannot resolve by configuration (torch/numpy/pyvips/PySide6 exist only inside the embedded env). Annotating them writes `list[Any]`: noise, not proof. |
| Per-archive `drain()` serialising encode behind the pack thread | measured at 1.0 % of wall time |
| `unit` / `task` dicts inside the planner | decided, not skipped: a frozen dataclass is the wrong tool (index injection and group accumulation are the design), and a `TypedDict` has nowhere to live without coupling the planner to the concurrency module |
| torch / torchvision / numpy / opencv / spandrel majors | pixel-path changes need a full real-GPU output comparison to accept honestly |
| The vendored chaiNNer inference code beyond the two fixes above | it is upstream's code; the smaller the delta, the cheaper future upstream merges |

## 15. Commit index

`e63e784..HEAD`, oldest first.

**Replacing the application** — `ec092f7` remove Avalonia/C#; `a0994ae` add the
Python rewrite; `c2a42fc` repo config and CI point at it; `9edcbef` install the
runtime inside the app; `5e9ba6c` model defaults, scale-mismatch warning,
tile-split restart; `89f460a` hold the proven tile across a chapter; `5a930a7`
measured performance numbers; `3b8fcc6` src layout, rules, presets; `6b72a0e`
Linux; `0611aff` CI; `1c26471` docs.

**Interface** — `6ac9a18` minimal surface, scrollable table, wheel-safe
controls; `ef87be5` rules table is the only model chooser; `c638f90` Upscale card
rebuilt around it; `9c71274` docs/harnesses; `e484579` pointer-scoped scrolling,
column fit; `74dbd2f` display scaling + indicators; `2e8b542` geometry harness;
`3045382` revert the scaling; `ef1788b` assert the font ramp; `6efd059` Tk → Qt;
`affe1a9` second review pass, adaptive tiling; `24f4930` collapse flash,
grid-aware re-tile, exclusion-aware dry run; `b1a2a50` profile then size tiles;
`807b0e1` stop capping for a cold allocator; `ca5936e` dead pipeline worker;
`50411f2` one monotonic counter; `9de2a6a` ignore agent working files;
`f46b78c` document the app as it is; `7e726c2` delete the unreachable CLI.

**Worker extraction** — `034921e` events; `3d2d532` control; `7e40554` pipeline;
`820a48f` planning; `c271b87` runtime; `b5d9e78` devices; `e7902ae` environment;
`22fcf93` capabilities; `55e10eb` tiling; `c93da2d` imageio; `d3d9f42` models;
`7f6a9d4` transforms; `bb3d70f` + `cb121fb` job; `1c4bfc0` hold; `a3c6cb6`
probe; `9303c67` profiling; later `f851104` PagePolicy; `1bfd51d` PageWorker;
`f14bbb1` JobReporter; `59d5f13` archives; `d427caf` orchestrate.

**Window split** — `ad839c5` option ladders; `298b259` run log; `91abd3a` input;
`6317501` output; `bf6079b` performance; `264bb3e` rules; `736df31` run
lifecycle.

**Bug fixes** — `6540548`, `198f080`, `3f9a05f`, `51fa031`, `a4955df`,
`2876c0f`, `768a4f7`, `bfadb9a`, `23ba72a`, `e934932`, `cebb1de`, `98e2183`,
`97df7eb`, `83078c9`, `944733c`, `8e0adad`, `acf995b`, `1d41b15`, `5467446`,
`772111a`, `5e938ce`.

**Gates and evidence** — `1c3dd33` promote throwaway harnesses into `scripts/`;
`c8f32aa` make `stdin_check` fail rather than print; `11ca36b` run the
concurrency gates in CI; `5100357` gate the archive done payload; `7bb76b9`
assert page order before that path gained a pack pool.

**Typing and lint** — `d037963`, `89ef378`, `34ed5be`, `60e772a`, `2723bd0`,
`82bf2a2`, `4143884`, `2d28e67`, `8ade94e`, `4140997`, `01cc7b4`, `6ecbb3b`,
`17ad60f`.

**Documentation truth** — `a8ef6c9` rewrite `AGENTS.md` against the real tree;
`9177ec5` Qt, not Tk; `2526d50` measure the module count instead of quoting it;
`706ac43` purge orphan bytecode.

**Deduplication** — `ce44ab9` one definition of the extension sets; `e77a2f1`
derive the Open dialog filter from them.

**Follow-up scope** — `0f01da1` headless driver; `7807581` + `af4dd0e` resume;
`f401d5c` pin pillow and assert the two dependency lists agree; `4d40f8a` mypy
2.3.1; `77c7049` take the byte-identical dependency bumps, hold the one that is
not; `f4b84f7` type-check in CI on the platform that ships; `50f2cf9` put the
output subfolder beside a folder input; `b125dae` report the packages this
install really has; `0fdec01` read the resume manifest in the dry run and for
whole archives; `19e95b8` name a single file's CBZ after the file; `fe1e239`
hold the tile size that fitted.
