# JaNai Upscaler

A small, fast desktop GUI for the MangaJaNai / IllustrationJaNai upscaling
models. It replaces the Avalonia/C# app with plain Python + Tk: no workflows,
no chains, no profiles. Pick an input, pick a target size, pick an output
format, press Start.

Everything lives in this one folder - interpreter, models, backend - so the
whole thing can be copied to another disk or machine and still run.

This repository is a fork of
[the-database/MangaJaNaiConverterGui](https://github.com/the-database/MangaJaNaiConverterGui).
The Avalonia/C# application has been replaced by this rewrite; what is left of
upstream is the part that does the actual work - the chaiNNer-derived backend
in `backend\src` (plus `spandrel_custom` for FDAT), the ICC profiles and the
model packs. See [What changed in this fork](#what-changed-in-this-fork).

```
MangaJaNaiConverterGui-MOD\
  JaNaiUpscaler.cmd      launcher (finds the venv's pythonw.exe)
  setup.cmd / setup.ps1  one-shot installer, uv-driven
  requirements.txt       the pinned worker dependencies
  janai.config.json      written by setup: python version, torch backend
  janai.runtime.txt      written by setup: which interpreter to launch
  settings.json          your last-used settings (created on first exit)
  app\                   the GUI - standard library only, never imports torch
  app\runlog.py          the pretty run log written to logs\Run_*.log
  common\formats.py      the encoder/option table shared by GUI and worker
  common\displays.py     e-reader / tablet screen presets for Fit mode
  worker\worker.py       the upscaling process (torch, pyvips, spandrel)
  logs\                  Run_YYYYMMDD-HHMMSS.log, one per run, auto-pruned
  tools\                 uv.exe, plus cjxl.exe / djxl.exe when available
  tools\selftest.py      end-to-end check: probe, encoders, models, upscale
  tools\bench.py         throughput harness (tile sizes, baseline compare)
  tools\smoke.py         dependency-free checks, also run by CI
  backend\
    python\              the virtual environment: torch, pyvips, spandrel
    pythons\             the CPython uv downloaded for that venv
    src\                 chaiNNer-derived backend, tracked in this repository
    ImageMagick\         ICC profiles used for grayscale resizing
    models\              *.pth / *.safetensors / *.onnx
    extras\              optional side-loaded packages, normally empty
    _cache\              uv cache + downloads, safe to delete
```

`backend\python`, `backend\pythons`, `backend\models`, `backend\extras` and
`backend\_cache` are per-machine and are not tracked by git: every clone starts
clean and builds its own runtime. Nothing is ever shared with, or borrowed
from, another application's install.

## Setup

Run **`setup.cmd`** once and let it finish. It is driven by
[uv](https://docs.astral.sh/uv/) and does this:

1. finds `uv.exe` - `tools\`, `PATH`, `~\.local\bin`, WinGet links - and
   downloads it into `tools\` if you do not have it (~15 MB);
2. checks that `backend\src`, `backend\ImageMagick` and `backend\resources`
   are in place - they are tracked here, so a clone already has them and
   nothing is copied;
3. `uv venv --python 3.13` into `backend\python`, fetching a managed CPython
   into `backend\pythons` when that version is not already available (~25 MB);
4. `uv pip install --torch-backend auto -r requirements.txt`, which resolves
   the CUDA wheels that match your driver (a few GB, nearly all of it torch);
5. downloads the model packs (~1.3 GB);
6. sets JXL up: copies `cjxl.exe`/`djxl.exe` into `tools\` when they are on
   `PATH`, and installs `pillow-jxl-plugin` as a fallback;
7. writes `janai.config.json` and `janai.runtime.txt`, then runs a self-test
   that prints the devices, encoders and models it can really see.

Useful options (pass them straight to `setup.cmd`):

| Option | Effect |
| --- | --- |
| `-Python 3.12` | build the venv on a different Python version |
| `-Torch cpu` \| `cu126` \| `cu128` \| `cu129` | override the auto-detected torch build |
| `-Models manga` \| `illustration` \| `none` | fetch only some model packs |
| `-JxlTools <dir>` | take `cjxl.exe`/`djxl.exe` from a specific folder |
| `-NoJxlPlugin` | skip `pillow-jxl-plugin` |
| `-Offline` | fail rather than download anything |
| `-Force` | redo every step and overwrite what is there |

Re-running setup is safe: finished steps are skipped.

Setup never borrows a runtime or weights from another application: everything
is installed into `backend\` and belongs to this folder alone. If a previous
version left `backend\python` or `backend\models` linked elsewhere, setup
unlinks it and builds a real folder in its place. To keep the weights on
another disk, point `models_dir` at them in `janai.config.json`.

One portability caveat: a uv venv stores the absolute path of its base
interpreter, so if you move the folder somewhere the old path no longer
exists, run `setup.cmd -Force` once to rebuild the venv (the downloads are
cached in `backend\_cache`).

## Platform support

Windows 10/11 x64 is what ships and what CI checks. The wrappers are Windows
scripts (`setup.cmd`, `setup.ps1`, `JaNaiUpscaler.cmd`) and the bundled JPEG XL
helpers are `cjxl.exe` / `djxl.exe`.

The Python side itself is portable: the GUI is Tk, the worker is torch +
pyvips, paths go through `pathlib`, and the Windows-only calls (DPI awareness,
WM_DROPFILES drag & drop, `CREATE_NO_WINDOW`, opening a folder) are all guarded
by platform checks with POSIX fallbacks. On Linux only the wrappers are
missing, so running it means creating the environment by hand:

```
uv venv backend/python --python 3.13
uv pip install --python backend/python/bin/python -r requirements.txt
backend/python/bin/python app/main.py
```

plus `libvips` and `libjxl-tools` from the distribution's package manager
(pyvips needs the system library, and the bundled `.exe` encoders will not
run). That path is not tested here, so treat Linux as unsupported until it is.

**NVIDIA T4** (Turing, sm_75, 16 GB) is a good fit. FP16 runs on its tensor
cores, so the default half precision is a genuine speed-up, and 16 GB lets the
tile planner keep whole pages in a single pass - the tuning that matters on a
6 GB laptop card rarely triggers there. BF16 needs Ampere or newer, so a T4
reports `bf16: false` in the probe; the upscaler never asks for BF16, and
BF16-trained weights (the `*_bf16.safetensors` packs) load and run in FP16 or
FP32 all the same. Any CUDA device torch supports works the same way.

## Using it

Start with **`JaNaiUpscaler.cmd`**.

**Input** - choose a file (single image or a `.cbz`/`.zip`) or a folder (bulk).
Drag and drop onto the window works too. Folders can optionally recurse and
optionally process archives; the header shows what the scan found.

**Upscale** - four target modes:

- `Scale` - multiply by 1.0x ... 8.0x
- `Width` / `Height` - resize so that one side hits an exact pixel value
- `Fit` - fit inside a width x height box, keeping aspect ratio, with a
  *Device* menu of e-reader, tablet and monitor presets (Kindle, Kobo,
  reMarkable, Boox, iPad, 1080p/1440p/4K) plus a portrait/landscape switch.
  Typing your own numbers flips the menu back to *Custom*.

**Models** - with *Grayscale detection* off, one model handles everything.
With it on you pick two: one for colour pages, one for grayscale pages, which
is the entire point of telling them apart. *Suggest* fills both from what is
installed (IllustrationJaNai for colour, the MangaJaNai model closest to your
target page height for grayscale); either box can stay on *Auto* to keep the
old per-image choice. Start refuses to run while a required model is empty and
says which one is missing.

Detection itself is no longer one saturation threshold. Every page is measured
for colour strength *and* for the share of pixels that are meaningfully
coloured, on a downsampled copy, so a manga page with a couple of coloured
bubbles or a yellowed scan still counts as grayscale while a pale colour
illustration does not. Both numbers are adjustable, and each per-file log line
prints the score that decided it. Auto levels and a pre-downscale cap are still
there for the cases where the automatic choice is wrong.

**Output** - folder, filename pattern, overwrite policy, keep-folder-structure,
the **package** mode, and the format: PNG, JPEG, WebP, AVIF or **JPEG XL**.

Package decides what lands on disk, independently of what went in:

| Package | Result |
| --- | --- |
| `Files` | loose images, mirroring the input layout |
| `CBZ per folder` | one `.cbz` per source folder or archive - point it at a series whose chapters are separate folders and each chapter comes out as its own CBZ |
| `One CBZ` | the whole run collected into a single `.cbz` |

So a folder of loose JPEG chapters can come out as upscaled JPEG XL inside one
CBZ per chapter, and a `.cbz` in can stay a `.cbz` out. Every encoder exposes its
real options (quality, lossless, effort, subsampling, bit depth, interlacing,
trellis quant, ...); the less common ones are behind the *Advanced* toggle and
options that do not apply to the current settings hide themselves.

The format row only offers what this install can genuinely write: at startup
the worker encodes a 1x1 test image with each encoder and reports how it
succeeded (libvips, pillow-jxl or `cjxl.exe`). Anything that fails is disabled
with the reason shown, so JXL support is reported, never assumed.

**Performance** - device (*Auto*, CPU or a specific GPU), FP16, tile size, VRAM
budget, cache wipe, torch thread count, I/O worker count, libvips concurrency,
cuDNN benchmark, TF32 and *Keep GPU awake*. *Auto* lets the worker pick the
fastest device it finds instead of pinning one.

**FP16 is on by default** and stays on for every device that reports half
precision support; it is disabled and greyed out on devices that do not, and
the worker downgrades to FP32 on its own (with a log line) if a device turns
out to be lying. Switching devices no longer silently clears the checkbox,
which it used to.

**Tile size** defaults to *Auto (adaptive)*, which is measured rather than
guessed. The first page is sized from free VRAM minus the model weights; the
worker then reads `torch.cuda.max_memory_allocated` and calibrates the real
per-pixel cost of *that* model at *that* precision, so every later page is
sized from measurement instead of a formula. Whole-page input and output
tensors are excluded from that calibration, and the tile only shrinks under
genuine pressure - a cudaMalloc retry, or a peak within 8% of free memory -
never merely for crossing a deliberately pessimistic safety estimate.

That restraint is the whole point. Measured on a 1920x1080 to 7680x4320 4x job
on a 6 GB RTX 3060 laptop:

| Tile | Time | ms/MP |
| --- | --- | --- |
| `Auto` -> 1248px | 74.5s | 2244 |
| fixed 1024px | 79.3s | 2391 (+6.5%) |
| fixed 768px | 81.5s | 2456 (+9.4%) |

Bigger tiles win, so a planner that shrinks needlessly costs throughput on
every page that follows. Run-to-run variance on the same laptop is about 5%
(73.7s vs 77.7s for identical repeats), which is worth knowing before chasing
small regressions. *Maximum*, *No tiling* and a fixed pixel size are all still
available when you want to force the issue.

*Keep GPU awake* (on by default) holds a 256 KB tensor on the GPU in a tiny
background process while the window is open. That keeps a CUDA context resident,
so the driver keeps the card powered, a laptop dGPU does not park, and the first
run of a session starts at full speed instead of waiting on context creation and
clock ramp. It is released while a job runs and when the window closes.

While a run is going you get per-file progress, throughput, ETA, a pause button
and a cancel button.

**Dry run** (*Ctrl+Shift+Enter*) answers "what would this actually do?" without
touching the GPU or writing a byte: how many pages and chapters were found,
which model each page would use and the grayscale score behind that choice, the
predicted output path and pixel size per file, which packages would be written,
and which existing files would be skipped or overwritten.

**Log** - the log is its own resizable region of the window and grows with it
(it used to be pinned to a nine-line strip in the footer, which is the bug you
saw). Lines are time-stamped, aligned and colour-coded by level, with file
results as `ok`/`skip`/`error` rows carrying size, tile, model and duration.
Noise such as torch's `meshgrid` deprecation warning is filtered at the source.
Debug lines and wrapping toggle from the toolbar, and the log can be copied,
saved elsewhere, cleared, or opened in Explorer.

Every run is also written to `logs\Run_YYYYMMDD-HHMMSS.log`: the same pretty
text, plus a header describing the job (input, output, package, format, models,
device, precision, tile) and a footer with the totals. The most recent 40 are
kept.

### Shortcuts

| Key | Action |
| --- | --- |
| `Ctrl+O` / `Ctrl+Shift+O` | choose file / choose folder |
| `Ctrl+Enter` | start |
| `Ctrl+Shift+Enter` | dry run |
| `Esc` | cancel |
| `Ctrl+L` | show/hide the log |
| `Ctrl+D` | dark/light theme |
| `F5` | re-detect devices, encoders and models |

## Command line

The worker is usable on its own, which is handy for scripting or debugging:

```bat
backend\python\Scripts\python.exe worker\worker.py --probe
backend\python\Scripts\python.exe worker\worker.py --job job.json
backend\python\Scripts\python.exe worker\worker.py --job job.json --dry-run
backend\python\Scripts\python.exe worker\worker.py --hold --device cuda:0
backend\python\Scripts\python.exe tools\selftest.py
backend\python\Scripts\python.exe tools\bench.py --input page.png --tiles auto,1024,768
```

`--probe` prints one JSON line describing devices, encoders, models and
library versions. `--job` takes the same JSON the GUI writes (input, output,
format, upscale, perf sections) and streams JSONL progress events on stdout;
send `cancel`, `pause` or `resume` on stdin to control it. `--dry-run` plans
the same job and reports what it would write without loading a model.
`--hold` is the wake lock: it keeps a context alive on `--device` (or the best
GPU when omitted) until `stop` arrives on stdin or stdin closes.

`tools\selftest.py` exercises the whole pipeline end to end - probe, encoders,
models, a real upscale, grayscale detection, CBZ packaging and a dry run - and
prints `ALL PASS` or the first failure. `tools\bench.py` runs the real worker
once per tile setting and prints a time / ms-per-megapixel table; `--repeat`,
`--warmup`, `--copies N` (duplicate one page into a chapter) and `--baseline
<ms>` are there for comparing against a known-good number.

## Parity with the original

Everything the Avalonia app could do to an image, this can do, minus its
workflow machinery:

| Original | Here |
| --- | --- |
| chains with per-chain resolution and scale ranges | one target mode (scale / width / height / fit) per run |
| `UseFp16` | FP16, on by default, capability-checked |
| `ModelTileSize` per chain | adaptive tile planner, or a fixed size |
| `IsGrayscale` / `IsColor` chain filters | grayscale detection with a colour model and a grayscale model |
| `GrayscaleDetectionThreshold` (12) | same threshold, plus a coloured-pixel share |
| `AutoAdjustLevels` | Auto levels |
| `ResizeHeightBeforeUpscale` | pre-downscale cap |
| `ResizeHeightAfterUpscale` | target height / fit modes |
| webp/jpeg/png output | png, jpeg, webp, avif, jpeg xl, all options exposed |
| archive in -> archive out | packages: files, CBZ per folder, one CBZ |
| `DisplayDeviceWidth` / `DisplayDeviceHeight` | Fit mode device presets |
| upscale images / archives / folders, recursion, overwrite | same |
| CPU / CUDA selection | Auto, CPU, or any detected GPU |

The deliberate omission is per-chain rules: the original let one run apply
different models at different resolutions through a chain list. Here the same
outcome comes from the colour/grayscale pair plus automatic per-page model
selection, which covers the manga case the chains existed for without a
workflow editor.

## Troubleshooting

- **"Portable runtime not found"** - setup has not run yet. The GUI still
  opens with the system Python, but upscaling needs `backend\python`.
- **No models listed** - drop `.pth`/`.safetensors` files into `backend\models`
  and press `F5`, or re-run `setup.cmd -Models all`.
- **JXL disabled** - the tooltip shows why. Dropping `cjxl.exe` into `tools\`
  fixes it; `setup.cmd` also tries `pillow-jxl-plugin`, which setup installs
  into the venv.
- **GPU not listed** - the CPU build got installed; re-run
  `setup.cmd -Torch cu128 -Force`. If CUDA is present but upscaling still runs
  on CPU, check that the device selector is not pinned to CPU; *Auto* always
  prefers a GPU. The log line at the start of every run states the device that
  was actually used.
- **`.cbr`/`.rar` input** - needs `unrar.exe` or `7z.exe` on `PATH`; the probe
  reports whether one was found.
- **Out of memory** - *Auto (adaptive)* already backs off after real pressure,
  and the log says so (`tile capped at ...`). If a run still dies, set a fixed
  tile size or a VRAM budget in Performance; both go straight to the backend's
  tiling logic. Leaving other GPU applications open reduces the free memory the
  planner has to work with.
- **`torch.meshgrid: ... indexing argument` warning** - harmless, and now
  filtered: it comes from inside torch's own positional-embedding code, not
  from this app or the models. Nothing was ever wrong with the run that printed
  it.
- **spandrel and `uv`** - `backend\python` carries stock `spandrel` 0.4.1 from
  PyPI, not a patched build, so `uv pip install` cannot break it. FDAT support
  lives separately in `backend\src\spandrel_custom\`, which registers the extra
  architectures at runtime. Keep that folder: FDAT models in `backend\models`
  do not load without it.

This app never reads from an install of the original `MangaJaNaiConverterGui`:
it resolves the interpreter, the weights, the backend source and the ICC
profiles inside its own folder, and nowhere else.

## What changed in this fork

Upstream is an Avalonia/C# front end wrapped around a chaiNNer-derived Python
backend. This fork keeps the backend and replaces everything above it.

**Gone:** the C# project and solution, the Avalonia views and view models, the
Velopack packaging, the bundled updater, the workflow/chain state in
`appstate2.json`, and the release workflow that published all of it.

**New or rewritten:**

- **the interface** - `app\`, Python + Tk, standard library only. It never
  imports torch; `worker\worker.py` is a separate process that does.
- **FP16 on by default**, decided by a runtime capability check rather than a
  checkbox that could silently do nothing.
- **an adaptive tile planner** - it measures what the loaded model actually
  costs per pixel on the first page, keeps the tile that worked for the rest of
  the run, and shrinks only after repeated allocator pressure. Measured at
  parity with a hand-picked fixed tile, and well ahead of the old 1024/768
  defaults on the same hardware.
- **grayscale detection from image statistics**, and when it is on you pick two
  models: one for colour pages, one for grayscale ones.
- **a scale/model sanity check** - every weight carries its factor in its name,
  so pointing a 4x model at a 2x run (or the reverse) is flagged in the panel
  before you start, and again in the log from the loaded weights themselves.
- **long-strip passthrough** - optional, off by default: webtoon-style mega
  strips are copied straight through in the chosen output format instead of
  being upscaled. Adopted from another fork's `SkipLargeLong*` settings, minus
  the clause that also caught ordinary large spreads.
- **cuDNN autotune off by default** - it measured 21-23 s per page here against
  ~17 s with it off, even across repeats of the same image, so it is now an
  opt-in rather than the default.
- **output packages** - loose files, one CBZ per source folder (one chapter per
  archive), or a single CBZ, in any supported format regardless of what went
  in.
- **a dry run** - the whole plan, per-page targets and output sizes, with
  nothing written and torch never loaded.
- **the run log** - aligned and grouped, saved to `logs\Run_<timestamp>.log`
  automatically, and the log pane now fills the window.
- **setup** - a single `uv`-driven `setup.cmd` that builds the environment,
  fetches the model packs and self-tests the result. Every install is clean
  and self-contained.
- **checks** - `tools\smoke.py` (no dependencies, runs in CI),
  `tools\selftest.py` (end to end, needs the runtime) and `tools\bench.py`
  (throughput).

`backend\resources` and the ICC profiles are byte-identical to upstream. The
backend under `backend\src` carries three deliberate changes, all in
`auto_split.py`. It called an undefined `safe_cuda_cache_empty()`, which raised
`NameError` the moment a job was paused mid-upscale; it now calls
`safe_accelerator_cache_empty(device)` like the rest of the file. A mid-image
tile split restarted from a row computed with the *horizontal* tile size and
then wrote the restart offset in input pixels rather than output pixels, which
duplicated or dropped a band of the page; both are fixed the same way another
fork fixed them. And the "did not fit in one pass" message now names the
planned tile next to the one being retried, because reporting only the
fallback made it look like it contradicted the tile in the run summary.
`backend\python` installs stock `spandrel` from
PyPI, with the FDAT architectures registered at runtime from
`backend\src\spandrel_custom\`, so `uv pip install` cannot break model
loading.

Upstream's model packs and the GPL license are unchanged - see `LICENSE` and
[the-database/MangaJaNaiConverterGui](https://github.com/the-database/MangaJaNaiConverterGui)
for the original project.
