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
  JaNaiUpscaler.cmd      Windows launcher (finds the venv's pythonw.exe)
  janai-upscaler.sh      Linux / macOS launcher
  setup.cmd / setup.ps1  one-shot installer for Windows, uv-driven
  setup.sh               the same installer for Linux / macOS
  pyproject.toml         package metadata and the ruff configuration
  requirements.txt       the pinned worker dependencies
  janai.config.json      written by setup: python version, torch backend
  janai.runtime.txt      written by setup: which interpreter to launch
  settings.json          your last-used settings (created on first exit)
  src\janai\app\         the GUI - standard library only, never imports torch
    runlog.py            the pretty run log written to logs\Run_*.log
  src\janai\core\        shared, torch-free logic used by GUI and worker
    formats.py           the encoder / option table
    displays.py          e-reader / tablet screen presets for Fit mode
    rules.py             the upscaling rule engine and default working set
    presets.py           export / import of settings presets
    paths.py             runtime, model and backend resolution
  src\janai\worker\      the upscaling process (torch, pyvips, spandrel)
  scripts\selftest.py    end-to-end check: probe, encoders, models, upscale
  scripts\bench.py       throughput harness (tile sizes, baseline compare)
  scripts\smoke.py       dependency-free checks, also run by CI
  logs\                  Run_YYYYMMDD-HHMMSS.log, one per run, auto-pruned
  presets\               your saved presets (*.janai.json)
  backend\
    python\              the virtual environment: torch, pyvips, spandrel
    pythons\             the CPython uv downloaded for that venv
    src\                 chaiNNer-derived backend, tracked in this repository
    ImageMagick\         ICC profiles used for grayscale resizing
    models\              *.pth / *.safetensors / *.onnx
    tools\               uv, plus cjxl / djxl when available
    extras\              optional side-loaded packages, normally empty
    _cache\              uv cache + downloads, safe to delete
```

`backend\python`, `backend\pythons`, `backend\models`, `backend\extras`,
`backend\_cache`, `logs\`, `presets\` and `settings.json` are per-machine and
are not tracked by git: every clone starts clean and builds its own runtime. Nothing is ever shared with, or borrowed
from, another application's install.

## Setup

Run **`setup.cmd`** once and let it finish. It is driven by
[uv](https://docs.astral.sh/uv/) and does this:

1. finds `uv.exe` - `backend\tools\`, `PATH`, `~\.local\bin`, WinGet links -
   and downloads it into `backend\tools\` if you do not have it (~15 MB);
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

Windows 10/11 x64 and Linux x64 are both supported, and CI checks both.

| | Windows | Linux |
| --- | --- | --- |
| install | `setup.cmd` (or `setup.ps1`) | `./setup.sh` |
| launch | `JaNaiUpscaler.cmd` | `./janai-upscaler.sh` |
| JPEG XL | bundled `cjxl.exe` / `djxl.exe` | `libjxl-tools`, or `pillow-jxl-plugin` in the venv |
| libvips | `pyvips-binary` wheel | `pyvips-binary` wheel, distro `libvips` as a fallback |

`setup.sh` mirrors `setup.ps1` flag for flag - `--python 3.13`, `--torch
auto|cpu|cu126|cu128|cu129`, `--models all|manga|illustration|none`,
`--force`, `--offline`, `--no-jxl-plugin` - detects an NVIDIA driver with
`nvidia-smi` and falls back to the CPU wheels when there is none, installs uv
into `backend/tools` if you do not have it, falls back to `python3 -m venv`
when even that is impossible, and finishes by probing the runtime exactly like
the Windows installer. Tk is the one thing it cannot install for you: the
CPython that uv downloads brings its own, but if setup falls back to a system
interpreter you may need `sudo apt install python3-tk` (or your
distribution's equivalent).

The Python side is platform-neutral by design: the GUI is Tk, the worker is
torch + pyvips, paths go through `pathlib`, and the Windows-only calls (DPI
awareness, WM_DROPFILES drag & drop, `CREATE_NO_WINDOW`) are guarded with
POSIX fallbacks - "open output folder" uses `xdg-open`, and the JPEG XL tool
finder falls back to `PATH`.

**NVIDIA T4** (Turing, sm_75, 16 GB) is a good fit. FP16 runs on its tensor
cores, so the default half precision is a genuine speed-up, and 16 GB lets the
tile planner keep whole pages in a single pass - the tuning that matters on a
6 GB laptop card rarely triggers there. BF16 needs Ampere or newer, so a T4
reports `bf16: false` in the probe; the upscaler never asks for BF16, and
BF16-trained weights (the `*_bf16.safetensors` packs) load and run in FP16 or
FP32 all the same. Any CUDA device torch supports works the same way.

## Using it

Start with **`JaNaiUpscaler.cmd`** on Windows or **`./janai-upscaler.sh`** on
Linux.

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

**Model rules** - the table is the only thing that chooses a model. There are
no separate model pickers and no `auto` entry: the app ships with a filled-in
table, and if you want something else you edit the row. A rule is a condition
and the model to use when it matches:

| On | When | Page size | Model | Auto levels |
| --- | --- | --- | --- | --- |
| * | grayscale | `1920p` | `2x_MangaJaNai_1920p_V1_ESRGAN_70k.pth` | default |
| * | grayscale | `1600-1759` | `2x_MangaJaNai_1600p_V1_ESRGAN_70k.pth` | default |
| * | colour | any | `4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16` | - |
| o | grayscale | any | `2x_MangaJaNai_2048p_V1_ESRGAN_95k.pth` | on |

The first column is the on/off state, drawn as a filled or hollow dot, so a
rule that is switched off is visible in the table itself rather than only in
the editor. Click the dot to toggle a row, press Space, or use the *Toggle*
button; double-click anywhere else to edit. Rows greyed out without being off
are grayscale rules while *Grayscale detection* is off - they cannot fire, and
the hint under the table says so.

Pages are matched from the top down, except that a rule with an explicit page
size always outranks an `any` rule, so a catch-all sitting too high cannot
silently shadow a sized rule. Sizes accept an exact height (`1920`, `1920p`),
a range (`1600-1920`), an open end (`1985-`, `-1250`) or `any`.

*Auto levels* applies to grayscale rules only and has three settings:
`default` follows the **Auto levels** checkbox on the card, while `on` and
`off` override it for the pages that rule claims. (`default` was called
`inherit` in the first build, which never said what it inherited from.)

The table flags anything that cannot work - a 4x model where the target is 2x,
auto levels on a colour rule, a model that is not installed, a row with no
model - on one line underneath it, and the log prints once per run which rule
claimed each page. Start stays disabled while no row is switched on, because
then nothing would run.

**Defaults** (beside the table) rewrites it as the shipped set: the MangaJaNai
height bands (1200p, 1300p, 1400p, 1500p, 1600p, 1920p, 2048p) for grayscale
pages, the IllustrationJaNai denoise model for colour pages at 2x and 4x, and
two unsized catch-alls - all built from the models you actually have. This is
exactly what used to be hidden inside "auto", now visible and editable. It
touches only the table; **Reset all** in the header puts every setting back to
its default, keeping your input and output folders.

Settings loaded from an older build are migrated rather than discarded: the
two old pickers become catch-all rules, and any `auto` row is resolved to the
file it would have picked, with a line in the log for each one.

Grayscale detection itself is not one saturation threshold. Every page is
measured for colour strength *and* for the share of pixels that are
meaningfully coloured, on a downsampled copy, so a manga page with a couple of
coloured bubbles or a yellowed scan still counts as grayscale while a pale
colour illustration does not. Both numbers are adjustable, and each per-file
log line prints the score that decided it. Auto levels and a pre-downscale cap
are still there for the cases where the automatic choice is wrong.

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
pressure that repeats - two or more cudaMalloc retries inside one page, or two
consecutive pages peaking within 8% of free memory - never merely for crossing
a deliberately pessimistic safety estimate. A single retry keeps the proven
tile, because one retry costs far less than a smaller tile does on every page
that follows. Later pages reuse the size that already finished, because the
driver counts page 1's cached blocks as used and budgeting from that number
alone was quietly collapsing a proven tile to a fraction of its size.

That restraint is the whole point. Measured on a 6 GB RTX 3060 laptop,
1920x1080 to 7680x4320 (4x), `4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16`,
FP16, autotune off, two pages per run:

| Tile | Time | ms/MP | vs best |
| --- | --- | --- | --- |
| fixed 1952px | 36.27s | 547 | best |
| fixed 1632px | 41.83s | 630 | +15.3% |
| fixed 1376px | 41.91s | 632 | +15.5% |
| fixed 1152px | 41.64s | 628 | +14.8% |

The largest tile is fastest *even while the allocator retries once per page*,
and every smaller one costs about 15%. Across a four-page chapter `Auto`
calibrates on page 1 and then holds 1952px for the rest of the run: **67.50s,
509 ms/MP**. An earlier build that stepped down after a single retry ratcheted
to 1632px and then 1376px, taking 72.77s (548 ms/MP) for the same four pages.
Run-to-run variance on this laptop is about 5%, which is worth knowing before
chasing small regressions. *Maximum*, *No tiling* and a fixed pixel size are
all still available when you want to force the issue.

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

**Presets** - *Presets* saves everything you have set up - output, format,
upscale settings including the whole rule table, performance and log options -
into `presets\<name>.janai.json`, and loads it back here or on another
machine. Machine-specific values are deliberately left out (the output folder,
the pinned device, the log folder), so applying someone else's preset never
redirects your output or pins a GPU you do not have; the menu lists whatever
is in `presets\`. Your settings are remembered between runs in `settings.json`
without saving anything.

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
set PY=backend\python\Scripts\python.exe
%PY% src\janai\worker\worker.py --probe
%PY% src\janai\worker\worker.py --job job.json
%PY% src\janai\worker\worker.py --job job.json --dry-run
%PY% src\janai\worker\worker.py --hold --device cuda:0
%PY% scripts\selftest.py
%PY% scripts\bench.py --input page.png --tiles auto,1024,768
```

A uv venv keeps the interpreter in `Scripts\`, an embedded runtime is flat,
and on Linux it is `backend/python/bin/python`; `janai.runtime.txt` records
whichever one setup found, which is also how the launchers locate it.

`--probe` prints one JSON line describing devices, encoders, models and
library versions. `--job` takes the same JSON the GUI writes (input, output,
format, upscale, perf sections) and streams JSONL progress events on stdout;
send `cancel`, `pause` or `resume` on stdin to control it. `--dry-run` plans
the same job and reports what it would write without loading a model.
`--hold` is the wake lock: it keeps a context alive on `--device` (or the best
GPU when omitted) until `stop` arrives on stdin or stdin closes.

`scripts\selftest.py` exercises the whole pipeline end to end - probe,
encoders, models, a real upscale, grayscale detection, the rule engine, CBZ
packaging and a dry run - and prints `ALL PASS` or the first failure.
`scripts\bench.py` runs the real worker once per tile setting and prints a
time / ms-per-megapixel table; `--repeat`, `--warmup`, `--copies N`
(duplicate one page into a chapter), `--cudnn` and `--baseline <ms>` are there
for comparing against a known-good number.

## Development

The code is a plain `src` layout package: `src\janai\app` (Tk GUI),
`src\janai\core` (torch-free shared logic) and `src\janai\worker` (the
upscaling process), described by `pyproject.toml`, with the harnesses in
`scripts\`. Nothing under `app` or `core` imports torch, which is what keeps
the window and the dry run instant.

Formatting and linting are [ruff](https://docs.astral.sh/ruff/), configured in
`pyproject.toml` (100 columns, `py311`, double quotes, LF):

```
uv tool run ruff format .
uv tool run ruff check .
```

The rule set is chosen for this kind of code rather than for maximal
strictness: `PERF`, `FURB`, `C4`, `SIM`, `RET` and `PIE` for hot-path
efficiency and dead weight, `PTH` for pathlib, `B`/`A`/`RUF`/`PL` for
correctness traps, `LOG`/`G` for logging, `NPY` for numpy, `I` for import
order and `TID` to ban relative imports. Four are deliberately off, each for a
performance or architecture reason: `PLC0415` (deferred imports are how torch
stays out of the GUI process), `SIM105` (`try`/`except`/`pass` is cheaper than
`contextlib.suppress` on the per-tile path), `PLW0603` (the worker's module
state is intentional) and `BLE001`. CI runs `ruff format --check`,
`ruff check` and the dependency-free smoke checks on both Windows and Ubuntu.

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
- **JXL disabled** - the tooltip shows why. Dropping `cjxl.exe` into
  `backend\tools\` fixes it; `setup.cmd` also tries `pillow-jxl-plugin`, which setup installs
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

- **the interface** - `src\janai\app`, Python + Tk, standard library only. It
  never imports torch; `src\janai\worker\worker.py` is a separate process that
  does.
- **a proper package layout** - `src\janai\{app,core,worker}` with
  `pyproject.toml` and `scripts\` for the harnesses, plus ruff enforcing
  format and lint in CI on Windows and Ubuntu.
- **rule-based upscaling** - an ordered *condition -> model* table (colour or
  grayscale, exact/ranged/any page size, model, auto levels) is the only thing
  that chooses a model: no model pickers, no `auto` entry, nothing hidden.
  Explicit sizes always outrank `any`, unusable rows are flagged under the
  table, each row can be switched off from the table itself, and the old auto
  behaviour ships as the editable default set.
- **a redesigned interface** - flatter palette and one type ramp, a rules table
  that scrolls by scrollbar and wheel, wheel-guarded dropdowns and number
  fields so the wheel scrolls the page instead of quietly changing a value,
  useful help text on every control, and a **Reset all** button.
- **drawn checkbox and radio indicators** - painted by the app instead of
  borrowed from the theme's font, so they cannot come out as missing-glyph
  boxes; they are sized from the body font's measured line height and carry
  the hover and disabled states of the surface they sit on. Point sizes stay a
  fixed ramp: Tk already applies the display's scaling to them, so the app
  never multiplies them again.
- **scrolling that follows the pointer** - the wheel scrolls whatever is under
  the cursor rather than whatever holds focus; the rules table takes the
  gesture while it has rows left and hands it back to the page at either end.
  Its columns are redistributed to the width the table actually has, so the
  last one always ends inside the frame instead of under the scrollbar.
- **presets** - export and import everything you have set up as
  `presets\<name>.janai.json`, minus machine-specific paths and the pinned
  device; settings are remembered between runs regardless.
- **Linux support** - `setup.sh` and `janai-upscaler.sh` beside the Windows
  wrappers, same flags, same closing probe.
- **FP16 on by default**, decided by a runtime capability check rather than a
  checkbox that could silently do nothing.
- **an adaptive tile planner** - it measures what the loaded model actually
  costs per pixel on the first page, keeps the tile that worked for the rest of
  the run, and shrinks only after pressure that repeats. Measured at parity
  with the best hand-picked fixed tile (509 ms/MP across a four-page chapter)
  and about 15% ahead of any smaller one.
- **grayscale detection from image statistics**, which is what lets one table
  send grayscale pages to MangaJaNai and colour pages to IllustrationJaNai.
- **a scale/model sanity check** - every weight carries its factor in its name,
  so pointing a 4x model at a 2x run (or the reverse) is flagged in the panel
  before you start, and again in the log from the loaded weights themselves.
- **long-strip passthrough** - optional, off by default: webtoon-style mega
  strips are copied straight through in the chosen output format instead of
  being upscaled. Adopted from another fork's `SkipLargeLong*` settings, minus
  the clause that also caught ordinary large spreads.
- **cuDNN autotune off by default** - a controlled A/B at a fixed 1952px tile,
  the same two pages back to back, measured 34.35s with it off against 39.78s
  with it on (518 against 600 ms/MP, about 16%). It is opt-in now rather than
  the default.
- **output packages** - loose files, one CBZ per source folder (one chapter per
  archive), or a single CBZ, in any supported format regardless of what went
  in.
- **a dry run** - the whole plan, per-page targets and output sizes, with
  nothing written and torch never loaded.
- **the run log** - aligned and grouped, saved to `logs\Run_<timestamp>.log`
  automatically, and the log pane now fills the window.
- **setup** - a single `uv`-driven installer (`setup.cmd` on Windows,
  `setup.sh` on Linux) that builds the environment, fetches the model packs
  and self-tests the result. Every install is clean and self-contained.
- **checks** - `scripts\smoke.py` (no dependencies, runs in CI),
  `scripts\selftest.py` (end to end, needs the runtime) and
  `scripts\bench.py` (throughput).

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
