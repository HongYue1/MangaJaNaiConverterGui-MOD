"""Regression checks for the tile planner. No GPU, no torch, no settings file.

The planner has produced three separate defects that only a real GPU run
exposed: a price that ignored the cut the splitter makes, a calibration taken
from a page that had already run out of memory, and a cap that stepped by a
ratio and so changed no work at all. Each was found by hand, after shipping.

These checks drive the real TilePlanner through choose -> before -> after with
the allocator, the free memory and the peak under test control, so the next
regression of that class fails here instead.

    python scripts/plannercheck.py
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janai.worker import runtime, tiling

# Measured on a 6 GB RTX 3060 laptop card by the app's own hardware profiler,
# so these checks reason about real numbers rather than invented ones.
NAME = "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.safetensors"
PROFILE = {
    NAME: {
        "fixed": 0,
        "per_px": 1579.2561,
        "max_pixels": 3345408,
        "failed_pixels": 0,
        "model_bytes": 8061246,
        "best_tile": 512,
        "largest_tile": 1024,
    }
}
MODEL_BYTES = 8_061_246
MIB = 1024**2
FREE = 6026 * MIB
# The long strip that exposed the cap defect: 0.85 x 1312px aligns to 1088px,
# and on this page both cut the very same four passes.
WIDTH, HEIGHT, CHANNELS = 720, 4000, 3

NOTES: list[str] = []
FAILS: list[str] = []


def fake_log(message: str, level: str = "info") -> None:
    NOTES.append(f"[{level}] {message}")


class FakeCuda:
    """Only the three calls the planner makes on torch.cuda."""

    peak = 0

    def max_memory_allocated(self, device: str) -> int:
        return self.peak

    def reset_peak_memory_stats(self, device: str) -> None:
        return None

    def memory_stats(self, device: str) -> dict:
        return {}


CUDA = FakeCuda()
# Patch each name on the module that owns it. The planner reads TILE and torch
# through `runtime.` at call time, so patching them here really reaches it;
# `log` is bound into tiling's own namespace by its import, so it has to be
# patched there. Patching the wrong module leaves these checks driving the real
# card, which is worse than not running them.
runtime.TILE = {"cls": int, "none": "none", "maximum": "maximum", "estimate": "estimate"}
runtime.torch = SimpleNamespace(cuda=CUDA)
tiling.log = fake_log


class Page:
    ndim = 3
    shape = (HEIGHT, WIDTH, CHANNELS)


class Model:
    scale = 4


class Planner(tiling.TilePlanner):
    """The real planner with the card's three answers under test control."""

    fake_retries = 0

    def _free_bytes(self) -> int:
        return FREE

    def model_bytes(self, model: object) -> int:
        self._model_bytes[id(model)] = MODEL_BYTES
        return MODEL_BYTES

    def _alloc_retries(self) -> int:
        return self.fake_retries


SCRATCH = Planner("auto", 0, "cuda:0", True, profile=PROFILE)


def cost(tile: int) -> int:
    """What one pass of this tile costs, in the units the planner budgets in."""
    return SCRATCH.tile_input_pixels(WIDTH, HEIGHT, int(tile), CHANNELS)


def passes(tile: int) -> int:
    return max(1, math.ceil(WIDTH / tile)) * max(1, math.ceil(HEIGHT / tile))


def new_planner() -> tuple[Planner, Model]:
    model = Model()
    planner = Planner("auto", 0, "cuda:0", True, profile=PROFILE)
    planner.note_model(model, NAME)
    return planner, model


def fixed_planner(size: int) -> tuple[Planner, Model]:
    """A planner with the tile pinned by hand, as the Performance tab does."""
    model = Model()
    planner = Planner("fixed", size, "cuda:0", True, profile=PROFILE)
    planner.note_model(model, NAME)
    return planner, model


def run_page(planner: Planner, model: Model, retries: int, peak_mib: int) -> int:
    """One page through choose -> before -> (upscale) -> after."""
    tile = int(planner.choose(model, Page()))
    planner.before()
    planner.fake_retries += retries
    CUDA.peak = int(peak_mib * MIB)
    planner.after()
    return tile


def notes() -> None:
    for note in NOTES:
        print("    " + note)
    NOTES.clear()


def check(label: str, ok: bool) -> None:
    print(f"  {'ok  ' if ok else 'FAIL'}  {label}")
    if not ok:
        FAILS.append(label)


def main() -> int:
    print(f"page {WIDTH}x{HEIGHT} colour through a x4 model, {FREE // MIB} MiB free")
    for tile in (1344, 1312, 1088, 992, 768):
        print(f"  {tile}px -> {passes(tile)} passes, {cost(tile):,} px-channels each")

    print("\nthe cut is a step function, so a ratio step can be a no-op")
    ratio = max(SCRATCH.MIN_TILE, SCRATCH._align(1312 * 0.85))
    check(f"0.85 x 1312px aligns to {ratio}px", ratio == 1088)
    check("which cuts exactly the same work", cost(ratio) == cost(1312))

    print("\nA: a cold allocator on the first page of a model")
    planner, model = new_planner()
    first = run_page(planner, model, retries=2, peak_mib=3513)
    second = int(planner.choose(model, Page()))
    notes()
    check(f"page 1 ran at {first}px", first == 1312)
    check("warm-up retries leave no lasting cap", id(model) not in planner._cap)
    check(f"page 2 is sized from measurement: {second}px", second >= first)
    check(f"and reaches the {passes(second)}-pass cut", passes(second) < passes(first))

    print("\nB: a settled allocator that repeats the retries")
    planner, model = new_planner()
    one = run_page(planner, model, retries=0, peak_mib=3513)
    two = run_page(planner, model, retries=2, peak_mib=3513)
    three = int(planner.choose(model, Page()))
    notes()
    # `is not None`, not truthiness: the cap is an `int | None`, and bool()
    # neither narrows it for a checker nor tells "no cap" from a 0px one.
    cap = planner._cap.get(id(model))
    check(f"pages 1-2 ran at {one}px and {two}px", one == 1312 and two > one)
    check(f"capped to {cap}px", cap is not None and cap < two)
    check("and the cap reaches a genuinely cheaper cut", cap is not None and cost(cap) < cost(two))
    check(f"page 3 runs at {three}px", three == cap)

    print("\nC: a peak against the ceiling still caps on page one")
    planner, model = new_planner()
    tight = run_page(planner, model, retries=2, peak_mib=5800)
    notes()
    cap = planner._cap.get(id(model))
    check(f"page 1 ran at {tight}px", tight == 1312)
    check(f"real capacity pressure caps, to {cap}px", cap is not None and cost(cap) < cost(tight))

    print("\nD: a clean page clears the slate")
    planner, model = new_planner()
    run_page(planner, model, retries=1, peak_mib=3513)
    run_page(planner, model, retries=0, peak_mib=3513)
    NOTES.clear()
    check("no pressure held against the model", id(model) not in planner._pressure)
    check("and no cap", id(model) not in planner._cap)

    print("\nE: every step down reaches a cheaper cut")
    tile = 1312
    chain = []
    for _ in range(4):
        step = SCRATCH._cheaper(WIDTH, HEIGHT, CHANNELS, tile)
        chain.append(step)
        check(f"{tile}px -> {step}px ({cost(tile):,} -> {cost(step):,})", cost(step) < cost(tile))
        tile = step
    print(f"  chain: 1312 -> {' -> '.join(str(size) for size in chain)}")

    print("\nF: a re-tiled page is not used to calibrate")
    planner, model = new_planner()
    planned = int(planner.choose(model, Page()))
    planner.before()
    planner.retiled(planned - 320)
    CUDA.peak = int(5800 * MIB)
    planner.after()
    notes()
    check("no cost learned from a page that missed", id(model) not in planner._per_px)
    check(f"but {planned}px is remembered as a ceiling", planner._ceiling(id(model)) < planned)

    print("\nG: a fixed tile the page cannot pay for is not paid for twice")
    # A fixed tile is a request: auto_split lowers it one grid step when the
    # page does not fit, and the planner used to hear nothing about it, so
    # every page re-attempted the size that had already failed -- 1536px asked
    # for, "tile 1056" reported, 24.0s against 17.3s.
    planner, model = fixed_planner(1536)
    asked = int(planner.choose(model, Page()))
    planner.before()
    planner.retiled(1056)
    CUDA.peak = int(5800 * MIB)
    planner.after()
    held = int(planner.choose(model, Page()))
    third = int(planner.choose(model, Page()))
    said = [note for note in NOTES if "fixed tile" in note]
    notes()
    check(f"page 1 asks for the {asked}px that was set", asked == 1536)
    check(f"page 2 holds the {held}px that fitted", held == 1056)
    check(f"page 3 too, not back to 1536px: {third}px", third == 1056)
    check(f"and the fallback is reported once, not per page: {len(said)} note(s)", len(said) == 1)

    print("\nH: a fixed tile that fits is never lowered")
    planner, model = fixed_planner(1024)
    sizes = [run_page(planner, model, retries=2, peak_mib=5800) for _ in range(3)]
    warned = [note for note in NOTES if "fixed tile" in note]
    proven = planner._good.get(id(model), 0)
    NOTES.clear()
    check(f"three pages at the size set: {sizes}", sizes == [1024, 1024, 1024])
    # Retries and a peak against the ceiling cap an *auto* tile. A hand-set one
    # is only ever lowered by a real miss, so nothing below it is on record.
    check(f"pressure records nothing below it: {proven}px proven", proven >= 1024)
    check(f"and no fallback is announced: {len(warned)} note(s)", not warned)

    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
