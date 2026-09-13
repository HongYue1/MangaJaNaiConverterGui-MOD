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

from janai.worker import worker

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
worker.TILE = {"cls": int, "none": "none", "maximum": "maximum", "estimate": "estimate"}
worker.log = fake_log
worker.torch = SimpleNamespace(cuda=CUDA)


class Page:
    ndim = 3
    shape = (HEIGHT, WIDTH, CHANNELS)


class Model:
    scale = 4


class Planner(worker.TilePlanner):
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
    cap = planner._cap.get(id(model))
    check(f"pages 1-2 ran at {one}px and {two}px", one == 1312 and two > one)
    check(f"capped to {cap}px", bool(cap) and cap < two)
    check("and the cap reaches a genuinely cheaper cut", bool(cap) and cost(cap) < cost(two))
    check(f"page 3 runs at {three}px", three == cap)

    print("\nC: a peak against the ceiling still caps on page one")
    planner, model = new_planner()
    tight = run_page(planner, model, retries=2, peak_mib=5800)
    notes()
    cap = planner._cap.get(id(model))
    check(f"page 1 ran at {tight}px", tight == 1312)
    check(f"real capacity pressure caps, to {cap}px", bool(cap) and cost(cap) < cost(tight))

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

    if FAILS:
        print(f"\n{len(FAILS)} FAILED: {FAILS}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
