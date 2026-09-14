"""Per-page pixel work: detect, transform, upscale, encode.

Lifted out of `run_job` because this is the only cluster that touches the GPU
context, the tile planner and the model cache while needing none of the job's
bookkeeping -- no counters, no progress events, no output paths. Keeping the two
apart means the reporting concern cannot quietly grow into the pixel path, and
"one page in, one page out" stays reviewable on its own.
"""

from dataclasses import dataclass
from typing import Any

from janai.worker import imageio, runtime, tiling
from janai.worker.control import CTRL
from janai.worker.events import log
from janai.worker.models import ModelCache, upscale_array
from janai.worker.selection import PagePolicy
from janai.worker.transforms import (
    auto_levels,
    final_resize,
    gray_stats,
    is_long_strip,
    standard_resize,
    to_grayscale,
)

# Arrays are numpy at runtime, but numpy is loaded lazily through `runtime` so a
# dry run never imports the imaging stack; `Any` is the honest annotation here.
# The info payload's keys are part of the JSONL `file` event the GUI parses.
PageResult = tuple[Any, bool, str, dict[str, Any]]


@dataclass(frozen=True, slots=True)
class PageEncoder:
    """The job's output format, bound once so every writer agrees on it.

    Frozen because `BundleWriter` and `PagePacker` each hold `encode` as a bare
    callable: if the format could be reassigned mid-job, pages already queued
    would encode with one format while their entry names claimed another.
    """

    fid: str
    opts: dict
    caps: dict

    def encode(self, image: Any) -> bytes:
        return imageio.encode(image, self.fid, self.opts, self.caps)


@dataclass(frozen=True, slots=True)
class PageWorker:
    """Runs one page through the full pixel pipeline.

    Frozen for the same reason as `PagePolicy`: these are job-wide settings and
    nothing may reassign them mid-run. The three collaborators are mutable by
    design -- `cache` keeps loaded models alive across pages and `planner`
    accumulates the VRAM/tile measurements that make later pages faster, which
    is the entire point of holding them for the job rather than per page.
    """

    policy: PagePolicy
    cache: ModelCache
    planner: tiling.TilePlanner
    ctx: Any

    # grayscale detection
    threshold: float
    colour_percent: float
    force_gray: bool
    do_gray: bool

    # exclusions
    skip_long: bool
    long_max_side: int
    long_aspect: float
    long_pixels: int

    # resizing
    pre_h: int
    t_scale: float
    t_w: int
    t_h: int

    def run(self, image: Any, src_name: str) -> PageResult:
        """Full single-image pipeline: (uint8 array, is_gray, model name, info)."""
        oh, ow = runtime.hwc(image)[:2]
        gray, score, coloured = gray_stats(image, self.threshold, self.colour_percent)
        if self.force_gray:
            gray = True
        by_rule = self.policy.excluded(gray, oh, ow)
        by_size = self.skip_long and is_long_strip(
            ow, oh, self.long_max_side, self.long_aspect, self.long_pixels
        )
        if by_rule or by_size:
            why = "excluded by a rule" if by_rule else "long strip"
            log(f"{src_name}: {ow}x{oh} {why}, passed through without upscaling", "warn")
            return (
                image,
                gray,
                "",
                {
                    "w": ow,
                    "h": oh,
                    "src_w": ow,
                    "src_h": oh,
                    "score": round(score, 2),
                    "colour": round(coloured, 2),
                    "tile": 0,
                    "passthrough": True,
                },
            )
        if gray and self.do_gray:
            image = to_grayscale(image)
        if self.pre_h and oh > self.pre_h:
            image = standard_resize(image, (round(ow * self.pre_h / oh), self.pre_h))

        pick, want_levels = self.policy.plan(gray, oh, ow)
        model = self.cache.get(pick["path"]) if pick else None

        image = auto_levels(image) if want_levels and image.ndim == 2 else runtime.normalize(image)
        CTRL.gate()
        self.planner.note_model(model, pick["name"] if pick else "")
        tile = self.planner.choose(model, image)
        self.planner.before()
        image = upscale_array(self.ctx, image, model, tile)
        self.planner.retiled(tiling.tile_actually_used())
        self.planner.after()
        image = runtime.to_uint8(image, normalized=True)
        as_gray = gray and self.do_gray
        image = final_resize(image, self.t_scale, self.t_w, self.t_h, ow, oh, as_gray)
        out_h, out_w = runtime.hwc(image)[:2]
        info = {
            "w": out_w,
            "h": out_h,
            "src_w": ow,
            "src_h": oh,
            "score": round(score, 2),
            "colour": round(coloured, 2),
            "tile": self.planner.last,
        }
        return image, gray, (pick["name"] if pick else ""), info
