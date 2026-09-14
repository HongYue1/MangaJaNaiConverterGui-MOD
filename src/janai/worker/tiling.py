"""Tile planning: how large a tile each page is upscaled in.

The planner decides the single number that most affects throughput and whether
a job survives a 6 GB card, so it owns its own module: the tile a page is cut
into is a memory decision, not an image-processing one.

Everything heavy is reached through ``runtime`` rather than imported here, so
this module can be imported (and the planner exercised) without torch or the
vendored backend being loaded. ``scripts/plannercheck.py`` relies on exactly
that: it patches ``runtime.TILE``, ``runtime.torch`` and ``tiling.log`` and
then drives the real class. Referencing those names through the module that
owns them, rather than copying them into this namespace, is what keeps those
patches effective - and a plannercheck that silently stops testing the real
planner is worse than no plannercheck at all.
"""

from __future__ import annotations

import math
from typing import Any

from janai.worker import runtime
from janai.worker.events import log


def parse_tile(value: Any) -> tuple[str, int]:
    """The tile setting as (mode, size): auto, maximum, none or a fixed size."""
    s = str(value if value is not None else "").strip().lower()
    if s in ("", "auto", "estimate", "auto (estimate)"):
        return "auto", 0
    if s in ("max", "maximum"):
        return "maximum", 0
    if s in ("none", "no tiling", "off", "0"):
        return "none", 0
    if s.isdecimal():
        return "fixed", max(16, int(s))
    return "auto", 0


class TilePlanner:
    """Works out the tile size handed to the upscaler for each image.

    The estimator inherited from chaiNNer budgets 60% of *total* VRAM, ignores
    what other processes already hold, and then rounds the tile down to a power
    of two, which throws away up to half of the usable area: a 700 px budget
    becomes 512 px, so a 1400 px page is cut into four tiles instead of two.
    Every extra tile is another forward pass plus blending, and a budget based
    on total memory is what makes a busy 6 GB laptop GPU fall into the
    out-of-memory retry path halfway through a chapter.

    This planner instead:

    * budgets memory that is actually free, minus the weights and a fixed
      headroom for the context and the allocator,
    * rounds to a multiple of 32 rather than a power of two,
    * never returns a tile bigger than the image, so a page that fits is done
      in one pass with no blending at all,
    * measures what the first image really allocated and calibrates the
      per-pixel cost from it, so later pages are sized for the model in use
      instead of for a constant,
    * only ever revises that cost upwards, and shrinks the tile for the rest
      of the run only on real memory pressure - an allocator retry once the
      allocator has settled, or a peak that came within a hair of the memory
      actually free - and then only as far as a cut that is really cheaper.
      A tile that fitted is a tile worth keeping: on a 6 GB laptop card,
      stepping 1248px down to 864px costs about 9% throughput on every page
      that follows.
    * prices a tile through the cut the splitter will really make - the tile
      grid, the per-edge overlap and the channel count - rather than pricing
      tile x tile, which overstated the affordable tile by about sqrt(3) on a
      colour page: that is what planned a 1930px whole-page pass on a 6 GB
      card and left auto_split to clean the miss up,
    * distrusts the inherited constant until it has measured the model once.
      For FDAT it is optimistic by roughly 9x, so the first page of an unseen
      model is clamped to UNVERIFIED_TILE instead of being sized from a guess.
      One cautious page is nearly free, because what costs memory is the grid
      the page is cut into and not the number asked for: a 1920x1080 page is
      cut into the same 2x2 grid of 960x540 tiles by anything from 960px to
      1079px, so 965px and 1024px are the very same work. The next coarser
      grid is 2x1, one 960x1080 tile, which needs about twice the activations
      and does not fit in 6 GB. A miss, by contrast, costs a whole wasted
      pass.
    """

    ALIGN = 32
    MIN_TILE = 128
    OVERLAP = 16  # auto_split's default padding per tile edge
    HEADROOM = 256 * 1024**2
    SAFETY = 0.85
    MARGIN = 1.10  # applied to a measured cost before reusing it
    UNVERIFIED_TILE = 1024  # ceiling until this model has been measured once

    def __init__(
        self,
        mode: str,
        fixed: int,
        device: str,
        fp16: bool,
        budget_limit_gib: int = 0,
        profile: dict | None = None,
    ) -> None:
        self.mode = mode
        self.fixed = int(fixed or 0)
        self.device = str(device or "")
        self.fp16 = bool(fp16)
        self.budget_limit = max(0, int(budget_limit_gib or 0)) * 1024**3
        self.elem = 2 if fp16 else 4
        self.last = 0
        self._said_clamp = False
        # What the one-off hardware profile measured, keyed by model file name.
        # A profile carries the two terms a single page can never separate:
        # the cost that exists before a tile is cut, and the cost each
        # channel-pixel of tile input adds. See load_profile().
        self._profile: dict[str, dict] = {}
        self._names: dict[int, str] = {}
        self._said_profile: set[str] = set()
        self.load_profile(profile)
        self._model_bytes: dict[int, int] = {}
        self._per_px: dict[int, float] = {}
        self._cap: dict[int, int] = {}
        self._pending: tuple[int, int, int, int, int, int] | None = None
        self._retries = 0
        self._pressure: dict[int, int] = {}
        # Pages this model has measured. The first one runs against a cold
        # allocator with the weights just uploaded, so its cudaMalloc retries
        # describe a cache that has not settled rather than a tile that is too
        # big. See after().
        self._pages: dict[int, int] = {}
        self._good: dict[int, int] = {}
        self._failed: dict[int, int] = {}
        self._last_page: tuple[int, int, int, int] | None = None
        self._contaminated = False

    # -- the hardware profile ---------------------------------------------- #
    def load_profile(self, profile: dict | None) -> None:
        """Take the measurements the app recorded for this machine.

        Only entries carrying a positive per-pixel cost are kept: a model
        whose profiling run failed has to fall back to the cautious
        first-page path, not to a cost of zero.
        """
        self._profile = {}
        for name, entry in (profile or {}).items():
            if isinstance(entry, dict) and float(entry.get("per_px") or 0.0) > 0:
                self._profile[str(name)] = entry

    def note_model(self, model: Any, name: str) -> None:
        """Tie a loaded model to the file name its measurements are keyed by."""
        if model is not None and name:
            self._names[id(model)] = str(name)

    def _measured(self, key: int) -> dict | None:
        """The profile entry for a loaded model, if this machine has one."""
        name = self._names.get(key)
        return self._profile.get(name) if name else None

    def tile_input_pixels(self, w: int, h: int, tile: int, c: int) -> int:
        """The cut-aware price of a tile, in the units a profile records.

        Public because the profiler has to measure in exactly the units the
        planner spends in, or the two would disagree about what a tile costs.
        """
        return self._tile_pixels(w, h, tile, c)

    # -- memory ------------------------------------------------------------ #
    def _free_bytes(self) -> int:
        # Deliberately the driver's number, not the driver's number plus
        # torch's reusable cache. Counting the cache made the planner size a
        # 1280px tile on a 6 GB card, which then hit a cudaMalloc retry on
        # every page: 97s per page against 74s at 1248px. Retries cost far
        # more than the tile gains.
        if self.device.startswith("cuda"):
            try:
                index = int(self.device.split(":")[1]) if ":" in self.device else 0
                free, total = runtime.torch.cuda.mem_get_info(index)
                return int(min(free, total))
            except Exception:
                return 0
        if self.device.startswith("xpu"):
            try:
                xpu = runtime.torch.xpu
                free, _total = xpu.mem_get_info(self.device)
                return int(free)
            except Exception:
                return 0
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:
            return 0

    def _budget(self, model_bytes: int) -> int:
        free = self._free_bytes()
        if free <= 0:
            return 0
        if self.budget_limit:
            free = min(free, self.budget_limit)
        return max(0, int((free - self.HEADROOM - model_bytes) * self.SAFETY))

    def _alloc_retries(self) -> int:
        """cudaMalloc retries so far: the allocator's own sign of pressure."""
        if not self.device.startswith("cuda"):
            return 0
        try:
            stats = runtime.torch.cuda.memory_stats(self.device)
            return int(stats.get("num_alloc_retries", 0))
        except Exception:
            return 0

    def model_bytes(self, model: Any) -> int:
        key = id(model)
        got = self._model_bytes.get(key)
        if got is None:
            try:
                params = sum(p.numel() for p in model.model.parameters())
            except Exception:
                params = 0
            got = int(params) * self.elem
            self._model_bytes[key] = got
        return got

    # -- planning ---------------------------------------------------------- #
    def _align(self, size: float) -> int:
        return max(self.MIN_TILE, (int(size) // self.ALIGN) * self.ALIGN)

    def _tile_pixels(self, w: int, h: int, tile: int, c: int) -> int:
        """Input pixels in the largest tile the splitter will actually cut."""
        count_x = max(1, math.ceil(w / tile))
        count_y = max(1, math.ceil(h / tile))
        size_x = min(w + 2 * self.OVERLAP, math.ceil(w / count_x) + 2 * self.OVERLAP)
        size_y = min(h + 2 * self.OVERLAP, math.ceil(h / count_y) + 2 * self.OVERLAP)
        return max(1, size_x * size_y * max(1, c))

    def _fit(self, w: int, h: int, c: int, budget: int, per_px: float, ceiling_px: int = 0) -> int:
        """The largest aligned tile whose real cut is predicted to fit.

        What costs memory is the grid the splitter cuts, not the number it was
        asked for, and _tile_pixels is a step function of that number: every
        tile from 672px to 960px cuts a 1920x1080 page into the same 3x2 grid.
        So walk the aligned sizes down from a whole-page pass and take the
        first that fits. Inverting tile x tile algebraically instead is what
        dropped the channel count and the overlap from the price.
        """
        affordable = budget / max(per_px, 1e-6)
        if ceiling_px > 0:
            # A tile input size that ran out of memory while profiling is a
            # hard ceiling, and unlike a tile *number* it carries across page
            # sizes: it is counted in the same channel-pixels _tile_pixels
            # returns, so a smaller page cutting fewer tiles cannot talk the
            # planner past what the card already refused.
            affordable = min(affordable, float(ceiling_px))
        tile = self._align(max(w, h) + self.ALIGN)
        while tile > self.MIN_TILE:
            if self._tile_pixels(w, h, tile, c) <= affordable:
                return tile
            tile -= self.ALIGN
        return self.MIN_TILE

    def _ceiling(self, key: int) -> int:
        """The largest aligned tile still below a size that missed, if any."""
        failed = self._failed.get(key, 0)
        if not failed:
            return 0
        return max(self.MIN_TILE, self._align(failed - self.ALIGN))

    def _cheaper(self, w: int, h: int, c: int, tile: int) -> int:
        """The largest aligned tile that cuts this page into cheaper passes.

        A step down is only worth taking if it reaches a different cut, and
        stepping by a ratio need not: 0.85 x 1312px aligns to 1088px, and on a
        720x4000 colour page both cut the same four passes of 752x1032. So the
        old cap shrank the number in the log and changed no work at all, while
        reading like a response to memory pressure. Walk the aligned sizes down
        until the price per pass really drops - the mirror of the blind halving
        that turned a 1152px request into 576px tiles.
        """
        pixels = self._tile_pixels(w, h, tile, c)
        step = self._align(tile) - self.ALIGN
        while step > self.MIN_TILE:
            if self._tile_pixels(w, h, step, c) < pixels:
                return step
            step -= self.ALIGN
        return self.MIN_TILE

    def choose(self, model: Any, image: Any):
        """A TileSize for this image, in the backend's own encoding."""
        h, w, c = runtime.hwc(image)
        self._pending = None
        if self.mode == "none":
            self.last = -1
            return runtime.TILE["none"]
        if self.mode == "maximum":
            self.last = -2
            return runtime.TILE["maximum"]
        if self.mode == "fixed":
            self.last = self.fixed
            return runtime.TILE["cls"](self.fixed)
        if model is None:
            self.last = 0
            return runtime.TILE["estimate"]

        key = id(model)
        model_bytes = self.model_bytes(model)
        budget = self._budget(model_bytes)
        if budget <= 0:
            # Page 2 and later usually land here: torch's caching allocator
            # still holds page 1's blocks, so the driver reports almost
            # nothing free even though that memory will be handed straight
            # back. Repeat the tile that already worked instead of falling
            # back to the backend's blind guess.
            proven = self._good.get(key, 0)
            if proven:
                limit = min(self._cap.get(key) or proven, self._ceiling(key) or proven)
                return self._arm(key, model, w, h, c, min(proven, limit), 0, 0)
            self.last = 0
            return runtime.TILE["estimate"]

        per_px = self._per_px.get(key)
        calibrated = per_px is not None
        measured = self._measured(key) if per_px is None else None
        ceiling_px = 0
        if measured is not None:
            # This machine has already been measured for this model, so the
            # first page is not a guess. Two things come from the profile that
            # a single page can never supply: the per-pixel slope, and the
            # fixed cost - weights, context, workspace - which is charged to
            # the budget once instead of being priced per pixel. Conflating
            # the two is what made a 1152px tile look like it needed ~4.8 GB
            # when the entire measured peak was 2586 MiB.
            per_px = float(measured.get("per_px") or 0.0) * self.MARGIN
            overhead = max(0, int(measured.get("fixed_bytes") or 0) - model_bytes)
            budget = max(0, budget - int(overhead * self.MARGIN))
            ceiling_px = int(measured.get("max_pixels") or 0)
            calibrated = per_px > 0
            name = self._names.get(key, "")
            if calibrated and name and name not in self._said_profile:
                self._said_profile.add(name)
                fixed_mib = int(measured.get("fixed_bytes") or 0) // 1024**2
                log(
                    f"sizing tiles from this machine's profile of {name}:"
                    f" {fixed_mib} MiB fixed plus"
                    f" {per_px / self.MARGIN:.1f} bytes per pixel",
                    "debug",
                )
        if per_px is None or per_px <= 0:
            # chaiNNer's calibration, put in the same units as a measurement:
            # bytes per channel-pixel of tile input. The `* c` it used to carry
            # is already inside _tile_pixels, so keeping both counted colour
            # twice while the algebraic inversion dropped it again.
            per_px = (model_bytes / (1024 * 52)) * self.elem
            calibrated = False
        tile = self._fit(w, h, c, budget, per_px, ceiling_px)
        if not calibrated:
            # An unmeasured model is a guess, and this is the guess that
            # planned 1930px and missed. Clamp the first page, measure it, then
            # trust the measurement for every page after it.
            if tile > self.UNVERIFIED_TILE and not self._said_clamp:
                # Say it once, so a single-image job that never gets to use
                # the measurement does not look like an unexplained ceiling.
                self._said_clamp = True
                log(
                    "first page of an unmeasured model: holding the tile at"
                    f" {self.UNVERIFIED_TILE}px rather than the estimated"
                    f" {tile}px until the cost has been measured"
                )
            tile = min(tile, self.UNVERIFIED_TILE)
        free = self._free_bytes()
        proven = self._good.get(key, 0)
        if proven > tile:
            # Page 2 and later: the driver still counts page 1's blocks as
            # used, because torch's allocator holds them to hand straight
            # back. Budgeting from that number alone quietly shrank a proven
            # 1952px tile to 864px, costing 23s a page against 17s. Repeat
            # the size that already finished instead - and only that size,
            # never larger: sizing from free memory plus the cache is what
            # made a 1280px tile retry on every page.
            tile = proven
            # Peak-against-free is meaningless once the cache is serving the
            # allocation, so leave real allocator retries as the only signal.
            free = 0
        cap = self._cap.get(key)
        if cap:
            tile = min(tile, cap)
        ceiling = self._ceiling(key)
        if ceiling:
            # a size that missed is a ceiling, not a target
            tile = min(tile, ceiling)
        return self._arm(key, model, w, h, c, tile, budget, free)

    def _arm(
        self, key: int, model: Any, w: int, h: int, c: int, tile: int, budget: int, free: int
    ) -> Any:
        """Note what is about to be attempted, then hand the tile over."""
        # one tile that covers the page is the fastest case there is
        tile = max(self.MIN_TILE, min(tile, self._align(max(w, h) + self.ALIGN)))
        self.last = tile
        self._last_page = (key, w, h, c)
        # The whole-page input and output tensors are allocated whatever the
        # tile size is. Keeping them out of the per-pixel cost stops a large
        # page from inflating the estimate and shrinking every later tile.
        try:
            model_scale = max(1, int(getattr(model, "scale", 1) or 1))
        except Exception:
            model_scale = 1
        page_bytes = int(w * h * max(1, c) * self.elem * (1 + model_scale**2))
        self._pending = (key, self._tile_pixels(w, h, tile, c), tile, budget, page_bytes, free)
        return runtime.TILE["cls"](tile)

    # -- measurement ------------------------------------------------------- #
    def before(self) -> None:
        self._retries = self._alloc_retries()
        self._contaminated = False
        if self._pending and self.device.startswith("cuda"):
            try:
                runtime.torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

    def after(self) -> None:
        """Calibrate the per-pixel cost from what the last upscale allocated."""
        pending, self._pending = self._pending, None
        if pending is None or not self.device.startswith("cuda"):
            return
        key, pixels, tile, _budget, page_bytes, free = pending
        if pixels < 256 * 256:
            return  # too small to measure anything useful
        try:
            peak = int(runtime.torch.cuda.max_memory_allocated(self.device))
        except Exception:
            return
        model_bytes = self._model_bytes.get(key, 0)
        activations = peak - model_bytes - page_bytes
        if activations <= 0:
            activations = peak - model_bytes
        if activations <= 0:
            return
        if self._contaminated:
            # This page's peak belongs partly to an attempt that ran out of
            # memory, so dividing it by the tile that did finish would inflate
            # the model's cost and shrink every page after it. Keep what the
            # page proved about what fits, discard its measurement.
            log("re-tiled page: not calibrating from its peak", "debug")
        else:
            measured = (activations / pixels) * self.MARGIN
            previous = self._per_px.get(key)
            self._per_px[key] = measured if previous is None else max(previous, measured)
            if previous is None:
                log(f"tile cost calibrated at {tile}px, peak {peak // 1024**2} MiB", "debug")
        # this tile finished, so it is the one to repeat when the driver stops
        # reporting usable free memory on later pages
        self._good[key] = max(self._good.get(key, 0), tile)
        # Shrink only when memory was genuinely tight. Going over the safety
        # budget does not qualify: that budget is deliberately pessimistic, and
        # a tile that finished without the allocator stalling is one worth
        # keeping, since every step down is paid on every later page.
        retried = max(0, self._alloc_retries() - self._retries)
        near_limit = bool(free) and peak > int(free * 0.92)
        pages = self._pages.get(key, 0) + 1
        self._pages[key] = pages
        if not (retried or near_limit):
            # A page that finished cleanly clears the slate: pressure has to be
            # consecutive to mean anything, or one fragmented page early in a
            # chapter ratchets the tile down for every page after it.
            self._pressure.pop(key, None)
            return
        if pages == 1 and retried and not near_limit:
            # The first page of a model pays for a cold allocator: the weights
            # have only just been uploaded and the cache has nothing to reuse,
            # so cudaMalloc retries here are warm-up and not a tile that is too
            # big. Proven on a 6 GB card with a 720x4000 colour page that
            # retried at 1312px: forcing 1088px across the batch produced the
            # very same 17.9s first page, and later pages 0.2s slower. Shrink
            # only if a settled allocator repeats the retries.
            log("first page of this model: allocator warming up, keeping the tile", "debug")
            return
        strikes = self._pressure.get(key, 0) + max(1, retried)
        self._pressure[key] = strikes
        # Measured on a 6 GB card, 1920x1080 -> 7680x4320 through the x4 FDAT
        # model: the proven 1952px tile holds 18.1s a page even while the
        # allocator retries once per page, against 20.9s at 1632px, 21.0s at
        # 1376px and 20.8s at 1152px. One retry a page is far cheaper than a
        # permanently smaller tile, and a genuine miss is already caught and
        # re-tiled for that page by auto_split. So only pressure that repeats
        # inside a single page earns a lasting step down.
        if retried >= 2:
            why = "repeated allocator retries in one page"
        elif not retried and strikes >= 2:
            why = "peaks near free memory"
        else:
            log("memory was tight, keeping the proven tile", "debug")
            return
        # Step to the next cut that is genuinely cheaper rather than stepping
        # by a ratio, which can land inside the same cut and cost a whole run
        # a smaller number for no change in the work done.
        last = self._last_page
        if last is not None and last[0] == key:
            capped = self._cheaper(last[1], last[2], last[3], tile)
        else:
            capped = max(self.MIN_TILE, self._align(tile * 0.85))
        if capped < (self._cap.get(key) or tile):
            self._cap[key] = capped
            log(f"tile capped at {capped}px after {why}", "debug")

    def retiled(self, tile: int) -> None:
        """Correct the plan with the tile auto_split actually finished with.

        `choose` hands out a plan. A page that turns out not to fit is re-tiled
        inside auto_split, which the planner never hears about: the result line
        then reported a tile that never ran, contradicting the splitter's own
        warning in the same log. Called before `after()` so the calibration
        measures the tiles that did run, and so the next page starts at the
        size that worked instead of repeating the failed full-page attempt.

        What a miss proves is that the *planned* size was too big, not that the
        size auto_split fell back to is the ceiling: halving 1930px to 965px
        says nothing against 1024px, which measures faster than both. So the
        planned size becomes a ceiling, the fallback becomes a proven floor,
        and the measured cost chooses between them.
        """
        attempted = self.last
        if tile <= 0 or attempted <= 0 or tile >= attempted:
            return
        self.last = tile
        self._contaminated = True
        if self._last_page is None:
            return
        key, w, h, c = self._last_page
        if self._pending is not None:
            pending_key, _pixels, _planned, budget, page_bytes, free = self._pending
            self._pending = (
                pending_key,
                self._tile_pixels(w, h, tile, c),
                tile,
                budget,
                page_bytes,
                free,
            )
        self._good[key] = max(self._good.get(key, 0), tile)
        known = self._failed.get(key, 0)
        if not known or attempted < known:
            self._failed[key] = attempted
            log(f"{attempted}px missed, {tile}px fitted: planning under {attempted}px", "debug")


# --------------------------------------------------------------------------- #
# what the splitter actually did
#
# auto_split re-tiles a page that does not fit without telling its caller, so
# these two read the note it leaves behind. Public rather than private because
# the upscale wrapper and the profiler both have to reconcile the plan with
# what really ran.
# --------------------------------------------------------------------------- #
def split_report() -> dict | None:
    """auto_split's note of a page it had to re-tile, when it is reachable."""
    try:
        from nodes.impl.upscale.auto_split import retile_report
    except Exception:
        return None
    return retile_report


def tile_actually_used() -> int:
    """The tile the last upscale finished with, or 0 if nothing was lowered."""
    report = split_report()
    return int((report or {}).get("tile", 0) or 0)
