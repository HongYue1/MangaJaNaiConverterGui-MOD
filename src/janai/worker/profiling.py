"""Measuring what this machine can actually do, ahead of any real job.

The tile planner sizes tiles from a model's memory cost, but a real job only
ever produces a single (pixels, peak) point per model, which cannot separate a
fixed cost from a per-pixel slope. That is why an unseen model is held at a
cautious tile until something better is known. This subcommand walks a ladder
of tile sizes on synthetic square pages to collect several points, fits both
terms, and reports them against the hardware fingerprint it was handed so the
numbers are re-measured when the machine changes rather than trusted forever.

It writes no images.

**No automated gate covers this path.** smoke, plannercheck, uicheck and
selftest all leave `--profile` untouched, so changes here must be verified with
a live `--profile` run against real models.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from janai.worker import devices, runtime, tiling
from janai.worker.control import CTRL
from janai.worker.environment import MODELS_DIR
from janai.worker.events import emit, log
from janai.worker.models import ModelCache, list_models, upscale_array

#: The ladder the profiler walks. Each page is square and exactly one tile
#: wide, so every measurement is one clean (pixels, peak) point with no grid
#: geometry folded into it.
PROFILE_TILES: tuple[int, ...] = (512, 768, 1024, 1280, 1536, 1792, 2048)


def _is_oom(exc: BaseException) -> bool:
    """Whether a failure was the card running out of memory, not a bug."""
    if "outofmemory" in type(exc).__name__.lower():
        return True
    text = str(exc).lower()
    return "out of memory" in text or ("alloc" in text and "fail" in text)


def _fit_cost(points: list[tuple[int, int]]) -> tuple[int, float]:
    """Split measured peaks into a fixed cost and a per-pixel cost.

    `points` is [(tile input channel-pixels, peak bytes)]. A real job can only
    ever produce one point per model, and dividing a single peak by its pixel
    count prices the resident weights, the CUDA context and the workspace as
    though they grew with the tile. That is exactly why a 1152px tile looked
    like it needed ~4.8 GB when the whole measured peak was 2586 MiB. Two or
    more points separate the constant from the slope by least squares.
    """
    usable = [(float(px), float(peak)) for px, peak in points if px > 0 and peak > 0]
    if not usable:
        return 0, 0.0
    if len(usable) == 1:
        px, peak = usable[0]
        return 0, peak / px
    n = float(len(usable))
    sx = sum(p for p, _ in usable)
    sy = sum(q for _, q in usable)
    sxx = sum(p * p for p, _ in usable)
    sxy = sum(p * q for p, q in usable)
    denom = n * sxx - sx * sx
    if denom <= 0:
        return 0, sy / max(sx, 1.0)
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    if slope <= 0:
        # Too flat to read a slope from. Charge it all per pixel rather than
        # invent a negative cost that would size tiles without limit.
        return 0, sy / max(sx, 1.0)
    return max(0, int(intercept)), slope


def _device_hardware(device: str) -> dict:
    """Name and total VRAM of the device being profiled, for the record."""
    out: dict[str, Any] = {"device": device, "name": "", "vram": 0}
    try:
        if device.startswith("cuda"):
            index = int(device.split(":")[1]) if ":" in device else 0
            props = runtime.torch.cuda.get_device_properties(index)
            out["name"] = str(getattr(props, "name", "") or "")
            out["vram"] = int(getattr(props, "total_memory", 0) or 0)
    except Exception:
        pass
    return out


def _profile_picks(job: dict, models_dir: Path) -> list[dict]:
    """Which models to measure: the ones asked for, else one per scale."""
    try:
        installed = list_models(models_dir)
    except Exception:
        installed = []
    by_path = {str(m.get("path")): m for m in installed if isinstance(m, dict)}
    wanted = [by_path[str(p)] for p in (job.get("models") or []) if str(p) in by_path]
    if wanted:
        return wanted
    # Nothing named: one model per distinct scale is enough, since what this
    # measures is per model and the shipped set is 2x and 4x.
    by_scale: dict[Any, dict] = {}
    for info in by_path.values():
        by_scale.setdefault(info.get("scale"), info)
    return list(by_scale.values())[:2]


def _profile_one(ctx, planner, model: Any, name: str, offset: int, total: int) -> dict | None:
    """Walk the tile ladder for one model and return what it measured."""
    import numpy as _np

    device = planner.device
    model_cost = planner.model_bytes(model)
    card = _device_hardware(device).get("vram") or 0
    steps: list[dict] = []
    points: list[tuple[int, int]] = []
    max_pixels = 0
    failed_pixels = 0
    step = offset
    for tile in PROFILE_TILES:
        if CTRL.cancelled:
            break
        pixels = planner.tile_input_pixels(tile, tile, tile, 3)
        if len(points) >= 2 and card:
            fixed_so_far, slope_so_far = _fit_cost(points)
            if fixed_so_far + slope_so_far * pixels > card:
                # Hopeless on this card by its own numbers, so stop rather
                # than spend a whole pass proving it.
                log(
                    f"{name}: {tile}px would need more than this card has,"
                    " so the ladder stops here",
                    "debug",
                )
                break
        step += 1
        emit("profile_progress", model=name, tile=tile, index=step, total=total)
        page = _np.random.default_rng(0x1A11).random((tile, tile, 3), dtype=_np.float32)
        try:
            runtime.torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
        result = None
        ok = True
        oom = False
        begin = time.time()
        try:
            result = upscale_array(ctx, page, model, runtime.TILE["cls"](tile))
        except Exception as exc:
            ok = False
            oom = _is_oom(exc)
            if not oom:
                log(f"{name}: {tile}px failed, {type(exc).__name__}: {exc}", "warn")
        seconds = time.time() - begin
        # A tile that does not fit does not always raise: auto_split catches
        # the miss itself and finishes the page at a smaller tile. Counting
        # that as a success would divide a peak measured at the lowered tile
        # by this tile's pixel count, understating the cost, and would claim a
        # largest single-pass tile that never ran in one pass. Measured here:
        # a 4x model asked for 768px was quietly finished at 736px.
        lowered = tiling.tile_actually_used()
        if ok and lowered and lowered < tile:
            log(
                f"{name}: {tile}px did not fit in one pass (auto_split finished it at {lowered}px)",
                "debug",
            )
            ok = False
            oom = True
        peak = 0
        try:
            peak = int(runtime.torch.cuda.max_memory_allocated(device))
        except Exception:
            peak = 0
        entry: dict[str, Any] = {
            "tile": tile,
            "pixels": pixels,
            "peak": peak,
            "seconds": round(seconds, 3),
            "ok": ok,
        }
        if ok and result is not None and seconds > 0:
            try:
                oh, ow, _oc = runtime.hwc(result)
                entry["mpx_per_s"] = round(int(ow) * int(oh) / seconds / 1e6, 2)
            except Exception:
                pass
        result = None
        page = None
        devices.release_cache(device)
        steps.append(entry)
        emit(
            "profile_progress",
            model=name,
            tile=tile,
            index=step,
            total=total,
            ok=ok,
            peak=peak,
            seconds=entry["seconds"],
        )
        if ok:
            points.append((pixels, peak))
            max_pixels = max(max_pixels, pixels)
            continue
        failed_pixels = pixels
        if not oom:
            # A failure that is not memory pressure says nothing about
            # capacity, so the rest of the ladder would measure noise.
            return None
        break
    fixed_bytes, per_px = _fit_cost(points)
    if per_px <= 0:
        return None
    ran = [s for s in steps if s.get("ok")]
    fastest = max(
        (s for s in ran if s.get("mpx_per_s")),
        key=lambda s: float(s["mpx_per_s"]),
        default=None,
    )
    largest = max((int(s["tile"]) for s in ran), default=0)
    tail = f", fastest {fastest['tile']}px at {fastest['mpx_per_s']:.1f} MPx/s" if fastest else ""
    log(
        f"{name}: {int(fixed_bytes) // 1024**2} MiB fixed plus {per_px:.1f} bytes"
        f" per pixel, largest single-pass tile {largest}px{tail}"
    )
    return {
        "fixed_bytes": int(fixed_bytes),
        "per_px": round(float(per_px), 4),
        "max_pixels": int(max_pixels),
        "failed_pixels": int(failed_pixels),
        "model_bytes": int(model_cost),
        "best_tile": int(fastest["tile"]) if fastest else 0,
        "largest_tile": largest,
        "steps": steps,
    }


def do_profile(job: dict) -> int:
    """Measure what each model costs and how fast it runs on this machine.

    A real job measures one page at one tile size, so it can only ever produce
    a single (pixels, peak) point per model. That is why the planner has to
    hold the first page of an unseen model at a cautious 1024px, and why it
    cannot tell a model's fixed cost apart from its per-pixel cost afterwards.

    This walks a ladder of tile sizes on synthetic square pages - one pass
    each, so the cut is not part of the measurement - and records both terms,
    the largest tile input that fits, and how fast each size actually ran.
    Everything it learns is stored against the hardware fingerprint it was
    given, so it is re-measured when the machine changes rather than trusted
    forever. It writes no images.
    """
    from janai.core import hardware as _hardware

    perf = dict(job.get("perf") or {})
    ups = dict(job.get("upscale") or {})
    models_dir = Path(str(ups.get("models_dir") or MODELS_DIR))
    started = time.time()
    try:
        runtime.load_backend(perf)
        runtime.apply_torch_perf(perf)
    except Exception as exc:
        emit("profile", ok=False, error=f"backend import failed: {type(exc).__name__}: {exc}")
        return 1
    CTRL.start()

    ctx, device, fp16 = devices.make_context(perf)
    if not device.startswith(("cuda", "xpu")):
        emit(
            "profile",
            ok=False,
            error="profiling measures VRAM, and this run is on the CPU",
        )
        return 1
    planner = tiling.TilePlanner("auto", 0, device, fp16, int(perf.get("budget_limit") or 0))
    cache = ModelCache(ctx)
    picks = _profile_picks(job, models_dir)
    if not picks:
        emit("profile", ok=False, error=f"no models found in {models_dir}")
        return 1

    total = len(picks) * len(PROFILE_TILES)
    log(
        f"profiling {len(picks)} model(s) on {device} in"
        f" {'FP16' if fp16 else 'FP32'} - this writes no images"
    )
    measured: dict[str, dict] = {}
    done = 0
    for pick in picks:
        if CTRL.cancelled:
            break
        name = str(pick.get("name") or Path(str(pick.get("path") or "")).name)
        try:
            model = cache.get(str(pick["path"]))
        except Exception as exc:
            log(f"{name}: will not load, skipping ({type(exc).__name__}: {exc})", "warn")
            done += len(PROFILE_TILES)
            continue
        got = _profile_one(ctx, planner, model, name, done, total)
        done += len(PROFILE_TILES)
        if got:
            measured[name] = got
        else:
            log(f"{name}: could not be measured on this device", "warn")

    profile = {
        "version": _hardware.PROFILE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "fingerprint": str(job.get("fingerprint") or ""),
        "device": device,
        "fp16": bool(fp16),
        "hardware": _device_hardware(device),
        "models": measured,
    }
    emit(
        "profile",
        ok=bool(measured),
        cancelled=CTRL.cancelled,
        elapsed=round(time.time() - started, 2),
        profile=profile,
        error="" if measured else "no model could be measured",
    )
    return 0 if measured else 1
