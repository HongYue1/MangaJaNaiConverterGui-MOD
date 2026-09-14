#!/usr/bin/env python
"""JaNaiUpscaler worker.

Runs the upscale pipeline out of process so the GUI stays responsive and never
imports torch. Three modes:

    worker.py --probe            print one JSON line describing this machine
    worker.py --job job.json     run a job, streaming JSONL events on stdout
    worker.py --hold [--device]  keep a GPU context awake until stdin says stop

Events (one JSON object per line, always with a "type" key):
    start     {total, out_dir, device, fp16, tile, format}
    progress  {i, total, path, sub_i, sub_n}
    file      {i, total, path, out, ms, bytes, w, h, gray, model, error}
    log       {level, message}
    done      {ok, processed, failed, skipped, cancelled, elapsed}
    probe     {...}
    hold      {ok, device, name, reserved, pid} / {ok, device, released}

Control commands arrive as lines on stdin: cancel, pause, resume (and stop,
which ends --hold).

The pipeline mirrors the original MangaJaNaiConverterGui backend
(read via libvips -> grayscale detection -> optional auto levels ->
spandrel/torch upscale -> dot-gain-aware final resize -> encode), minus the
chain/workflow machinery.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1]  # <app folder>/src, the import root

# Run as a script, sys.path[0] is this directory, so the package itself would
# not be importable. Put the import root in front before anything of ours.
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from janai.core import rules as _rules
from janai.core.formats import (
    CONTAINERS,
    FORMATS,
    merged,
)
from janai.worker import capabilities, devices, imageio, runtime, tiling
from janai.worker.control import CTRL, Cancelled

# Importing environment resolves the install layout, puts the vendored backend
# on sys.path and the bundled tools on PATH. It must happen before any heavy
# import, which is why nothing below may be reordered above it.
from janai.worker.environment import MODELS_DIR, PATHS, ROOT
from janai.worker.events import emit, log
from janai.worker.job import dry_run, open_archive
from janai.worker.models import ModelCache, choose_model, list_models, upscale_array
from janai.worker.pipeline import BundleWriter, WritePool, prefetch
from janai.worker.planning import (
    build_tasks,
    format_name,
    gather_units,
    resolve_out,
    unique_path,
)
from janai.worker.transforms import (
    auto_levels,
    final_resize,
    gray_stats,
    is_long_strip,
    standard_resize,
    to_grayscale,
)

# --------------------------------------------------------------------------- #
# heavy handles, mirrored from janai.worker.runtime
#
# Transitional scaffolding for the Phase 1 split. runtime.py owns the lazy
# imports now, but the job and profiling code that reads these
# names still lives further down this file. Each extraction repoints one group
# of consumers at ``runtime.<name>``; the last one deletes this block.
#
# Mirroring is safe because the loaders are idempotent and the handles are
# module objects, and it is what lets every intermediate commit stay runnable
# and bisectable instead of forcing one unreviewable mega-move.
# --------------------------------------------------------------------------- #
np = None
cv2 = None
pyvips = None
torch = None
_normalize = None
_to_uint8 = None
TILE: dict[str, Any] = {}


def _mirror_runtime() -> None:
    """Publish runtime's loaded handles under the names this file still uses."""
    global np, cv2, pyvips, torch, _normalize, _to_uint8, TILE
    np, cv2, pyvips, torch = runtime.np, runtime.cv2, runtime.pyvips, runtime.torch
    _normalize, _to_uint8 = runtime.normalize, runtime.to_uint8
    TILE = runtime.TILE


def load_imaging(perf: dict | None = None) -> None:
    runtime.load_imaging(perf)
    _mirror_runtime()


def load_backend(perf: dict | None = None) -> None:
    runtime.load_backend(perf)
    _mirror_runtime()


# --------------------------------------------------------------------------- #
# probe report
# --------------------------------------------------------------------------- #
def do_probe(models_dir: Path) -> int:
    info: dict[str, Any] = {
        "root": str(ROOT),
        "models_dir": str(models_dir),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "paths": PATHS.as_dict(),
        "ok": True,
        "errors": [],
    }
    try:
        load_backend()
    except Exception as exc:
        info["ok"] = False
        info["errors"].append(f"backend import failed: {exc}")
        emit("probe", **info)
        return 1

    try:
        info["libvips"] = ".".join(str(pyvips.version(i)) for i in range(3))
    except Exception:
        info["libvips"] = "?"
    info["pyvips"] = getattr(pyvips, "__version__", "?")
    info["torch"] = getattr(torch, "__version__", "?")
    try:
        info["cuda"] = torch.version.cuda or ""
    except Exception:
        info["cuda"] = ""
    try:
        info["numpy"] = np.__version__
        info["opencv"] = cv2.__version__
    except Exception:
        pass

    info["devices"] = devices.device_objects()
    gpu = next((d for d in info["devices"] if d["value"] != "cpu"), None)
    info["default_device"] = gpu["value"] if gpu else "cpu"
    info["formats"] = imageio.encode_capabilities()
    info["read_jxl"] = capabilities.vips_has("jxlload") or bool(capabilities.find_djxl())
    info["read_heif"] = capabilities.vips_has("heifload")
    info["models"] = list_models(models_dir)
    info["icc"] = PATHS.icc() is not None
    info["tools"] = {name: capabilities.find_tool(name) for name in ("cjxl", "djxl")}
    try:
        import rarfile

        info["rar"] = bool(
            rarfile.tool_setup(sevenzip=True, sevenzip2=True, unrar=True, bsdtar=True)
        )
    except Exception:
        info["rar"] = False
    info["cpu_count"] = os.cpu_count() or 1
    emit("probe", **info)
    return 0


# --------------------------------------------------------------------------- #
# job execution
# --------------------------------------------------------------------------- #
def run_job(job: dict) -> int:
    inp = job.get("input", {}) or {}
    outp = job.get("output", {}) or {}
    fmt_cfg = job.get("format", {}) or {}
    ups = job.get("upscale", {}) or {}
    perf = job.get("perf", {}) or {}
    dry = bool(job.get("dry_run"))

    fid = str(fmt_cfg.get("id") or "png")
    if fid not in FORMATS:
        log(f"unknown format {fid}, using png", "warn")
        fid = "png"
    opts = merged(fid, fmt_cfg.get("options"))
    ext = FORMATS[fid].ext
    container_id = str(outp.get("container") or "files")
    if container_id not in CONTAINERS:
        log(f"unknown output container {container_id}, writing loose files", "warn")
        container_id = "files"

    if dry:
        load_imaging(perf)  # a dry run never imports torch
    else:
        load_backend(perf)
        runtime.apply_torch_perf(perf)
    CTRL.start()  # only safe after the imports; see the note in main()

    models_dir = Path(str(ups.get("models_dir") or MODELS_DIR))
    caps = imageio.encode_capabilities()
    if not caps.get(fid, {}).get("ok"):
        emit(
            "done",
            ok=False,
            processed=0,
            failed=0,
            skipped=0,
            cancelled=False,
            elapsed=0,
            error=(
                f"{FORMATS[fid].label} cannot be written here: "
                f"{caps.get(fid, {}).get('reason', '')}"
            ),
        )
        return 2

    units = gather_units(inp)
    total = len(units)
    out_dir = Path(str(outp.get("dir") or "")).expanduser()
    if not out_dir:
        raise ValueError("output.dir is required")

    models = list_models(models_dir)
    mode = str(ups.get("mode") or "scale")
    scale = float(ups.get("scale") or 2.0)
    width = int(ups.get("width") or 0)
    height = int(ups.get("height") or 0)
    threshold = float(ups.get("grayscale_threshold", 12))
    colour_percent = float(ups.get("grayscale_colour_percent", 0.25))
    # Three exclusive cases, with the old flag as the fallback so a settings
    # file written by an earlier build still means what it meant.
    page_kind = str(ups.get("page_kind") or "").strip().lower()
    if page_kind not in {"detect", "grayscale", "colour"}:
        page_kind = "detect" if bool(ups.get("grayscale_convert", True)) else "colour"
    do_gray = page_kind != "colour"
    force_gray = page_kind == "grayscale"
    do_levels = bool(ups.get("auto_levels", True))
    pre_h = int(ups.get("pre_downscale_height") or 0)
    # The rules table is the only chooser the interface exposes. These two
    # names remain as an internal fallback for a page no rule claims, and for
    # job files written before the table existed; "auto" hands the page to the
    # built-in height-band picker.
    model_colour = str(ups.get("model") or "auto")
    model_gray = str(ups.get("model_gray") or "auto") if do_gray else model_colour
    rule_set = _rules.RuleSet.from_dicts(ups.get("rules"))
    rules_logged: set[str] = set()
    overwrite = bool(outp.get("overwrite", False))
    pattern = str(outp.get("pattern") or "{name}")
    keep_structure = bool(outp.get("keep_structure", True))
    io_workers = int(perf.get("io_workers") or 2)
    tile_mode, tile_fixed = tiling.parse_tile(perf.get("tile"))
    tile_label = str(perf.get("tile", "auto"))
    skip_long = bool(ups.get("skip_long_strips", False))
    long_max_side = int(ups.get("long_strip_max_side") or 3000)
    long_aspect = float(ups.get("long_strip_min_aspect") or 2.8)
    long_pixels = int(ups.get("long_strip_min_pixels") or 9_000_000)

    if mode == "scale":
        t_scale, t_w, t_h = scale, 0, 0
    elif mode == "width":
        t_scale, t_w, t_h = 1.0, width, 0
    elif mode == "height":
        t_scale, t_w, t_h = 1.0, 0, height
    else:  # fit
        t_scale, t_w, t_h = 1.0, width, height

    tasks = build_tasks(units, out_dir, keep_structure, container_id)

    def page_factor(oh: int, ow: int) -> float:
        """The factor this page will actually be upscaled by."""
        if mode == "height" and t_h:
            return t_h / max(1, oh)
        if mode == "width" and t_w:
            return t_w / max(1, ow)
        if mode == "fit" and t_w and t_h:
            return min(t_w / max(1, ow), t_h / max(1, oh))
        return t_scale

    def resolve_model(wanted: str, gray: bool, oh: int, factor: float) -> dict | None:
        """Turn a model name (or "auto") into an installed model."""
        name = (wanted or "").strip()
        if name.lower() in ("", "auto"):
            return choose_model(models, gray, oh, factor)
        found = next((m for m in models if m["name"] == name or m["path"] == name), None)
        if found is None:
            found = choose_model(models, gray, oh, factor)
            if found:
                log(f"model {name} not found, using {found['name']}", "warn")
        return found

    def note_rule(hit) -> None:
        """Say which rule fired, once per distinct rule, not once per page."""
        text = hit.describe()
        if text not in rules_logged:
            rules_logged.add(text)
            log(f"rule: {text}")

    def page_plan(gray: bool, oh: int, ow: int) -> tuple[dict | None, bool]:
        """What happens to one page: which model, and whether to auto-level.

        A matching rule decides. A page no rule claims falls back to the
        built-in picker, which is exactly what the shipped table's catch-all
        rows do explicitly.
        """
        is_gray = force_gray or (gray and do_gray)
        levels = do_levels
        if not models:
            return None, levels
        factor = page_factor(oh, ow)
        hit = rule_set.match(gray=is_gray, width=ow, height=oh, scale=factor)
        if hit is None:
            wanted = model_gray if is_gray else model_colour
        else:
            note_rule(hit)
            wanted = hit.model
            if hit.auto_levels is not None:
                levels = bool(hit.auto_levels)
        return resolve_model(wanted, is_gray, oh, factor), levels

    def pick_model(gray: bool, oh: int, ow: int) -> dict | None:
        """Model only; the dry run reports models without touching levels."""
        return page_plan(gray, oh, ow)[0]

    def excluded_by_rule(gray: bool, oh: int, ow: int) -> bool:
        """A rule can exclude a page from the model instead of choosing one.

        The row's page-size condition decides which pages skip upscaling and
        are only re-encoded - the same outcome as the old long-strip switch,
        with the sizes visible and editable instead of hardcoded.
        """
        hit = rule_set.match(
            gray=force_gray or (gray and do_gray),
            width=ow,
            height=oh,
            scale=page_factor(oh, ow),
        )
        return hit is not None and str(getattr(hit, "action", "upscale")) == "passthrough"

    if dry:
        return dry_run(
            tasks,
            total,
            {
                "out_dir": out_dir,
                "ext": ext,
                "pattern": pattern,
                "overwrite": overwrite,
                "keep_structure": keep_structure,
                "threshold": threshold,
                "colour_percent": colour_percent,
                "pick_model": pick_model,
                "excluded": excluded_by_rule,
                "t_scale": t_scale,
                "t_w": t_w,
                "t_h": t_h,
                "fid": fid,
                "container": container_id,
                "model_count": len(models),
                "device": str(perf.get("device") or ""),
                "fp16": devices.wants_fp16(perf.get("use_fp16", True)),
                "tile_label": tile_label,
            },
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx, device, fp16 = devices.make_context(perf)
    stored_profile = perf.get("profile")
    planner = tiling.TilePlanner(
        tile_mode,
        tile_fixed,
        device,
        fp16,
        int(perf.get("budget_limit") or 0),
        # The app only forwards measurements taken on this machine in this
        # precision; anything stale is dropped there rather than here.
        profile=stored_profile if isinstance(stored_profile, dict) else None,
    )
    # Only a plain scale target has one fixed factor to compare models against;
    # width/height/fit factors depend on each page, so no warning there.
    cache = ModelCache(ctx, t_scale if mode == "scale" else 0.0)

    emit(
        "start",
        total=total,
        out_dir=str(out_dir),
        device=device,
        fp16=fp16,
        tile=tile_label,
        format=fid,
        models=len(models),
        container=container_id,
        bundles=sum(1 for t in tasks if t["kind"] == "bundle"),
    )
    if not models:
        log(f"no models found in {models_dir}; images will only be resized", "warn")

    writer = WritePool(io_workers)
    counters = {"processed": 0, "failed": 0, "skipped": 0}
    clock = time.perf_counter()

    def process_array(image, src_name: str):
        """Full single-image pipeline: (uint8 array, is_gray, model name, info)."""
        oh, ow = runtime.hwc(image)[:2]
        gray, score, coloured = gray_stats(image, threshold, colour_percent)
        if force_gray:
            gray = True
        by_rule = excluded_by_rule(gray, oh, ow)
        by_size = skip_long and is_long_strip(ow, oh, long_max_side, long_aspect, long_pixels)
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
        if gray and do_gray:
            image = to_grayscale(image)
        if pre_h and oh > pre_h:
            image = standard_resize(image, (round(ow * pre_h / oh), pre_h))

        pick, want_levels = page_plan(gray, oh, ow)
        model = cache.get(pick["path"]) if pick else None

        image = auto_levels(image) if want_levels and image.ndim == 2 else _normalize(image)
        CTRL.gate()
        planner.note_model(model, pick["name"] if pick else "")
        tile = planner.choose(model, image)
        planner.before()
        image = upscale_array(ctx, image, model, tile)
        planner.retiled(tiling.tile_actually_used())
        planner.after()
        image = _to_uint8(image, normalized=True)
        image = final_resize(image, t_scale, t_w, t_h, ow, oh, gray and do_gray)
        out_h, out_w = runtime.hwc(image)[:2]
        info = {
            "w": out_w,
            "h": out_h,
            "src_w": ow,
            "src_h": oh,
            "score": round(score, 2),
            "colour": round(coloured, 2),
            "tile": planner.last,
        }
        return image, gray, (pick["name"] if pick else ""), info

    def encode_now(image) -> bytes:
        return imageio.encode(image, fid, opts, caps)

    def write_result(
        index: int,
        src: Path,
        dest: Path,
        image,
        gray: bool,
        model_name: str,
        info: dict,
        started: float,
    ):
        try:
            data = encode_now(image)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            counters["processed"] += 1
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                ms=int((time.perf_counter() - started) * 1000),
                bytes=len(data),
                gray=bool(gray),
                model=model_name,
                **info,
            )
        except Exception as exc:
            counters["failed"] += 1
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                error=f"{type(exc).__name__}: {exc}",
            )

    def on_bundle_page(meta: dict, name: str, size: int) -> None:
        counters["processed"] += 1
        emit(
            "file",
            i=meta.get("i"),
            total=total,
            path=meta.get("src"),
            out=meta.get("bundle"),
            entry=name,
            bytes=size,
            ms=int((time.perf_counter() - float(meta.get("started") or 0)) * 1000),
            gray=bool(meta.get("gray")),
            model=meta.get("model"),
            **(meta.get("info") or {}),
        )

    def on_bundle_fail(meta: dict, name: str, error: str) -> None:
        counters["failed"] += 1
        emit(
            "file",
            i=meta.get("i"),
            total=total,
            path=meta.get("src"),
            out=meta.get("bundle"),
            entry=name,
            error=error,
        )

    def on_bundle_done(key: str, dest: Path, entries: int, failed: int, elapsed: float) -> None:
        emit(
            "bundle",
            key=key,
            out=str(dest),
            entries=entries,
            failed=failed,
            bytes=(dest.stat().st_size if dest.exists() else 0),
            ms=int(elapsed * 1000),
        )

    bundle = BundleWriter(encode_now, on_bundle_page, on_bundle_fail, on_bundle_done)

    def handle_archive(index: int, unit: dict) -> None:
        src: Path = unit["path"]
        dest = resolve_out(unit, out_dir, pattern, ".cbz", keep_structure, index, total)
        if dest.exists() and not overwrite:
            counters["skipped"] += 1
            emit(
                "file", i=index, total=total, path=str(src), out=str(dest), error="exists, skipped"
            )
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        opener = open_archive(src)
        if opener is None:
            counters["failed"] += 1
            emit("file", i=index, total=total, path=str(src), error="unsupported archive")
            return
        names, reader = opener
        tmp = dest.with_suffix(".cbz.part")
        written = 0
        try:
            with ZipFile(tmp, "w", ZIP_STORED) as zf:
                for k, name in enumerate(names, 1):
                    if CTRL.cancelled:
                        raise Cancelled
                    CTRL.gate()
                    emit("progress", i=index, total=total, path=str(src), sub_i=k, sub_n=len(names))
                    try:
                        raw = reader(name)
                        image, _gray, _model, _info = process_array(
                            imageio.read_image_bytes(raw, name), name
                        )
                        data = encode_now(image)
                        zf.writestr(str(Path(name).with_suffix(FORMATS[fid].ext).as_posix()), data)
                        written += 1
                    except Cancelled:
                        raise
                    except Exception as exc:
                        log(f"{src.name}:{name}: {exc}", "warn")
            tmp.replace(dest)
            counters["processed"] += 1
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                ms=int((time.perf_counter() - started) * 1000),
                bytes=dest.stat().st_size,
                entries=written,
            )
        except Cancelled:
            tmp.unlink(missing_ok=True)
            raise
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            counters["failed"] += 1
            emit("file", i=index, total=total, path=str(src), error=f"{type(exc).__name__}: {exc}")

    def reader(unit: dict):
        if unit["kind"] == "image":
            return imageio.read_image(unit["path"])
        return None

    def run_images(items: list[dict], into: dict | None) -> None:
        """Upscale a run of images, either to loose files or into one archive."""
        count = len(items)
        seen: set[str] = set()
        position = 0
        for unit, payload in prefetch(items, io_workers, reader):
            position += 1
            if CTRL.cancelled:
                raise Cancelled
            CTRL.gate()
            index = int(unit.get("index") or position)
            src: Path = unit["path"]
            emit(
                "progress",
                i=index,
                total=total,
                path=str(src),
                sub_i=position if into else 0,
                sub_n=count if into else 0,
            )
            if isinstance(payload, Exception):
                counters["failed"] += 1
                emit(
                    "file",
                    i=index,
                    total=total,
                    path=str(src),
                    error=f"read failed: {type(payload).__name__}: {payload}",
                )
                continue
            dest: Path | None = None
            if into is None:
                dest = resolve_out(unit, out_dir, pattern, ext, keep_structure, index, total)
                if dest.exists() and not overwrite:
                    counters["skipped"] += 1
                    emit(
                        "file",
                        i=index,
                        total=total,
                        path=str(src),
                        out=str(dest),
                        error="exists, skipped",
                    )
                    continue
                if dest.resolve() == src.resolve():
                    dest = unique_path(dest)
            started = time.perf_counter()
            try:
                image, gray, model_name, info = process_array(payload, src.name)
            except Exception as exc:
                if CTRL.cancelled:
                    raise Cancelled from exc
                counters["failed"] += 1
                emit(
                    "file",
                    i=index,
                    total=total,
                    path=str(src),
                    error=f"{type(exc).__name__}: {exc}",
                )
                log(traceback.format_exc(limit=4), "debug")
                continue
            if into is None and dest is not None:
                writer.submit(
                    write_result, index, src, dest, image, gray, model_name, info, started
                )
                continue
            entry = format_name(pattern, src, position, count) + ext
            while entry.lower() in seen:
                entry = f"{entry[: -len(ext)]}_{position}{ext}"
            seen.add(entry.lower())
            bundle.add(
                entry,
                image,
                {
                    "i": index,
                    "src": str(src),
                    "gray": gray,
                    "model": model_name,
                    "info": info,
                    "started": started,
                    "bundle": str(into["dest"]),
                },
            )

    cancelled = False
    try:
        for task in tasks:
            if CTRL.cancelled:
                cancelled = True
                break
            if task["kind"] == "archive":
                unit = task["unit"]
                emit("progress", i=int(unit.get("index") or 0), total=total, path=str(unit["path"]))
                handle_archive(int(unit.get("index") or 0), unit)
                continue
            if task["kind"] == "images":
                run_images(task["units"], None)
                continue
            dest_bundle: Path = task["dest"]
            units_here: list[dict] = task["units"]
            if dest_bundle.exists() and not overwrite:
                counters["skipped"] += len(units_here)
                emit(
                    "file",
                    i=int(units_here[0].get("index") or 0),
                    total=total,
                    path=str(units_here[0]["path"].parent),
                    out=str(dest_bundle),
                    entries=len(units_here),
                    error="exists, skipped",
                )
                continue
            bundle.open(task["key"], dest_bundle)
            run_images(units_here, task)
            bundle.close()
    except Cancelled:
        cancelled = True
    except KeyboardInterrupt:
        cancelled = True
    finally:
        try:
            bundle.close(keep=not CTRL.cancelled)
        except Exception as exc:
            log(f"could not finish the archive: {exc}", "error")
        bundle.shutdown()
        writer.close()
        if CTRL.cancelled:
            cancelled = True
        try:
            for fn in list(getattr(ctx, "chain_cleanup_fns", ())):
                fn()
        except Exception:
            pass
        try:
            if device != "cpu" and torch is not None:
                torch.cuda.empty_cache()
        except Exception:
            pass

    emit(
        "done",
        ok=counters["failed"] == 0 and not cancelled,
        cancelled=cancelled,
        elapsed=round(time.perf_counter() - clock, 2),
        **counters,
    )
    return 0 if counters["failed"] == 0 else 1


# --------------------------------------------------------------------------- #
# GPU wake lock
# --------------------------------------------------------------------------- #
def resolve_hold_device(torch_mod: Any, device: str) -> str:
    """The accelerator to keep warm. Empty result means 'nothing to hold'."""
    want = (device or "").strip()
    if want == "cpu":
        return ""
    if want:
        return want
    try:
        if torch_mod.cuda.is_available() and torch_mod.cuda.device_count():
            return "cuda:0"
    except Exception:
        pass
    try:
        if hasattr(torch_mod, "xpu") and torch_mod.xpu.is_available():
            return "xpu:0"
    except Exception:
        pass
    return ""


def do_hold(device: str = "", interval: float = 15.0) -> int:
    """Keep a GPU context alive until stdin says stop.

    Creating a CUDA context is what actually wakes the card: without one the
    driver leaves it in a low power state (and on laptops the dGPU parks
    entirely), so the first upscale of a session pays several seconds of
    context creation and clock ramp before any real work starts. This holds a
    tiny tensor on the device and touches it every `interval` seconds, which
    costs a few MB of VRAM and effectively no power, but keeps the context
    resident. Only torch is imported, never the model backend.
    """
    try:
        import torch
    except Exception as exc:
        emit("hold", ok=False, device=device or "auto", error=f"torch unavailable: {exc}")
        return 1

    target = resolve_hold_device(torch, device)
    if not target:
        emit("hold", ok=False, device=device or "auto", error="no GPU available to hold")
        return 0

    try:
        dev = torch.device(target)
        pin = torch.zeros(256, 256, dtype=torch.float32, device=dev)
        name, reserved = target, 0
        if dev.type == "cuda":
            name = torch.cuda.get_device_name(dev)
            torch.cuda.synchronize(dev)
            reserved = int(torch.cuda.memory_reserved(dev))
        elif dev.type == "xpu":
            try:
                name = torch.xpu.get_device_name(dev)
            except Exception:
                pass
    except Exception as exc:
        emit("hold", ok=False, device=target, error=f"{type(exc).__name__}: {exc}")
        return 1

    emit("hold", ok=True, device=target, name=name, reserved=reserved, pid=os.getpid())

    stop = threading.Event()

    def watch() -> None:
        try:
            for line in sys.stdin:
                if line.strip().lower() in ("stop", "cancel", "quit", "exit"):
                    break
        except Exception:
            pass
        stop.set()

    threading.Thread(target=watch, name="hold-stdin", daemon=True).start()

    while not stop.wait(max(1.0, float(interval))):
        try:
            pin.add_(1.0)
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
        except Exception as exc:
            emit("hold", ok=False, device=target, error=f"lost the device: {exc}")
            return 1

    try:
        del pin
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        elif dev.type == "xpu":
            torch.xpu.empty_cache()
    except Exception:
        pass
    emit("hold", ok=True, device=target, released=True)
    return 0


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# hardware profile
# --------------------------------------------------------------------------- #
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
            props = torch.cuda.get_device_properties(index)
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
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
        result = None
        ok = True
        oom = False
        begin = time.time()
        try:
            result = upscale_array(ctx, page, model, TILE["cls"](tile))
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
            peak = int(torch.cuda.max_memory_allocated(device))
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
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
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
        load_backend(perf)
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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="worker", description="JaNaiUpscaler worker")
    ap.add_argument("--job", help="path to a job JSON file")
    ap.add_argument("--probe", action="store_true", help="report devices, encoders and models")
    ap.add_argument("--models-dir", default=str(MODELS_DIR))
    ap.add_argument(
        "--hold", action="store_true", help="hold a GPU context awake until stdin says stop"
    )
    ap.add_argument("--device", default="", help="device for --hold, e.g. cuda:0")
    ap.add_argument(
        "--hold-interval",
        type=float,
        default=15.0,
        help="seconds between keep-alive touches (default 15)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="with --job: report what would happen, write nothing"
    )
    ap.add_argument(
        "--profile",
        action="store_true",
        help="with --job: measure this machine's tile cost and speed, write nothing",
    )
    args = ap.parse_args(argv)

    runtime.install_warning_filters()  # before torch is imported anywhere in this process

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", newline="\n")
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    if args.probe:
        return do_probe(Path(args.models_dir))
    if args.hold:
        return do_hold(args.device, args.hold_interval)
    if not args.job:
        ap.error("one of --job, --probe or --hold is required")

    # The stdin control reader is deliberately NOT started here. On Windows a
    # thread parked in a blocking read on a *piped* stdin while numpy's
    # extension modules are being loaded deadlocks the loader: "import numpy"
    # never returns, so a GUI-launched job hung forever before printing a
    # single event. run_job() starts the reader once the backend is imported;
    # anything the GUI sends in the meantime waits in the pipe and is handled
    # as soon as the reader comes up, so no cancel/pause is lost.
    try:
        job = json.loads(Path(args.job).read_text(encoding="utf-8"))
    except Exception as exc:
        emit(
            "done",
            ok=False,
            processed=0,
            failed=0,
            skipped=0,
            cancelled=False,
            elapsed=0,
            error=f"bad job file: {exc}",
        )
        return 2
    if args.dry_run:
        job["dry_run"] = True
    if args.profile:
        try:
            return do_profile(job)
        except Exception as exc:
            emit("log", level="error", message=traceback.format_exc(limit=8))
            emit("profile", ok=False, error=f"{type(exc).__name__}: {exc}")
            return 1
    try:
        return run_job(job)
    except Exception as exc:
        emit("log", level="error", message=traceback.format_exc(limit=8))
        emit(
            "done",
            ok=False,
            processed=0,
            failed=0,
            skipped=0,
            cancelled=CTRL.cancelled,
            elapsed=0,
            error=f"{type(exc).__name__}: {exc}",
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
