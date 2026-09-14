"""Job execution: reading a planned job's sources and predicting its output.

:mod:`janai.worker.planning` decides *which* files a job covers; this module
decides what happens to them. The dry run belongs beside the real run rather
than in a module of its own, because the two have to agree on output paths, on
entry names inside a bundle and on predicted sizes. When they drift the user is
shown a plan the run will not honour, which is worse than no plan at all.

``open_archive`` is the read side of a CBZ/CBR source, and both paths use it.
"""

import time
import traceback
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from janai.core import rules as _rules
from janai.core.formats import CONTAINERS, FORMATS, merged
from janai.worker import devices, imageio, runtime, tiling
from janai.worker.control import CTRL, Cancelled
from janai.worker.environment import MODELS_DIR
from janai.worker.events import emit, log
from janai.worker.models import ModelCache, choose_model, list_models, upscale_array
from janai.worker.pipeline import BundleWriter, WritePool, prefetch
from janai.worker.planning import (
    IMAGE_EXTS,
    build_tasks,
    format_name,
    gather_units,
    natural_key,
    resolve_out,
    unique_path,
)
from janai.worker.transforms import (
    GRAY_SAMPLE,
    auto_levels,
    final_resize,
    gray_stats,
    is_long_strip,
    predict_size,
    standard_resize,
    to_grayscale,
)


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
        runtime.load_imaging(perf)  # a dry run never imports torch
    else:
        runtime.load_backend(perf)
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

        image = auto_levels(image) if want_levels and image.ndim == 2 else runtime.normalize(image)
        CTRL.gate()
        planner.note_model(model, pick["name"] if pick else "")
        tile = planner.choose(model, image)
        planner.before()
        image = upscale_array(ctx, image, model, tile)
        planner.retiled(tiling.tile_actually_used())
        planner.after()
        image = runtime.to_uint8(image, normalized=True)
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
            # The event carries only "OSError: ...", and encode/write failures
            # land in library frames, so the message alone rarely says where.
            # Same limit as the read/upscale path so both read alike.
            log(traceback.format_exc(limit=4), "debug")

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
        failed_entries = 0
        seen: set[str] = set()
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
                        # Re-encoding collapses distinct source names onto one
                        # output name - a.jpg and a.png both become a.png - and
                        # a zip stores two entries under the identical name
                        # without complaint, so readers show one page twice or
                        # drop one and nothing in the log says which. Same
                        # de-dup the loose-file bundle path already applies.
                        entry = str(Path(name).with_suffix(ext).as_posix())
                        while entry.lower() in seen:
                            entry = f"{entry[: -len(ext)]}_{k}{ext}"
                        seen.add(entry.lower())
                        zf.writestr(entry, data)
                        written += 1
                    except Cancelled:
                        raise
                    except Exception as exc:
                        # A page that fails here is dropped and the CBZ is
                        # silently short: the entry count was the only trace, and
                        # a short chapter looks like a short chapter. Count it so
                        # the summary line can say so.
                        #
                        # Deliberately NOT counters["failed"]: that would flip
                        # done.ok and the process exit code for a chapter with one
                        # unreadable page, which is a policy change rather than a
                        # reporting fix.
                        failed_entries += 1
                        log(f"{src.name}:{name}: {exc}", "warn")
                        log(traceback.format_exc(limit=4), "debug")
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
                failed=failed_entries,
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
        # One try per hook: a single failing hook used to abandon every later
        # one, so a leaked model handle or open file could hide behind the
        # first error - and the error itself was swallowed as well.
        for fn in list(getattr(ctx, "chain_cleanup_fns", ())):
            try:
                fn()
            except Exception as exc:
                log(f"cleanup hook failed: {type(exc).__name__}: {exc}", "warn")
        devices.release_cache(device)

    emit(
        "done",
        ok=counters["failed"] == 0 and not cancelled,
        cancelled=cancelled,
        elapsed=round(time.perf_counter() - clock, 2),
        **counters,
    )
    return 0 if counters["failed"] == 0 else 1


def probe_image(path: Path, threshold: float, colour_percent: float):
    """(width, height, is_gray, score, coloured percent) without a full decode.

    The size comes from the header and the colour verdict from a thumbnail,
    which libvips produces with shrink-on-load, so a dry run over a folder of
    8000 px scans costs a fraction of a second per page.
    """
    header = runtime.pyvips.Image.new_from_file(str(path), access="sequential")
    w, h = int(header.width), int(header.height)
    gray, score, coloured = None, 0.0, 0.0
    try:
        sample = runtime.pyvips.Image.thumbnail(str(path), GRAY_SAMPLE).numpy()
        if sample.ndim == 3 and sample.shape[2] >= 3:
            gray, score, coloured = gray_stats(sample[:, :, :3], threshold, colour_percent)
        else:
            gray = True
    except Exception:
        pass
    return w, h, gray, score, coloured


def dry_run(tasks: list[dict], total: int, cfg: dict) -> int:
    """Report exactly what a real run would produce, writing nothing at all."""
    out_dir: Path = cfg["out_dir"]
    ext: str = cfg["ext"]
    pattern: str = cfg["pattern"]
    overwrite: bool = cfg["overwrite"]
    counters = {"processed": 0, "failed": 0, "skipped": 0}
    clock = time.perf_counter()
    cancelled = False

    emit(
        "start",
        total=total,
        out_dir=str(out_dir),
        device=cfg["device"] or "auto",
        fp16=cfg["fp16"],
        tile=cfg["tile_label"],
        format=cfg["fid"],
        container=cfg["container"],
        models=cfg["model_count"],
        dry=True,
        bundles=sum(1 for t in tasks if t["kind"] == "bundle"),
    )

    for task in tasks:
        if CTRL.cancelled:
            cancelled = True
            break
        if task["kind"] == "archive":
            unit = task["unit"]
            src: Path = unit["path"]
            index = int(unit.get("index") or 0)
            dest = resolve_out(unit, out_dir, pattern, ".cbz", cfg["keep_structure"], index, total)
            entries = 0
            try:
                opener = open_archive(src)
                entries = len(opener[0]) if opener else 0
            except Exception as exc:
                log(f"{src.name}: {exc}", "warn")
            exists = dest.exists() and not overwrite
            counters["skipped" if exists else "processed"] += 1
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                dry=True,
                entries=entries,
                error="exists, would skip" if exists else None,
            )
            continue

        bundle = task["kind"] == "bundle"
        units: list[dict] = task["units"]
        dest_bundle: Path | None = task.get("dest")
        if bundle and dest_bundle is not None and dest_bundle.exists() and not overwrite:
            counters["skipped"] += len(units)
            emit(
                "file",
                i=int(units[0].get("index") or 0),
                total=total,
                path=str(units[0]["path"].parent),
                out=str(dest_bundle),
                dry=True,
                entries=len(units),
                error="exists, would skip",
            )
            continue
        if bundle and dest_bundle is not None:
            emit("bundle", out=str(dest_bundle), entries=len(units), planned=True, dry=True)

        seen: set[str] = set()
        for position, unit in enumerate(units, 1):
            if CTRL.cancelled:
                cancelled = True
                break
            CTRL.gate()
            src = unit["path"]
            index = int(unit.get("index") or position)
            emit(
                "progress",
                i=index,
                total=total,
                path=str(src),
                sub_i=position if bundle else 0,
                sub_n=len(units) if bundle else 0,
            )
            try:
                w, h, gray, score, coloured = probe_image(
                    src, cfg["threshold"], cfg["colour_percent"]
                )
            except Exception as exc:
                counters["failed"] += 1
                emit(
                    "file",
                    i=index,
                    total=total,
                    path=str(src),
                    dry=True,
                    error=f"cannot read: {type(exc).__name__}: {exc}",
                )
                continue
            # Ask for the model first, because that is what logs the matched
            # rule, then let an exclusion overrule the answer. The plan used
            # to name a model the real run would never load, and predict a
            # size the page was never going to reach.
            pick = cfg["pick_model"](bool(gray), h, w)
            excluded = bool(cfg.get("excluded") and cfg["excluded"](bool(gray), h, w))
            if excluded:
                pick = None
                pw, ph = w, h
                log(
                    f"{src.name}: {w}x{h} excluded by a rule,"
                    " would be re-encoded without upscaling",
                    "warn",
                )
            else:
                pw, ph = predict_size(w, h, cfg["t_scale"], cfg["t_w"], cfg["t_h"])
            entry = ""
            if bundle and dest_bundle is not None:
                entry = format_name(pattern, src, position, len(units)) + ext
                while entry.lower() in seen:
                    entry = f"{entry[: -len(ext)]}_{position}{ext}"
                seen.add(entry.lower())
                dest = dest_bundle
                exists = False
            else:
                dest = resolve_out(unit, out_dir, pattern, ext, cfg["keep_structure"], index, total)
                exists = dest.exists() and not overwrite
            counters["skipped" if exists else "processed"] += 1
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                entry=entry,
                w=pw,
                h=ph,
                src_w=w,
                src_h=h,
                gray=bool(gray),
                score=round(score, 2),
                colour=round(coloured, 2),
                model=(pick["name"] if pick else ""),
                passthrough=excluded,
                dry=True,
                error="exists, would skip" if exists else None,
            )
        if cancelled:
            break

    emit(
        "done",
        ok=counters["failed"] == 0 and not cancelled,
        cancelled=cancelled,
        elapsed=round(time.perf_counter() - clock, 2),
        dry=True,
        **counters,
    )
    return 0 if counters["failed"] == 0 else 1


def open_archive(path: Path):
    """Return (sorted entry names, read(name) -> bytes) or None."""
    ext = path.suffix.lower()
    if ext in (".zip", ".cbz"):
        zf = ZipFile(path)
        names = sorted(
            (
                n
                for n in zf.namelist()
                if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS
            ),
            key=natural_key,
        )
        return names, zf.read
    if ext in (".rar", ".cbr"):
        try:
            import rarfile

            rf = rarfile.RarFile(str(path))
            names = sorted(
                (
                    n
                    for n in rf.namelist()
                    if not n.endswith("/") and Path(n).suffix.lower() in IMAGE_EXTS
                ),
                key=natural_key,
            )
            return names, rf.read
        except Exception as exc:
            log(f"cannot open {path.name}: {exc} (RAR needs unrar/7z on PATH)", "warn")
            return None
    return None
