"""Job execution: reading a planned job's sources and predicting its output.

:mod:`janai.worker.planning` decides *which* files a job covers; this module
decides what happens to them. The dry run belongs beside the real run rather
than in a module of its own, because the two have to agree on output paths, on
entry names inside a bundle and on predicted sizes. When they drift the user is
shown a plan the run will not honour, which is worse than no plan at all.

Both paths read their sources through :mod:`janai.worker.archives`, which sits
below this module precisely because the run and the prediction share it.
"""

import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile

from janai.core import rules as _rules
from janai.core.formats import CONTAINERS, FORMATS, merged
from janai.core.fspath import io_path, path_too_long
from janai.worker import devices, imageio, runtime, tiling
from janai.worker.archives import ArchiveReader, open_archive
from janai.worker.control import CTRL, Cancelled
from janai.worker.environment import MODELS_DIR
from janai.worker.events import emit, log
from janai.worker.models import ModelCache, list_models
from janai.worker.page import PageEncoder, PageWorker
from janai.worker.pipeline import BundleWriter, Counters, PagePacker, WritePool, prefetch
from janai.worker.planning import (
    build_tasks,
    format_name,
    gather_units,
    path_key,
    resolve_out,
    unique_path,
)
from janai.worker.reporting import JobReporter
from janai.worker.selection import PagePolicy
from janai.worker.transforms import (
    GRAY_SAMPLE,
    gray_stats,
    predict_size,
)


class NoPagesError(Exception):
    """An archive finished with nothing worth publishing.

    Raised rather than handled inline so the abandoned ``.part`` is discarded,
    the unit is counted failed and the ``file`` event is emitted by the *one*
    existing failure path in ``handle_archive``. Duplicating that cleanup risks
    a second ``file`` event for the same archive, and the interface counts one
    unit of work per ``file`` event.
    """


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
    # Guard the raw string, never the Path: Path("") normalises to Path("."),
    # which is truthy, so `if not out_dir` could never fire. A job that arrived
    # without output.dir therefore wrote every page into the worker's current
    # directory -- the app folder, when the GUI launches it -- and still
    # reported success. The Path is still built from the unstripped value, so
    # every destination that worked before is byte-for-byte unchanged.
    out_dir_raw = str(outp.get("dir") or "")
    if not out_dir_raw.strip():
        raise ValueError("output.dir is required")
    out_dir = Path(out_dir_raw).expanduser()

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

    # One policy object for the whole job: it owns the "which model, levelled or
    # not, or excluded entirely" decision, and the dry run below is handed its
    # bound methods so a plan cannot disagree with the run it predicts.
    policy = PagePolicy(
        models=models,
        rule_set=rule_set,
        model_colour=model_colour,
        model_gray=model_gray,
        mode=mode,
        t_scale=t_scale,
        t_w=t_w,
        t_h=t_h,
        force_gray=force_gray,
        do_gray=do_gray,
        do_levels=do_levels,
    )

    if dry:
        return dry_run(
            tasks,
            total,
            DryRunPlan(
                out_dir=out_dir,
                ext=ext,
                pattern=pattern,
                overwrite=overwrite,
                keep_structure=keep_structure,
                threshold=threshold,
                colour_percent=colour_percent,
                pick_model=policy.model,
                excluded=policy.excluded,
                t_scale=t_scale,
                t_w=t_w,
                t_h=t_h,
                fid=fid,
                container=container_id,
                model_count=len(models),
                device=str(perf.get("device") or ""),
                fp16=devices.wants_fp16(perf.get("use_fp16", True)),
                tile_label=tile_label,
            ),
        )

    io_path(out_dir).mkdir(parents=True, exist_ok=True)
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
    counters = Counters()
    clock = time.perf_counter()

    # The pixel path gets the collaborators it needs and nothing else: it never
    # sees `counters`, `total` or the output paths, so a change to how a page is
    # reported cannot reach how a page is upscaled.
    encoder = PageEncoder(fid=fid, opts=opts, caps=caps)
    pager = PageWorker(
        policy=policy,
        cache=cache,
        planner=planner,
        ctx=ctx,
        threshold=threshold,
        colour_percent=colour_percent,
        force_gray=force_gray,
        do_gray=do_gray,
        skip_long=skip_long,
        long_max_side=long_max_side,
        long_aspect=long_aspect,
        long_pixels=long_pixels,
        pre_h=pre_h,
        t_scale=t_scale,
        t_w=t_w,
        t_h=t_h,
    )

    # Reporting is one object so this job's slice of the wire format lives in
    # one module: every method only moves a counter and emits one event. It is
    # given `counters` by reference on purpose -- that locked tally is the
    # shared state, and `write_page` runs on a write-pool thread.
    reporter = JobReporter(counters=counters, total=total, encoder=encoder)
    bundle = BundleWriter(
        encoder.encode, reporter.bundle_page, reporter.bundle_failed, reporter.bundle_done
    )

    def handle_archive(index: int, unit: dict) -> None:
        src: Path = unit["path"]
        dest = resolve_out(unit, out_dir, pattern, ".cbz", keep_structure, index, total)
        if io_path(dest).exists() and not overwrite:
            counters.bump("skipped")
            emit(
                "file", i=index, total=total, path=str(src), out=str(dest), error="exists, skipped"
            )
            return
        too_long = path_too_long(dest)
        if too_long:
            # Fail this chapter, not the run, and name the culprit -- the same
            # shape as the unsupported-archive branch just below.
            counters.bump("failed")
            emit("file", i=index, total=total, path=str(src), error=f"OSError: {too_long}")
            return
        io_path(dest).parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        # Scoped to exactly the span that reads from it: `pack_archive` pulls
        # every page through `reader` and nothing touches the source after it
        # returns, so the handle is released the moment packing is done.
        with open_archive(src) as opened:
            if opened is None:
                counters.bump("failed")
                emit("file", i=index, total=total, path=str(src), error="unsupported archive")
                return
            names, reader = opened
            pack_archive(index, src, dest, names, reader, started)

    def pack_archive(
        index: int,
        src: Path,
        dest: Path,
        names: list[str],
        reader: ArchiveReader,
        started: float,
    ) -> None:
        """Encode every page of one already-open archive into its output CBZ.

        Split from `handle_archive` so the source can be held by a `with` for
        precisely as long as it is read: this half is the only code that calls
        `reader`. The failure accounting lives here because the `.part` file it
        has to clean up is created here too.
        """
        tmp = dest.with_suffix(".cbz.part")
        # `dest` and `tmp` stay plain: they are what the `file` event and the
        # log carry. Only these two handles cross into the file system, so only
        # they may wear the extended-length prefix.
        tmp_io, dest_io = io_path(tmp), io_path(dest)
        written = 0
        failed_entries = 0
        seen: set[str] = set()

        def on_page_fail(entry: str, exc: BaseException) -> None:
            """Report a page the packer could not encode. Runs on the pack thread.

            The same two tallies the decode half below keeps, for the same
            reason. Both are safe off-thread: ``counters`` is locked and
            ``emit`` holds the stdout lock.
            """
            counters.bump("pages_failed")
            log(f"{src.name}:{entry}: {exc}", "warn")
            log(traceback.format_exc(limit=4), "debug")

        try:
            with ZipFile(tmp_io, "w", ZIP_STORED) as zf:
                # Encoding is not cheap beside the upscale it follows -- ~9% of a
                # page for PNG, ~72% for AVIF -- and it used to run inline, so
                # the GPU idled through all of it. The packer overlaps it with
                # the next page's upscale, with one worker so pages still land
                # in the order they were read.
                packer = PagePacker(zf, encoder.encode, on_page_fail)
                try:
                    for k, name in enumerate(names, 1):
                        if CTRL.cancelled:
                            raise Cancelled
                        CTRL.gate()
                        emit(
                            "progress",
                            i=index,
                            total=total,
                            path=str(src),
                            sub_i=k,
                            sub_n=len(names),
                        )
                        try:
                            raw = reader(name)
                            image, _gray, _model, _info = pager.run(
                                imageio.read_image_bytes(raw, name), name
                            )
                            # Re-encoding collapses distinct source names onto
                            # one output name - a.jpg and a.png both become
                            # a.png - and a zip stores two entries under the
                            # identical name without complaint, so readers show
                            # one page twice or drop one and nothing in the log
                            # says which. Same de-dup the loose-file bundle path
                            # already applies. Named here, on the producer
                            # thread, so the suffix follows page order rather
                            # than whichever encode happened to finish first.
                            entry = str(Path(name).with_suffix(ext).as_posix())
                            while entry.lower() in seen:
                                entry = f"{entry[: -len(ext)]}_{k}{ext}"
                            seen.add(entry.lower())
                            packer.add(entry, image)
                        except Cancelled:
                            raise
                        except Exception as exc:
                            # A page that fails here is dropped and the CBZ is
                            # silently short: the entry count was the only
                            # trace, and a short chapter looks like a short
                            # chapter. Count it so the summary line can say so.
                            #
                            # Two tallies on purpose: this archive's own count
                            # feeds its `file` line, and the job-level
                            # `pages_failed` puts the loss in the `done`
                            # summary, so the run as a whole admits it. NEITHER
                            # is counters["failed"], which alone sets done.ok
                            # and the exit code -- a chapter that lost one
                            # unreadable page still converted, so the process
                            # still exits 0. That split is the chosen policy,
                            # not an oversight. Losing *every* page is a
                            # different case, and is caught below.
                            failed_entries += 1
                            counters.bump("pages_failed")
                            log(f"{src.name}:{name}: {exc}", "warn")
                            log(traceback.format_exc(limit=4), "debug")
                finally:
                    # The zip must not close under an in-flight writestr, and
                    # the packer's tallies are only readable once every task has
                    # joined. A cancelled run discards the .part, so its queued
                    # pages are dropped rather than encoded for nothing.
                    packer.close(drain=not CTRL.cancelled)
                    written = packer.written
                    failed_entries += packer.failed
            if written == 0:
                # Publishing now would put an EMPTY .cbz where a chapter
                # belongs - and with overwrite on, over a good one - while the
                # run still reported success. Nothing was converted, so this is
                # a failed unit: the far end of the policy above. Same answer
                # when the archive held no page to begin with, because an empty
                # output is never the right one.
                raise NoPagesError(
                    f"{failed_entries} of {len(names)} pages failed"
                    if failed_entries
                    else "no pages in archive"
                )
            tmp_io.replace(dest_io)
            counters.bump("processed")
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                ms=int((time.perf_counter() - started) * 1000),
                bytes=dest_io.stat().st_size,
                entries=written,
                failed=failed_entries,
            )
        except Cancelled:
            tmp_io.unlink(missing_ok=True)
            raise
        except Exception as exc:
            tmp_io.unlink(missing_ok=True)
            counters.bump("failed")
            emit("file", i=index, total=total, path=str(src), error=f"{type(exc).__name__}: {exc}")

    def reader(unit: dict):
        if unit["kind"] == "image":
            return imageio.read_image(unit["path"])
        return None

    def run_images(items: list[dict], into: dict | None) -> None:
        """Upscale a run of images, either to loose files or into one archive."""
        count = len(items)
        # `seen` claims entry names inside a bundle; `taken` claims paths on
        # disk. Both exist because re-encoding is not injective, and the two
        # namespaces de-dupe differently.
        seen: set[str] = set()
        taken: set[str] = set()
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
                counters.bump("failed")
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
                # a.jpg and a.png both resolve to a.png, and a {parent} pattern
                # collapses a whole folder onto one name. A name this run has
                # already handed out must NOT take the skip branch: it belongs
                # to a sibling page whose write may still be queued, so exists()
                # cannot tell it apart from output left by an earlier run.
                # path_key stays on the PLAIN path: a reservation keyed on a
                # prefixed string would never match the same name again.
                claimed = path_key(dest) in taken
                if not claimed and io_path(dest).exists() and not overwrite:
                    counters.bump("skipped")
                    emit(
                        "file",
                        i=index,
                        total=total,
                        path=str(src),
                        out=str(dest),
                        error="exists, skipped",
                    )
                    continue
                if claimed or dest.resolve() == src.resolve():
                    dest = unique_path(dest, taken)
                taken.add(path_key(dest))
            started = time.perf_counter()
            try:
                image, gray, model_name, info = pager.run(payload, src.name)
            except Exception as exc:
                if CTRL.cancelled:
                    raise Cancelled from exc
                counters.bump("failed")
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
                    reporter.write_page, index, src, dest, image, gray, model_name, info, started
                )
                continue
            if into is None:
                # A page with neither a bundle nor a destination has nowhere to
                # go. This used to surface two lines below as "NoneType is not
                # subscriptable", which named the wrong cause entirely.
                raise RuntimeError(f"no output destination for {src}")
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
                counters.bump("skipped", len(units_here))
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
        **counters.snapshot(),
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
    except Exception as exc:
        # Stays broad, but must not stay silent. The header already parsed, so
        # re-raising would report a page the run might well convert as
        # unreadable. But an unmeasured verdict leaves gray None, which the
        # caller emits as bool(None) - "colour" - with score 0.0, byte-identical
        # to a genuine grayscale page: the preview then names the colour model
        # and nothing says the sample never happened.
        # Nothing inside the try gates, so no Cancelled can be swallowed here.
        # Keep it that way if this block grows.
        detail = " ".join(str(exc).split())
        log(
            f"{path.name}: colour sample failed ({type(exc).__name__}: {detail});"
            " previewing it as colour, which the real run may disagree with",
            "warn",
        )
    return w, h, gray, score, coloured


ModelPicker = Callable[[bool, int, int], dict | None]
"""Chooses the model for one page, from (gray, height, width)."""

PagePredicate = Callable[[bool, int, int], bool]
"""Answers one yes/no question about a page, from (gray, height, width)."""


@dataclass(frozen=True, slots=True)
class DryRunPlan:
    """Everything the dry run needs to predict what a real run would produce.

    A dataclass rather than the 18-key dict this used to be: the plan and the
    run have to agree, and a mistyped or forgotten key surfaced as a
    ``KeyError`` mid-preview, *after* the ``start`` event had already promised
    the user a plan. The one call site now either builds a complete plan or
    fails before anything is emitted.

    ``pick_model`` and ``excluded`` are the real run's own choosers, passed in
    rather than reimplemented, so a preview cannot name a model the run would
    not load or predict a size the page would never reach. Frozen because the
    dry run only ever reads it.
    """

    out_dir: Path
    ext: str
    pattern: str
    overwrite: bool
    keep_structure: bool
    threshold: float
    colour_percent: float
    pick_model: ModelPicker
    excluded: PagePredicate
    t_scale: float
    t_w: int
    t_h: int
    fid: str
    container: str
    model_count: int
    device: str
    fp16: bool
    tile_label: str


def dry_run(tasks: list[dict], total: int, plan: DryRunPlan) -> int:
    """Report exactly what a real run would produce, writing nothing at all."""
    out_dir = plan.out_dir
    ext = plan.ext
    pattern = plan.pattern
    overwrite = plan.overwrite
    counters = Counters()
    clock = time.perf_counter()
    cancelled = False

    emit(
        "start",
        total=total,
        out_dir=str(out_dir),
        device=plan.device or "auto",
        fp16=plan.fp16,
        tile=plan.tile_label,
        format=plan.fid,
        container=plan.container,
        models=plan.model_count,
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
            dest = resolve_out(unit, out_dir, pattern, ".cbz", plan.keep_structure, index, total)
            entries = 0
            try:
                with open_archive(src) as opened:
                    entries = len(opened[0]) if opened else 0
            except Exception as exc:
                log(f"{src.name}: {exc}", "warn")
            exists = dest.exists() and not overwrite
            counters.bump("skipped" if exists else "processed")
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
            counters.bump("skipped", len(units))
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
        taken: set[str] = set()
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
                w, h, gray, score, coloured = probe_image(src, plan.threshold, plan.colour_percent)
            except Exception as exc:
                counters.bump("failed")
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
            pick = plan.pick_model(bool(gray), h, w)
            excluded = plan.excluded(bool(gray), h, w)
            if excluded:
                pick = None
                pw, ph = w, h
                log(
                    f"{src.name}: {w}x{h} excluded by a rule,"
                    " would be re-encoded without upscaling",
                    "warn",
                )
            else:
                pw, ph = predict_size(w, h, plan.t_scale, plan.t_w, plan.t_h)
            entry = ""
            if bundle and dest_bundle is not None:
                entry = format_name(pattern, src, position, len(units)) + ext
                while entry.lower() in seen:
                    entry = f"{entry[: -len(ext)]}_{position}{ext}"
                seen.add(entry.lower())
                dest = dest_bundle
                exists = False
            else:
                dest = resolve_out(unit, out_dir, pattern, ext, plan.keep_structure, index, total)
                # Mirror run_images' reservation, or the plan promises one file
                # per colliding name while the run writes a de-duped second one.
                claimed = path_key(dest) in taken
                if claimed:
                    dest = unique_path(dest, taken)
                taken.add(path_key(dest))
                exists = not claimed and dest.exists() and not overwrite
            counters.bump("skipped" if exists else "processed")
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
        **counters.snapshot(),
    )
    return 0 if counters["failed"] == 0 else 1
