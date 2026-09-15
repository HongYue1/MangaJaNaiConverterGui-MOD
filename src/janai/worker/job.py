"""Job execution: reading a planned job's sources and predicting its output.

:mod:`janai.worker.planning` decides *which* files a job covers; this module
decides what happens to them. The dry run belongs beside the real run rather
than in a module of its own, because the two have to agree on output paths, on
entry names inside a bundle and on predicted sizes. When they drift the user is
shown a plan the run will not honour, which is worse than no plan at all.

Both paths read their sources through :mod:`janai.worker.archives`, which sits
below this module precisely because the run and the prediction share it. The
real run's per-unit work -- one archive, or one run of loose images -- lives in
:mod:`janai.worker.orchestrate`; what stays here is the setup that decides what
a job means and the loop that dispatches its tasks.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from janai.core import rules as _rules
from janai.core.formats import CONTAINERS, FORMATS, merged
from janai.core.fspath import io_path
from janai.worker import devices, imageio, runtime, tiling
from janai.worker.archives import open_archive
from janai.worker.control import CTRL, Cancelled
from janai.worker.environment import MODELS_DIR
from janai.worker.events import emit, log
from janai.worker.models import ModelCache, list_models
from janai.worker.orchestrate import UnitRunner
from janai.worker.page import PageEncoder, PageWorker
from janai.worker.pipeline import BundleWriter, Counters, WritePool
from janai.worker.planning import (
    build_tasks,
    format_name,
    gather_units,
    path_key,
    resolve_out,
    unique_path,
)
from janai.worker.reporting import JobReporter
from janai.worker.resume import ResumeLog, fingerprint, unit_key
from janai.worker.selection import PagePolicy
from janai.worker.transforms import (
    GRAY_SAMPLE,
    gray_stats,
    predict_size,
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
        # The preview has to answer the same question the run will, so it
        # reads the same manifest under the same fingerprint. Read-only by
        # construction: nothing in dry_run marks or flushes, so previewing a
        # half-finished run cannot leave behind a record of work never done.
        return dry_run(
            tasks,
            total,
            DryRunPlan(
                resume=ResumeLog.load(
                    out_dir, fingerprint(job), enabled=bool(job.get("resume", True))
                ),
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

    # Which sources this output folder has already finished, read from the
    # manifest that sits beside them. Default on, because a resume record that
    # only exists when the user thought to ask for it is no use to the user who
    # cancelled without planning to. The fingerprint is what keeps it honest: a
    # record written under different settings is ignored, not trusted.
    # Built before the reporter, which records each finished loose page into it:
    # one resume writer for the run, shared by reference exactly like
    # `counters`, so no two objects can disagree about what is done.
    resume = ResumeLog.load(out_dir, fingerprint(job), enabled=bool(job.get("resume", True)))
    already = resume.finished_count()
    if already:
        log(f"resuming: {already} of {total} already done", "info")

    # Reporting is one object so this job's slice of the wire format lives in
    # one module: every method only moves a counter and emits one event. It is
    # given `counters` by reference on purpose -- that locked tally is the
    # shared state, and `write_page` runs on a write-pool thread.
    reporter = JobReporter(counters=counters, total=total, encoder=encoder, resume=resume)
    bundle = BundleWriter(
        encoder.encode, reporter.bundle_page, reporter.bundle_failed, reporter.bundle_done
    )

    # One object for everything that happens to a single planned unit, so the
    # loop below is dispatch and nothing else. The collaborators go in by
    # reference -- `counters` above all, which is locked and is this run's only
    # tally -- and the settings by value, because none of them may change once
    # the job has started.
    runner = UnitRunner(
        pager=pager,
        encoder=encoder,
        reporter=reporter,
        writer=writer,
        bundle=bundle,
        counters=counters,
        out_dir=out_dir,
        pattern=pattern,
        ext=ext,
        keep_structure=keep_structure,
        overwrite=overwrite,
        total=total,
        io_workers=io_workers,
        resume=resume,
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
                runner.handle_archive(int(unit.get("index") or 0), unit)
                continue
            if task["kind"] == "images":
                runner.run_images(task["units"], None)
                continue
            dest_bundle: Path = task["dest"]
            units_here: list[dict] = task["units"]
            # Ask the manifest before the filesystem. An archive whose every
            # page is recorded and whose file is still there is finished work:
            # without this, "resuming: 1 of 1 already done" printed and the
            # page was converted and packed all over again, because the only
            # question asked here was whether the .cbz existed -- and with
            # overwrite on, not even that.
            if bundle_done(resume, units_here, dest_bundle):
                counters.bump("skipped", len(units_here))
                emit(
                    "file",
                    i=int(units_here[0].get("index") or 0),
                    total=total,
                    path=str(units_here[0]["path"].parent),
                    out=str(dest_bundle),
                    entries=len(units_here),
                    error="already done, skipped",
                )
                continue
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
            runner.run_images(units_here, task)
            if bundle.close():
                # Recorded per SOURCE, never under `task["key"]`: planning
                # leaves that key empty for a single-archive run, so a record
                # under it would claim "the bundle for this folder is done" and
                # silently skip a different input converted into the same folder
                # later. The atomic publish inside close() is what makes every
                # member true at once -- hence only when it returns True.
                for member in units_here:
                    resume.mark_done(
                        unit_key(member["path"], Path(str(member["base"]))),
                        dest_bundle,
                        defer=True,
                    )
                # One manifest write per archive rather than per member: a
                # cbz_single run groups every page of a folder into one bundle.
                resume.flush()
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
        # After `writer.close()` has joined every queued page write, so each
        # page that reached disk has already recorded itself. Page records are
        # batched, and a deferred record that never reaches disk is work the
        # next run repeats for nothing -- on the cancel path above all.
        resume.flush()
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


def bundle_done(resume: ResumeLog, units: list[dict], dest: Path) -> bool:
    """Is every page of this archive recorded, with the archive still there?

    Bundle members are recorded per source, all at once, and only after the
    atomic publish inside ``close()`` -- so "every member recorded" means this
    exact archive was published under these settings. The existence test is
    what keeps a stale record from skipping a chapter the user has since
    deleted: the record alone would report a resume and leave no file behind.
    """
    if not units:
        return False
    if not all(resume.is_done(unit_key(u["path"], Path(str(u["base"])))) for u in units):
        return False
    return dest.exists()


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
    # The same manifest the run would consult, loaded read-only. The dry run
    # used to return before resume was ever loaded, so a preview of a
    # half-finished run listed every finished page as work still to do.
    resume: ResumeLog


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

    # Same wording and same position in the log as the real run, so a preview
    # and the run it predicts read identically.
    resume = plan.resume
    already = resume.finished_count()
    if already:
        log(f"resuming: {already} of {total} already done", "info")

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
            # handle_archive asks the manifest first, and regardless of
            # `overwrite`; the preview asks in the same order.
            key = unit_key(src, Path(str(unit["base"])))
            done = resume.is_done(key)
            carried = 0 if done else len(resume.partial_entries(key))
            if carried:
                log(
                    f"{src.name}: {carried} page(s) packed by an earlier run;"
                    " it would carry on from there if the partial still matches",
                    "info",
                )
            exists = not done and dest.exists() and not overwrite
            counters.bump("skipped" if done or exists else "processed")
            emit(
                "file",
                i=index,
                total=total,
                path=str(src),
                out=str(dest),
                dry=True,
                entries=entries,
                error=(
                    "already done, would skip" if done else "exists, would skip" if exists else None
                ),
            )
            continue

        bundle = task["kind"] == "bundle"
        units: list[dict] = task["units"]
        dest_bundle: Path | None = task.get("dest")
        if bundle and dest_bundle is not None:
            # The run's two questions, in the run's order: a fully recorded
            # archive that is still on disk is skipped whatever `overwrite`
            # says, and only then does an unrecorded file in the way count.
            packed = bundle_done(resume, units, dest_bundle)
            if packed or (dest_bundle.exists() and not overwrite):
                counters.bump("skipped", len(units))
                emit(
                    "file",
                    i=int(units[0].get("index") or 0),
                    total=total,
                    path=str(units[0]["path"].parent),
                    out=str(dest_bundle),
                    dry=True,
                    entries=len(units),
                    error="already done, would skip" if packed else "exists, would skip",
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
                # Membership is decided for the whole archive above, so a page
                # inside one is never skipped on its own.
                done = False
                exists = False
            else:
                dest = resolve_out(unit, out_dir, pattern, ext, plan.keep_structure, index, total)
                done = resume.is_done(unit_key(src, Path(str(unit["base"]))))
                # Mirror run_images' reservation, or the plan promises one file
                # per colliding name while the run writes a de-duped second one.
                # A page the run would skip reserves nothing, because run_images
                # skips before it claims a name: reserving it here would predict
                # a de-duped suffix for a page that keeps its own.
                claimed = False
                if not done:
                    claimed = path_key(dest) in taken
                    if claimed:
                        dest = unique_path(dest, taken)
                    taken.add(path_key(dest))
                exists = not done and not claimed and dest.exists() and not overwrite
            counters.bump("skipped" if done or exists else "processed")
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
                error=(
                    "already done, would skip" if done else "exists, would skip" if exists else None
                ),
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
