"""Job execution: reading a planned job's sources and predicting its output.

:mod:`janai.worker.planning` decides *which* files a job covers; this module
decides what happens to them. The dry run belongs beside the real run rather
than in a module of its own, because the two have to agree on output paths, on
entry names inside a bundle and on predicted sizes. When they drift the user is
shown a plan the run will not honour, which is worse than no plan at all.

``open_archive`` is the read side of a CBZ/CBR source, and both paths use it.
"""

import time
from pathlib import Path
from zipfile import ZipFile

from janai.worker import runtime
from janai.worker.control import CTRL
from janai.worker.events import emit, log
from janai.worker.planning import IMAGE_EXTS, format_name, natural_key, resolve_out
from janai.worker.transforms import GRAY_SAMPLE, gray_stats, predict_size


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
