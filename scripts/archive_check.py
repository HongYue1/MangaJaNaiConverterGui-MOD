"""Live proof for the archive-input path (F9, F10, F13, F23, F24).

`selftest.py` covers loose files and CBZ *output*; nothing exercises
`handle_archive`, the CBZ *input* path. These fixtures cover what that code
is actually responsible for:

  Corrupt.cbz    two readable pages and one entry that is not an image
  Collision.cbz  a.jpg and a.png, which re-encode onto the SAME output name
  Ordered.cbz    six pages stored out of order, under natural-sort-sensitive
                 names, and more pages than the pack queue is deep
  AllBad.cbz     every entry is a page name whose bytes cannot be decoded
  NoPages.cbz    a valid zip holding no page at all, only a metadata sidecar

Asserted, in the terms the GUI sees:
  * a bad page does not fail the chapter - exit 0, done.ok true, processed=1
  * exactly ONE `file` event per archive. run_panel counts one unit per file
    event, so a per-page event would inflate the progress bar and counter.
  * the losing chapter reports failed=1 and renders "1 failed"
  * the job summary reports pages_failed=1 and reads as a warning, while done.ok
    stays true and the process still exits 0 - the chosen policy for a page a
    chapter could not keep
  * colliding names produce two distinct entries, not one name written twice
  * page ORDER survives. Order inside a .cbz is what the reader sees, yet every
    other check here asserts entry *identity* and de-dup only, so nothing would
    have caught a reordering. Asserted BEFORE the archive path gained a pack
    pool, and it has to keep passing after. The fixture is stored scrambled and
    named so natural order (p2 before p10) differs from lexicographic order, so
    a stored-order pass-through and a plain sort both fail it; it also holds
    more pages than PACK_SLOTS, so the producer really does wait for a slot.
  * an archive that could keep NO page publishes nothing at all, counts as a
    failed unit and exits 1 - the far end of the same policy: losing some pages
    is a warning, losing every page is a failure, and an empty .cbz must never
    replace a good one. The same guard, for the same reason, when the archive
    simply held no page: an empty output is never the right answer.

AllBad.cbz runs as a SECOND worker invocation over its own tree, deliberately:
it must exit 1, while everything above is the F10 policy and has to keep
passing untouched in the same file.

Run: backend/python/python.exe scripts/archive_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "janai" / "worker" / "worker.py"
sys.path.insert(0, str(ROOT / "src"))

from janai.app.runlog import format_done, format_file

BAD_ENTRY = "page-002.png"
# Page *names*, so `is_page_entry` counts them as pages and the run really does
# try them; the bytes are what makes every single one fail.
ALL_BAD_ENTRIES = ("page-001.png", "page-002.png")
# Not a page name at all, so `is_page_entry` filters it out and the archive is
# left with nothing to convert - the other way to end up writing no pages.
NO_PAGE_ENTRY = "ComicInfo.xml"
# Stored deliberately scrambled, and named so that natural order differs from
# lexicographic order: a reorder, a plain sort and a stored-order pass-through
# each produce a different list, so one assertion catches all three.
ORDERED_STORED = ("p10.png", "p2.png", "p20.png", "p1.png", "p11.png", "p3.png")
ORDERED_EXPECTED = ["p1.png", "p2.png", "p3.png", "p10.png", "p11.png", "p20.png"]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def pages() -> tuple[bytes, bytes]:
    """One readable page encoded two ways, so both decode but share a stem."""
    import cv2
    import numpy as np

    w, h = 240, 320
    ramp = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    img = cv2.cvtColor(np.repeat(ramp, h, axis=0), cv2.COLOR_GRAY2BGR)
    ok_png, png = cv2.imencode(".png", img)
    ok_jpg, jpg = cv2.imencode(".jpg", img)
    if not (ok_png and ok_jpg):
        raise SystemExit("could not encode the fixture pages")
    return png.tobytes(), jpg.tobytes()


def build_job(src: Path, out: Path) -> dict:
    return {
        "input": {"path": str(src), "mode": "bulk", "recursive": True, "include_archives": True},
        "output": {
            "dir": str(out),
            "container": "files",
            "pattern": "{name}",
            "overwrite": True,
            "keep_structure": True,
        },
        "format": {"id": "png", "options": {}},
        "upscale": {
            "mode": "scale",
            "scale": 2.0,
            "grayscale_convert": True,
            "auto_levels": True,
            "grayscale_threshold": 12,
            "grayscale_colour_percent": 0.25,
            "pre_downscale_height": 0,
            "skip_long_strips": False,
            "rules": [{"kind": "grayscale", "model": "auto"}, {"kind": "colour", "model": "auto"}],
        },
        "perf": {
            "device": "",
            "use_fp16": True,
            "tile": "auto",
            "io_workers": 2,
            "cudnn_benchmark": False,
            "allow_tf32": True,
        },
        "dry_run": False,
    }


def names_in(path: Path) -> list[str]:
    with ZipFile(path) as zf:
        return zf.namelist()


def run_worker(
    src: Path, out: Path, job_path: Path
) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    """Run the real worker over `src` and return its process plus parsed events."""
    job_path.write_text(json.dumps(build_job(src, out)), encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(WORKER), "--job", str(job_path)],
        check=False,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    events: list[dict] = []
    for line in (proc.stdout or "").splitlines():
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return proc, events


def main() -> int:
    png, jpg = pages()
    tmp = Path(tempfile.mkdtemp(prefix="janai-archive-"))
    src, out = tmp / "library", tmp / "out"
    src.mkdir()

    with ZipFile(src / "Corrupt.cbz", "w") as zf:
        zf.writestr("page-001.png", png)
        zf.writestr(BAD_ENTRY, b"this is not a png at all")
        zf.writestr("page-003.png", png)
    with ZipFile(src / "Collision.cbz", "w") as zf:
        zf.writestr("a.jpg", jpg)
        zf.writestr("a.png", png)
    with ZipFile(src / "Ordered.cbz", "w") as zf:
        for entry in ORDERED_STORED:
            zf.writestr(entry, png)

    proc, events = run_worker(src, out, tmp / "job.json")

    files = [e for e in events if e.get("type") == "file"]
    logs = [e for e in events if e.get("type") == "log"]
    done = next((e for e in events if e.get("type") == "done"), {})

    def event_for(stem: str) -> dict:
        return next((e for e in files if stem in str(e.get("path"))), {})

    print(f"\n  exit={proc.returncode}  events={len(events)}  file={len(files)}")

    check("worker exits 0 despite the bad page", proc.returncode == 0, str(proc.returncode))
    check("done.ok stays true", bool(done.get("ok")), json.dumps(done))
    check("every archive counted as processed", int(done.get("processed") or 0) == 3)
    check("no archive counted as failed", int(done.get("failed") or 0) == 0)
    check(
        "one file event per archive, so the progress counter cannot inflate",
        len(files) == 3,
        f"{len(files)} event(s)",
    )

    # --- the chapter that lost a page (F9 + F10 visibility) ---
    bad = event_for("Corrupt.cbz")
    check("Corrupt.cbz kept its two good pages", int(bad.get("entries") or 0) == 2)
    check("Corrupt.cbz reports failed=1", int(bad.get("failed") or 0) == 1)
    warned = [e for e in logs if e.get("level") == "warn" and BAD_ENTRY in str(e.get("message"))]
    check("a warn line names the bad entry", bool(warned))
    traced = [e for e in logs if e.get("level") == "debug" and "Traceback" in str(e.get("message"))]
    check("a debug traceback explains why", bool(traced))
    line, level = format_file(bad) if bad else ("", "")
    check("the run log line says '1 failed'", "1 failed" in line, line.strip())
    check("that line is still an ok line, not an error", level == "ok", level)

    # --- the same loss, carried up into the job summary (F10) ---
    check(
        "the done payload reports the lost page",
        int(done.get("pages_failed") or 0) == 1,
        repr(done.get("pages_failed")),
    )
    done_line, done_level = format_done(done) if done else ("", "")
    check("the summary line names it", "1 page failed" in done_line, done_line.strip())
    check("the summary is a warning, not a clean green line", done_level == "warn", done_level)

    # --- the chapter whose names collide (F13) ---
    good = event_for("Collision.cbz")
    check("Collision.cbz wrote both pages", int(good.get("entries") or 0) == 2)
    check("Collision.cbz reports no loss", not good.get("failed"))

    produced = {p.stem: p for p in out.rglob("*.cbz")}
    check(
        "every archive was written",
        set(produced) == {"Corrupt", "Collision", "Ordered"},
        str(set(produced)),
    )

    bad_names = names_in(produced["Corrupt"]) if "Corrupt" in produced else []
    check(
        "Corrupt.cbz holds exactly the readable pages, in order",
        bad_names == ["page-001.png", "page-003.png"],
        str(bad_names),
    )
    good_names = names_in(produced["Collision"]) if "Collision" in produced else []
    check(
        "colliding names were de-duped rather than written twice",
        good_names == ["a.png", "a_2.png"],
        str(good_names),
    )

    # --- page order, which the reader sees directly (F24) ---
    ordered_names = names_in(produced["Ordered"]) if "Ordered" in produced else []
    check(
        "pages are packed in natural order, not stored or lexicographic order",
        ordered_names == ORDERED_EXPECTED,
        str(ordered_names),
    )
    for stem, path in sorted(produced.items()):
        got = [n.lower() for n in names_in(path)]
        check(f"{stem}.cbz has no duplicate entry names", len(got) == len(set(got)), str(got))

    check("no .part file left behind", not list(out.rglob("*.part")))

    # --- archives that could keep no page at all (F23) ---
    bad_src, bad_out = tmp / "allbad", tmp / "allbad-out"
    bad_src.mkdir()
    with ZipFile(bad_src / "AllBad.cbz", "w") as zf:
        for entry in ALL_BAD_ENTRIES:
            zf.writestr(entry, b"this is not a png at all")
    with ZipFile(bad_src / "NoPages.cbz", "w") as zf:
        zf.writestr(NO_PAGE_ENTRY, b"<ComicInfo />")

    bad_proc, bad_events = run_worker(bad_src, bad_out, tmp / "allbad-job.json")
    bad_done = next((e for e in bad_events if e.get("type") == "done"), {})
    bad_files = [e for e in bad_events if e.get("type") == "file"]
    bad_produced = sorted(p.name for p in bad_out.rglob("*.cbz"))

    print(f"\n  [allbad] exit={bad_proc.returncode}  file={len(bad_files)}")

    check(
        "neither archive that kept no page is published at all",
        bad_produced == [],
        str(bad_produced),
    )
    check(
        "both count as failed units, not processed ones",
        int(bad_done.get("failed") or 0) == 2 and int(bad_done.get("processed") or 0) == 0,
        json.dumps({k: bad_done.get(k) for k in ("processed", "failed", "pages_failed")}),
    )
    check(
        "every lost page is still reported, and an absent page is not invented",
        int(bad_done.get("pages_failed") or 0) == len(ALL_BAD_ENTRIES),
        repr(bad_done.get("pages_failed")),
    )
    check("done.ok is false for that run", not bad_done.get("ok"), json.dumps(bad_done))
    check("and the worker exits 1", bad_proc.returncode == 1, str(bad_proc.returncode))
    check(
        "one file event each, both carrying an error rather than reading as a conversion",
        len(bad_files) == 2 and all(e.get("error") for e in bad_files),
        json.dumps(bad_files),
    )
    check("no abandoned .part is left behind", not list(bad_out.rglob("*.part")))

    print("\n  " + ("ALL PASS" if not failures else f"FAILED: {', '.join(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
