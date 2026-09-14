"""Live proof for the archive-input path (F9, F10-visibility, F13).

`selftest.py` covers loose files and CBZ *output*; nothing exercises
`handle_archive`, the CBZ *input* path. These two fixtures cover what that code
is actually responsible for:

  Corrupt.cbz    two readable pages and one entry that is not an image
  Collision.cbz  a.jpg and a.png, which re-encode onto the SAME output name

Asserted, in the terms the GUI sees:
  * a bad page does not fail the chapter - exit 0, done.ok true, processed=1
  * exactly ONE `file` event per archive. run_panel counts one unit per file
    event, so a per-page event would inflate the progress bar and counter.
  * the losing chapter reports failed=1 and renders "1 failed"
  * colliding names produce two distinct entries, not one name written twice

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

from janai.app.runlog import format_file

BAD_ENTRY = "page-002.png"

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

    job_path = tmp / "job.json"
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

    files = [e for e in events if e.get("type") == "file"]
    logs = [e for e in events if e.get("type") == "log"]
    done = next((e for e in events if e.get("type") == "done"), {})

    def event_for(stem: str) -> dict:
        return next((e for e in files if stem in str(e.get("path"))), {})

    print(f"\n  exit={proc.returncode}  events={len(events)}  file={len(files)}")

    check("worker exits 0 despite the bad page", proc.returncode == 0, str(proc.returncode))
    check("done.ok stays true", bool(done.get("ok")), json.dumps(done))
    check("both archives counted as processed", int(done.get("processed") or 0) == 2)
    check("no archive counted as failed", int(done.get("failed") or 0) == 0)
    check(
        "one file event per archive, so the progress counter cannot inflate",
        len(files) == 2,
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

    # --- the chapter whose names collide (F13) ---
    good = event_for("Collision.cbz")
    check("Collision.cbz wrote both pages", int(good.get("entries") or 0) == 2)
    check("Collision.cbz reports no loss", not good.get("failed"))

    produced = {p.stem: p for p in out.rglob("*.cbz")}
    check(
        "both archives were written", set(produced) == {"Corrupt", "Collision"}, str(set(produced))
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
    for stem, path in sorted(produced.items()):
        got = [n.lower() for n in names_in(path)]
        check(f"{stem}.cbz has no duplicate entry names", len(got) == len(set(got)), str(got))

    check("no .part file left behind", not list(out.rglob("*.part")))

    print("\n  " + ("ALL PASS" if not failures else f"FAILED: {', '.join(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
