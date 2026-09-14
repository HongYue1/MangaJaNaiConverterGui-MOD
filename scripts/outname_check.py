"""Live proof that two sources cannot collapse onto one output file (F20).

`archive_check.py` covers this collapse *inside* a CBZ. This covers the
loose-file output path, where it is worse: `run_images` de-dupes bundle entry
names but nothing de-dupes loose dest paths, and the write is asynchronous, so
the `dest.exists()` skip test races the still-queued earlier write.

Fixture: one folder holding a.jpg and a.png. Under the default `{name}` pattern
both re-encode onto a.png, so a run that does not reserve names writes one file
for two pages and still reports processed=2.

Asserted, in the terms the GUI sees:
  * exit 0, done.ok true, processed=2, failed=0 - the run still succeeds
  * the two `file` events name two DISTINCT outputs, and both exist on disk
  * overwrite=False reaches the same result, because the name was claimed by
    THIS run: a page must not be dropped as "already converted" when the file
    it collides with is its own sibling, and which of the two wins must not
    depend on write-pool timing
  * the dry run predicts exactly the files the real run writes, so the preview
    cannot promise a different set of names

Run: backend/python/python.exe scripts/outname_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "janai" / "worker" / "worker.py"

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


def build_job(src: Path, out: Path, *, overwrite: bool, dry: bool) -> dict:
    return {
        "input": {"path": str(src), "mode": "bulk", "recursive": True, "include_archives": True},
        "output": {
            "dir": str(out),
            "container": "files",
            "pattern": "{name}",
            "overwrite": overwrite,
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
        "dry_run": dry,
    }


def run(
    tmp: Path, src: Path, out: Path, *, overwrite: bool = True, dry: bool = False
) -> tuple[int, list[dict], dict]:
    """Run one real worker process and return its exit code, file events and done."""
    job = build_job(src, out, overwrite=overwrite, dry=dry)
    job_path = tmp / f"job-{out.name}.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
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
    done = next((e for e in events if e.get("type") == "done"), {})
    return proc.returncode, files, done


def written(out: Path) -> list[str]:
    return sorted(p.name for p in out.rglob("*.png"))


def announced(files: list[dict]) -> list[str]:
    return [str(e["out"]) for e in files if e.get("out")]


def main() -> int:
    png, jpg = pages()
    tmp = Path(tempfile.mkdtemp(prefix="janai-outname-"))
    src = tmp / "chapter"
    src.mkdir()
    (src / "a.png").write_bytes(png)
    (src / "a.jpg").write_bytes(jpg)

    print("\n[control] a run of two colliding pages still succeeds")
    out = tmp / "overwrite"
    rc, files, done = run(tmp, src, out)
    outs, produced = announced(files), written(out)
    print(f"  rc={rc} events={len(files)} on_disk={produced}")
    print(f"  announced={sorted(set(outs))}")
    check("worker exits 0", rc == 0, str(rc))
    check("done.ok stays true", bool(done.get("ok")), json.dumps(done))
    check("both pages counted as processed", int(done.get("processed") or 0) == 2)
    check("neither page counted as failed", int(done.get("failed") or 0) == 0)
    check("one file event per page", len(files) == 2, f"{len(files)} event(s)")
    check("both sources survive", {p.name for p in src.iterdir()} == {"a.png", "a.jpg"})

    print("\n[collide] two sources must not collapse onto one output file")
    check("the events name two distinct outputs", len(set(outs)) == 2, str(sorted(set(outs))))
    check("two files exist on disk, not one", len(produced) == 2, str(produced))
    check("every written file has bytes", all(p.stat().st_size for p in out.rglob("*.png")))
    check(
        "what was announced is what exists",
        {Path(o).name for o in outs} == set(produced),
        f"{sorted({Path(o).name for o in outs})} vs {produced}",
    )

    print("\n[skip] overwrite=False must not drop a page as 'already converted'")
    out2 = tmp / "skip"
    rc2, _files2, done2 = run(tmp, src, out2, overwrite=False)
    produced2 = written(out2)
    print(f"  rc={rc2} skipped={done2.get('skipped')} on_disk={produced2}")
    check("worker exits 0 with overwrite off", rc2 == 0, str(rc2))
    check("two files exist on disk", len(produced2) == 2, str(produced2))
    check(
        "nothing was skipped: the collision is a sibling, not a prior run",
        int(done2.get("skipped") or 0) == 0,
        repr(done2.get("skipped")),
    )

    print("\n[dry] the preview must name exactly the files the run writes")
    out3 = tmp / "dry"
    rc3, files3, _done3 = run(tmp, src, out3, dry=True)
    predicted = sorted({Path(o).name for o in announced(files3)})
    print(f"  rc={rc3} predicted={predicted}")
    check("dry run exits 0", rc3 == 0, str(rc3))
    check("the plan predicts two distinct files", len(predicted) == 2, str(predicted))
    check(
        "the plan matches what the real run produced",
        predicted == produced,
        f"{predicted} vs {produced}",
    )

    print("\n  " + ("ALL PASS" if not failures else f"FAILED: {', '.join(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
