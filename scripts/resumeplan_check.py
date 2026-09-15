"""Live proof that a preview reports resume, and that a packed archive is left alone.

Two defects, one manifest. `dry_run` returned before the resume manifest was
ever loaded, so a preview of a half-finished job listed every finished page as
work still to do -- the only skip it could report was "exists, would skip" from
a file in the way. And the bundle branch of the real run asked only whether the
.cbz existed, so with overwrite on it printed "resuming: 1 of 1 already done"
and then converted and re-packed the page anyway.

Both are proved here against the real worker, with overwrite ON so that nothing
but the manifest can justify a skip:

  1. a real run packs a two-page folder into one .cbz and records it
  2. a dry run over that output reports the resume, skips the archive, and
     leaves the manifest byte-identical -- a preview may not write
  3. the real run skips it too, and does not touch the file it packed
  4. deleting the .cbz makes the preview and the run build it again, because a
     record whose file is gone is stale, not done
  5. loose files answer the same question per page

Run: backend/python/python.exe scripts/resumeplan_check.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "janai" / "worker" / "worker.py"
MANIFEST = ".janai-resume.json"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def pages() -> dict[str, bytes]:
    """Two small grayscale pages: enough for a two-member archive."""
    import cv2
    import numpy as np

    w, h = 160, 240
    ramp = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    flat = np.repeat(ramp, h, axis=0)
    ok, enc = cv2.imencode(".png", flat)
    if not ok:
        raise SystemExit("could not encode the fixture page")
    data = enc.tobytes()
    return {"page_01.png": data, "page_02.png": data}


def build_job(src: Path, out: Path, *, container: str, dry: bool) -> dict:
    """One job, run twice. `dry_run` is not part of the resume fingerprint."""
    return {
        "input": {"path": str(src), "mode": "bulk", "recursive": True, "include_archives": True},
        "output": {
            "dir": str(out),
            "container": container,
            "pattern": "{name}",
            "overwrite": True,
            "keep_structure": False,
        },
        "format": {"id": "png", "options": {}},
        "upscale": {
            "mode": "scale",
            "scale": 2.0,
            "grayscale_convert": False,
            "auto_levels": False,
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
        "resume": True,
        "dry_run": dry,
    }


def run(src: Path, out: Path, tmp: Path, *, container: str, dry: bool, tag: str) -> list[dict]:
    """One real worker process, returning the events it reported."""
    job_path = tmp / f"job-{tag}.json"
    payload = build_job(src, out, container=container, dry=dry)
    job_path.write_text(json.dumps(payload), encoding="utf-8")
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
    if proc.returncode != 0:
        print(f"    {tag}: worker exit={proc.returncode} {(proc.stderr or '')[-300:]}")
    return events


def done_of(events: list[dict]) -> dict:
    return next((e for e in events if e.get("type") == "done"), {})


def files_of(events: list[dict]) -> list[dict]:
    return [e for e in events if e.get("type") == "file"]


def reasons(events: list[dict]) -> list[str | None]:
    return [e.get("error") for e in files_of(events)]


def messages(events: list[dict]) -> list[str]:
    return [str(e.get("message")) for e in events if e.get("type") == "log"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else ""


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="janai-resumeplan-"))
    src = tmp / "library"
    src.mkdir()
    for name, data in pages().items():
        (src / name).write_bytes(data)

    out = tmp / "out-cbz"
    archive = out / "library.cbz"
    manifest = out / MANIFEST

    print("\n  1. a real run packs the folder into one archive")
    first = run(src, out, tmp, container="cbz_single", dry=False, tag="real-1")
    check("the archive is packed", archive.exists(), str(archive))
    check("and recorded", manifest.exists())
    check("two pages processed", done_of(first).get("processed") == 2, json.dumps(done_of(first)))

    recorded = digest(manifest)
    stamp = archive.stat().st_mtime_ns if archive.exists() else 0

    print("\n  2. a dry run over that output")
    dry = run(src, out, tmp, container="cbz_single", dry=True, tag="dry-1")
    said = messages(dry)
    check(
        "the resume is reported, in the run's own words",
        any("resuming: 2 of 2 already done" in m for m in said),
        str([m for m in said if "resum" in m or "changed" in m]),
    )
    check(
        "the archive is announced as a skip",
        reasons(dry) == ["already done, would skip"],
        str(reasons(dry)),
    )
    check(
        "counted as skipped, not as planned work",
        done_of(dry).get("skipped") == 2 and not done_of(dry).get("processed"),
        json.dumps(done_of(dry)),
    )
    check("a preview writes nothing", digest(manifest) == recorded)

    print("\n  3. the real run agrees, with overwrite on")
    again = run(src, out, tmp, container="cbz_single", dry=False, tag="real-2")
    check(
        "the packed archive is skipped",
        reasons(again) == ["already done, skipped"],
        str(reasons(again)),
    )
    check("nothing was re-packed", archive.stat().st_mtime_ns == stamp)

    print("\n  4. a record whose archive is gone is not trusted")
    archive.unlink()
    gone = run(src, out, tmp, container="cbz_single", dry=True, tag="dry-2")
    check(
        "the preview plans it again",
        done_of(gone).get("processed") == 2,
        json.dumps(done_of(gone)),
    )
    rebuilt = run(src, out, tmp, container="cbz_single", dry=False, tag="real-3")
    check(
        "and the run rebuilds it",
        archive.exists() and done_of(rebuilt).get("processed") == 2,
        json.dumps(done_of(rebuilt)),
    )

    print("\n  5. loose files answer the same question per page")
    loose = tmp / "out-files"
    written = run(src, loose, tmp, container="files", dry=False, tag="loose-real")
    check(
        "two pages written",
        done_of(written).get("processed") == 2,
        json.dumps(done_of(written)),
    )
    loose_manifest = digest(loose / MANIFEST)
    preview = run(src, loose, tmp, container="files", dry=True, tag="loose-dry")
    check(
        "each finished page is announced as a skip",
        reasons(preview) == ["already done, would skip"] * 2,
        str(reasons(preview)),
    )
    check(
        "and counted as skipped",
        done_of(preview).get("skipped") == 2 and not done_of(preview).get("processed"),
        json.dumps(done_of(preview)),
    )
    check("the preview left the manifest alone", digest(loose / MANIFEST) == loose_manifest)

    print("\n  " + ("ALL PASS" if not failures else f"FAILED: {', '.join(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
