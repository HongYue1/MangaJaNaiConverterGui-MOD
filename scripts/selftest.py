"""End-to-end smoke test for the worker.

Generates a small fake manga library (chapter folders with gray pages plus a
colour cover), then drives the worker three times and checks what it produced:

1. a dry run, which must report every page and write nothing at all
2. a real run with loose files
3. a real run with "CBZ per folder", which must produce one archive per
   chapter folder with every page inside it

Run it with the backend interpreter from the project root:

    backend\\python\\python.exe scripts\\selftest.py

Exit code 0 means everything passed. ``--keep`` leaves the temporary tree in
place so you can look at the output.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "janai" / "worker" / "worker.py"

DASH = "\u2014"

# A small rule set, so the rules path is exercised end to end. "auto" keeps it
# independent of whichever models happen to be installed.
RULES: list[dict] = [
    {"kind": "grayscale", "model": "auto", "auto_levels": True},
    {"kind": "colour", "model": "auto"},
]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def make_library(
    root: Path, chapters: int = 2, pages: int = 3, size: tuple[int, int] = (360, 520)
) -> None:
    """Write `chapters` folders of tiny PNG pages, one colour page each."""
    import cv2
    import numpy as np

    w, h = size
    for c in range(1, chapters + 1):
        folder = root / f"Chapter {c:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        for p in range(1, pages + 1):
            if p == 1:  # a colour cover, so both models get exercised
                img = np.zeros((h, w, 3), np.uint8)
                img[:, :, 0] = np.linspace(20, 240, w, dtype=np.uint8)[None, :]
                img[:, :, 1] = np.linspace(240, 20, h, dtype=np.uint8)[:, None]
                img[:, :, 2] = 128
            else:  # grayscale line art
                ramp = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
                gray = np.repeat(ramp, h, axis=0)
                gray[::16, :] = 0
                gray[:, ::24] = 255
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(folder / f"page-{p:03d}.png"), img)


def build_job(
    src: Path,
    out: Path,
    container: str,
    dry: bool,
    fmt: str = "png",
    rule_dicts: list[dict] | None = None,
) -> dict:
    return {
        "input": {"path": str(src), "mode": "bulk", "recursive": True, "include_archives": True},
        "output": {
            "dir": str(out),
            "container": container,
            "pattern": "{name}",
            "overwrite": True,
            "keep_structure": True,
        },
        "format": {"id": fmt, "options": {}},
        "upscale": {
            "mode": "scale",
            "scale": 2.0,
            "grayscale_convert": True,
            "auto_levels": True,
            "grayscale_threshold": 12,
            "grayscale_colour_percent": 0.25,
            "pre_downscale_height": 0,
            "skip_long_strips": False,
            "rules": list(rule_dicts or []),
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


# --------------------------------------------------------------------------- #
# running the worker
# --------------------------------------------------------------------------- #
def run_worker(job: dict, tmp: Path, label: str, verbose: bool) -> dict:
    """Run one job and collect its events by type."""
    path = tmp / f"job-{label}.json"
    path.write_text(json.dumps(job), encoding="utf-8")
    started = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, str(WORKER), "--job", str(path)],
        check=False,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    events: dict[str, list[dict]] = {}
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue
        events.setdefault(str(ev.get("type")), []).append(ev)
    if verbose:
        for kind, items in events.items():
            print(f"    {kind}: {len(items)}")
        for ev in events.get("log", []):
            print(f"    log[{ev.get('level')}]: {ev.get('message')}")
    if proc.returncode != 0:
        print(f"    worker exited {proc.returncode}")
        tail = (proc.stderr or "").strip().splitlines()[-6:]
        for line in tail:
            print(f"    stderr: {line}")
    events["_meta"] = [{"code": proc.returncode, "secs": round(time.perf_counter() - started, 2)}]
    return events


def check(name: str, ok: bool, detail: str = "") -> bool:
    # The dash lives outside the f-string: escapes inside one need Python 3.12.
    suffix = f" {DASH} {detail}" if detail else ""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{suffix}")
    return ok


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="JaNai Upscaler worker self-test")
    ap.add_argument("--chapters", type=int, default=2)
    ap.add_argument("--pages", type=int, default=3)
    ap.add_argument("--keep", action="store_true", help="keep the temporary tree")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only", choices=["dry", "files", "cbz"], help="run one case")
    args = ap.parse_args(argv)

    tmp = Path(tempfile.mkdtemp(prefix="janai-selftest-"))
    src = tmp / "library"
    total = args.chapters * args.pages
    passed = True
    try:
        make_library(src, args.chapters, args.pages)
        print(f"fixtures: {total} pages in {args.chapters} chapters under {src}")

        if args.only in (None, "dry"):
            print("dry run (CBZ per folder)")
            out = tmp / "out-dry"
            ev = run_worker(
                build_job(src, out, "cbz", True, rule_dicts=RULES), tmp, "dry", args.verbose
            )
            done = (ev.get("done") or [{}])[0]
            files = ev.get("file") or []
            bundles = ev.get("bundle") or []
            passed &= check("exit code 0", ev["_meta"][0]["code"] == 0)
            passed &= check("marked as a dry run", bool(done.get("dry")))
            passed &= check("every page reported", len(files) == total, f"{len(files)}/{total}")
            passed &= check(
                "one archive planned per chapter", len(bundles) == args.chapters, f"{len(bundles)}"
            )
            passed &= check(
                "predicted sizes present", all(f.get("w") and f.get("h") for f in files)
            )
            passed &= check("nothing written to disk", not out.exists())
            print(f"  took {ev['_meta'][0]['secs']}s (no torch import)")

        if args.only in (None, "files"):
            print("real run (loose files)")
            out = tmp / "out-files"
            ev = run_worker(
                build_job(src, out, "files", False, rule_dicts=RULES), tmp, "files", args.verbose
            )
            done = (ev.get("done") or [{}])[0]
            written = sorted(out.rglob("*.png"))
            passed &= check("exit code 0", ev["_meta"][0]["code"] == 0)
            passed &= check(
                "no failures", int(done.get("failed") or 0) == 0, str(done.get("error") or "")
            )
            passed &= check("one file per page", len(written) == total, f"{len(written)}/{total}")
            passed &= check("pages were upscaled", all(p.stat().st_size > 0 for p in written))
            print(f"  took {ev['_meta'][0]['secs']}s")

        if args.only in (None, "cbz"):
            print("real run (CBZ per folder)")
            out = tmp / "out-cbz"
            ev = run_worker(build_job(src, out, "cbz", False), tmp, "cbz", args.verbose)
            done = (ev.get("done") or [{}])[0]
            archives = sorted(out.rglob("*.cbz"))
            leftovers = sorted(out.rglob("*.part"))
            entries = []
            for arc in archives:
                with ZipFile(arc) as zf:
                    entries.append(len(zf.namelist()))
            passed &= check("exit code 0", ev["_meta"][0]["code"] == 0)
            passed &= check(
                "no failures", int(done.get("failed") or 0) == 0, str(done.get("error") or "")
            )
            passed &= check(
                "one archive per chapter",
                len(archives) == args.chapters,
                ", ".join(a.name for a in archives),
            )
            passed &= check(
                "every page inside", entries == [args.pages] * args.chapters, str(entries)
            )
            passed &= check("no .part files left behind", not leftovers)
            passed &= check("bundle events emitted", len(ev.get("bundle") or []) == args.chapters)
            print(f"  took {ev['_meta'][0]['secs']}s")
    finally:
        if args.keep:
            print(f"kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    print("ALL PASS" if passed else "FAILURES")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
