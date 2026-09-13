"""Benchmark the worker across tile settings.

The point of this tool is to keep "is it still fast?" an answerable question.
It runs the real worker on a real image once per tile setting and prints the
per-file time, so a tile-planner change can be judged instead of guessed.

Example:

    backend/python/python.exe scripts/bench.py \\
        --input C:\\path\\to\\page.jpg \\
        --model 4x-UltraSharpV2.safetensors --scale 4 \\
        --tiles auto,1024,768 --baseline 68897
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

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "janai" / "worker" / "worker.py"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:,.1f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.1f} GB"


def build_job(args: argparse.Namespace, tile: str, out_dir: Path) -> dict:
    return {
        "input": {
            "path": str(Path(args.input).resolve()),
            "mode": "single" if Path(args.input).is_file() else "folder",
            "recursive": True,
            "include_archives": True,
        },
        "output": {
            "dir": str(out_dir),
            "container": "files",
            "pattern": "{name}",
            "overwrite": True,
            "keep_structure": True,
        },
        "format": {"id": args.format, "options": {}},
        "upscale": {
            "mode": "scale",
            "scale": args.scale,
            "model": args.model,
            "model_gray": args.model_gray,
            "grayscale_convert": not args.no_grayscale,
            "auto_levels": True,
            "grayscale_threshold": 12,
            "grayscale_colour_percent": 0.25,
            "pre_downscale_height": 0,
            "skip_long_strips": False,
        },
        "perf": {
            "device": args.device,
            "use_fp16": not args.no_fp16,
            "tile": tile,
            "io_workers": 2,
            # Off by default, exactly like the app, so a bench reflects what
            # users actually get. Pass --cudnn to measure autotune instead.
            "cudnn_benchmark": bool(args.cudnn),
            "allow_tf32": True,
        },
    }


def run_once(job: dict, verbose: bool) -> dict:
    """Run the worker on one job and collect what the events tell us."""
    tmp = Path(tempfile.mkdtemp(prefix="janai-bench-job-"))
    job_path = tmp / "job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    result: dict = {"files": [], "elapsed": None, "device": "", "fp16": None, "errors": []}
    started = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(WORKER), "--job", str(job_path)],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(ROOT),
        )
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event.get("type")
            if kind == "start":
                result["device"] = event.get("device", "")
                result["fp16"] = event.get("fp16")
            elif kind == "file":
                result["files"].append(event)
                if event.get("error"):
                    result["errors"].append(str(event["error"]))
            elif kind == "done":
                result["elapsed"] = event.get("elapsed")
            elif kind == "log" and verbose:
                print(f"    [{event.get('level', 'info')}] {event.get('message', '')}")
        if proc.returncode != 0:
            result["errors"].append(f"worker exited {proc.returncode}: {proc.stderr[-400:]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    result["wall"] = time.time() - started
    return result


def main() -> int:
    p = argparse.ArgumentParser(description="Benchmark worker tile settings.")
    p.add_argument("--input", required=True, help="image file or folder")
    p.add_argument("--model", default="auto")
    p.add_argument("--model-gray", dest="model_gray", default="auto")
    p.add_argument("--scale", type=float, default=4.0)
    p.add_argument("--format", default="jxl")
    p.add_argument("--device", default="")
    p.add_argument("--tiles", default="auto", help="comma list, e.g. auto,1024,768,max")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument(
        "--copies",
        type=int,
        default=0,
        help="duplicate a single input N times into a temp folder, to check "
        "that the tile planner keeps its size across a whole chapter",
    )
    p.add_argument("--baseline", type=float, default=0.0, help="reference ms to compare against")
    p.add_argument("--warmup", action="store_true", help="discard one run before measuring")
    p.add_argument("--no-fp16", action="store_true")
    p.add_argument("--no-grayscale", action="store_true")
    p.add_argument(
        "--cudnn",
        action="store_true",
        help="turn cuDNN autotune on (off by default, as in the app)",
    )
    p.add_argument("--keep", action="store_true", help="keep upscaled output")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if not Path(args.input).exists():
        print(f"input not found: {args.input}")
        return 2

    pages_dir: Path | None = None
    if args.copies and Path(args.input).is_file():
        src = Path(args.input)
        pages_dir = Path(tempfile.mkdtemp(prefix="janai-bench-pages-"))
        for n in range(args.copies):
            shutil.copyfile(src, pages_dir / f"page-{n + 1:02d}{src.suffix}")
        args.input = str(pages_dir)
        print(f"benchmarking {args.copies} copies of {src.name} as a folder")

    out_dir = Path(tempfile.mkdtemp(prefix="janai-bench-out-"))
    tiles = [t.strip() for t in args.tiles.split(",") if t.strip()]
    rows: list[tuple[str, float, int, float, str]] = []
    try:
        if args.warmup:
            print("warmup run (not measured)")
            run_once(build_job(args, tiles[0], out_dir), args.verbose)
        for tile in tiles:
            for attempt in range(args.repeat):
                label = tile if args.repeat == 1 else f"{tile} #{attempt + 1}"
                print(f"running tile={tile} ...", flush=True)
                res = run_once(build_job(args, tile, out_dir), args.verbose)
                if res["errors"]:
                    for err in res["errors"]:
                        print(f"  ERROR {err}")
                    continue
                files = res["files"]
                if not files:
                    print("  no files processed")
                    continue
                total_ms = sum(float(f.get("ms") or 0) for f in files)
                pixels = sum(int(f.get("w") or 0) * int(f.get("h") or 0) for f in files)
                used = ", ".join(sorted({str(f.get("tile")) for f in files}))
                size = sum(int(f.get("bytes") or 0) for f in files)
                mp = pixels / 1e6 or 1.0
                rows.append((label, total_ms, pixels, total_ms / mp, used))
                print(
                    f"  {total_ms / 1000:.2f}s for {len(files)} file(s), "
                    f"{mp:.1f} MP out, tile used {used}, {human_bytes(size)}, "
                    f"device {res['device']} fp16={res['fp16']}"
                )

        if not rows:
            print("nothing measured")
            return 1
        print()
        print(f"{'setting':<12}{'time':>10}{'ms/MP':>10}{'tile used':>14}{'vs best':>10}")
        best = min(r[1] for r in rows)
        for label, ms, _pixels, per_mp, used in rows:
            delta = (ms / best - 1) * 100
            print(
                f"{label:<12}{ms / 1000:>9.2f}s{per_mp:>10.0f}{used:>14}"
                f"{('best' if ms == best else f'+{delta:.1f}%'):>10}"
            )
        if args.baseline:
            per_file = max(1, args.copies) if args.copies else 1
            best_each = best / per_file
            diff = (best_each / args.baseline - 1) * 100
            verdict = "faster" if diff < 0 else "slower"
            each = " per page" if per_file > 1 else ""
            print(
                f"\nbest {best_each / 1000:.2f}s{each} vs baseline "
                f"{args.baseline / 1000:.2f}s -> {abs(diff):.1f}% {verdict}"
            )
    finally:
        if pages_dir is not None:
            shutil.rmtree(pages_dir, ignore_errors=True)
        if not args.keep:
            shutil.rmtree(out_dir, ignore_errors=True)
        else:
            print(f"\noutput kept in {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
