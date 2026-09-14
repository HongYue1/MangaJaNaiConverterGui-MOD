"""Live proof for the dry-run colour probe (F7).

`probe_image` takes the size from the header and the colour verdict from a
thumbnail, wrapping only the thumbnail in a broad `except`. A page whose header
parses but whose pixels do not decode therefore yields gray=None, which
`dry_run` emits as `gray=False` - i.e. "colour" - and routes to the colour
model. Measured pre-fix: score/colour come out 0.0, byte-identical to a genuine
grayscale page, so nothing in the payload OR the log distinguishes "measured"
from "never measured".

Fixtures, all in one folder, one real dry run over them:
  colour.png     genuinely coloured          -> gray false, score > 0 (measured)
  gray.png       genuinely grayscale         -> gray true
  truncated.png  signature + IHDR only       -> header parses, pixels cannot
  badidat.png    IHDR intact, IDAT scrambled -> header parses, pixels cannot

Both undecodable fixtures are built from the GRAYSCALE page, so the preview
calling them colour is demonstrably wrong, not merely unknown.

Run: backend/python/python.exe scripts/dryprobe_check.py
Add --compare to also run the same fixtures for real and print what the run
actually does with them, which is how F7's user-visible impact was measured.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "janai" / "worker" / "worker.py"

HEADER_BYTES = 80  # PNG signature (8) + IHDR (25) + the start of the first IDAT
UNDECODABLE = ("truncated.png", "badidat.png")

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def fixtures() -> dict[str, bytes]:
    import cv2
    import numpy as np

    w, h = 240, 320
    ramp = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    flat = np.repeat(ramp, h, axis=0)

    colour = np.zeros((h, w, 3), dtype=np.uint8)
    colour[:, :, 0] = 30
    colour[:, :, 1] = 90
    colour[:, :, 2] = 220

    ok_c, enc_colour = cv2.imencode(".png", colour)
    ok_g, enc_gray = cv2.imencode(".png", flat)
    if not (ok_c and ok_g):
        raise SystemExit("could not encode the fixture pages")

    good = enc_gray.tobytes()
    bad_idat = bytearray(good)
    at = bad_idat.find(b"IDAT")
    if at < 0:
        raise SystemExit("fixture has no IDAT chunk")
    bad_idat[at + 4 : at + 28] = b"\xde\xad\xbe\xef" * 6

    return {
        "colour.png": enc_colour.tobytes(),
        "gray.png": good,
        "truncated.png": good[:HEADER_BYTES],
        "badidat.png": bytes(bad_idat),
    }


def build_job(src: Path, out: Path, *, dry: bool) -> dict:
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
        "dry_run": dry,
    }


def run_job(src: Path, out: Path, tmp: Path, *, dry: bool) -> tuple[int, list[dict]]:
    job_path = tmp / ("job-dry.json" if dry else "job-real.json")
    job_path.write_text(json.dumps(build_job(src, out, dry=dry)), encoding="utf-8")
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
    return proc.returncode, events


def main() -> int:
    compare = "--compare" in sys.argv
    tmp = Path(tempfile.mkdtemp(prefix="janai-dryprobe-"))
    src = tmp / "library"
    src.mkdir()
    for name, data in fixtures().items():
        (src / name).write_bytes(data)

    code, events = run_job(src, tmp / "out-dry", tmp, dry=True)
    files = [e for e in events if e.get("type") == "file"]
    logs = [e for e in events if e.get("type") == "log"]
    done = next((e for e in events if e.get("type") == "done"), {})
    by_name = {Path(str(e.get("path"))).name: e for e in files}

    print(f"\n  dry: exit={code} events={len(events)} file={len(files)}")
    print(f"  done={json.dumps(done)}\n")

    check("dry run exits 0", code == 0, str(code))
    check("every page got a file event", len(files) == 4, f"{len(files)} event(s)")
    check("genuine grayscale reads gray", by_name.get("gray.png", {}).get("gray") is True)
    colour_ev = by_name.get("colour.png", {})
    check(
        "genuine colour reads colour, with a measured score",
        colour_ev.get("gray") is False and float(colour_ev.get("score") or 0) > 0,
        f"gray={colour_ev.get('gray')} score={colour_ev.get('score')}",
    )

    # The heart of F7: a page whose sample could not be taken must not pass
    # itself off as a measured verdict.
    for name in UNDECODABLE:
        ev = by_name.get(name, {})
        said = f"gray={ev.get('gray')} score={ev.get('score')} model={ev.get('model')!r}"
        print(f"  probe {name:<16} {said}")
        explained = [
            e for e in logs if e.get("level") in {"warn", "error"} and name in str(e.get("message"))
        ]
        check(
            f"{name}: the unmeasurable verdict is explained in the log",
            bool(explained),
            str([str(e.get("message"))[:90] for e in explained]),
        )

    if compare:
        real_code, real_events = run_job(src, tmp / "out-real", tmp, dry=False)
        real_files = [e for e in real_events if e.get("type") == "file"]
        real_done = next((e for e in real_events if e.get("type") == "done"), {})
        print(f"\n  real: exit={real_code}")
        print(f"  done={json.dumps(real_done)}")
        for ev in real_files:
            nm = Path(str(ev.get("path"))).name
            outcome = ev.get("error") or f"ok gray={ev.get('gray')} model={ev.get('model')}"
            print(f"  run   {nm:<16} {str(outcome)[:110]}")
        print(
            "\n  IMPACT: the dry run promised"
            f" processed={done.get('processed')} failed={done.get('failed')};"
            f" the run delivered processed={real_done.get('processed')}"
            f" failed={real_done.get('failed')}"
        )

    print("\n  " + ("ALL PASS" if not failures else f"FAILED: {', '.join(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
