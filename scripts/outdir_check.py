"""Proof that a job with no ``output.dir`` is refused, not silently redirected (F31).

``run_job`` built the destination as ``Path(str(outp.get("dir") or ""))`` and then
guarded it with ``if not out_dir``. ``Path("")`` normalises to ``Path(".")``,
which is **truthy**, so that guard could never fire: a job that forgot
``output.dir`` did not fail with the intended message, it quietly wrote every
upscaled page into the worker's current directory. When the GUI launches the
worker that directory is the app folder, so the pages landed somewhere the user
never chose and the run still reported success.

Asserted:
  * ``[control]`` a dry run with a real output dir still succeeds -- this must
    pass before *and* after the fix, or the guard is rejecting valid jobs
  * ``[missing]`` an empty ``output.dir`` is refused: non-zero exit, ``done.ok``
    false, and an error that actually names the missing setting
  * ``[blank]`` a whitespace-only ``output.dir`` is refused the same way; a path
    of spaces is exactly as unusable as an empty one, and Windows trims it
  * ``[harm]`` the same job run for real writes **nothing** into the process's
    current directory -- the assertion that measures F31's true cost, and the
    one that fails loudest against the unfixed tree

Every worker here is spawned with ``cwd`` set to a throwaway temp directory, so
the unfixed behaviour dirties that directory instead of the repository.

Needs the imaging stack (and a model, for ``[harm]``), so this gate is
local-only by design.

Run: backend/python/python.exe scripts/outdir_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "janai" / "worker" / "worker.py"

# The setting the guard is about. Asserted as a substring of the reported error
# so the message stays useful to a human, not just non-empty.
SETTING = "output.dir"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def page() -> bytes:
    """One small readable page; enough to get a real run as far as a write."""
    import cv2
    import numpy as np

    w, h = 240, 320
    ramp = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    ok, enc = cv2.imencode(".png", np.repeat(ramp, h, axis=0))
    if not ok:
        raise SystemExit("could not encode the fixture page")
    return enc.tobytes()


def build_job(src: Path, out_dir: str, *, dry: bool) -> dict:
    """A normal job, with ``output.dir`` passed through verbatim as a string.

    The output dir is deliberately NOT a Path here: the whole point is what the
    worker does with the raw value that arrives over the JSON wire.
    """
    return {
        "input": {"path": str(src), "mode": "bulk", "recursive": True, "include_archives": True},
        "output": {
            "dir": out_dir,
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


def run(tmp: Path, src: Path, out_dir: str, label: str, *, dry: bool) -> tuple[int, dict, Path]:
    """Run one real worker in its own throwaway cwd; return rc, done payload, cwd."""
    job_path = tmp / f"job-{label}.json"
    job_path.write_text(json.dumps(build_job(src, out_dir, dry=dry)), encoding="utf-8")
    cwd = tmp / f"cwd-{label}"
    cwd.mkdir()
    proc = subprocess.run(
        [sys.executable, str(WORKER), "--job", str(job_path)],
        check=False,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    done: dict = {}
    for line in (proc.stdout or "").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # The event key is "type" and payloads are flat; see events.py emit().
        if event.get("type") == "done":
            done = event
    return proc.returncode, done, cwd


def written(cwd: Path) -> list[str]:
    """Anything the worker left behind in its current directory."""
    return sorted(p.name for p in cwd.rglob("*") if p.is_file())


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="outdir-") as raw:
        tmp = Path(raw)
        src = tmp / "src"
        src.mkdir()
        (src / "page-001.png").write_bytes(page())

        print("[control] a real output dir is still accepted")
        good = tmp / "out"
        rc, done, cwd = run(tmp, src, str(good), "control", dry=True)
        check("exit code 0", rc == 0, f"rc={rc}")
        check("done.ok", bool(done.get("ok")), f"error={done.get('error', '')!r}")
        check("no guard complaint", SETTING not in str(done.get("error") or ""))
        check("cwd untouched", written(cwd) == [], f"{written(cwd)}")

        print("[missing] output.dir is absent")
        rc, done, cwd = run(tmp, src, "", "missing", dry=True)
        check("non-zero exit", rc != 0, f"rc={rc}")
        check("done.ok is false", done.get("ok") is False, f"ok={done.get('ok')!r}")
        check(
            "error names the setting",
            SETTING in str(done.get("error") or ""),
            f"error={done.get('error', '')!r}",
        )

        print("[blank] output.dir is whitespace only")
        rc, done, cwd = run(tmp, src, "   ", "blank", dry=True)
        check("non-zero exit", rc != 0, f"rc={rc}")
        check(
            "error names the setting",
            SETTING in str(done.get("error") or ""),
            f"error={done.get('error', '')!r}",
        )

        print("[harm] a real run must not fall back to the current directory")
        rc, done, cwd = run(tmp, src, "", "harm", dry=False)
        check("non-zero exit", rc != 0, f"rc={rc}")
        check("nothing written to cwd", written(cwd) == [], f"{written(cwd)}")
        check(
            "no pages processed", int(done.get("processed") or 0) == 0, f"{done.get('processed')}"
        )

    print()
    if failures:
        print(f"{len(failures)} FAIL: " + ", ".join(failures))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
