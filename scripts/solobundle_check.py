"""Live proof that a single chosen file names its own archive.

`gather_units` sets a file's `base` to its parent folder, because both
`keep_structure` and the resume key are measured from it. Both packaging modes
then read that base for the archive name, so converting
Downloads\\test_2.jpg into one CBZ produced "Downloads.cbz" -- named after a
folder the user never chose, and shared with every other file converted out of
that folder.

Checked against the real worker, for the run and for the preview, in both
packaging modes, with a folder input as the control:

  a chosen file   + one CBZ        -> test_2.cbz   (one archive, one page)
  a chosen file   + CBZ per folder -> test_2.cbz   (one archive, one page)
  a chosen folder + one CBZ        -> library.cbz  (unchanged)
  a chosen folder + CBZ per folder -> library.cbz  (unchanged)

The file fixture sits in a folder whose name differs from its own, which is the
whole point: if the two matched, the defect would be invisible.

Run: backend/python/python.exe scripts/solobundle_check.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "src" / "janai" / "worker" / "worker.py"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(name)


def page_bytes() -> bytes:
    import cv2
    import numpy as np

    w, h = 160, 240
    ramp = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    ok, enc = cv2.imencode(".png", np.repeat(ramp, h, axis=0))
    if not ok:
        raise SystemExit("could not encode the fixture page")
    return enc.tobytes()


def build_job(src: Path, out: Path, *, container: str, dry: bool) -> dict:
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
        "resume": False,
        "dry_run": dry,
    }


def run(src: Path, out: Path, tmp: Path, *, container: str, dry: bool, tag: str) -> list[dict]:
    job_path = tmp / f"job-{tag}.json"
    job_path.write_text(json.dumps(build_job(src, out, container=container, dry=dry)), "utf-8")
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


def archives_announced(events: list[dict]) -> list[str]:
    """Every .cbz a bundle event named, by basename, whatever key carries it."""
    names: list[str] = []
    for event in events:
        if event.get("type") != "bundle":
            continue
        names.extend(
            Path(value).name
            for value in event.values()
            if isinstance(value, str) and value.lower().endswith(".cbz")
        )
    return names


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="janai-solobundle-"))
    page = page_bytes()

    folder = tmp / "library"
    folder.mkdir()
    for name in ("page_01.png", "page_02.png"):
        (folder / name).write_bytes(page)

    # The parent folder is deliberately not named after the file.
    downloads = tmp / "Downloads"
    downloads.mkdir()
    chosen = downloads / "test_2.png"
    chosen.write_bytes(page)

    cases = (
        ("a chosen file, one CBZ", chosen, "cbz_single", "test_2.cbz", 1),
        ("a chosen file, CBZ per folder", chosen, "cbz", "test_2.cbz", 1),
        ("a chosen folder, one CBZ", folder, "cbz_single", "library.cbz", 2),
        ("a chosen folder, CBZ per folder", folder, "cbz", "library.cbz", 2),
    )

    for index, (label, src, container, expected, count) in enumerate(cases, 1):
        print(f"\n  {index}. {label}")
        out = tmp / f"out-{index}"
        events = run(src, out, tmp, container=container, dry=False, tag=f"real-{index}")
        announced = archives_announced(events)
        made = sorted(path.name for path in out.glob("*.cbz"))
        check(f"packs exactly {expected}", made == [expected], str(made))
        # A run that names the archive in the log must name the same one it wrote.
        check("the run says that name too", set(announced) <= {expected}, str(announced))
        if made:
            with zipfile.ZipFile(out / made[0]) as zf:
                entries = zf.namelist()
            check(f"holding {count} page(s)", len(entries) == count, str(entries))
            check(
                "page entries are named from the page, not the archive",
                all(Path(entry).stem != Path(expected).stem for entry in entries) or count == 1,
                str(entries),
            )
        else:
            check(f"holding {count} page(s)", False, "no archive to open")

        preview = tmp / f"plan-{index}"
        planned = run(src, preview, tmp, container=container, dry=True, tag=f"dry-{index}")
        announced = archives_announced(planned)
        check("the preview promises the same name", announced == [expected], str(announced))
        check(
            "and promises exactly one archive",
            len(announced) == 1,
            json.dumps(next((e for e in planned if e.get("type") == "done"), {})),
        )

    print("\n  " + ("ALL PASS" if not failures else f"FAILED: {', '.join(failures)}"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
