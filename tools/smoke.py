#!/usr/bin/env python
"""Dependency-free checks for the parts that must never break.

Nothing here imports torch, pyvips or tkinter, so it runs on a bare Python
install in a second or two - which is what makes it usable in CI::

    python tools/smoke.py

Exit code 0 means every check passed. For the real end-to-end run (models,
encoders, GPU) use ``tools/selftest.py`` with the backend interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import runlog  # noqa: E402
from common import displays, formats, paths  # noqa: E402

FAILED: list[str] = []


def check(name: str, fn) -> None:
    """Run one check and keep going, so one failure does not hide the rest."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001 - report every failure, never raise
        FAILED.append(name)
        print(f"  FAIL  {name}: {exc.__class__.__name__}: {exc}")
    else:
        print(f"  ok    {name}")


def output_formats() -> None:
    assert formats.FORMAT_IDS, "no output formats registered"
    everything = formats.all_defaults()
    for fid in formats.FORMAT_IDS:
        assert formats.fmt(fid) is not None, f"{fid} has no descriptor"
        assert fid in everything, f"{fid} is missing from all_defaults()"
        assert isinstance(formats.defaults(fid), dict), f"{fid} defaults are not a dict"
        assert isinstance(formats.merged(fid, None), dict), f"{fid} merged() is not a dict"
        assert isinstance(formats.save_kwargs(fid, None), dict), f"{fid} save_kwargs() is not a dict"
        summary = formats.summary(fid, None)
        assert isinstance(summary, str) and summary, f"{fid} has an empty summary"


def containers() -> None:
    assert formats.CONTAINER_IDS, "no output containers registered"
    for cid in formats.CONTAINER_IDS:
        assert formats.container(cid) is not None, f"{cid} has no descriptor"
        assert isinstance(formats.packs_archive(cid), bool), f"{cid} packs_archive() is not a bool"
    assert any(formats.packs_archive(cid) for cid in formats.CONTAINER_IDS), \
        "no container packs an archive, so CBZ output would be unreachable"


def display_presets() -> None:
    assert displays.DISPLAYS, "no display presets registered"
    assert displays.labels(), "no display labels"
    for d in displays.DISPLAYS:
        label = displays.label_for_id(d.id)
        assert label, f"{d.id} has no label"
        assert displays.id_for_label(label) == d.id, f"{d.id} does not round-trip through its label"
        if d.id == displays.CUSTOM:
            continue
        portrait = displays.size(d.id, True)
        landscape = displays.size(d.id, False)
        assert portrait and landscape, f"{d.id} has no size"
        assert sorted(portrait) == sorted(landscape), f"{d.id} landscape is not a rotation"
        assert portrait[0] <= portrait[1], f"{d.id} portrait is not the taller orientation"


def log_formatting() -> None:
    assert runlog.fmt_secs(0), "fmt_secs(0) is empty"
    assert runlog.fmt_secs(95.4), "fmt_secs(95.4) is empty"
    assert runlog.fmt_bytes(0), "fmt_bytes(0) is empty"
    assert runlog.fmt_bytes(1536), "fmt_bytes(1536) is empty"
    assert isinstance(runlog.fmt_ms(None), str), "fmt_ms(None) is not a string"
    assert isinstance(runlog.fmt_ms(1234), str), "fmt_ms(1234) is not a string"
    assert runlog.counter(1, 10), "counter(1, 10) is empty"
    long_name = "a-very-long-chapter-page-name-" * 5
    assert len(runlog.elide(long_name)) <= runlog.NAME_WIDTH, "elide() exceeded NAME_WIDTH"


def path_resolution() -> None:
    resolved = paths.resolve()
    for name in paths.LOCATIONS:
        assert hasattr(resolved, name), f"resolve() has no {name}"
    text = paths.report(resolved)
    assert isinstance(text, str) and text, "paths.report() produced nothing"


def main() -> int:
    print(f"smoke test in {ROOT}")
    check("output formats", output_formats)
    check("containers", containers)
    check("display presets", display_presets)
    check("log formatting", log_formatting)
    check("path resolution", path_resolution)
    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
