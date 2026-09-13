#!/usr/bin/env python
"""Dependency-free checks for the parts that must never break.

Nothing here imports torch, pyvips or tkinter, so it runs on a bare Python
install in a second or two - which is what makes it usable in CI::

    python scripts/smoke.py

Exit code 0 means every check passed. For the real end-to-end run (models,
encoders, GPU) use ``scripts/selftest.py`` with the backend interpreter.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janai.app import runlog
from janai.core import displays, formats, paths, presets, rules

FAILED: list[str] = []


def check(name: str, fn) -> None:
    """Run one check and keep going, so one failure does not hide the rest."""
    try:
        fn()
    except Exception as exc:  # report every failure, never raise
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
        assert isinstance(formats.save_kwargs(fid, None), dict), (
            f"{fid} save_kwargs() is not a dict"
        )
        summary = formats.summary(fid, None)
        assert isinstance(summary, str) and summary, f"{fid} has an empty summary"


def containers() -> None:
    assert formats.CONTAINER_IDS, "no output containers registered"
    for cid in formats.CONTAINER_IDS:
        assert formats.container(cid) is not None, f"{cid} has no descriptor"
        assert isinstance(formats.packs_archive(cid), bool), f"{cid} packs_archive() is not a bool"
    assert any(formats.packs_archive(cid) for cid in formats.CONTAINER_IDS), (
        "no container packs an archive, so CBZ output would be unreachable"
    )


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


def rule_engine() -> None:
    installed = [
        "2x_MangaJaNai_1200p_V1_ESRGAN_70k.pth",
        "2x_MangaJaNai_1920p_V1_ESRGAN_70k.pth",
        "4x_MangaJaNai_2048p_V1_ESRGAN_95k.pth",
        "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.safetensors",
        "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.safetensors",
    ]
    working = rules.default_working_set(installed)
    assert working, "the default working set is empty"
    for rule in working:
        assert rules.Rule.from_dict(rule.to_dict()) == rule, f"{rule.describe()} did not round-trip"
        assert len(rule.columns()) == 4, f"{rule.describe()} did not produce four columns"

    ruleset = rules.RuleSet(working)
    gray = ruleset.match(gray=True, width=1350, height=1920, scale=2.0)
    assert gray is not None, "a 1920p grayscale page matched no rule"
    assert gray.is_auto or "1920p" in gray.model, f"a 1920p page went to {gray.model}"
    assert ruleset.match(gray=False, width=1920, height=1080, scale=4.0) is not None, (
        "a 4x colour page matched no rule"
    )

    # A rule that names a size has to win over an "any" rule sitting above it.
    catch_all = rules.Rule(kind=rules.GRAYSCALE, model="any.pth")
    sized = rules.Rule(kind=rules.GRAYSCALE, height="1920", model="sized.pth")
    won = rules.RuleSet([catch_all, sized]).match(gray=True, width=1350, height=1920, scale=2.0)
    assert won is not None and won.model == "sized.pth", "an unsized rule beat a sized rule"

    for text, expected in (
        ("1920", (1920, 1920)),
        ("1600-1920", (1600, 1920)),
        ("1985-", (1985, 0)),
        ("-1250", (0, 1250)),
        ("any", (0, 0)),
    ):
        assert rules.parse_dim(text) == expected, f"parse_dim({text!r}) is wrong"
        assert rules.parse_dim(rules.dim_spec(*expected)) == expected, (
            f"{text!r} did not survive dim_spec()"
        )

    assert rules.problems(rules.Rule(kind=rules.COLOUR, auto_levels=True)), (
        "auto-levels on a colour rule went unreported"
    )
    assert rules.problems(rules.Rule(model="4x_missing.pth", scale=2.0), installed), (
        "a missing 4x model on a 2x rule went unreported"
    )


def preset_round_trip() -> None:
    from janai.app.state import defaults  # the settings shape lives with the app

    data = defaults()
    data["output"]["dir"] = "/somewhere/private"
    data["perf"]["device"] = "cuda:1"
    data["upscale"]["rules"] = rules.default_dicts()
    preset = presets.build("Manga 2x", data, app_version="test")
    kept = preset["settings"]
    assert presets.summary(preset), "a preset produced no summary"
    assert "dir" not in kept.get("output", {}), "a preset carried an output folder"
    assert "device" not in kept.get("perf", {}), "a preset carried a device"
    assert kept["upscale"]["rules"], "a preset dropped the rules"

    with tempfile.TemporaryDirectory() as tmp:
        written = presets.write(Path(tmp) / presets.filename("Manga 2x"), preset)
        assert written.is_file(), "presets.write() wrote nothing"
        again = presets.read(written)
        assert again["settings"] == kept, "a preset did not survive a write and a read"

        fresh = defaults()
        changed = presets.apply(fresh, again)
        assert changed, "applying a preset changed nothing"
        assert len(fresh["upscale"]["rules"]) == len(kept["upscale"]["rules"]), (
            "applying a preset lost rules"
        )
        assert fresh["output"]["dir"] == defaults()["output"]["dir"], (
            "applying a preset overwrote the output folder"
        )

        junk = Path(tmp) / "junk.json"
        junk.write_text('{"hello": true}', encoding="utf-8")
        try:
            presets.read(junk)
        except presets.PresetError:
            pass
        else:
            raise AssertionError("a junk file was accepted as a preset")


def main() -> int:
    print(f"smoke test in {ROOT}")
    check("output formats", output_formats)
    check("containers", containers)
    check("display presets", display_presets)
    check("log formatting", log_formatting)
    check("path resolution", path_resolution)
    check("rule engine", rule_engine)
    check("presets", preset_round_trip)
    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
