#!/usr/bin/env python
"""Dependency-free checks for the parts that must never break.

Nothing here imports torch, pyvips or Qt, so it runs on a bare Python
install in a second or two - which is what makes it usable in CI::

    python scripts/smoke.py

Exit code 0 means every check passed. For the real end-to-end run (models,
encoders, GPU) use ``scripts/selftest.py`` with the backend interpreter.
"""

from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janai.app import runlog
from janai.core import displays, filetypes, formats, paths, presets, rules
from janai.worker import planning

FAILED: list[str] = []

#: A representative model set, so the rule checks do not depend on what this
#: machine happens to have installed. The shipped table only ever names files
#: that exist, so every rule check has to supply the files.
INSTALLED = [
    "2x_MangaJaNai_1200p_V1_ESRGAN_70k.pth",
    "2x_MangaJaNai_1920p_V1_ESRGAN_70k.pth",
    "4x_MangaJaNai_2048p_V1_ESRGAN_95k.pth",
    "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.safetensors",
    "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.safetensors",
]


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


def output_naming() -> None:
    """Two sources must never resolve onto one output path.

    Re-encoding is not injective: a.jpg and a.png both become a.png. The run
    reserves every path it hands out, and `unique_path` has to honour those
    reservations even for names that do not exist on disk yet, because the
    write is queued rather than immediate - the live proof is
    ``scripts/outname_check.py``, which needs a GPU; these are the same
    guarantees at the function level.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        first = out / "a.png"
        assert planning.unique_path(first) == first, "an unused name was renamed"

        taken = {planning.path_key(first)}
        second = planning.unique_path(first, taken)
        assert second != first, "a reserved name was handed out a second time"
        assert not second.exists(), "unique_path returned a path that already exists"

        taken.add(planning.path_key(second))
        third = planning.unique_path(first, taken)
        assert third not in (first, second), "a third collision reused an earlier name"

        # Windows treats A.png and a.png as one file, so the reservation is
        # case-folded; raw string comparison would lose a page on the platform
        # this app actually ships on.
        upper = out / "A.png"
        assert planning.path_key(upper) == planning.path_key(first), (
            "path_key is case-sensitive, so A.png and a.png could collide on Windows"
        )
        assert planning.unique_path(upper, taken) != upper, (
            "a reservation made under a.png did not cover A.png"
        )

        # An existing file still wins, so output from an earlier run is never
        # handed out as if it were free.
        first.write_bytes(b"x")
        assert planning.unique_path(first) != first, "an existing file was handed out"

    # The collapse starts in the pattern: the default keeps only the stem, and
    # {parent} maps a whole folder onto one name. Both are deliberate, which is
    # why de-dup belongs at the path and not in the pattern.
    jpg, png = Path("/src/ch/a.jpg"), Path("/src/ch/a.png")
    for pattern in ("{name}", "{parent}"):
        one = planning.format_name(pattern, jpg, 1, 9)
        two = planning.format_name(pattern, png, 2, 9)
        assert one == two, f"{pattern} no longer collides, so the reservation proves nothing"


def extension_sets() -> None:
    """One definition of what counts as a page, read by both processes.

    The GUI's pre-run scan and the worker's real walk used to keep private
    copies of the same suffix sets - identical only by luck, with nothing
    holding them in step (F6/F11). Adding a format on one side would have made
    the preview promise "12 images" and the run convert 11.

    Two guards, because neither alone is enough: the importers must resolve to
    the *same object* (a re-added local copy changes identity), and no module
    may quietly grow a second definition somewhere the identity check does not
    look. The source scan covers the GUI without importing Qt, so this stays
    dependency-free.
    """
    # Read the module namespace rather than attributes: after the move these
    # names are imports, not part of planning's public surface, so attribute
    # access is a re-export error under no_implicit_reexport - and it is that
    # namespace a re-added private copy would rebind anyway.
    worker_ns = vars(planning)
    assert worker_ns["IMAGE_EXTS"] is filetypes.IMAGE_EXTS, (
        "the worker no longer reads the shared page-extension set"
    )
    assert worker_ns["ARCHIVE_EXTS"] is filetypes.ARCHIVE_EXTS, (
        "the worker no longer reads the shared archive-extension set"
    )

    pkg = ROOT / "src" / "janai"
    owners = {
        "IMAGE_EXTS": pkg / "core" / "filetypes.py",
        "ARCHIVE_EXTS": pkg / "core" / "filetypes.py",
        "MODEL_EXTS": pkg / "core" / "paths.py",
    }
    for name, owner in owners.items():
        pattern = rf"^{name}\s*="
        # Without this the scan below would pass vacuously if a set were renamed.
        assert re.search(pattern, owner.read_text(encoding="utf-8"), re.MULTILINE), (
            f"{owner.name} no longer defines {name}, so the scan proves nothing"
        )
        for path in sorted(pkg.rglob("*.py")):
            if path == owner:
                continue
            if re.search(pattern, path.read_text(encoding="utf-8"), re.MULTILINE):
                raise AssertionError(f"{path.relative_to(ROOT)} defines a second {name}")

    reads = {
        pkg / "app" / "input_panel.py": "from janai.core.filetypes import",
        pkg / "worker" / "models.py": "from janai.core.paths import MODEL_EXTS",
    }
    for path, line in reads.items():
        assert line in path.read_text(encoding="utf-8"), (
            f"{path.relative_to(ROOT)} no longer reads the shared set"
        )

    # The Qt open-dialog filter was a third hand-written copy of this same
    # knowledge, and it had already drifted (F33). A literal glob list cannot
    # be held in step by identity, so the shape itself is forbidden here. That
    # the derived string matches the sets is proved in uicheck.py, which may
    # import Qt; this check must stay dependency-free.
    panel = (pkg / "app" / "input_panel.py").read_text(encoding="utf-8")
    assert re.search(r"^FILE_FILTER\s*=", panel, re.MULTILINE), (
        "input_panel.py no longer defines FILE_FILTER, so this scan proves nothing"
    )
    globs = sorted(set(re.findall(r"\*\.[A-Za-z0-9]+", panel)))
    assert not globs, f"input_panel.py hard-codes page suffixes again: {globs}"


def orphan_bytecode() -> None:
    """A `__pycache__` entry whose source is gone is rot, not a cache.

    Deleting `app/dnd.py` (the Tk drag-and-drop shim Qt's `DropZone` replaced)
    and the pre-split `app/ui.py` left 165 KB of bytecode behind, and AGENTS.md
    ended up documenting one of the two as a known leftover - a document wrong
    about its own tree (F34). Python cannot import these, because a sourceless
    import has to sit in the source location rather than in `__pycache__`, so
    the cost is not behaviour: it is that `grep` keeps reporting modules which
    no longer exist, and a reader trusts it.

    CI checks out a clean tree and finds nothing here. That is the point - this
    guards working copies, which is where the rot actually accumulates.
    """
    orphans: list[str] = []
    for root in (ROOT / "src", ROOT / "scripts"):
        for cache in sorted(root.rglob("__pycache__")):
            for pyc in sorted(cache.glob("*.pyc")):
                # foo.cpython-313.pyc belongs to foo.py one directory up.
                stem = pyc.name.split(".", 1)[0]
                if not (cache.parent / f"{stem}.py").exists():
                    orphans.append(pyc.relative_to(ROOT).as_posix())
    assert not orphans, f"bytecode whose source is gone (delete the files): {orphans}"


def page_entries() -> None:
    """An archive entry is a page only if it is really an image.

    A zip written on macOS carries an AppleDouble sidecar per file, with the
    page's own extension. Treating those as pages makes a healthy chapter
    report failed pages and inflates its progress total. The live proof is
    ``scripts/archname_check.py``, which needs the imaging stack; these are the
    same guarantees at the function level.
    """
    for name in ("page-001.jpg", "sub/page-002.png", ".hidden/cover.png"):
        assert planning.is_page_entry(name), f"{name} should count as a page"

    for name in (
        "__MACOSX/._page-001.jpg",
        "__MACOSX/sub/._page-002.jpg",
        "sub/._page-002.jpg",
        "._cover.png",
        "ComicInfo.xml",
        "pages/",
    ):
        assert not planning.is_page_entry(name), f"{name} should not count as a page"

    # Over-filtering would silently drop real pages, which is worse than the
    # junk it is meant to remove.
    for name in ("page._final.jpg", "__MACOSX_fanbook/page-001.jpg"):
        assert planning.is_page_entry(name), f"{name} was over-filtered"


def entry_name_decoding() -> None:
    """An entry name stored without the UTF-8 flag must be recovered, or left.

    `zipfile` decodes an unflagged name as cp437, so a Japanese page name
    arrives as mojibake - and `handle_archive` writes the name it is handed into
    the output CBZ, which makes the corruption permanent. The recovery is
    self-proving (cp437 maps all 256 bytes, and they only decode as UTF-8 if
    they were UTF-8), so it must fire for real UTF-8 and never on a guess. The
    live proof is ``scripts/archname_check.py``, which needs the imaging stack.
    """
    true_name = "\u7b2c01\u8a71.jpg"

    kept = planning.decode_entry_name(true_name, utf8_flag=True)
    assert kept == true_name, "a flagged name was rewritten"

    mojibake = true_name.encode("utf-8").decode("cp437")
    recovered = planning.decode_entry_name(mojibake, utf8_flag=False)
    assert recovered == true_name, "unflagged UTF-8 was not recovered"

    # Shift-JIS needs a detector or a user-set codepage; guessing would rename
    # pages silently, so it has to survive untouched.
    sjis = true_name.encode("shift_jis").decode("cp437")
    left = planning.decode_entry_name(sjis, utf8_flag=False)
    assert left == sjis, "a name that is not UTF-8 was altered on a guess"

    for flag in (True, False):
        plain = planning.decode_entry_name("page-001.jpg", utf8_flag=flag)
        assert plain == "page-001.jpg", "an ASCII name was rewritten"


def rule_engine() -> None:
    installed = INSTALLED
    working = rules.default_working_set(installed)
    assert working, "the default working set is empty"
    assert not rules.default_working_set([]), (
        "the shipped table named a model with nothing installed"
    )
    for rule in working:
        assert rules.Rule.from_dict(rule.to_dict()) == rule, f"{rule.describe()} did not round-trip"
        assert len(rule.columns()) == 4, f"{rule.describe()} did not produce four columns"
        assert len(rule.cells()) == 5, f"{rule.describe()} did not produce five table cells"
        assert rule.enabled_mark(), f"{rule.describe()} has no on/off indicator"
        # The table is the only chooser, so nothing may ship unresolved.
        assert not rule.is_auto, f"{rule.describe()} left its model on auto"

    ruleset = rules.RuleSet(working)
    gray = ruleset.match(gray=True, width=1350, height=1920, scale=2.0)
    assert gray is not None, "a 1920p grayscale page matched no rule"
    assert "1920p" in gray.model, f"a 1920p page went to {gray.model}"

    # Settings from an older build carry "auto" rows; they have to resolve to a
    # real file on load, or the table would still be hiding the choice.
    legacy = rules.Rule(kind=rules.GRAYSCALE, scale=2.0, height="1920")
    assert legacy.is_auto, "a rule with no model should read as auto"
    resolved, notes = rules.materialise([legacy], installed)
    assert resolved and not any(r.is_auto for r in resolved), "materialise() kept an auto rule"
    assert all(r.model in installed for r in resolved), "materialise() named a missing model"
    assert notes, "materialise() resolved a rule without reporting it"
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
    data["upscale"]["rules"] = rules.default_dicts(INSTALLED)
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
    check("output naming", output_naming)
    check("extension sets", extension_sets)
    check("orphan bytecode", orphan_bytecode)
    check("page entries", page_entries)
    check("entry name decoding", entry_name_decoding)
    check("rule engine", rule_engine)
    check("presets", preset_round_trip)
    if FAILED:
        print(f"\n{len(FAILED)} check(s) failed: {', '.join(FAILED)}")
        return 1
    print("\nall checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
