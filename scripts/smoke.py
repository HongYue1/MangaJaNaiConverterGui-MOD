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
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from janai.app import runlog
from janai.core import displays, filetypes, formats, paths, presets, rules
from janai.worker import control, models, planning, resume, runtime

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


def dependency_pins() -> None:
    """Both dependency lists must pin exactly, and must not disagree.

    `setup.cmd` feeds `requirements.txt` to uv, and that file's own header says
    it mirrors `backend/src/pyproject.toml` "so the upscaling results stay
    identical". Neither half of that was enforced (F37). `pillow>=11.0.0` was
    the one line that did not pin, so a setup run today installs Pillow 12.3.0
    while this machine has 11.3.0 - a floating version in the decode and ICC
    path, which is exactly where a silent pixel change comes from. And nothing
    compared the two files, so bumping one and forgetting the other drifts in
    silence.

    The shared-pin count is asserted too: if the parsing below ever stops
    matching, the comparison would pass by finding nothing to compare.
    """

    def normalize(name: str) -> str:
        # PEP 503: spandrel_extra_arches and Spandrel-Extra-Arches are one name.
        return re.sub(r"[-_.]+", "-", name).lower()

    pinned = re.compile(r"([A-Za-z0-9._-]+)==([A-Za-z0-9._+!-]+)")
    unpinned: list[str] = []
    wanted: dict[str, str] = {}
    for raw in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        found = pinned.fullmatch(line)
        if found is None:
            unpinned.append(line)
            continue
        wanted[normalize(found.group(1))] = found.group(2)
    assert not unpinned, f"requirements.txt lines that do not pin with ==: {unpinned}"

    toml = (ROOT / "backend" / "src" / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"\ndependencies = \[(.*?)\n\]", toml, re.DOTALL)
    assert block is not None, "backend/src/pyproject.toml lost its dependencies array"
    mirror: dict[str, str] = {normalize(n): v for n, v in pinned.findall(block.group(1))}

    shared = sorted(wanted.keys() & mirror.keys())
    assert len(shared) >= 10, f"only {len(shared)} shared pins parsed - the parser broke"
    drift = [
        f"{name}: requirements.txt {wanted[name]} vs pyproject.toml {mirror[name]}"
        for name in shared
        if wanted[name] != mirror[name]
    ]
    assert not drift, f"the two dependency lists disagree: {drift}"


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


def _assert_free_vram_clamp(text: str) -> None:
    """Every accelerator branch of the estimator must clamp its budget to free VRAM.

    Split out from the file read so a negative control can hand this the
    pre-fix bytes straight from git and prove the guard still bites.
    """
    # Comments name mem_get_info() too - including the one explaining this very
    # clamp - so count call sites in code only, never in prose. An earlier
    # version counted raw string occurrences and miscounted its own comment.
    sites: list[int] = []
    for number, line in enumerate(text.splitlines(), 1):
        if "mem_get_info(" in line.split("#", 1)[0]:
            sites.append(number)
    assert sites, "no mem_get_info() call site found; has the estimator moved?"
    assert "_free, total = mem_info" not in text, (
        "a mem_get_info() branch discards `free` and budgets from the card total"
    )
    clamped = text.count("total = min(total, free)")
    assert clamped == len(sites), (
        f"mem_get_info() is called at lines {sites} but only {clamped} branch(es) "
        "clamp the budget to free VRAM"
    )


def vram_budget_clamp() -> None:
    """The vendored tile estimator must budget from FREE VRAM, not card size.

    ``mem_get_info()`` returns ``(free, total)``. Discarding ``free`` budgets a
    share of the whole device even when most of it is already held, and this
    estimator is only reached when no proven tile exists - precisely when VRAM
    is scarce. Measured on a 6 GB card: at >=75% held, budgeting from total
    chose tile 512 where only 256 fits. Counting the call sites means a new
    accelerator branch cannot be added without the clamp.
    """
    src = (
        ROOT
        / "backend"
        / "src"
        / "packages"
        / "chaiNNer_pytorch"
        / "pytorch"
        / "processing"
        / "upscale_image.py"
    )
    assert src.is_file(), f"the vendored tile estimator is missing: {src}"
    _assert_free_vram_clamp(src.read_text(encoding="utf-8"))


def _assert_oom_arm_frees(text: str) -> None:
    """The OOM recovery arm must free the tile and report what goes wrong.

    Split out from the file read so a negative control can hand this the
    pre-fix bytes straight from git and prove the guard still bites.
    """
    marker = "except RuntimeError as e:"
    assert marker in text, "the OOM recovery arm has moved; re-point this guard"
    # Scan code only: the comment at that site names the copy it forbids, and a
    # guard that reads its own documentation as code miscounts - which is
    # exactly what _assert_free_vram_clamp did before it was fixed.
    arm = "\n".join(line.split("#", 1)[0] for line in text.split(marker, 1)[1].splitlines())
    assert ".cpu()" not in arm, (
        "the OOM recovery arm copies the tile to host RAM while out of memory"
    )
    assert "except Exception" not in arm, (
        "the OOM recovery arm swallows exceptions raised while freeing memory"
    )
    assert "del input_tensor" in arm, "the OOM recovery arm no longer releases the input tensor"


def oom_recovery_frees_without_copying() -> None:
    """OOM recovery must release the tile, not copy it back to host RAM.

    ``input_tensor.detach().cpu()`` asked the host for a whole tile's worth of
    RAM - and synchronised the device to do it - at the exact moment an
    allocation had just failed, then threw the result away. ``del`` plus
    ``gc.collect()`` plus the cache-empty call are what actually free the
    accelerator. The copy was also wrapped in ``except Exception: pass``, so
    anything that went wrong while freeing memory vanished silently. Both read
    like cleanup, which is why they need a guard and not just a comment.
    """
    src = ROOT / "backend" / "src" / "nodes" / "impl" / "pytorch" / "auto_split.py"
    assert src.is_file(), f"the vendored OOM recovery is missing: {src}"
    _assert_oom_arm_frees(src.read_text(encoding="utf-8"))


def _assert_job_payload(job: dict[str, Any]) -> None:
    """Assert a driver payload carries everything ``worker/job.py`` reads.

    job.py reads ``input``/``output``/``format``/``upscale``/``perf`` off the
    wire and defaults whatever is missing, so a driver that drops a section
    does not fail loudly - it quietly runs a different job than the one asked
    for. The empty ``output.dir`` case (F31) wrote every page into the worker's
    own folder and still reported success, which is why this is asserted
    rather than trusted.
    """
    for section in ("input", "output", "format", "upscale", "perf"):
        assert isinstance(job.get(section), dict), f"the payload has no {section} section"
    assert str(job["input"]["path"]).strip(), "the payload carries an empty input.path"
    assert str(job["output"]["dir"]).strip(), "the payload carries an empty output.dir"
    assert job["upscale"].get("rules"), "the payload carries no rules, so no model would run"
    fid = str(job["format"]["id"])
    assert fid in formats.FORMATS, f"the payload names an unknown format: {fid}"


def headless_driver() -> None:
    """The driver mirrors the GUI's payload, and never drags Qt in with it.

    Qt is the load-bearing half: ``janai.cli`` reaches the worker through
    ``janai.app.runner``, so one accidental PySide6 import anywhere in that
    chain would make the driver unusable on exactly the headless machines that
    want it. Importing it here and inspecting ``sys.modules`` is the only way
    to catch that, because the import itself is what does the damage.
    """
    from janai import cli
    from janai.app.state import defaults

    assert "PySide6" not in sys.modules, "importing the headless driver pulled in Qt"
    data = defaults()
    # The shipped table is seeded from the installed models, so use the fixed
    # set above rather than whatever this machine happens to have.
    data["upscale"]["rules"] = rules.default_dicts(INSTALLED)
    src = ROOT / "src"
    out = cli.resolve_out_dir(src, "", data)
    # Beside a folder input, not inside it: output written into the tree being
    # scanned is read back as input by the next run, resume above all.
    assert out == ROOT / "upscaled", f"the output folder resolved to {out}"
    beside = cli.resolve_out_dir(ROOT / "README.md", "", data)
    assert beside == ROOT / "upscaled", f"a file input resolved to {beside}"
    job = cli.build_job(data, src, out, None)
    _assert_job_payload(job)
    assert set(job) == {"input", "output", "format", "upscale", "perf"}, sorted(job)
    plan = cli.build_job(data, src, out, None, dry=True)
    assert plan.get("dry_run") is True, "plan must ask the worker for a dry run"


def resume_manifest() -> None:
    """The manifest may forget a finished unit; it must never invent one.

    Both halves of that asymmetry are asserted here, plus the fingerprint rule
    that stops a stale record from skipping exactly the work the user's new
    settings would have changed.
    """
    job: dict[str, Any] = {
        "input": {"dir": "in"},
        "output": {"dir": "out", "format": "cbz", "overwrite": False},
        "format": {"kind": "jpeg", "quality": 90},
        "upscale": {"scale": 2},
        "perf": {"threads": 4},
    }
    fp = resume.fingerprint(job)
    assert resume.fingerprint({**job, "perf": {"threads": 1}}) == fp, (
        "perf decides how long a page takes, not what it contains"
    )
    moved = {**job, "output": {**job["output"], "dir": r"D:\elsewhere"}}
    assert resume.fingerprint(moved) == fp, "the same output spelled differently is one job"
    worse = {**job, "format": {"kind": "jpeg", "quality": 60}}
    assert resume.fingerprint(worse) != fp, "a quality change rewrites the bytes"
    assert resume.fingerprint({**job, "upscale": {"scale": 4}}) != fp, "scale changes the output"

    base = Path(tempfile.gettempdir(), "manga")
    key = resume.unit_key(base / "Vol 01" / "Ch 11.cbz", base)
    assert key == "vol 01/ch 11.cbz", key
    outside = resume.unit_key(Path(tempfile.gettempdir(), "Loose.cbz"), base)
    assert outside.endswith("loose.cbz") and "\\" not in outside, outside

    assert resume.planned_entries(["b.png", "a.jpg", "b.jpeg", "b.bmp"], ".png") == [
        "b.png",
        "a.png",
        "b_3.png",
        "b_4.png",
    ], "the prediction must de-dup exactly as the packer does"

    with tempfile.TemporaryDirectory() as raw:
        out = Path(raw)
        state = resume.ResumeLog.load(out, fp)
        assert not state.is_done(key) and state.finished_count() == 0
        state.mark_partial(key, out / "Ch 11.cbz.part", ["001.png", "002.png"])
        assert not state.is_done(key), "a partial is progress, not a finished unit"
        assert (out / resume.MANIFEST_NAME).exists(), "the manifest sits in the output root"

        reopened = resume.ResumeLog.load(out, fp)
        assert reopened.partial_entries(key) == ["001.png", "002.png"], "page order survives"
        reopened.mark_done(key, out / "Ch 11.cbz", 24)
        assert reopened.is_done(key) and reopened.finished_count() == 1
        assert reopened.partial_entries(key) == [], "publishing retires the partial"
        assert resume.ResumeLog.load(out, fp).is_done(key), "a finished unit survives a restart"

        changed = resume.ResumeLog.load(out, "0" * resume.FINGERPRINT_CHARS)
        assert not changed.is_done(key), "a record from other settings is not trusted"
        assert changed.finished_count() == 0

        off = resume.ResumeLog.load(out, fp, enabled=False)
        assert not off.is_done(key), "--no-resume must convert everything again"
        assert off.partial_entries(key) == []
        off.mark_done("other.cbz", out / "other.cbz", 1)
        assert resume.ResumeLog.load(out, fp).is_done(key), (
            "a disabled log must not write, or opting out would erase real progress"
        )

        resume.ResumeLog.load(out, fp).drop(key)
        assert not resume.ResumeLog.load(out, fp).is_done(key), "drop is persistent"

        (out / resume.MANIFEST_NAME).write_text("{ truncated", encoding="utf-8")
        assert not resume.ResumeLog.load(out, fp).is_done(key), (
            "an unreadable manifest is ignored, never fatal"
        )


#: The publish-then-record order and the cancel arm cannot be exercised from
#: here -- importing the orchestrator pulls in the decoder and torch -- so they
#: are pinned by reading its source instead.
ORCHESTRATE = ROOT / "src" / "janai" / "worker" / "orchestrate.py"


def _code_only(text: str) -> str:
    """Source with comments stripped, so prose can never satisfy a guard.

    A WHY comment at one of these very sites once matched a guard that was
    counting call sites, and the guard passed while the code was still wrong.
    """
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _assert_resume_wiring(text: str) -> None:
    code = _code_only(text)
    publish = code.find("tmp_io.replace(dest_io)")
    record = code.find("self.resume.mark_done(")
    assert publish != -1, "the publishing rename is gone"
    assert record != -1, "a finished archive no longer records itself"
    assert publish < record, (
        "mark_done must follow the publishing replace(), or a crash between the "
        "two would leave a record claiming a chapter that was never published"
    )
    assert code.count("self.resume.mark_done(") == 1, (
        "one recorder per archive, so no other path can claim an unpublished one"
    )
    # The cancel arm of the *publishing* try block. `except Cancelled:` occurs
    # twice in that function -- the inner one drops a single page -- and only
    # the outer arm is the one that has to leave the .part on disk, so the
    # search starts at the rename rather than at the top of the file.
    cancelled = code.index("except Cancelled:", publish)
    arm = code[cancelled : code.index("except Exception", cancelled)]
    assert "keep_partial" in arm, "a cancelled archive must record how far it got"
    assert "unlink" not in arm, (
        "the .part has to survive a cancel, or resuming inside chapter 11 starts "
        "chapter 11 from its first page"
    )
    assert "def planned_entries" not in code, (
        "planned_entries lives in janai.worker.resume so this dependency-free "
        "gate can test it; a local copy would drift from the packer it mirrors"
    )


def resume_wiring() -> None:
    _assert_resume_wiring(ORCHESTRATE.read_text(encoding="utf-8"))


#: The other two paths that finish work record it from their own modules, and
#: `resume wiring` above reads the orchestrator ONLY -- so without these pins
#: both new recorders would sit here completely unpoliced while its count of
#: one recorder per archive stayed true.
REPORTING = ROOT / "src" / "janai" / "worker" / "reporting.py"
JOB = ROOT / "src" / "janai" / "worker" / "job.py"
PIPELINE = ROOT / "src" / "janai" / "worker" / "pipeline.py"


def _assert_loose_page_recorded(text: str) -> None:
    code = _code_only(text)
    write = code.find("target.write_bytes(data)")
    record = code.find("self.resume.mark_done(")
    assert write != -1, "the loose-page write is gone"
    assert record != -1, "a written loose page no longer records itself"
    assert write < record, (
        "the write IS the publish for a loose page, so recording first would "
        "claim a page that never reached disk"
    )
    assert "if resume_key:" in code, (
        "an empty key is planning's bundle placeholder, and a record under it "
        "would claim a source nobody converted"
    )
    assert "defer=True" in code, (
        "every flush rewrites the whole manifest, so flushing per page is "
        "quadratic: 1.4 GiB rewritten across a 4000-page run"
    )


def _assert_loose_page_key_passed(text: str) -> None:
    code = _code_only(text)
    submit = code.find("self.writer.submit(")
    assert submit != -1, "the loose-page submit is gone"
    call = code[submit : code.index(")", code.index("started,", submit))]
    assert "resume_key," in call, (
        "write_page takes the key positionally through submit(fn, *args), which "
        "mypy cannot check, so a dropped key would surface only as a resumed "
        "run silently re-converting every loose page"
    )
    assert "self.resume.is_done(" in code, (
        "loose pages have to ask the manifest, or resuming redoes all of them"
    )


def _assert_bundle_recorded(text: str) -> None:
    code = _code_only(text)
    guard = code.find("if bundle.close():")
    assert guard != -1, (
        "a bundle may only be recorded once close() reports it published: the "
        ".part is the work, and the rename inside close() is the publish"
    )
    body = code[guard : code.index("resume.flush()", guard)]
    assert "unit_key(" in body, "bundle members are recorded per source file"
    assert '"key"' not in body, (
        "planning leaves the task key empty for a single-archive run, so a "
        "record under it would skip a different input into the same folder"
    )
    assert code.count("resume.flush()") >= 2, (
        "one flush per published bundle and one on the way out, so a cancel "
        "cannot discard the deferred page records"
    )


def _assert_publish_reported(text: str) -> None:
    code = _code_only(text)
    close = code.index("def close(self, keep: bool = True) -> bool:")
    body = code[close : code.index("def shutdown", close)]
    assert body.count("return True") == 1, "one success path, so one meaning"
    assert body.index("return True") > body.index("replace(io_path(dest))"), (
        "close() may only report success after the rename that publishes the "
        "archive, because its caller records resume state on that answer"
    )


def resume_records_every_path() -> None:
    """Whatever finishes work records it, and never before it is published.

    The archive path is pinned by ``resume wiring``; these are the other two.
    A loose page is published by its own write and a bundle by the rename
    inside ``BundleWriter.close()``, so each records at a different moment --
    and each is invisible to a guard that reads only the orchestrator.
    Pinned by source because importing either module pulls in the decoder.
    """
    _assert_loose_page_recorded(REPORTING.read_text(encoding="utf-8"))
    _assert_loose_page_key_passed(ORCHESTRATE.read_text(encoding="utf-8"))
    _assert_bundle_recorded(JOB.read_text(encoding="utf-8"))
    _assert_publish_reported(PIPELINE.read_text(encoding="utf-8"))


def cancel_not_a_lost_page() -> None:
    """A user stop must never be reported as a page the run failed to convert.

    The vendored nodes raise their own ``api.node_context.Aborted`` when our
    ``ExecutorNodeContext`` reports ``aborted``, and the archive page loop
    counts anything that is not ``Cancelled`` as a lost page - so an
    untranslated abort makes a cancel look like data loss in the ``done``
    summary. Measured before the fix: ``pages_failed: 1`` alongside
    ``failed: 0``, plus a warning line with no message at all, because
    ``Aborted`` carries no text.

    Both boundaries that call into a vendored node are asserted, and so is the
    opposite direction: a genuine upscale failure must NOT become a cancel, or
    the translation would be a blanket swallow hiding real page loss.
    """

    class VendoredAbortError(Exception):
        """Stands in for `api.node_context.Aborted`, which needs the backend."""

    class RealFailureError(Exception):
        """A genuine page failure, which must stay a failure."""

    def abort(*_args: object, **_kwargs: object) -> None:
        raise VendoredAbortError

    def boom(*_args: object, **_kwargs: object) -> None:
        raise RealFailureError

    saved_abort = getattr(runtime, "Aborted", None)
    saved_upscale = runtime.upscale_image_node
    saved_load = runtime.load_model_node
    try:
        runtime.Aborted = VendoredAbortError
        runtime.upscale_image_node = abort
        try:
            models.upscale_array(None, object(), object(), 0)
        except control.Cancelled:
            pass
        except VendoredAbortError:
            raise AssertionError(
                "upscale_array leaks the vendored Aborted, so the page loop counts a "
                "cancel as a lost page"
            ) from None
        else:
            raise AssertionError("upscale_array swallowed the abort instead of translating it")

        runtime.upscale_image_node = boom
        try:
            models.upscale_array(None, object(), object(), 0)
        except control.Cancelled:
            raise AssertionError("a real upscale failure is being reported as a cancel") from None
        except RealFailureError:
            pass

        runtime.load_model_node = abort
        try:
            models.ModelCache(None).get("4x_model.pth")
        except control.Cancelled:
            pass
        except VendoredAbortError:
            raise AssertionError("the model load path leaks the vendored Aborted") from None
        else:
            raise AssertionError("the model load swallowed the abort instead of translating it")

        # Before the heavy stack is loaded the handle is still None, and
        # `except None` raises TypeError over the top of the real error.
        runtime.Aborted = None
        runtime.upscale_image_node = boom
        try:
            models.upscale_array(None, object(), object(), 0)
        except RealFailureError:
            pass
        except TypeError as exc:
            raise AssertionError(f"the translation breaks with no backend loaded: {exc}") from None
    finally:
        runtime.Aborted = saved_abort
        runtime.upscale_image_node = saved_upscale
        runtime.load_model_node = saved_load


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
    check("dependency pins", dependency_pins)
    check("vram budget clamp", vram_budget_clamp)
    check("oom recovery", oom_recovery_frees_without_copying)
    check("headless driver", headless_driver)
    check("resume manifest", resume_manifest)
    check("resume wiring", resume_wiring)
    check("resume records every path", resume_records_every_path)
    check("cancel is not a lost page", cancel_not_a_lost_page)
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
