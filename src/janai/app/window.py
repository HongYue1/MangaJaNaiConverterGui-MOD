"""The main window: four cards, a footer, and the run log.

Read it top to bottom - what goes in, how far to upscale it, what comes out,
and (folded away) how hard to push the hardware. Everything heavy happens in
the worker process; this file collects settings, renders events and keeps a log.

The worker contract is untouched: :class:`janai.app.runner.Runner` still spawns
the same process and still hands back the same JSONL events, pumped here by a
QTimer instead of a Tk ``after`` loop. Nothing in ``janai.core`` or
``janai.worker`` had to move for this interface.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QGuiApplication, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from janai import __version__
from janai.app.fields import (
    MODE_OPTIONS,
    TILE_CHOICES,
    tile_label,
    tile_value,
)
from janai.app.input_panel import InputPanelMixin
from janai.app.log_panel import LogPanelMixin
from janai.app.output_panel import OutputPanelMixin
from janai.app.rules_table import EXCLUSION_HEADERS, RuleDialog, RulesModel, RulesTable
from janai.app.runlog import (
    RunLog,
    fmt_bytes,
    fmt_secs,
    format_bundle,
    format_done,
    format_file,
    format_start,
)
from janai.app.runner import Runner, open_in_explorer
from janai.app.state import Settings, defaults
from janai.app.theme import Theme
from janai.app.widgets import (
    Banner,
    Card,
    Collapsible,
    Segmented,
    button,
    checkbox,
    combo,
    label,
    row,
    set_combo,
    spin_float,
    spin_int,
)
from janai.core import displays, hardware, presets, rules
from janai.core.formats import (
    CONTAINERS,
    FORMAT_IDS,
    FORMATS,
    packs_archive,
    summary,
)

#: An empty device string means "let the worker pick the best one".
AUTO_DEVICE = "Auto (best available)"
DEVICE_RE = re.compile(r"^(cpu|cuda|xpu|mps|dml|privateuseone)(:\d+)?$")

GEOMETRY_RE = re.compile(r"^(\d+)x(\d+)(?:\+(-?\d+)\+(-?\d+))?$")


class MainWindow(InputPanelMixin, LogPanelMixin, OutputPanelMixin, QMainWindow):
    """The whole interface. One instance, one settings file, one worker."""

    def __init__(self, root_dir: Path, theme: Theme, app: QApplication) -> None:
        super().__init__()
        self.app = app
        self.theme = theme
        self.dir = root_dir
        self.settings = Settings(root_dir / "settings.json").load()
        self.runner = Runner(root_dir)

        log_cfg = self.settings.data.get("log") or {}
        self.runlog = RunLog(
            self._log_dir(),
            keep=int(log_cfg.get("keep", 30) or 0),
            enabled=bool(log_cfg.get("auto_save", True)),
        )

        probe = self.settings.data.get("probe") or {}
        self.probe: dict = probe if isinstance(probe, dict) else {}
        self.caps: dict = self.probe.get("formats", {}) or {}
        self.models: list = self.probe.get("models", []) or []
        self.devices: list = self.probe.get("devices", []) or []

        # What this machine measured about itself, if it ever has. It sits
        # next to the probe because it answers the same question - what the
        # hardware can do - only by measurement rather than by asking.
        stored = self.settings.data.get("profile") or {}
        self.profile: dict = stored if isinstance(stored, dict) else {}
        self._profiling = False
        self._profile_offered = False

        self.total = 0
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.started_index = 0
        self.progress_sub = ""
        self.started_at = 0.0
        self.dry = False
        self.last_out_dir: Path | None = None
        self.scan_text = ""
        self._log_visible = False

        data = self.settings.data
        self._in_path = str((data.get("input") or {}).get("path", ""))
        self._saved_device = str((data.get("perf") or {}).get("device", "") or "")
        seeded = [rules.Rule.from_dict(r) for r in (data["upscale"].get("rules") or [])]
        self._rules_seeded = bool(seeded)

        #: Encoder options live here rather than in the widgets, so switching
        #: format keeps what was typed into the one you left.
        self.fmt_values: dict[str, dict[str, Any]] = {}
        for fid, spec in FORMATS.items():
            saved = self.settings.format_options(fid)
            self.fmt_values[fid] = {opt.key: saved.get(opt.key, opt.default) for opt in spec.opts}

        self._build_ui(seeded)
        self._bind_keys()

        self.apply_probe(self.probe, cached=True)
        self.runner.probe()
        self.render_format_options()
        self.render_target()
        self.seed_rules()
        self.render_rules()
        self.update_start_state()
        self.update_wake_lock()
        self.render_profile()

        self._timer = QTimer(self)
        self._timer.setInterval(90)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

        if self._in_path:
            self.scan_input_async(self._in_path)

    # ------------------------------------------------------------------ #
    # layout
    # ------------------------------------------------------------------ #
    def _build_ui(self, seeded: list[rules.Rule]) -> None:
        self.setWindowTitle("JaNai Upscaler")
        self.setMinimumSize(920, 620)
        self._apply_geometry(str(self.settings.data["ui"].get("geometry") or ""))

        central = QWidget(self)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_header())

        self.banner = Banner(central)
        holder = QWidget(central)
        holder_box = QVBoxLayout(holder)
        holder_box.setContentsMargins(18, 0, 18, 0)
        holder_box.setSpacing(0)
        holder_box.addWidget(self.banner)
        outer.addWidget(holder)

        self.splitter = QSplitter(Qt.Orientation.Vertical, central)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(8)

        self.scroll = QScrollArea(self.splitter)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget()
        self.page = QVBoxLayout(page)
        self.page.setContentsMargins(18, 12, 18, 16)
        self.page.setSpacing(12)
        self._build_input()
        self._build_upscale(seeded)
        self._build_output()
        self._build_perf()
        self.page.addStretch(1)
        self.scroll.setWidget(page)
        self.splitter.addWidget(self.scroll)

        self._build_log()
        self.splitter.addWidget(self.log_panel)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        outer.addWidget(self.splitter, 1)

        outer.addWidget(self._build_footer())
        self.setCentralWidget(central)

        self.show_log(bool(self.settings.data["ui"].get("log_open", False)))

    def _build_header(self) -> QWidget:
        head = QWidget(self)
        box = QHBoxLayout(head)
        box.setContentsMargins(18, 14, 18, 10)
        box.setSpacing(12)

        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(2)
        name = QHBoxLayout()
        name.setContentsMargins(0, 0, 0, 0)
        name.setSpacing(8)
        name.addWidget(label("JaNai Upscaler", "display"))
        name.addWidget(label(f"v{__version__}", "hint"), 0, Qt.AlignmentFlag.AlignBottom)
        name.addStretch(1)
        titles.addLayout(name)
        self.lbl_env = label("detecting hardware\u2026", "muted")
        titles.addWidget(self.lbl_env)
        box.addLayout(titles, 1)

        self.btn_theme = button(
            "Theme",
            self.toggle_theme,
            variant="ghost",
            tip="Switch between the dark and light palette.  (Ctrl+D)",
        )
        self.btn_refresh = button(
            "Re-detect",
            self.refresh_probe,
            variant="ghost",
            tip=(
                "Ask the backend again which GPU, encoders and models are available. "
                "Use it after installing models or changing drivers \u2014 the answer "
                "is cached between runs so the window can open instantly.  (F5)"
            ),
        )
        self.btn_presets = button(
            "Presets",
            variant="ghost",
            tip=(
                "Save the current settings as a preset, load one from a file, or "
                "switch to one you already saved. A preset carries the target, the "
                "rules table, the format, the output layout and the performance "
                "options \u2014 never your folders or your device, so someone "
                "else's preset cannot redirect your output."
            ),
        )
        self.preset_menu = QMenu(self.btn_presets)
        self.preset_menu.aboutToShow.connect(self._fill_preset_menu)
        self.btn_presets.setMenu(self.preset_menu)
        self.btn_reset = button(
            "Reset all",
            self.reset_all,
            variant="ghost",
            tip=(
                "Put every setting back to the shipped defaults: target, rules table, "
                "output format and layout, performance options and log view. Your "
                "input and output folders are kept, and you get a confirmation "
                "prompt first."
            ),
        )
        for btn in (self.btn_theme, self.btn_refresh, self.btn_presets, self.btn_reset):
            box.addWidget(btn, 0, Qt.AlignmentFlag.AlignTop)
        return head

    def _build_upscale(self, seeded: list[rules.Rule]) -> None:
        card = Card("Upscale", "How big the result is, and which model each page gets.")
        self.card_upscale = card
        u = self.settings.data["upscale"]
        body = card.body

        self.seg_mode = Segmented(MODE_OPTIONS, str(u.get("mode", "scale")))
        self.seg_mode.changed.connect(lambda _value: self.render_target())
        body.field("Target", "How large the result should be.", self.seg_mode)

        self.sp_scale = spin_float(
            1.0,
            8.0,
            float(u.get("scale", 2.0)),
            0.25,
            2,
            self.update_summary,
            suffix="\u00d7",
        )
        self.w_scale = row(
            self.sp_scale, label("of the original size", "muted"), spacing=8, stretch=False
        )
        self.sp_width = spin_int(
            64, 30000, int(u.get("width", 2048)), 16, self.update_summary, suffix=" px"
        )
        self.w_width = row(label("Width", "muted"), self.sp_width, spacing=8, stretch=False)
        self.sp_height = spin_int(
            64, 30000, int(u.get("height", 2160)), 16, self.update_summary, suffix=" px"
        )
        self.w_height = row(label("Height", "muted"), self.sp_height, spacing=8, stretch=False)

        # Fit mode also offers the original fork's display-device list, so a
        # chapter can be sized for a specific reader in one click.
        self.cb_display = combo(
            displays.labels(),
            displays.label_for_id(str(u.get("display", displays.CUSTOM))),
            self.on_display_change,
            width=250,
        )
        self.seg_orient = Segmented(
            (("Portrait", True), ("Landscape", False)),
            bool(u.get("display_portrait", True)),
        )
        self.seg_orient.changed.connect(lambda _value: self.on_display_change())
        self.w_fit = row(
            label("Device", "muted"), self.cb_display, self.seg_orient, spacing=10, stretch=False
        )

        targets = QWidget()
        tbox = QVBoxLayout(targets)
        tbox.setContentsMargins(0, 0, 0, 0)
        tbox.setSpacing(8)
        sizes = QWidget()
        sbox = QHBoxLayout(sizes)
        sbox.setContentsMargins(0, 0, 0, 0)
        sbox.setSpacing(16)
        sbox.addWidget(self.w_scale)
        sbox.addWidget(self.w_width)
        sbox.addWidget(self.w_height)
        sbox.addStretch(1)
        tbox.addWidget(sizes)
        tbox.addWidget(self.w_fit)
        body.control(targets)

        # Page kind comes before the table, because it decides which rules can
        # ever fire. The three cases are exclusive, so they are one control:
        # measure every page, or declare the whole run grayscale or colour.
        self._page_kinds = (
            ("Detect per page", "detect"),
            ("All grayscale", "grayscale"),
            ("All colour", "colour"),
        )
        saved_kind = str(u.get("page_kind") or "").strip().lower()
        if saved_kind not in {"detect", "grayscale", "colour"}:
            saved_kind = "detect" if bool(u.get("grayscale_convert", True)) else "colour"
        self.cb_pagekind = combo(
            [text for text, _ in self._page_kinds],
            next(text for text, value in self._page_kinds if value == saved_kind),
            width=200,
            tip=(
                "How each page's kind is decided. \u201cDetect per page\u201d measures "
                "every page: a grayscale one is stored as a single channel (smaller, "
                "faster), downscaled with the dot-gain aware filter, and matched "
                "against the grayscale rules. \u201cAll grayscale\u201d and \u201cAll "
                "colour\u201d skip the measurement and declare the whole run, for "
                "pages you already know - an all-colour artbook, or a monochrome "
                "volume on tinted paper that detection would call colour."
            ),
        )
        self.cb_pagekind.currentTextChanged.connect(self.on_pagekind_change)
        self.chk_levels = checkbox(
            "Auto levels",
            bool(u.get("auto_levels", True)),
            self.on_levels_toggle,
            tip=(
                "Stretches the black and white points of grayscale pages before "
                "upscaling, which lifts washed-out scans. Colour pages are never "
                "touched. This is the setting a rule follows when its Levels cell "
                "says \u201cdefault\u201d; a rule can override it per page size."
            ),
        )
        body.field(
            "Page type",
            "Whether each page is judged on its own, or the whole run is declared.",
            row(self.cb_pagekind, self.chk_levels, spacing=18),
        )

        body.full(self._build_rules_block(seeded))
        body.full(self._build_exclusions_block(seeded))

        self.sp_threshold = spin_int(
            0,
            64,
            int(u.get("grayscale_threshold", 12)),
            1,
            self.update_summary,
            tip=(
                "How far a pixel's red, green and blue may drift apart (0-255) before "
                "it counts as coloured. 12 matches the original app. Raise it to send "
                "yellowed or sepia scans to the grayscale model anyway; lower it if "
                "faintly tinted pages should be treated as colour."
            ),
        )
        body.field(
            "Gray threshold", "How far R, G and B may drift before colour.", self.sp_threshold
        )

        self.sp_colour = spin_float(
            0.0,
            25.0,
            float(u.get("grayscale_colour_percent", 0.25)),
            0.05,
            2,
            self.update_summary,
            suffix=" %",
            tip=(
                "The second test, for pages whose average still looks gray: once this "
                "share of pixels is clearly coloured, the page is treated as colour. "
                "0.25% catches a coloured title or one spot-colour panel on an "
                "otherwise black-and-white page. Set it to 0 to judge by the "
                "threshold alone."
            ),
        )
        body.field(
            "Colour pixels", "Share of coloured pixels that makes a page colour.", self.sp_colour
        )

        self.sp_pre_h = spin_int(
            0,
            20000,
            int(u.get("pre_downscale_height", 0)),
            100,
            self.update_summary,
            suffix=" px",
            special="off",
            tip=(
                "Shrinks an oversized page to this height first and then upscales it "
                "as usual. Worth using when raws are far larger than the model was "
                "trained for - a 3000px scan through a 1600p model - since the model "
                "sees fewer pixels, which is both faster and often cleaner. The page "
                "is still upscaled, unlike an exclusion rule in the table."
            ),
        )
        body.field("Pre-downscale height", "Shrink huge pages before the model.", self.sp_pre_h)

        # Kept so the detection numbers can be greyed out together with their
        # labels when the run declares its pages instead of measuring them.
        self.upscale_body = body
        self.page.addWidget(card)

    def _rule_buttons(self, which: str) -> QWidget:
        """The column of buttons beside one of the two rule tables."""
        side = QWidget()
        sbox = QVBoxLayout(side)
        sbox.setContentsMargins(0, 0, 0, 0)
        sbox.setSpacing(6)
        actions: tuple[tuple[str, Callable[[], None], str], ...] = (
            ("Add", lambda: self.rule_add(which), "Add a row below the selected one."),
            (
                "Edit",
                lambda: self.rule_edit(which),
                "Edit the selected row (or double-click it).",
            ),
            (
                "Toggle",
                lambda: self.rule_toggle(which),
                (
                    "Switch the selected row off without deleting it. The checkbox "
                    "in the first column shows the state, and the space bar does the "
                    "same thing."
                ),
            ),
            ("Remove", lambda: self.rule_remove(which), "Delete the selected row."),
            (
                "Up",
                lambda: self.rule_move(-1, which),
                (
                    "Move the row up (Alt+Up). Order only decides between rows that "
                    "are equally specific."
                ),
            ),
            ("Down", lambda: self.rule_move(1, which), "Move the row down (Alt+Down)."),
        )
        if which == "rules":
            actions = (
                *actions,
                (
                    "Defaults",
                    self.rules_reset,
                    (
                        "Rewrite this table as the shipped set: the MangaJaNai height "
                        "bands for grayscale pages and the IllustrationJaNai denoise "
                        "models for colour, built from the models you have installed. "
                        "Only this table is touched - your exclusions and the rest of "
                        "your settings are left alone."
                    ),
                ),
            )
        for text, action, hint in actions:
            btn = button(text, action, variant="ghost", tip=hint)
            btn.setMinimumWidth(96)
            sbox.addWidget(btn)
        sbox.addStretch(1)
        return side

    def _table_row(self, view: RulesTable, side: QWidget) -> QWidget:
        """A table with its buttons beside it."""
        holder = QWidget()
        tbox = QHBoxLayout(holder)
        tbox.setContentsMargins(0, 0, 0, 0)
        tbox.setSpacing(10)
        tbox.addWidget(view, 1)
        tbox.addWidget(side, 0)
        return holder

    def _build_exclusions_block(self, seeded: list[rules.Rule]) -> QWidget:
        """Size exclusions, in their own panel, closed until they are wanted.

        They answer a different question from the model rules - which pages
        should not go through a model at all - and most runs never touch them,
        so they sit in a panel that starts collapsed rather than taking up half
        of the card.
        """
        panel = Collapsible(
            "Size exclusions",
            "pages that skip the model",
            expanded=False,
        )
        self.panel_excl = panel
        panel.setToolTip(
            "Pages that skip the model and are only re-encoded: the long webtoon "
            "strips the old build skipped with numbers nobody could see. An "
            "exclusion names a page size, and a sized row always beats a model "
            "rule that says \u201cany\u201d, so those pages pass through untouched."
        )
        body = panel.body

        self.excl_model = RulesModel(
            self.theme.p,
            [r for r in seeded if r.action == rules.PASSTHROUGH],
            bool(self.settings.data["upscale"].get("grayscale_convert", True)),
            self.model_names(),
            parent=self,
            headers=EXCLUSION_HEADERS,
        )
        self.excl_model.edited.connect(self.on_rule_checked)
        self.excl_view = RulesTable(self.excl_model, rows=3)
        self.excl_view.doubleClicked.connect(lambda _index: self.rule_edit("excl"))
        body.full(self._table_row(self.excl_view, self._rule_buttons("excl")))

        self.lbl_excl_hint = label("", "hint", wrap=True)
        body.full(self.lbl_excl_hint)
        return panel

    def _build_rules_block(self, seeded: list[rules.Rule]) -> QWidget:
        """The model-rules table, its side buttons, and the lines beneath it."""
        block = QWidget()
        box = QVBoxLayout(block)
        box.setContentsMargins(0, 4, 0, 0)
        box.setSpacing(8)

        how = (
            "page kind + size decide the model \u00b7 a sized row always beats "
            "an \u201cany\u201d row"
        )
        tip = (
            "Every page is matched against this table. The first row whose "
            "conditions fit decides which model runs, and for grayscale pages "
            "whether auto levels is applied. A row that names a page size wins "
            "over a row that says \u201cany\u201d wherever the two sit, so a "
            "catch-all at the top cannot swallow everything by accident. "
            "Double-click a row to edit it, or clear its checkbox to switch it off."
        )
        box.addWidget(row(label("Model rules", "field", tip=tip), label(how, "hint", tip=tip)))

        self.rules_model = RulesModel(
            self.theme.p,
            [r for r in seeded if r.action != rules.PASSTHROUGH],
            bool(self.settings.data["upscale"].get("grayscale_convert", True)),
            self.model_names(),
            parent=self,
        )
        self.rules_model.edited.connect(self.on_rule_checked)
        self.rules_view = RulesTable(self.rules_model, rows=6)
        self.rules_view.doubleClicked.connect(lambda _index: self.rule_edit("rules"))
        box.addWidget(self._table_row(self.rules_view, self._rule_buttons("rules")))

        self.lbl_rules_hint = label("", "hint", wrap=True)
        box.addWidget(self.lbl_rules_hint)
        self.lbl_rules_warn = label("", "warn", wrap=True)
        self.lbl_rules_warn.setVisible(False)
        box.addWidget(self.lbl_rules_warn)
        return block

    def _build_perf(self) -> None:
        panel = Collapsible(
            "Performance",
            "device, precision, tiling, threads",
            # Always closed on startup. It is the panel of last resort, and a
            # window that opens with it expanded buries the settings that are
            # actually used on every run.
            expanded=False,
        )
        self.panel_perf = panel
        body = panel.body
        p = self.settings.data["perf"]

        self.cb_device = combo(
            self._device_labels(),
            self._device_label(self._saved_device),
            self.on_device_change,
            width=330,
        )
        body.field("Device", "Auto picks the fastest GPU it can find.", self.cb_device)

        self.chk_fp16 = checkbox(
            "FP16 (half precision)",
            bool(p.get("use_fp16", True)),
            self.update_summary,
            tip=(
                "On by default: roughly twice as fast and half the VRAM on supported "
                "GPUs. The worker falls back to FP32 by itself on hardware or models "
                "that cannot do it."
            ),
        )
        body.control(self.chk_fp16)
        self.lbl_fp16 = label("", "hint")
        body.control(self.lbl_fp16)

        self.cb_tile = combo(
            [text for text, _ in TILE_CHOICES],
            tile_label(str(p.get("tile", "auto"))),
            self.update_summary,
            width=210,
            tip=(
                "Auto measures what the model actually costs on the first tiles and "
                "grows to the largest tile that fits the free VRAM \u2014 fewer, "
                "bigger tiles means fewer seams and less overhead."
            ),
        )
        body.field("Tile size", "Splits large images so they fit in VRAM.", self.cb_tile)

        self.btn_profile = button(
            "Measure this machine",
            self.on_profile_clicked,
            tip=(
                "Runs each installed model over a ladder of tile sizes and records what "
                "it costs and how fast it is. Auto then sizes the first page from "
                "measurements instead of holding it at 1024px, and a tile this card "
                "has already refused is never planned again. Writes no images, and is "
                "only needed once per machine."
            ),
        )
        body.field(
            "Hardware profile",
            "Measured once per machine, then reused.",
            row(self.btn_profile),
        )
        self.lbl_profile = label("", "hint", wrap=True)
        body.control(self.lbl_profile)

        self.sp_budget = spin_int(
            0,
            128,
            int(p.get("budget_limit", 0)),
            1,
            self.update_summary,
            suffix=" GiB",
            special="no cap",
        )
        body.field("VRAM budget", "An upper bound on what a job may reserve.", self.sp_budget)

        self.sp_threads = spin_int(
            0, 256, int(p.get("torch_threads", 0)), 1, self.update_summary, special="auto"
        )
        body.field("CPU threads", "Leave on auto to let torch decide.", self.sp_threads)

        self.sp_io = spin_int(1, 16, max(1, int(p.get("io_workers", 2))), 1, self.update_summary)
        body.field("I/O workers", "Threads that decode and encode while the GPU works.", self.sp_io)

        self.sp_vips = spin_int(
            0, 64, int(p.get("vips_concurrency", 0)), 1, self.update_summary, special="default"
        )
        body.field("libvips concurrency", "Leave on default unless tuning.", self.sp_vips)

        self.chk_cudnn = checkbox(
            "cuDNN autotune",
            bool(p.get("cudnn_benchmark", True)),
            tip=(
                "Benchmarks convolution algorithms once per shape. Faster for long "
                "runs of same-sized pages."
            ),
        )
        self.chk_tf32 = checkbox(
            "TF32 matmuls",
            bool(p.get("allow_tf32", False)),
            tip=(
                "Ampere and newer: faster matmuls at slightly reduced precision. Off "
                "by default because the measured gain on a single 17 s image is "
                "0.1-0.2 s, which is inside the noise - not worth trading precision "
                "for unasked. Worth turning on for long batches, where it compounds."
            ),
        )
        self.chk_wipe = checkbox(
            "Wipe cache between images",
            bool(p.get("force_cache_wipe", False)),
            tip=("Frees VRAM after every image. Slower, but avoids fragmentation on small GPUs."),
        )
        body.control(row(self.chk_cudnn, self.chk_tf32, self.chk_wipe, spacing=18))

        self.chk_wake = checkbox(
            "Keep GPU awake",
            bool(p.get("gpu_wake_lock", True)),
            self.on_wake_toggle,
            tip=(
                "Holds a tiny context on the GPU while this window is open, so the "
                "driver keeps the card powered (and a laptop dGPU does not park) and "
                "the first run starts at full speed. It holds a CUDA context, which "
                "costs about 120 MB of VRAM while the app sits idle, and it is "
                "released automatically while a job runs. Turn it off to leave the "
                "GPU completely alone."
            ),
        )
        body.control(self.chk_wake)
        self.page.addWidget(panel)

    def _build_footer(self) -> QWidget:
        foot = QWidget(self)
        box = QVBoxLayout(foot)
        box.setContentsMargins(18, 8, 18, 12)
        box.setSpacing(8)

        self.bar = QProgressBar(foot)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        box.addWidget(self.bar)

        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(10)
        status = QVBoxLayout()
        status.setContentsMargins(0, 0, 0, 0)
        status.setSpacing(1)
        self.lbl_status = label("Ready", "field")
        self.lbl_detail = label("", "hint")
        status.addWidget(self.lbl_status)
        status.addWidget(self.lbl_detail)
        line.addLayout(status, 1)

        self.btn_log = button("Log", self.toggle_log, variant="ghost", tip="Ctrl+L")
        self.btn_open = button(
            "Open output", self.open_output, variant="ghost", tip="Show the output folder."
        )
        self.btn_dry = button(
            "Dry run",
            self.start_dry,
            variant="ghost",
            tip=(
                "Walks the whole job and reports every file it would write, the size, "
                "the model and the archives it would build \u2014 without touching "
                "the disk or the GPU.  (Ctrl+Shift+Enter)"
            ),
        )
        self.btn_pause = button("Pause", self.toggle_pause)
        self.btn_start = button("Start", self.on_start_clicked, variant="accent", tip="Ctrl+Enter")
        self.btn_start.setMinimumWidth(110)
        for btn in (self.btn_log, self.btn_open, self.btn_dry, self.btn_pause, self.btn_start):
            line.addWidget(btn)
        box.addLayout(line)
        return foot

    # ------------------------------------------------------------------ #
    # keyboard
    # ------------------------------------------------------------------ #
    def _shortcut(self, keys: str, slot: Callable[[], None]) -> None:
        short = QShortcut(QKeySequence(keys), self)
        short.activated.connect(slot)

    def _bind_keys(self) -> None:
        self._shortcut("Ctrl+O", self.choose_file)
        self._shortcut("Ctrl+Shift+O", self.choose_folder)
        self._shortcut("Ctrl+Return", self.start)
        self._shortcut("Ctrl+Shift+Return", self.start_dry)
        self._shortcut("Esc", self.cancel)
        self._shortcut("Ctrl+L", self.toggle_log)
        self._shortcut("Ctrl+D", self.toggle_theme)
        self._shortcut("F5", self.refresh_probe)
        # These follow the focus, so they move a row in whichever of the two
        # tables the keyboard is actually in.
        self._shortcut("Alt+Up", lambda: self.rule_move(-1, self._focused_table()))
        self._shortcut("Alt+Down", lambda: self.rule_move(1, self._focused_table()))

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #
    def _apply_geometry(self, text: str) -> None:
        """Restore the saved size, in the same string the Tk build wrote.

        The size is clamped to a share of the desktop, and an unplaced window is
        centred. A window as large as the screen cannot be moved or aligned, and
        that is exactly what a geometry saved by the Tk build - or on a larger
        monitor - asks for.
        """
        screen = QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        area = screen.availableGeometry() if screen is not None else None
        match = GEOMETRY_RE.match(text.strip())
        if match is None:
            width, height = 1180, 900
        else:
            width, height = int(match.group(1)), int(match.group(2))
        trimmed = False
        if area is not None:
            fit_w = min(width, int(area.width() * 0.88))
            fit_h = min(height, int(area.height() * 0.90))
            trimmed = (fit_w, fit_h) != (width, height)
            width, height = fit_w, fit_h
        self.resize(max(920, width), max(620, height))
        if match is None or match.group(3) is None or trimmed:
            # Centred on both axes: with no saved position, or a size that had
            # to be trimmed to fit this desktop, the stored corner belongs to a
            # window that no longer exists.
            if area is not None:
                self.move(
                    area.x() + max(0, (area.width() - self.width()) // 2),
                    area.y() + max(0, (area.height() - self.height()) // 2),
                )
            return
        x, y = int(match.group(3)), int(match.group(4))
        if area is not None:
            x = min(max(x, area.x()), max(area.x(), area.right() - self.width() + 1))
            y = min(max(y, area.y()), max(area.y(), area.bottom() - self.height() + 1))
        self.move(x, y)

    def _geometry_text(self) -> str:
        rect = self.frameGeometry() if self.isVisible() else self.geometry()
        return f"{self.width()}x{self.height()}+{rect.x()}+{rect.y()}"

    # ------------------------------------------------------------------ #
    # dynamic rendering
    # ------------------------------------------------------------------ #
    def render_target(self) -> None:
        mode = str(self.seg_mode.value())
        self.w_scale.setVisible(mode == "scale")
        self.w_width.setVisible(mode in {"width", "fit"})
        self.w_height.setVisible(mode in {"height", "fit"})
        self.w_fit.setVisible(mode == "fit")
        self.update_summary()

    def on_display_change(self) -> None:
        """Fill the fit box from a device preset (or leave it alone for Custom)."""
        did = displays.id_for_label(self.cb_display.currentText())
        size = displays.size(did, bool(self.seg_orient.value()))
        if size is not None:
            self.sp_width.setValue(size[0])
            self.sp_height.setValue(size[1])
        self.update_summary()

    def page_kind(self) -> str:
        """detect | grayscale | colour - the three exclusive page-kind cases."""
        text = self.cb_pagekind.currentText()
        return next((value for name, value in self._page_kinds if name == text), "detect")

    def set_page_kind(self, value: str) -> None:
        first = self._page_kinds[0][0]
        set_combo(self.cb_pagekind, next((n for n, v in self._page_kinds if v == value), first))

    def gray_rules_live(self) -> bool:
        """Grayscale rows can only fire while some page can be grayscale."""
        return self.page_kind() != "colour"

    def on_pagekind_change(self, _text: str = "") -> None:
        """The page kind decides which rows can fire, and which fields apply."""
        self.set_gray_rows(self.gray_rules_live())
        self.sync_detection_fields()
        self.render_rules()
        self.update_summary()

    def set_gray_rows(self, live: bool) -> None:
        """Both tables grey out their grayscale rows together."""
        self.rules_model.set_gray(live)
        self.excl_model.set_gray(live)

    def sync_detection_fields(self) -> None:
        """Grey out the detection numbers when the run declares its pages.

        They only decide how a page is measured, so once every page is declared
        grayscale or colour they cannot change anything. A control that looks
        live but does nothing is worse than a disabled one.
        """
        body = getattr(self, "upscale_body", None)
        if body is None:
            return  # still building the card
        live = self.page_kind() == "detect"
        body.set_row_enabled(self.sp_threshold, live)
        body.set_row_enabled(self.sp_colour, live)

    def on_levels_toggle(self) -> None:
        self.render_rules()
        self.update_summary()

    # ------------------------------------------------------------------ #
    # rules
    # ------------------------------------------------------------------ #
    @property
    def rules(self) -> list[rules.Rule]:
        """Every rule from both tables: exclusions first, then the model rules.

        The interface splits them - model rules in the card, size exclusions in
        their own panel - but matching, saving and the job payload all want the
        whole set, and exclusions come first so that reading the saved file top
        to bottom follows the order a page is decided in.
        """
        return [*self.excl_model.rules, *self.rules_model.rules]

    def set_all_rules(self, items: list[rules.Rule]) -> None:
        """Deal one saved list into the two tables, keeping relative order."""
        self.rules_model.set_rules([r for r in items if r.action != rules.PASSTHROUGH])
        self.excl_model.set_rules([r for r in items if r.action == rules.PASSTHROUGH])

    def _table(self, which: str = "rules") -> tuple[RulesModel, RulesTable]:
        if which == "excl":
            return self.excl_model, self.excl_view
        return self.rules_model, self.rules_view

    def _focused_table(self) -> str:
        """Which table a keyboard shortcut should act on."""
        return "excl" if self.excl_view.hasFocus() else "rules"

    def model_names(self) -> list[str]:
        return [str(m.get("name")) for m in self.models if m.get("name")]

    def seed_rules(self) -> None:
        """Make sure the table is filled in and names real files, once.

        Both cases need the probe to have reported the installed models: a
        first run gets the shipped working set written out in full, and a
        settings file from an older build gets its legacy "auto" rows resolved
        to the file they would have picked.
        """
        if self._rules_seeded or not self.models:
            return
        installed = self.model_names()
        current = list(self.rules)
        if not current:
            current = rules.default_working_set(installed)
        elif any(r.is_auto for r in current):
            current, notes = rules.materialise(current, installed)
            for note in notes:
                self.log(f"rule resolved: {note}", "debug")
            if not current:
                current = rules.default_working_set(installed)
        self.set_all_rules(current)
        self._rules_seeded = True
        self.save_rules()

    def render_rules(self) -> None:
        """Refresh the lines under the two tables. The models paint the rows."""
        names = self.model_names()
        self.rules_model.set_installed(names)
        self.excl_model.set_installed(names)
        kind = self.page_kind()
        items = self.rules_model.rules
        if not items:
            hint = (
                "The table is empty, so nothing can run. \u201cDefaults\u201d fills it "
                "with the shipped set, built from the models you have installed."
            )
        else:
            active = sum(1 for r in items if r.enabled)
            hint = f"{active} of {len(items)} rules on"
            if kind == "colour":
                hint += "  \u00b7  grayscale rules are idle: every page is colour"
            elif kind == "grayscale":
                hint += "  \u00b7  colour rules are idle: every page is grayscale"
            hint += "  \u00b7  space toggles a row, double-click edits it"
        self.lbl_rules_hint.setText(hint)

        excluded = self.excl_model.rules
        live = sum(1 for r in excluded if r.enabled)
        if not excluded:
            self.lbl_excl_hint.setText(
                "Nothing is excluded, so every page goes through a model. Add a row "
                "to let pages of a given size skip the model and only be re-encoded."
            )
            self.panel_excl.set_hint("none")
        else:
            self.lbl_excl_hint.setText(
                f"{live} of {len(excluded)} exclusions on  \u00b7  matching pages skip "
                "the model and are only re-encoded"
            )
            self.panel_excl.set_hint(f"{live} on" if live else f"{len(excluded)} off")
        self.refresh_rule_warnings()

    def refresh_rule_warnings(self) -> None:
        """Everything wrong with either table, on one line under the rules.

        Each table numbers its own rows, so a warning has to say which table it
        points at or the number would send you to the wrong row.
        """
        installed = self.model_names()
        notes: list[str] = []
        for what, items in (
            ("row", self.rules_model.rules),
            ("exclusion", self.excl_model.rules),
        ):
            for index, rule in enumerate(items):
                if not rule.enabled:
                    continue
                notes.extend(
                    f"{what} {index + 1}: {note}" for note in rules.problems(rule, installed)
                )
        notes.extend(self.scale_mismatches())
        self.lbl_rules_warn.setText("\u26a0  " + "; ".join(notes[:4]) if notes else "")
        self.lbl_rules_warn.setVisible(bool(notes))

    def scale_mismatches(self) -> list[str]:
        """Rows whose model name advertises a factor the target will not use."""
        if str(self.seg_mode.value()) != "scale":
            return []
        want = float(self.sp_scale.value())
        out: list[str] = []
        for index, rule in enumerate(self.rules_model.rules):
            if not rule.enabled or rule.is_auto:
                continue
            if rule.scale and abs(rule.scale - want) > 0.01:
                continue
            found = rules.model_scale(rule.model)
            if found and abs(found - want) > 0.01:
                out.append(
                    f"row {index + 1} runs a {found}\u00d7 model but the target is "
                    f"{want:g}\u00d7, so the result gets resampled"
                )
        return out

    def save_rules(self) -> None:
        """Rules are remembered as they are edited, not only on Start."""
        self.settings.set("upscale", "rules", [r.to_dict() for r in self.rules])
        self.settings.save()

    def on_rule_checked(self) -> None:
        """The checkbox column changed a row in place."""
        self.render_rules()
        self.save_rules()
        self.update_summary()

    def rules_changed(self, which: str = "rules", select: int = -1) -> None:
        model, view = self._table(which)
        if 0 <= select < model.rowCount():
            view.select_row(select)
        self.render_rules()
        self.save_rules()
        self.update_summary()

    def default_rule_model(self) -> str:
        """A sensible model to open a new rule with - never a placeholder."""
        names = self.model_names()
        if not names:
            return ""
        scale = max(1, round(float(self.sp_scale.value())))
        return rules.gray_model(names, scale, rules.GRAY_TOP_BUCKET) or names[0]

    def rule_add(self, which: str = "rules") -> None:
        exclusion = which == "excl"
        draft = rules.Rule(
            kind=rules.ANY if exclusion else rules.GRAYSCALE,
            scale=0.0 if exclusion else float(self.sp_scale.value()),
            # A new exclusion opens on the case it exists for: pages far taller
            # than a page, which is what the old hardcoded switch matched.
            height=rules.dim_spec(3000, 0) if exclusion else rules.ANY,
            auto_levels=None if exclusion else True,
            model=self.default_rule_model(),
            action=rules.PASSTHROUGH if exclusion else rules.UPSCALE,
        )
        title = "Add exclusion" if exclusion else "Add rule"
        made = RuleDialog.edit(self, title, draft, self.model_names())
        if made is None:
            return
        self._insert_rule(which, made)

    def _insert_rule(self, which: str, made: rules.Rule) -> None:
        """File an edited rule in whichever table its action belongs to."""
        target = "excl" if made.action == rules.PASSTHROUGH else "rules"
        model, _view = self._table(target)
        items = list(model.rules)
        index = self._table(which)[1].current_row() if target == which else -1
        at = len(items) if index < 0 else index + 1
        items.insert(at, made)
        model.set_rules(items)
        self.rules_changed(target, at)

    def rule_edit(self, which: str = "rules") -> None:
        model, view = self._table(which)
        index = view.current_row()
        if index < 0:
            return
        title = "Edit exclusion" if which == "excl" else "Edit rule"
        made = RuleDialog.edit(self, title, model.rules[index], self.model_names())
        if made is None:
            return
        items = list(model.rules)
        if (made.action == rules.PASSTHROUGH) != (which == "excl"):
            # Its action changed, so the row now belongs in the other table.
            del items[index]
            model.set_rules(items)
            self._insert_rule(which, made)
            return
        items[index] = made
        model.set_rules(items)
        self.rules_changed(which, index)

    def rule_remove(self, which: str = "rules") -> None:
        model, view = self._table(which)
        index = view.current_row()
        if index < 0:
            return
        items = list(model.rules)
        del items[index]
        model.set_rules(items)
        self.rules_changed(which, min(index, len(items) - 1))

    def rule_move(self, delta: int, which: str = "rules") -> None:
        model, view = self._table(which)
        index = view.current_row()
        if index < 0:
            return
        items = rules.move(model.rules, index, delta)
        model.set_rules(items)
        self.rules_changed(which, max(0, min(index + delta, len(items) - 1)))

    def rule_toggle(self, which: str = "rules") -> None:
        model, view = self._table(which)
        index = view.current_row()
        if index >= 0:
            model.toggle(index)

    def rules_reset(self) -> None:
        if self.rules_model.rules and not self._confirm(
            "Reset the rules table",
            "Replace every row with the shipped set, built from the models you "
            "have installed?\n\nOnly this table changes \u2014 your exclusions and "
            "the rest of your settings are left alone.",
        ):
            return
        self.rules_model.set_rules(rules.default_working_set(self.model_names()))
        self._rules_seeded = True
        self.rules_changed("rules", 0)
        self.log(f"rules reset to the shipped set ({len(self.rules_model.rules)} rows)")

    def _confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    # ------------------------------------------------------------------ #
    # presets
    # ------------------------------------------------------------------ #
    def _fill_preset_menu(self) -> None:
        menu = self.preset_menu
        menu.clear()
        menu.addAction("Save current settings\u2026", self.preset_save)
        menu.addAction("Load from file\u2026", self.preset_load)
        saved = presets.available(self.dir)
        if saved:
            menu.addSeparator()
            for name, path in saved:
                menu.addAction(name, lambda target=path: self.preset_apply(target))

    def preset_save(self) -> None:
        self.sync_settings()
        name, ok = QInputDialog.getText(
            self, "Save preset", "Name this preset:", text="My settings"
        )
        if not ok or not name.strip():
            return
        payload = presets.build(name.strip(), self.settings.data, app_version=__version__)
        try:
            written = presets.write(presets.folder(self.dir) / presets.filename(name), payload)
        except OSError as exc:
            self.show_banner(f"Could not write the preset: {exc}")
            return
        self.settings.save()
        self.log(f"preset saved: {written.name}  \u00b7  {presets.summary(payload)}")

    def preset_load(self) -> None:
        start = presets.folder(self.dir)
        chosen, _filter = QFileDialog.getOpenFileName(
            self,
            "Load preset",
            str(start if start.is_dir() else self.dir),
            "JaNai preset (*.json);;All files (*)",
        )
        if chosen:
            self.preset_apply(Path(chosen))

    def preset_apply(self, path: Path) -> None:
        try:
            preset = presets.read(path)
        except presets.PresetError as exc:
            self.show_banner(str(exc))
            return
        self.sync_settings()  # so untouched sections survive the merge
        changed = presets.apply(self.settings.data, preset)
        self.reload_widgets()
        name = str(preset.get("name") or path.stem)
        if changed:
            self.hide_banner()
            self.log(f"preset applied: {name}  \u00b7  {', '.join(changed)}")
        else:
            self.log(f"preset {name} already matches the current settings")

    def reload_widgets(self) -> None:
        """Push the settings dict back into the widgets, after a preset.

        The device and the folders are deliberately left alone: a preset
        describes how to convert, not where this machine keeps things.
        """
        d = self.settings.data
        u, o, p = d["upscale"], d["output"], d["perf"]
        lg = d.get("log") or {}

        self.seg_mode.set_value(str(u.get("mode", "scale")))
        self.sp_scale.setValue(float(u.get("scale", 2.0)))
        self.sp_width.setValue(int(u.get("width", 2048)))
        self.sp_height.setValue(int(u.get("height", 2160)))
        set_combo(self.cb_display, displays.label_for_id(str(u.get("display", displays.CUSTOM))))
        self.seg_orient.set_value(bool(u.get("display_portrait", True)))
        self.chk_levels.setChecked(bool(u.get("auto_levels", True)))
        saved_kind = str(u.get("page_kind") or "").strip().lower()
        if saved_kind not in {"detect", "grayscale", "colour"}:
            saved_kind = "detect" if bool(u.get("grayscale_convert", True)) else "colour"
        self.set_page_kind(saved_kind)
        self.sp_threshold.setValue(int(u.get("grayscale_threshold", 12)))
        self.sp_colour.setValue(float(u.get("grayscale_colour_percent", 0.25)))
        self.sp_pre_h.setValue(int(u.get("pre_downscale_height", 0)))
        self.set_all_rules([rules.Rule.from_dict(r) for r in (u.get("rules") or [])])
        self._rules_seeded = bool(self.rules)

        self.seg_container.set_value(str(o.get("container", "files")))
        self.chk_same.setChecked(bool(o.get("same_as_input", True)))
        self.ed_sub.setText(str(o.get("subfolder", "upscaled")))
        self.ed_pattern.setText(str(o.get("pattern") or "{name}_JaNai"))
        self.chk_overwrite.setChecked(bool(o.get("overwrite", False)))
        self.chk_keep_tree.setChecked(bool(o.get("keep_structure", True)))

        self.seg_fmt.set_value(str(d["format"].get("id", "png")))
        for fid, spec in FORMATS.items():
            saved = self.settings.format_options(fid)
            for opt in spec.opts:
                self.fmt_values[fid][opt.key] = saved.get(opt.key, opt.default)

        self.chk_fp16.setChecked(bool(p.get("use_fp16", True)))
        set_combo(self.cb_tile, tile_label(str(p.get("tile", "auto"))))
        self.sp_budget.setValue(int(p.get("budget_limit", 0)))
        self.chk_wipe.setChecked(bool(p.get("force_cache_wipe", False)))
        self.sp_threads.setValue(int(p.get("torch_threads", 0)))
        self.sp_io.setValue(max(1, int(p.get("io_workers", 2))))
        self.sp_vips.setValue(int(p.get("vips_concurrency", 0)))
        self.chk_cudnn.setChecked(bool(p.get("cudnn_benchmark", False)))
        self.chk_tf32.setChecked(bool(p.get("allow_tf32", False)))
        self.chk_wake.setChecked(bool(p.get("gpu_wake_lock", True)))
        self.chk_wrap.setChecked(bool(lg.get("wrap", False)))
        self.chk_debug.setChecked(bool(lg.get("show_debug", False)))

        self.set_gray_rows(self.gray_rules_live())
        self.sync_detection_fields()
        self.render_format_options()
        self.render_target()
        self.render_rules()
        self.apply_log_wrap()
        self.on_dest_change()
        self.update_summary()
        self.update_start_state()
        self.update_wake_lock()
        self.settings.save()

    # ------------------------------------------------------------------ #
    # devices
    # ------------------------------------------------------------------ #
    def _known_devices(self) -> list[dict]:
        """Probed devices, falling back to the cached probe from last launch."""
        if self.devices:
            return [d for d in self.devices if isinstance(d, dict)]
        cached = (self.settings.data.get("probe") or {}).get("devices") or []
        return [d for d in cached if isinstance(d, dict)]

    def _device_label(self, value: str) -> str:
        value = (value or "").strip()
        if not value:
            return AUTO_DEVICE
        for device in self._known_devices():
            if str(device.get("value")) == value:
                return str(device.get("label"))
        return value

    def _device_labels(self) -> list[str]:
        labels = [AUTO_DEVICE] + [str(d.get("label")) for d in self._known_devices()]
        current = self.cb_device.currentText() if hasattr(self, "cb_device") else ""
        if current and current not in labels:
            labels.append(current)
        return labels

    def device_value(self) -> str:
        """The device string for the job. Empty means auto, never a silent CPU
        fallback: before the probe lands the saved choice is kept."""
        text = self.cb_device.currentText().strip()
        if not text or text == AUTO_DEVICE:
            return ""
        for device in self._known_devices():
            if str(device.get("label")) == text:
                return str(device.get("value"))
        if DEVICE_RE.match(text):
            return text
        return self._saved_device

    def on_device_change(self) -> None:
        """Keep the user's FP16 preference; only report what the device can do."""
        value = self.device_value()
        self._saved_device = value
        known = self._known_devices()
        if value:
            device = next((d for d in known if str(d.get("value")) == value), None)
        else:
            device = next((d for d in known if str(d.get("value")) != "cpu"), None)
        note = ""
        if device is not None:
            supported = bool(device.get("fp16")) and str(device.get("value")) != "cpu"
            self.chk_fp16.setEnabled(supported)
            if not supported and self.chk_fp16.isChecked():
                note = (
                    "this device runs FP32 \u2014 the preference is kept for GPUs that support it"
                )
        self.lbl_fp16.setText(note)
        # The line exists only to warn. When FP16 simply works, the checkbox
        # already says so, so nothing is added and the line disappears.
        self.lbl_fp16.setVisible(bool(note))
        self.update_wake_lock()
        self.update_summary()

    # ------------------------------------------------------------------ #
    # summaries
    # ------------------------------------------------------------------ #
    def update_summary(self) -> None:
        if str(self.seg_mode.value()) == "fit":
            # The display preset box has to follow a size typed in by hand.
            found = displays.match(self.sp_width.value(), self.sp_height.value())
            if displays.id_for_label(self.cb_display.currentText()) != found:
                set_combo(self.cb_display, displays.label_for_id(found))
        # There is deliberately no summary line under this card. The target, the
        # rule count and the exclusion count each restated a control a few pixels
        # above them, and both tables now carry their own count.
        self.sync_detection_fields()
        self.refresh_rule_warnings()

        cid = self.container_value()
        self.lbl_container_hint.setText(CONTAINERS[cid].hint)
        packs = packs_archive(cid)

        fid = str(self.seg_fmt.value())
        if fid in FORMATS:
            try:
                enc = summary(fid, self.format_values(fid))
            except Exception:
                enc = FORMATS[fid].label
            dest = self.resolved_out_dir()
            dest_text = str(dest) if dest else "choose a destination"
            pack = f"  \u2192  {CONTAINERS[cid].label}" if packs else ""
            self.lbl_out_sum.setText(f"{enc}{pack}  \u2192  {dest_text}")

        device = self.cb_device.currentText() or AUTO_DEVICE
        hint = [device, "FP16" if self.chk_fp16.isChecked() else "FP32"]
        hint.append(f"tile {self.cb_tile.currentText().lower()}")
        if self.chk_wake.isChecked():
            hint.append("GPU kept awake")
        self.panel_perf.set_hint(" \u00b7 ".join(hint))

    def update_start_state(self) -> None:
        ok = bool(self._in_path.strip()) and self.resolved_out_dir() is not None
        # An empty table means no model would run, so there is nothing to start.
        ok = ok and any(r.enabled for r in self.rules)
        running = self.runner.running
        if running:
            self.btn_start.setText("Cancel")
            self.btn_start.setProperty("variant", "")
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(not self.dry)
            self.btn_dry.setEnabled(False)
        else:
            self.btn_start.setText("Start")
            self.btn_start.setProperty("variant", "accent")
            self.btn_start.setEnabled(ok)
            self.btn_dry.setEnabled(ok)
            self.btn_pause.setEnabled(False)
            self.btn_pause.setText("Pause")
        style = self.btn_start.style()
        style.unpolish(self.btn_start)
        style.polish(self.btn_start)
        self.btn_open.setEnabled(self.last_out_dir is not None)

    # ------------------------------------------------------------------ #
    # GPU wake lock
    # ------------------------------------------------------------------ #
    def wake_lock_device(self) -> str | None:
        """Device to hold awake: "" for auto, None when it should not be held."""
        if not self.chk_wake.isChecked():
            return None
        value = self.device_value()
        if value == "cpu":
            return None
        if value:
            return value
        known = self._known_devices()
        if known and not any(str(d.get("value")) != "cpu" for d in known):
            return None
        return ""

    def update_wake_lock(self) -> None:
        """Start or release the idle hold. Never held while a job is running,
        so the worker gets the whole card to itself."""
        want = self.wake_lock_device()
        if self.runner.running or want is None:
            if self.runner.holding:
                self.runner.hold_stop()
            return
        if self.runner.holding and self.runner.hold_device == want:
            return
        self.runner.hold_start(want)

    def on_wake_toggle(self) -> None:
        if not self.chk_wake.isChecked() and self.runner.holding:
            self.log("GPU wake lock off")
        self.update_wake_lock()
        self.update_summary()

    def on_hold(self, event: dict) -> None:
        if event.get("released"):
            return
        if event.get("ok"):
            held = int(event.get("reserved") or 0)
            where = str(event.get("name") or event.get("device") or "GPU")
            extra = f" ({fmt_bytes(held)} reserved)" if held else ""
            self.log(f"GPU wake lock on {where}{extra}", "debug")
        else:
            self.log(f"GPU wake lock unavailable: {event.get('error', 'unknown reason')}", "warn")

    # ------------------------------------------------------------------ #
    # hardware profile
    # ------------------------------------------------------------------ #
    def profile_for_run(self) -> dict:
        """The measurements this run may use, or ``{}`` to plan cautiously."""
        return hardware.profile_for_run(self.profile, self.probe, self.chk_fp16.isChecked())

    def profile_model_paths(self) -> list[str]:
        """The models this table would actually run.

        What gets measured is per model, so measuring a model no rule names
        buys nothing. Measured on a real machine: picking one model per scale
        chose the file that sorts first at 4x, while the table ran a different
        4x file, so every page still fell back to the cautious first-page tile
        and the measurement bought nothing at all.

        So: one model per way a page can be routed - grayscale or colour, at
        each factor - taken from the rules that are switched on. Then one per
        factor the rules never mention, so a table of only 2x rows still
        learns something about a 4x file. Capped, because every model costs a
        ladder of real passes.
        """
        by_name: dict[str, str] = {}
        scale_of: dict[str, Any] = {}
        for entry in self.models:
            if isinstance(entry, dict) and entry.get("path"):
                by_name[str(entry.get("name"))] = str(entry["path"])
                scale_of[str(entry["path"])] = entry.get("scale")
        routes: dict[tuple[str, int], str] = {}
        for rule in self.rules:
            if not rule.enabled or rule.action == rules.PASSTHROUGH or rule.is_auto:
                continue
            path = by_name.get(rule.model)
            if not path:
                continue
            factor = rules.bucket_scale(rule.scale or rules.model_scale(rule.model))
            routes.setdefault((rule.kind, factor), path)
        paths: list[str] = []
        for path in routes.values():
            if path not in paths:
                paths.append(path)
        covered = {scale_of.get(path) for path in paths}
        for entry in self.models:
            if not (isinstance(entry, dict) and entry.get("path")):
                continue
            if entry.get("scale") in covered:
                continue
            covered.add(entry.get("scale"))
            paths.append(str(entry["path"]))
        # Four is the number of routes a page can take (grayscale/colour at
        # 2x/4x); past that the wait stops being worth the measurement.
        return paths[:4]

    def build_profile_job(self) -> dict:
        """A job that measures this machine and converts nothing.

        Only ``perf`` matters here: there is no input, output or format,
        because nothing is written. The models come from the probe, and the
        fingerprint records which machine the numbers belong to so they are
        dropped rather than trusted once it changes.
        """
        self.sync_settings()
        perf = dict(self.settings.data["perf"])
        perf["profile"] = None
        return {
            "perf": perf,
            "models": self.profile_model_paths(),
            "fingerprint": hardware.fingerprint(self.probe),
        }

    def on_profile_clicked(self) -> None:
        if self.runner.running:
            self.log("something is already running, so the measurement has to wait", "warn")
            return
        job = self.build_profile_job()
        if not job["models"]:
            self.show_banner("No models are installed, so there is nothing to measure.")
            return
        self._profiling = True
        self.btn_profile.setEnabled(False)
        self.lbl_profile.setText("measuring\u2026")
        self.started_at = time.time()
        self.log("measuring this machine - no images are written")
        if not self.runner.start_profile(job):
            self._profiling = False
            self.btn_profile.setEnabled(True)
            self.render_profile()
            self.log("the worker would not start", "error")
            return
        self.update_start_state()

    def on_profile_progress(self, event: dict) -> None:
        """One line per step, in the panel rather than the log.

        The worker reports each step twice: once on the way in, which is what
        the label follows, and once on the way out carrying the result.
        """
        index = int(event.get("index") or 0)
        total = max(1, int(event.get("total") or 1))
        name = str(event.get("model") or "")
        tile = int(event.get("tile") or 0)
        if "ok" not in event:
            self.lbl_profile.setText(f"measuring {index}/{total}: {name} at {tile}px")
        elif not event.get("ok"):
            self.log(f"{name}: {tile}px did not fit, so that is the ceiling", "debug")

    def on_profile(self, event: dict) -> None:
        self._profiling = False
        self.btn_profile.setEnabled(True)
        profile = event.get("profile")
        if not event.get("ok") or not isinstance(profile, dict):
            reason = "cancelled" if event.get("cancelled") else ""
            reason = reason or str(event.get("error") or "nothing could be measured")
            self.log(f"the measurement did not finish: {reason}", "warn")
            self.render_profile()
            return
        self.profile = profile
        self.settings.data["profile"] = profile
        self.settings.save()
        count = len(hardware.profile_models(profile))
        elapsed = fmt_secs(float(event.get("elapsed") or 0.0))
        self.log(
            f"measured {count} model(s) in {elapsed} \u2014 Auto now sizes tiles from this",
            "ok",
        )
        self.render_profile()
        self.update_summary()

    def render_profile(self) -> None:
        """Say what the measurements know, and whether they still apply."""
        models = hardware.profile_models(self.profile)
        if not models:
            self.btn_profile.setText("Measure this machine")
            self.lbl_profile.setText(
                "not measured \u2014 Auto holds the first page of an unseen model at "
                "1024px until it has measured it"
            )
            return
        if not hardware.profile_is_current(self.profile, self.probe):
            self.btn_profile.setText("Measure this machine")
            was = str((self.profile.get("hardware") or {}).get("name") or "another machine")
            self.lbl_profile.setText(
                f"measured on {was}, which is not what is here now \u2014 not in use"
            )
            return
        self.btn_profile.setText("Measure again")
        best = max(
            (int(e.get("best_tile") or 0) for e in models.values() if isinstance(e, dict)),
            default=0,
        )
        bits = [f"{len(models)} model(s) measured"]
        if best:
            bits.append(f"fastest tile {best}px")
        created = str(self.profile.get("created") or "")[:10]
        if created:
            bits.append(created)
        self.lbl_profile.setText("   \u00b7   ".join(bits))

    def offer_profile(self) -> None:
        """A first run, or new hardware: say so once, and never block on it."""
        if self._profile_offered or self._profiling or not self.models:
            return
        if hardware.profile_is_current(self.profile, self.probe):
            return
        gpus = [d for d in self.devices if isinstance(d, dict) and str(d.get("value")) != "cpu"]
        if not gpus:
            # Nothing to size against: the CPU path has no VRAM budget.
            return
        self._profile_offered = True
        what = (
            "The hardware changed since the last measurement"
            if hardware.profile_models(self.profile)
            else "This machine has not been measured yet"
        )
        press = self.btn_profile.text()
        self.log(
            f"{what}. Open Performance and press \u201c{press}\u201d so Auto can size"
            " tiles from measurements instead of a cautious guess.",
            "warn",
        )
        if not (self.probe.get("errors") or []):
            self.show_banner(
                f"{what}. Performance \u203a Hardware profile \u2192 \u201c{press}\u201d"
                " measures it once, in about a minute, and writes no images."
            )

    # ------------------------------------------------------------------ #
    # probe
    # ------------------------------------------------------------------ #
    def refresh_probe(self) -> None:
        self.lbl_env.setText("detecting hardware\u2026")
        self.btn_refresh.setEnabled(False)
        self.runner.probe()

    def apply_probe(self, probe: dict, cached: bool = False) -> None:
        if not probe:
            self._refresh_device_list()
            return
        self.probe = probe
        self.caps = probe.get("formats", {}) or {}
        self.models = probe.get("models", []) or []
        self.devices = probe.get("devices", []) or []

        self._refresh_device_list()

        # The shipped table is built from the models actually installed, so it
        # can only be seeded once the probe has reported them.
        self.seed_rules()
        self.render_rules()

        for fid in FORMAT_IDS:
            self.seg_fmt.set_option_enabled(fid, bool(self.caps.get(fid, {}).get("ok", True)))
        current = str(self.seg_fmt.value())
        if not self.caps.get(current, {}).get("ok", True):
            fallback = next((f for f in FORMAT_IDS if self.caps.get(f, {}).get("ok")), "png")
            self.log(
                f"{FORMATS[current].label} is unavailable, switching to {FORMATS[fallback].label}",
                "warn",
            )
            self.seg_fmt.set_value(fallback)

        self.lbl_env.setText(self._environment_line(probe, cached))

        if not cached:
            self.settings.data["probe"] = probe
            self.btn_refresh.setEnabled(True)
            errors = probe.get("errors") or []
            if errors:
                self.show_banner("; ".join(str(e) for e in errors[:2]))
            else:
                self.hide_banner()
            self.settings.save()

        self.render_format_options()
        self.update_summary()
        self.update_start_state()
        self.update_wake_lock()
        self.render_profile()
        if not cached:
            # Only meaningful once the real probe has arrived: a cached probe
            # cannot tell whether the hardware changed underneath it.
            self.offer_profile()

    def _refresh_device_list(self) -> None:
        """Refill the device picker, keeping the current choice selected."""
        wanted = self._device_label(self._saved_device)
        self.cb_device.blockSignals(True)
        self.cb_device.clear()
        self.cb_device.addItems(self._device_labels())
        set_combo(self.cb_device, wanted)
        self.cb_device.blockSignals(False)
        self.on_device_change()

    def _environment_line(self, probe: dict, cached: bool) -> str:
        """The line under the title: what this machine can actually do."""
        devices = [d for d in (probe.get("devices") or []) if isinstance(d, dict)]
        gpu = next((d for d in devices if str(d.get("value")) != "cpu"), None)
        bits: list[str] = []
        if gpu is not None:
            name = str(gpu.get("label") or gpu.get("value"))
            vram = int(gpu.get("vram") or 0)
            bits.append(f"{name}  \u00b7  {fmt_bytes(vram)}" if vram else name)
        else:
            bits.append("CPU only")
        if self.models:
            bits.append(f"{len(self.models)} models")
        working = [f for f in FORMAT_IDS if self.caps.get(f, {}).get("ok")]
        if working:
            bits.append(f"{len(working)} encoders")
        if cached:
            bits.append("cached \u00b7 press Re-detect to refresh")
        return "   \u00b7   ".join(bits)

    def on_probe_error(self, event: dict) -> None:
        self.btn_refresh.setEnabled(True)
        self.lbl_env.setText("backend not ready")
        message = str(event.get("message") or "").strip()
        self.show_banner(
            (
                "The Python backend is not ready. Run setup.cmd in this folder "
                f"to create it.\n{message}"
            ).strip()
        )

    # ------------------------------------------------------------------ #
    # banner
    # ------------------------------------------------------------------ #
    def show_banner(self, text: str) -> None:
        self.banner.show_text(text)

    def hide_banner(self) -> None:
        self.banner.hide()

    # ------------------------------------------------------------------ #
    # worker events
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        """Drain whatever the worker said since the last frame."""
        self.runner.drain(self.on_event)

    def on_event(self, event: dict) -> None:
        kind = str(event.get("type") or "")
        if kind == "scan":
            self.on_scan(event)
        elif kind == "probe":
            self.apply_probe(event)
        elif kind == "probe_error":
            self.on_probe_error(event)
        elif kind == "start":
            self.on_job_start(event)
        elif kind == "progress":
            self.on_progress(event)
        elif kind == "file":
            self.on_file(event)
        elif kind == "bundle":
            text, level = format_bundle(event)
            self.log(text, level)
        elif kind == "profile":
            self.on_profile(event)
        elif kind == "profile_progress":
            self.on_profile_progress(event)
        elif kind == "log":
            self.log(str(event.get("message", "")), str(event.get("level", "info")))
        elif kind == "done":
            self.on_done(event)
        elif kind == "hold":
            self.on_hold(event)
        elif kind == "exit":
            self.on_exit(event)

    def on_scan(self, event: dict) -> None:
        if str(event.get("path") or "") != self._in_path:
            return
        kind = str(event.get("kind") or "")
        images = int(event.get("images") or 0)
        archives = int(event.get("archives") or 0)
        folders = int(event.get("folders") or 0)
        if kind == "missing":
            self.card_input.set_badge("not found")
            self.scan_text = ""
            self.update_start_state()
            return
        parts = []
        if images:
            parts.append(f"{images} image{'s' if images != 1 else ''}")
        if archives:
            parts.append(f"{archives} archive{'s' if archives != 1 else ''}")
        if folders > 1:
            parts.append(f"{folders} folders")
        self.scan_text = ", ".join(parts) or "nothing to do"
        head = "Single" if kind == "single" else "Bulk"
        self.card_input.set_badge(f"{head} \u00b7 {self.scan_text}")
        self.update_summary()

    def on_job_start(self, event: dict) -> None:
        self.total = int(event.get("total") or 0)
        self.dry = bool(event.get("dry")) or self.dry
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.started_index = 0
        self.progress_sub = ""
        self.render_status()
        self.lbl_detail.setText(
            f"{event.get('device')} \u00b7 "
            f"{'FP16' if event.get('fp16') else 'FP32'} \u00b7 "
            f"tile {event.get('tile')}"
        )
        for text, level in format_start(event):
            self.log(text, level)

    def render_status(self, total: int = 0) -> None:
        """Write the status line from one place.

        Two counters used to fight over this label: the worker's progress
        index, which counts the files it has *started*, and the finished count.
        The worker reads, upscales and writes on separate threads, so by the
        time a file is finished it has already announced the next one or two -
        which made the number jump forward and then fall back a moment later.
        Show the oldest file still in flight instead (finished + 1, never past
        what the worker has actually started): that only ever moves forward.
        """
        total = total or self.total or 0
        finished = self.completed + self.failed + self.skipped
        current = min(finished + 1, self.started_index) if self.started_index else finished
        current = max(current, finished)
        if total:
            current = min(current, total)
        elapsed = max(0.001, time.time() - self.started_at)
        rate = self.completed / elapsed if self.completed else 0.0
        left = max(0, total - finished)
        eta = f" \u00b7 ETA {fmt_secs(left / rate)}" if rate > 0 and left > 0 else ""
        head = "Planning" if self.dry else "Upscaling"
        self.lbl_status.setText(f"{head} {current}/{total}{self.progress_sub}{eta}")

    def on_progress(self, event: dict) -> None:
        name = Path(str(event.get("path", ""))).name
        index = int(event.get("i") or 0)
        total = int(event.get("total") or 0)
        if total:
            self.total = total
        # keep the furthest file the worker has started, but let render_status
        # decide which number to show, so read-ahead cannot reach the label
        self.started_index = max(self.started_index, index)
        self.progress_sub = (
            f" (page {event.get('sub_i')}/{event.get('sub_n')})" if event.get("sub_n") else ""
        )
        self.render_status(total)
        self.lbl_detail.setText(name)

    def on_file(self, event: dict) -> None:
        total = int(event.get("total") or self.total or 1)
        error = str(event.get("error") or "")
        if error:
            if "skip" in error.lower() or "exists" in error.lower():
                self.skipped += 1
            else:
                self.failed += 1
        else:
            self.completed += 1
        text, level = format_file(event)
        self.log(text, level)
        finished = self.completed + self.failed + self.skipped
        self.bar.setValue(int(min(100.0, 100.0 * finished / max(1, total))))
        self.render_status(total)

    def on_done(self, event: dict) -> None:
        elapsed = float(event.get("elapsed") or (time.time() - self.started_at))
        processed = int(event.get("processed") or self.completed)
        failed = int(event.get("failed") or self.failed)
        skipped = int(event.get("skipped") or self.skipped)
        dry = bool(event.get("dry")) or self.dry
        text, level = format_done(event)
        self.log(text, level)
        self.bar.setRange(0, 100)
        if event.get("error"):
            self.lbl_status.setText("Failed")
            self.lbl_detail.setText(str(event["error"]))
        elif event.get("cancelled"):
            self.lbl_status.setText(f"Cancelled after {processed} file(s)")
            self.lbl_detail.setText(fmt_secs(elapsed))
        else:
            bits = [f"{processed} file(s) in {fmt_secs(elapsed)}"]
            if failed:
                bits.append(f"{failed} failed")
            if skipped:
                bits.append(f"{skipped} skipped")
            self.lbl_status.setText("Dry run complete" if dry else "Done")
            self.lbl_detail.setText(" \u00b7 ".join(bits))
            self.bar.setValue(100)
        self.update_start_state()
        self.settings.save()

    def on_exit(self, event: dict) -> None:
        self.update_start_state()
        self.update_wake_lock()
        if self._profiling:
            # The worker left without reporting. Whatever went wrong, the
            # control must not be left dead.
            self._profiling = False
            self.btn_profile.setEnabled(True)
            self.render_profile()
        if int(event.get("code") or 0) not in (0, 1, 2):
            self.lbl_status.setText("Worker stopped unexpectedly")
            self.log(f"worker exited with code {event.get('code')}", "error")
        self.end_run_log()

    # ------------------------------------------------------------------ #
    # settings and jobs
    # ------------------------------------------------------------------ #
    def sync_settings(self) -> None:
        """Pull every widget back into the settings dict, ready to save."""
        d = self.settings.data
        d["theme"] = self.theme.p.name
        d["input"] = {
            "path": self._in_path,
            "recursive": self.chk_recursive.isChecked(),
            "include_archives": self.chk_archives.isChecked(),
        }
        d["upscale"] = {
            "mode": str(self.seg_mode.value()),
            "scale": float(self.sp_scale.value()),
            "width": int(self.sp_width.value()),
            "height": int(self.sp_height.value()),
            "display": displays.id_for_label(self.cb_display.currentText()),
            "display_portrait": bool(self.seg_orient.value()),
            "auto_levels": self.chk_levels.isChecked(),
            "page_kind": self.page_kind(),
            # Kept in step with page_kind so the worker, and any settings file
            # read by an older build, still sees the flag it understands.
            "grayscale_convert": self.page_kind() != "colour",
            "grayscale_threshold": int(self.sp_threshold.value()),
            "grayscale_colour_percent": float(self.sp_colour.value()),
            "pre_downscale_height": int(self.sp_pre_h.value()),
            "rules": [r.to_dict() for r in self.rules],
        }
        d["format"]["id"] = str(self.seg_fmt.value())
        for fid in FORMATS:
            d["format"]["options"][fid] = self.format_values(fid)
        d["output"] = {
            "dir": self.ed_out.text(),
            "same_as_input": self.chk_same.isChecked(),
            "subfolder": self.ed_sub.text(),
            "container": self.container_value(),
            "pattern": self.ed_pattern.text(),
            "overwrite": self.chk_overwrite.isChecked(),
            "keep_structure": self.chk_keep_tree.isChecked(),
        }
        d["perf"] = {
            "device": self.device_value(),
            "use_fp16": self.chk_fp16.isChecked(),
            "tile": tile_value(self.cb_tile.currentText()),
            "budget_limit": int(self.sp_budget.value()),
            "force_cache_wipe": self.chk_wipe.isChecked(),
            "torch_threads": int(self.sp_threads.value()),
            "io_workers": max(1, int(self.sp_io.value())),
            "vips_concurrency": int(self.sp_vips.value()),
            "cudnn_benchmark": self.chk_cudnn.isChecked(),
            "allow_tf32": self.chk_tf32.isChecked(),
            "gpu_wake_lock": self.chk_wake.isChecked(),
        }
        log_cfg = dict(d.get("log") or {})
        log_cfg.update(
            {"wrap": self.chk_wrap.isChecked(), "show_debug": self.chk_debug.isChecked()}
        )
        d["log"] = log_cfg
        # Only keys the settings file already knows: unknown ones are dropped
        # by the merge on load, and the geometry keeps the old "WxH+X+Y" form
        # so a settings file written by either build still opens correctly.
        d["ui"] = dict(
            d.get("ui") or {},
            advanced_format=self.chk_adv.isChecked(),
            perf_open=self.panel_perf.is_open(),
            log_open=self._log_visible,
            geometry=self._geometry_text(),
        )

    def build_job(self, dry: bool = False) -> dict | None:
        """The job payload for the worker, or None with the reason on screen."""
        self.sync_settings()
        d = self.settings.data
        src = Path(d["input"]["path"])
        if not src.exists():
            self.show_banner(f"Input not found: {src}")
            return None
        out_dir = self.resolved_out_dir()
        if out_dir is None:
            self.show_banner("Choose an output folder.")
            return None
        fid = d["format"]["id"]
        if self.caps and not self.caps.get(fid, {}).get("ok", False):
            self.show_banner(f"{FORMATS[fid].label} cannot be written in this install.")
            return None
        if not [r for r in d["upscale"]["rules"] if r.get("enabled", True)]:
            self.show_banner(
                "The rules table has no rows switched on, so no model would run. "
                "Add a rule, or press Defaults beside the table."
            )
            return None
        self.hide_banner()
        job = {
            "input": {
                "path": str(src),
                "mode": "single" if src.is_file() else "bulk",
                "recursive": d["input"]["recursive"],
                "include_archives": d["input"]["include_archives"],
            },
            "output": {
                "dir": str(out_dir),
                "container": d["output"]["container"],
                "pattern": d["output"]["pattern"],
                "overwrite": d["output"]["overwrite"],
                "keep_structure": d["output"]["keep_structure"],
            },
            "format": {"id": fid, "options": d["format"]["options"][fid]},
            "upscale": dict(d["upscale"], models_dir=str(self.runner.paths().models_dir or "")),
            # Measurements taken on this machine, in this precision - or {},
            # which leaves the worker on its cautious first-page path.
            "perf": dict(d["perf"], profile=self.profile_for_run()),
        }
        if dry:
            job["dry_run"] = True
        return job

    def on_start_clicked(self) -> None:
        """The primary button is Start, and Cancel while a job is running."""
        if self.runner.running:
            self.cancel()
        else:
            self.start()

    def start(self, dry: bool = False) -> None:
        if self.runner.running:
            return
        job = self.build_job(dry=dry)
        if job is None:
            return
        self.settings.save()
        self.dry = dry
        self.total = 0
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.started_index = 0
        self.progress_sub = ""
        self.started_at = time.time()
        self.bar.setValue(0)
        self.last_out_dir = Path(job["output"]["dir"])
        self.lbl_status.setText("Planning\u2026" if dry else "Starting worker\u2026")
        self.lbl_detail.setText("")
        self.begin_run_log(job, dry)
        if not dry and not self.runner.has_portable_python():
            self.log("no backend runtime found, using the interpreter running the GUI", "warn")
        self.runner.hold_stop()  # the job gets the whole card to itself
        if self.runner.start(job):
            self.update_start_state()

    def start_dry(self) -> None:
        self.start(dry=True)

    def cancel(self) -> None:
        if self.runner.running:
            self.runner.cancel()
            self.lbl_status.setText("Cancelling\u2026")

    def toggle_pause(self) -> None:
        if not self.runner.running:
            return
        if self.runner.paused:
            self.runner.resume()
            self.btn_pause.setText("Pause")
            self.lbl_status.setText("Resumed")
        else:
            self.runner.pause()
            self.btn_pause.setText("Resume")
            self.lbl_status.setText("Paused")

    def open_output(self) -> None:
        target = self.last_out_dir or self.resolved_out_dir()
        if target is not None and Path(target).exists():
            open_in_explorer(Path(target))

    # ------------------------------------------------------------------ #
    # run log file
    # ------------------------------------------------------------------ #
    def begin_run_log(self, job: dict, dry: bool) -> None:
        u, o, p = job["upscale"], job["output"], job["perf"]
        fid = job["format"]["id"]
        rows = [r for r in (u.get("rules") or []) if r.get("enabled", True)]
        used = sorted({str(r.get("model") or "") for r in rows})
        models = f"{len(rows)} rule(s)"
        if used:
            models += "  \u00b7  " + ", ".join(used[:3])
            if len(used) > 3:
                models += f", +{len(used) - 3} more"
        header = [
            (
                f"JaNai Upscaler \u2014 {'dry run' if dry else 'run'} "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            ),
            f"input    {job['input']['path']}",
            f"output   {o['dir']}",
            f"format   {FORMATS[fid].label}  \u00b7  package {CONTAINERS[o['container']].label}",
            f"rules    {models}",
            (
                f"device   {p.get('device') or 'auto'}  \u00b7  "
                f"{'FP16' if p.get('use_fp16') else 'FP32'}  \u00b7  tile {p.get('tile')}"
            ),
        ]
        path = self.runlog.begin(header, dry=dry)
        self.lbl_log_file.setText(str(path) if path else "not saved")
        self.log(
            f"{'dry run' if dry else 'job'}: {job['input']['path']} \u2192 {o['dir']} "
            f"[{FORMATS[fid].label}]"
        )
        if path:
            self.log(f"log: {path}", "debug")

    def end_run_log(self) -> None:
        if self.runlog.path is None:
            return
        self.runlog.end(
            [
                "-" * 78,
                (
                    f"ended {time.strftime('%Y-%m-%d %H:%M:%S')}  \u00b7  "
                    f"{self.completed} done, {self.failed} failed, "
                    f"{self.skipped} skipped"
                ),
            ]
        )

    # ------------------------------------------------------------------ #
    # theme, reset, close
    # ------------------------------------------------------------------ #
    def toggle_theme(self) -> None:
        """One stylesheet swap repaints the window - no walk over widgets."""
        mode = self.theme.toggle(self.app)
        self.settings.data["theme"] = mode
        self.rules_model.set_palette(self.theme.p)  # row colours come from it
        self.excl_model.set_palette(self.theme.p)
        self.log_view.setFont(self.theme.fonts["mono"])
        self.render_format_options()
        self.render_rules()
        self.settings.save()

    def reset_all(self) -> None:
        """Every setting back to the shipped defaults, folders excluded.

        Deliberately keeps the input and output paths and the cached hardware
        probe: nobody presses this wanting to retype where their manga lives or
        to wait for the backend to be detected again.
        """
        if not self._confirm(
            "Reset all settings",
            "Put every setting back to its default?\n\nThe rules table, target, "
            "output format and layout, performance options and log view are "
            "reset. Your input and output folders are kept.",
        ):
            return
        old = self.settings.data
        fresh = defaults()
        fresh["theme"] = old.get("theme", fresh.get("theme"))
        fresh["probe"] = old.get("probe", {})
        fresh["input"]["path"] = str((old.get("input") or {}).get("path", ""))
        out = old.get("output") or {}
        fresh["output"]["dir"] = str(out.get("dir", ""))
        fresh["output"]["same_as_input"] = bool(out.get("same_as_input", True))
        fresh["ui"] = dict(
            fresh.get("ui") or {}, geometry=(old.get("ui") or {}).get("geometry", "")
        )
        self.settings.data = fresh
        self._rules_seeded = False
        self.reload_widgets()
        self.seed_rules()
        self.render_rules()
        self.update_summary()
        self.log("all settings reset to defaults")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Save, close the run log, and let go of the GPU before quitting."""
        try:
            self.sync_settings()
            self.settings.save()
        except Exception:
            pass
        self._timer.stop()
        self.end_run_log()
        self.runner.hold_stop()
        if self.runner.running:
            self.runner.cancel()
            QTimer.singleShot(400, self.runner.kill)
        event.accept()
