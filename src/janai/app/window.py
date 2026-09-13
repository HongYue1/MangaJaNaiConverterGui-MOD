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
import threading
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
from janai.app.rules_table import RuleDialog, RulesModel, RulesTable
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
    DropZone,
    FieldGrid,
    LogView,
    Segmented,
    button,
    checkbox,
    combo,
    label,
    line_edit,
    row,
    set_combo,
    spin_float,
    spin_int,
)
from janai.core import displays, presets, rules
from janai.core.formats import (
    CONTAINER_IDS,
    CONTAINERS,
    FORMAT_IDS,
    FORMATS,
    Opt,
    is_active,
    packs_archive,
    summary,
)

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".jfif",
    ".webp",
    ".avif",
    ".jxl",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".heic",
    ".heif",
    ".ppm",
    ".pgm",
}
ARCHIVE_EXTS = {".zip", ".cbz", ".rar", ".cbr"}

FILE_FILTER = (
    "Images and archives (*.png *.jpg *.jpeg *.jfif *.webp *.avif *.jxl *.bmp "
    "*.tif *.tiff *.gif *.heic *.heif *.cbz *.zip *.cbr *.rar);;All files (*)"
)

TILE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Auto (adaptive)", "auto"),
    ("Maximum", "maximum"),
    ("No tiling", "none"),
    ("128 px", "128"),
    ("192 px", "192"),
    ("256 px", "256"),
    ("384 px", "384"),
    ("512 px", "512"),
    ("768 px", "768"),
    ("1024 px", "1024"),
)

MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Scale", "scale"),
    ("Width", "width"),
    ("Height", "height"),
    ("Fit", "fit"),
)

LOG_TAGS = ("info", "debug", "warn", "error", "ok", "skip", "dry")

#: An empty device string means "let the worker pick the best one".
AUTO_DEVICE = "Auto (best available)"
DEVICE_RE = re.compile(r"^(cpu|cuda|xpu|mps|dml|privateuseone)(:\d+)?$")

GEOMETRY_RE = re.compile(r"^(\d+)x(\d+)(?:\+(-?\d+)\+(-?\d+))?$")


def choice_label(opt: Opt, value: Any) -> str:
    for text, val in opt.choices:
        if val == value:
            return text
    return opt.choices[0][0] if opt.choices else str(value)


def choice_value(opt: Opt, text: str) -> Any:
    for lbl, val in opt.choices:
        if lbl == text:
            return val
    return opt.default


def tile_label(value: str) -> str:
    for text, val in TILE_CHOICES:
        if val == value:
            return text
    return TILE_CHOICES[0][0]


def tile_value(text: str) -> str:
    for lbl, val in TILE_CHOICES:
        if lbl == text:
            return val
    return "auto"


class MainWindow(QMainWindow):
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

        self.total = 0
        self.completed = 0
        self.failed = 0
        self.skipped = 0
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

    def _build_input(self) -> None:
        card = Card("Input", "A folder, an archive, or single images. Drop them here.")
        self.card_input = card
        self.drop = DropZone("Drop images, a folder or a .cbz here")
        self.drop.dropped.connect(self.on_drop)
        self.drop.set_path(self._in_path)
        card.body.full(self.drop)

        card.body.full(
            row(
                button("Choose file\u2026", self.choose_file, tip="Ctrl+O"),
                button("Choose folder\u2026", self.choose_folder, tip="Ctrl+Shift+O"),
                button("Clear", self.clear_input, variant="ghost"),
            )
        )

        i = self.settings.data["input"]
        self.chk_recursive = checkbox(
            "Include subfolders",
            bool(i.get("recursive", True)),
            self.on_input_options,
            tip=(
                "Walk the whole tree. With a CBZ package this is what turns each "
                "chapter folder into its own archive."
            ),
        )
        self.chk_archives = checkbox(
            "Include archives (cbz/zip/cbr/rar)",
            bool(i.get("include_archives", True)),
            self.on_input_options,
            tip=(
                "Comic archives found in the input are re-packed as .cbz with every page upscaled."
            ),
        )
        card.body.full(row(self.chk_recursive, self.chk_archives, spacing=18))
        self.page.addWidget(card)

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
            "Pages",
            "Whether each page is judged on its own, or the whole run is declared.",
            row(self.cb_pagekind, self.chk_levels, spacing=18),
        )

        body.full(self._build_rules_block(seeded))

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

        self.lbl_upscale_sum = label("", "muted", wrap=True)
        body.control(self.lbl_upscale_sum)
        self.page.addWidget(card)

    def _build_rules_block(self, seeded: list[rules.Rule]) -> QWidget:
        """The rules table, its side buttons, and the two lines beneath it."""
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
            seeded,
            bool(self.settings.data["upscale"].get("grayscale_convert", True)),
            self.model_names(),
            parent=self,
        )
        self.rules_model.edited.connect(self.on_rule_checked)
        self.rules_view = RulesTable(self.rules_model)
        self.rules_view.doubleClicked.connect(lambda _index: self.rule_edit())

        side = QWidget()
        sbox = QVBoxLayout(side)
        sbox.setContentsMargins(0, 0, 0, 0)
        sbox.setSpacing(6)
        actions: tuple[tuple[str, Callable[[], None], str], ...] = (
            ("Add", self.rule_add, "Add a rule below the selected one."),
            ("Edit", self.rule_edit, "Edit the selected rule (or double-click it)."),
            (
                "Toggle",
                self.rule_toggle,
                (
                    "Switch the selected rule off without deleting it. The checkbox "
                    "in the first column shows the state, and the space bar does the "
                    "same thing."
                ),
            ),
            ("Remove", self.rule_remove, "Delete the selected rule."),
            (
                "Up",
                lambda: self.rule_move(-1),
                (
                    "Move the rule up (Alt+Up). Order only decides between rules that "
                    "are equally specific."
                ),
            ),
            ("Down", lambda: self.rule_move(1), "Move the rule down (Alt+Down)."),
            (
                "Defaults",
                self.rules_reset,
                (
                    "Rewrite this table as the shipped set: the MangaJaNai height "
                    "bands for grayscale pages and the IllustrationJaNai denoise "
                    "models for colour, built from the models you have installed. "
                    "Only the table is touched, nothing else."
                ),
            ),
        )
        for text, action, hint in actions:
            btn = button(text, action, variant="ghost", tip=hint)
            btn.setMinimumWidth(96)
            sbox.addWidget(btn)
        sbox.addStretch(1)

        table_row = QWidget()
        tbox = QHBoxLayout(table_row)
        tbox.setContentsMargins(0, 0, 0, 0)
        tbox.setSpacing(10)
        tbox.addWidget(self.rules_view, 1)
        tbox.addWidget(side, 0)
        box.addWidget(table_row)

        self.lbl_rules_hint = label("", "hint", wrap=True)
        box.addWidget(self.lbl_rules_hint)
        self.lbl_rules_warn = label("", "warn", wrap=True)
        self.lbl_rules_warn.setVisible(False)
        box.addWidget(self.lbl_rules_warn)
        return block

    def _build_output(self) -> None:
        card = Card("Output", "The encoder, how pages are packaged, and where they land.")
        self.card_output = card
        body = card.body
        d = self.settings.data
        o = d["output"]

        self.seg_fmt = Segmented(
            tuple((FORMATS[fid].label, fid) for fid in FORMAT_IDS),
            str(d["format"].get("id", "png")),
        )
        self.seg_fmt.changed.connect(lambda _value: self.render_format_options())
        body.field("Format", "The encoder used for every page.", self.seg_fmt)

        self.lbl_fmt_hint = label("", "muted", wrap=True)
        body.control(self.lbl_fmt_hint)

        self.opt_box = FieldGrid()
        body.control(self.opt_box)

        self.chk_adv = checkbox(
            "Show advanced encoder options",
            bool(d["ui"].get("advanced_format", False)),
            self.render_format_options,
        )
        body.control(self.chk_adv)
        body.rule()

        self.seg_container = Segmented(
            tuple((CONTAINERS[cid].label, cid) for cid in CONTAINER_IDS),
            str(o.get("container", "files")),
        )
        self.seg_container.changed.connect(lambda _value: self.on_container_change())
        body.field(
            "Package", "Loose files, or pack the pages into comic archives.", self.seg_container
        )
        self.lbl_container_hint = label("", "muted", wrap=True)
        body.control(self.lbl_container_hint)
        body.rule()

        self.chk_same = checkbox(
            "Next to the input, in subfolder",
            bool(o.get("same_as_input", True)),
            self.on_dest_change,
        )
        self.ed_sub = line_edit(
            str(o.get("subfolder", "upscaled")), on_change=self.update_summary, width=190
        )
        body.field(
            "Destination",
            "Where the finished pages are written.",
            row(self.chk_same, self.ed_sub, spacing=10),
        )

        self.ed_out = line_edit(
            str(o.get("dir", "")),
            placeholder="Choose a folder\u2026",
            on_change=self.update_summary,
        )
        self.w_dest_custom = row(
            self.ed_out,
            button("Browse\u2026", self.choose_out_dir, variant="ghost"),
            spacing=8,
            stretch=False,
        )
        body.control(self.w_dest_custom)

        self.ed_pattern = line_edit(
            str(o.get("pattern") or "{name}_JaNai"),
            on_change=self.update_summary,
            tip=(
                "Tokens: {name} original name, {parent} folder name, {index} "
                "position, {index0} zero-padded position. With a CBZ package this "
                "names the pages inside the archive."
            ),
        )
        body.field("File names", "Tokens: {name} {parent} {index}", self.ed_pattern)

        self.chk_overwrite = checkbox(
            "Overwrite existing", bool(o.get("overwrite", False)), self.update_summary
        )
        self.chk_keep_tree = checkbox(
            "Mirror folder structure",
            bool(o.get("keep_structure", True)),
            self.update_summary,
            tip="Recreate the input's subfolder layout inside the output folder.",
        )
        body.control(row(self.chk_overwrite, self.chk_keep_tree, spacing=18))

        self.lbl_out_sum = label("", "muted", wrap=True)
        body.control(self.lbl_out_sum)
        self.page.addWidget(card)

    def _build_perf(self) -> None:
        panel = Collapsible(
            "Performance",
            "device, precision, tiling, threads",
            expanded=bool(self.settings.data["ui"].get("perf_open", False)),
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
                "the first run starts at full speed. Costs a few MB of VRAM and is "
                "released automatically while a job runs."
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

    def _build_log(self) -> None:
        panel = QWidget(self.splitter)
        box = QVBoxLayout(panel)
        box.setContentsMargins(18, 0, 18, 0)
        box.setSpacing(8)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(10)
        bar.addWidget(label("Run log", "title"))
        self.lbl_log_file = label("", "hint")
        bar.addWidget(self.lbl_log_file, 1)

        lg = self.settings.data.get("log") or {}
        self.chk_wrap = checkbox("Wrap", bool(lg.get("wrap", False)), self.apply_log_wrap)
        self.chk_debug = checkbox(
            "Debug",
            bool(lg.get("show_debug", False)),
            tip=(
                "Show the noisy lines (library warnings, tracebacks). They are always "
                "written to the run file either way."
            ),
        )
        bar.addWidget(self.chk_wrap)
        bar.addWidget(self.chk_debug)
        for text, action in (
            ("Copy", self.copy_log),
            ("Save as\u2026", self.save_log_as),
            ("Log folder", self.open_log_folder),
            ("Clear", self.clear_log),
            ("Hide", self.toggle_log),
        ):
            bar.addWidget(button(text, action, variant="ghost"))
        box.addLayout(bar)

        self.log_view = LogView(self.theme.fonts["mono"], panel)
        self.log_view.set_wrap(bool(lg.get("wrap", False)))
        box.addWidget(self.log_view, 1)
        self.log_panel = panel

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
        self._shortcut("Alt+Up", lambda: self.rule_move(-1))
        self._shortcut("Alt+Down", lambda: self.rule_move(1))

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
        if area is not None:
            width = min(width, int(area.width() * 0.88))
            height = min(height, int(area.height() * 0.90))
        self.resize(max(920, width), max(620, height))
        if match is None or match.group(3) is None:
            if area is not None:
                self.move(
                    area.x() + max(0, (area.width() - self.width()) // 2),
                    area.y() + max(0, (area.height() - self.height()) // 3),
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

    def _log_dir(self) -> Path:
        raw = str((self.settings.data.get("log") or {}).get("dir") or "logs")
        path = Path(raw).expanduser()
        return path if path.is_absolute() else self.dir / path

    # ------------------------------------------------------------------ #
    # input handling
    # ------------------------------------------------------------------ #
    def choose_file(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Choose an image or archive", self._in_path or "", FILE_FILTER
        )
        if path:
            self.set_input(path)

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a folder", self._in_path or "")
        if path:
            self.set_input(path)

    def choose_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose an output folder", self.ed_out.text() or ""
        )
        if path:
            self.chk_same.setChecked(False)
            self.ed_out.setText(path)
            self.on_dest_change()

    def clear_input(self) -> None:
        self._in_path = ""
        self.scan_text = ""
        self.drop.set_path("")
        self.card_input.set_badge("")
        self.update_start_state()
        self.update_summary()

    def on_drop(self, paths: list) -> None:
        if not paths:
            return
        self.set_input(str(paths[0]))
        if len(paths) > 1:
            self.log(f"{len(paths)} items dropped; using {Path(paths[0]).name}", "warn")

    def set_input(self, path: str) -> None:
        self._in_path = path
        self.drop.set_path(path)
        self.scan_input_async(path)
        self.update_start_state()
        self.update_summary()

    def on_input_options(self) -> None:
        if self._in_path:
            self.scan_input_async(self._in_path)
        self.update_summary()

    def scan_input_async(self, path: str) -> None:
        """Count what the input holds without blocking the window."""
        recursive = self.chk_recursive.isChecked()
        archives = self.chk_archives.isChecked()

        def work() -> None:
            target = Path(path)
            images = arch = 0
            folders: set[str] = set()
            kind = "single"
            try:
                if target.is_file():
                    if target.suffix.lower() in ARCHIVE_EXTS:
                        arch = 1
                    else:
                        images = 1
                elif target.is_dir():
                    kind = "bulk"
                    walk = target.rglob("*") if recursive else target.glob("*")
                    for item in walk:
                        if not item.is_file():
                            continue
                        ext = item.suffix.lower()
                        if ext in IMAGE_EXTS:
                            images += 1
                            folders.add(str(item.parent))
                        elif archives and ext in ARCHIVE_EXTS:
                            arch += 1
                else:
                    kind = "missing"
            except Exception:
                pass
            self.runner.events.put(
                {
                    "type": "scan",
                    "kind": kind,
                    "images": images,
                    "archives": arch,
                    "folders": len(folders),
                    "path": path,
                }
            )

        threading.Thread(target=work, name="scan", daemon=True).start()

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
        """The page kind decides which half of the table can fire, so recolour it."""
        self.rules_model.set_gray(self.gray_rules_live())
        self.render_rules()
        self.update_summary()

    def on_levels_toggle(self) -> None:
        self.render_rules()
        self.update_summary()

    def on_container_change(self) -> None:
        self.update_summary()
        self.update_start_state()

    def on_dest_change(self) -> None:
        same = self.chk_same.isChecked()
        self.ed_sub.setEnabled(same)
        self.w_dest_custom.setVisible(not same)
        self.update_start_state()
        self.update_summary()

    def render_format_options(self) -> None:
        """Rebuild the encoder options for the chosen format."""
        self.opt_box.reset()
        fid = str(self.seg_fmt.value())
        if fid not in FORMATS:
            fid = "png"
            self.seg_fmt.set_value(fid)
        spec = FORMATS[fid]
        values = self.fmt_values[fid]
        show_adv = self.chk_adv.isChecked()

        cap = self.caps.get(fid, {})
        broken = bool(self.caps) and not cap.get("ok", False)
        hint = spec.hint
        if broken:
            hint = f"Not available in this install \u2014 {cap.get('reason', 'unsupported')}"
        elif cap.get("via") and cap.get("via") != "libvips":
            hint = f"{spec.hint}  (encoded with {cap['via']})"
        self.lbl_fmt_hint.setText(hint)
        self.lbl_fmt_hint.setProperty("role", "err" if broken else "muted")
        self.lbl_fmt_hint.style().unpolish(self.lbl_fmt_hint)
        self.lbl_fmt_hint.style().polish(self.lbl_fmt_hint)

        hidden = False
        for opt in spec.opts:
            if not is_active(opt, values):
                continue
            if opt.advanced and not show_adv:
                hidden = True
                continue
            self.opt_box.field(opt.label, opt.hint, self._option_widget(fid, opt))
        if hidden:
            self.opt_box.control(label("More options are hidden", "hint"))
        self.update_summary()

    def _option_widget(self, fid: str, opt: Opt) -> QWidget:
        """One encoder option, wired straight into :attr:`fmt_values`."""
        values = self.fmt_values[fid]
        value = values.get(opt.key, opt.default)

        if opt.kind == "bool":
            box = checkbox("", bool(value))
            box.toggled.connect(lambda checked: self._set_option(fid, opt.key, checked, True))
            return box
        if opt.kind == "choice":
            picker = combo([text for text, _ in opt.choices], choice_label(opt, value), width=240)
            picker.currentTextChanged.connect(
                lambda text: self._set_option(fid, opt.key, choice_value(opt, text), True)
            )
            return picker
        if opt.kind == "float":
            number = spin_float(
                opt.lo, opt.hi, float(value), opt.step or 0.1, 2, suffix="", tip=opt.hint
            )
            number.valueChanged.connect(
                lambda number_value: self._set_option(fid, opt.key, float(number_value))
            )
            return number
        number_int = spin_int(
            int(opt.lo), int(opt.hi), int(value), int(opt.step or 1), tip=opt.hint
        )
        number_int.valueChanged.connect(
            lambda number_value: self._set_option(fid, opt.key, int(number_value))
        )
        return number_int

    def _set_option(self, fid: str, key: str, value: Any, rerender: bool = False) -> None:
        self.fmt_values[fid][key] = value
        if rerender:
            # A switch can reveal or hide other options, and the widget that
            # fired is about to be replaced, so rebuild once control returns.
            QTimer.singleShot(0, self.render_format_options)
        else:
            self.update_summary()

    def format_values(self, fid: str) -> dict:
        """The options for ``fid``, coerced to the types the worker expects."""
        out: dict[str, Any] = {}
        values = self.fmt_values.get(fid, {})
        for opt in FORMATS[fid].opts:
            raw = values.get(opt.key, opt.default)
            try:
                if opt.kind == "bool":
                    out[opt.key] = bool(raw)
                elif opt.kind == "int":
                    out[opt.key] = int(raw)
                elif opt.kind == "float":
                    out[opt.key] = float(raw)
                else:
                    out[opt.key] = raw
            except (TypeError, ValueError):
                out[opt.key] = opt.default
        return out

    # ------------------------------------------------------------------ #
    # rules
    # ------------------------------------------------------------------ #
    @property
    def rules(self) -> list[rules.Rule]:
        return self.rules_model.rules

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
        self.rules_model.set_rules(current)
        self._rules_seeded = True
        self.save_rules()

    def render_rules(self) -> None:
        """Refresh the line under the table. The model paints the rows."""
        self.rules_model.set_installed(self.model_names())
        kind = self.page_kind()
        if not self.rules:
            hint = (
                "The table is empty, so nothing can run. \u201cDefaults\u201d fills it "
                "with the shipped set, built from the models you have installed."
            )
        else:
            active = sum(1 for r in self.rules if r.enabled)
            hint = f"{active} of {len(self.rules)} rules on"
            if kind == "colour":
                hint += "  \u00b7  grayscale rules are idle: every page is colour"
            elif kind == "grayscale":
                hint += "  \u00b7  colour rules are idle: every page is grayscale"
            hint += "  \u00b7  space toggles a row, double-click edits it"
        self.lbl_rules_hint.setText(hint)
        self.refresh_rule_warnings()

    def refresh_rule_warnings(self) -> None:
        """Everything wrong with the table, on one line underneath it."""
        installed = self.model_names()
        notes: list[str] = []
        for index, rule in enumerate(self.rules):
            if not rule.enabled:
                continue
            notes.extend(f"row {index + 1}: {note}" for note in rules.problems(rule, installed))
        notes.extend(self.scale_mismatches())
        self.lbl_rules_warn.setText("\u26a0  " + "; ".join(notes[:4]) if notes else "")
        self.lbl_rules_warn.setVisible(bool(notes))

    def rules_summary(self) -> str:
        if not self.rules:
            return "no rules \u2014 the table is empty"
        active = sum(1 for r in self.rules if r.enabled)
        return f"{active} of {len(self.rules)} rules on"

    def scale_mismatches(self) -> list[str]:
        """Rows whose model name advertises a factor the target will not use."""
        if str(self.seg_mode.value()) != "scale":
            return []
        want = float(self.sp_scale.value())
        out: list[str] = []
        for index, rule in enumerate(self.rules):
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

    def rules_changed(self, select: int = -1) -> None:
        self.rules_model.set_rules(list(self.rules))
        if 0 <= select < len(self.rules):
            self.rules_view.select_row(select)
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

    def rule_add(self) -> None:
        draft = rules.Rule(
            kind=rules.GRAYSCALE,
            scale=float(self.sp_scale.value()),
            auto_levels=True,
            model=self.default_rule_model(),
        )
        made = RuleDialog.edit(self, "Add rule", draft, self.model_names())
        if made is None:
            return
        index = self.rules_view.current_row()
        items = list(self.rules)
        at = len(items) if index < 0 else index + 1
        items.insert(at, made)
        self.rules_model.set_rules(items)
        self.rules_changed(at)

    def rule_edit(self) -> None:
        index = self.rules_view.current_row()
        if index < 0:
            return
        made = RuleDialog.edit(self, "Edit rule", self.rules[index], self.model_names())
        if made is None:
            return
        items = list(self.rules)
        items[index] = made
        self.rules_model.set_rules(items)
        self.rules_changed(index)

    def rule_remove(self) -> None:
        index = self.rules_view.current_row()
        if index < 0:
            return
        items = list(self.rules)
        del items[index]
        self.rules_model.set_rules(items)
        self.rules_changed(min(index, len(items) - 1))

    def rule_move(self, delta: int) -> None:
        index = self.rules_view.current_row()
        if index < 0:
            return
        items = rules.move(self.rules, index, delta)
        self.rules_model.set_rules(items)
        self.rules_changed(max(0, min(index + delta, len(items) - 1)))

    def rule_toggle(self) -> None:
        index = self.rules_view.current_row()
        if index >= 0:
            self.rules_model.toggle(index)

    def rules_reset(self) -> None:
        if self.rules and not self._confirm(
            "Reset the rules table",
            "Replace every row with the shipped set, built from the models you "
            "have installed?\n\nOnly this table changes \u2014 the rest of your "
            "settings are left alone.",
        ):
            return
        self.rules_model.set_rules(rules.default_working_set(self.model_names()))
        self._rules_seeded = True
        self.rules_changed(0)
        self.log(f"rules reset to the shipped set ({len(self.rules)} rows)")

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
        self.rules_model.set_rules([rules.Rule.from_dict(r) for r in (u.get("rules") or [])])
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

        self.rules_model.set_gray(self.gray_rules_live())
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
        mode = str(self.seg_mode.value())
        if mode == "scale":
            target = f"{float(self.sp_scale.value()):g}\u00d7"
        elif mode == "width":
            target = f"{self.sp_width.value()} px wide"
        elif mode == "height":
            target = f"{self.sp_height.value()} px tall"
        else:
            width, height = self.sp_width.value(), self.sp_height.value()
            target = f"fit {width}\u00d7{height}"
            found = displays.match(width, height)
            if displays.id_for_label(self.cb_display.currentText()) != found:
                set_combo(self.cb_display, displays.label_for_id(found))
            if found != displays.CUSTOM:
                target += f" ({self.cb_display.currentText()})"
        bits = [target, self.rules_summary()]
        bits.append(
            {
                "grayscale": "every page grayscale",
                "colour": "every page colour",
            }.get(self.page_kind(), "grayscale detection")
        )
        if self.chk_levels.isChecked():
            bits.append("auto levels")
        if self.sp_pre_h.value():
            bits.append(f"pre-downscale {self.sp_pre_h.value()}px")
        excluded = sum(1 for r in self.rules if r.enabled and r.action == rules.PASSTHROUGH)
        if excluded:
            bits.append(f"{excluded} exclusion rule{'s' if excluded > 1 else ''}")
        self.lbl_upscale_sum.setText(" \u00b7 ".join(bits))
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

    def container_value(self) -> str:
        cid = str(self.seg_container.value() or "files")
        return cid if cid in CONTAINERS else "files"

    def resolved_out_dir(self) -> Path | None:
        src = self._in_path.strip()
        if self.chk_same.isChecked():
            if not src:
                return None
            path = Path(src)
            base = path.parent if path.is_file() else path
            sub = self.ed_sub.text().strip() or "upscaled"
            return base / sub
        custom = self.ed_out.text().strip()
        return Path(custom) if custom else None

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
    # banner and log
    # ------------------------------------------------------------------ #
    def show_banner(self, text: str) -> None:
        self.banner.show_text(text)

    def hide_banner(self) -> None:
        self.banner.hide()

    def log_colour(self, level: str) -> str:
        p = self.theme.p
        return {
            "info": p.text,
            "debug": p.muted,
            "skip": p.muted,
            "ok": p.ok,
            "dry": p.accent,
            "warn": p.warn,
            "error": p.err,
        }.get(level, p.text)

    def log(self, message: str, level: str = "info") -> None:
        """One line in the panel and, unless it is noise, in the run file."""
        if not message:
            return
        stamp = time.strftime("%H:%M:%S")
        tag = level if level in LOG_TAGS else "info"
        self.runlog.write(stamp, message)
        if tag == "debug" and not self.chk_debug.isChecked():
            return
        # LogView caps itself at 4000 blocks, so nothing is trimmed by hand.
        self.log_view.add_line(f"{stamp}  {message}", self.log_colour(tag))

    def log_text(self) -> str:
        return self.log_view.toPlainText()

    def show_log(self, visible: bool) -> None:
        """The log shares a splitter with the cards, so it can be dragged."""
        self._log_visible = bool(visible)
        self.log_panel.setVisible(self._log_visible)
        self.btn_log.setText("Hide log" if self._log_visible else "Log")
        if self._log_visible:
            weight = max(1, int(self.settings.data["ui"].get("log_weight", 2) or 2))
            total = max(self.splitter.height(), 520)
            share = min(max(total * weight // (weight + 4), 180), total - 240)
            self.splitter.setSizes([total - share, share])

    def toggle_log(self) -> None:
        self.show_log(not self._log_visible)

    def apply_log_wrap(self) -> None:
        self.log_view.set_wrap(self.chk_wrap.isChecked())

    def copy_log(self) -> None:
        text = self.log_text()
        clipboard = QGuiApplication.clipboard()
        if text and clipboard is not None:
            clipboard.setText(text)

    def save_log_as(self) -> None:
        path, _chosen = QFileDialog.getSaveFileName(
            self,
            "Save log",
            str(self.dir / f"Run_{time.strftime('%Y%m%d-%H%M%S')}.log"),
            "Log files (*.log);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self.log_text() + "\n", encoding="utf-8")
        except OSError as exc:
            self.log(f"could not save the log: {exc}", "error")
            return
        self.log(f"log saved to {path}")

    def open_log_folder(self) -> None:
        target = self._log_dir()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        if target.exists():
            open_in_explorer(target)

    def clear_log(self) -> None:
        self.log_view.clear()

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
        head = "Planning" if self.dry else "Upscaling"
        self.lbl_status.setText(f"{head} 0/{self.total}")
        self.lbl_detail.setText(
            f"{event.get('device')} \u00b7 "
            f"{'FP16' if event.get('fp16') else 'FP32'} \u00b7 "
            f"tile {event.get('tile')}"
        )
        for text, level in format_start(event):
            self.log(text, level)

    def on_progress(self, event: dict) -> None:
        name = Path(str(event.get("path", ""))).name
        index = int(event.get("i") or 0)
        total = int(event.get("total") or self.total or 0)
        sub = f" (page {event.get('sub_i')}/{event.get('sub_n')})" if event.get("sub_n") else ""
        head = "Planning" if self.dry else "Upscaling"
        self.lbl_status.setText(f"{head} {index}/{total}{sub}")
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
        elapsed = max(0.001, time.time() - self.started_at)
        rate = self.completed / elapsed if self.completed else 0.0
        left = total - finished
        eta = f" \u00b7 ETA {fmt_secs(left / rate)}" if rate > 0 and left > 0 else ""
        head = "Planning" if self.dry else "Upscaling"
        self.lbl_status.setText(f"{head} {finished}/{total}{eta}")

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
            "perf": d["perf"],
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
