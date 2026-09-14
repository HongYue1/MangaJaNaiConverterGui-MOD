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
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from janai import __version__
from janai.app.fields import (
    MODE_OPTIONS,
    tile_label,
)
from janai.app.input_panel import InputPanelMixin
from janai.app.log_panel import LogPanelMixin
from janai.app.output_panel import OutputPanelMixin
from janai.app.perf_panel import AUTO_DEVICE, PerfPanelMixin
from janai.app.rules_panel import RulesPanelMixin
from janai.app.run_panel import RunPanelMixin
from janai.app.runlog import (
    RunLog,
)
from janai.app.runner import Runner
from janai.app.state import Settings, defaults
from janai.app.theme import Theme
from janai.app.widgets import (
    Banner,
    Card,
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
from janai.core import displays, presets, rules
from janai.core.formats import (
    CONTAINERS,
    FORMATS,
    packs_archive,
    summary,
)

GEOMETRY_RE = re.compile(r"^(\d+)x(\d+)(?:\+(-?\d+)\+(-?\d+))?$")


class MainWindow(
    InputPanelMixin,
    LogPanelMixin,
    OutputPanelMixin,
    PerfPanelMixin,
    RulesPanelMixin,
    RunPanelMixin,
    QMainWindow,
):
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

    def on_pagekind_change(self, _text: str = "") -> None:
        """The page kind decides which rows can fire, and which fields apply."""
        self.set_gray_rows(self.gray_rules_live())
        self.sync_detection_fields()
        self.render_rules()
        self.update_summary()

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
    # dialogs
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # banner
    # ------------------------------------------------------------------ #
    def show_banner(self, text: str) -> None:
        self.banner.show_text(text)

    def hide_banner(self) -> None:
        self.banner.hide()

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
