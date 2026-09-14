"""The Output card: encoder, packaging, destination and file names.

The encoder options are not a fixed form. Each format in :mod:`janai.core.formats`
declares its own options, some of which only apply when another option has a
particular value, so the grid is rebuilt from the spec rather than written out
once per format. ``fmt_values`` keeps every format's answers alive while the
user switches between them, and ``format_values`` is the single place that
coerces them to the types the worker expects - the GUI must never send a string
where the worker will do arithmetic.

``container_value`` and ``resolved_out_dir`` live here rather than with the
summaries that call them: they are pure readers of this card's widgets, and
``resolved_out_dir`` is the one definition of where output lands, consulted by
the summary line, the Start button's enabled state and the job builder alike.

Mixed into :class:`janai.app.window.MainWindow`, so ``self`` is the window.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFileDialog, QWidget

from janai.app.fields import choice_label, choice_value
from janai.app.widgets import (
    Card,
    FieldGrid,
    Segmented,
    button,
    checkbox,
    combo,
    label,
    line_edit,
    row,
    spin_float,
    spin_int,
)
from janai.core.formats import CONTAINER_IDS, CONTAINERS, FORMAT_IDS, FORMATS, Opt, is_active

if TYPE_CHECKING:
    from janai.app.surface import WindowSurface as _Base
else:  # type-only: at runtime the base is object, so the MRO is untouched
    _Base = object


class OutputPanelMixin(_Base):
    """The Output card, its dynamic encoder grid, and the destination logic."""

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

    def choose_out_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Choose an output folder", self.ed_out.text() or ""
        )
        if path:
            self.chk_same.setChecked(False)
            self.ed_out.setText(path)
            self.on_dest_change()

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
