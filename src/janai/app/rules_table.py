"""The rules table: a real model over :class:`janai.core.rules.Rule` rows.

The table is the only thing that chooses a model, so it earns a proper Qt
model/view pair rather than a redrawn list: rows keep their selection while
the page changes around them, the On column is a genuine checkbox the keyboard
can reach, and the Model column takes the slack when the window grows.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QObject, Qt, Signal
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from janai.app.theme import Palette
from janai.app.widgets import FieldGrid, checkbox, combo, label, line_edit
from janai.core import rules

#: Column order. The first one is the on/off checkbox and has no heading.
HEADERS: tuple[str, ...] = ("", "When", "Page size", "Model", "Auto levels")

#: Presets for the target-scale box. The box is editable, so a custom model
#: factor - 3x, 1.5x, 8x - can simply be typed in.
SCALES: tuple[tuple[str, float], ...] = (
    ("any", 0.0),
    ("1x", 1.0),
    ("2x", 2.0),
    ("3x", 3.0),
    ("4x", 4.0),
    ("8x", 8.0),
)


def scale_text(value: float) -> str:
    """``2.0`` -> ``2x``; ``0`` -> ``any``."""
    return "any" if not value else f"{float(value):g}x"


def parse_scale(text: str) -> float:
    """Read a typed factor: ``3x``, ``3``, ``1.5x``. Anything else means any."""
    cleaned = str(text or "").strip().lower().replace("\u00d7", "x").rstrip("x").strip()
    if not cleaned or cleaned in {"any", "*"}:
        return 0.0
    try:
        return max(0.0, float(cleaned))
    except ValueError:
        return 0.0


#: What a matching rule does with the page. The second case turns the row into a
#: size exclusion: give it a page-size condition and those pages skip the model
#: entirely, which is what the old hardcoded long-strip switch did.
ACTIONS: tuple[tuple[str, str], ...] = (
    ("upscale with the model", rules.UPSCALE),
    ("skip the model, re-encode only", rules.PASSTHROUGH),
)


def action_text(value: str) -> str:
    return next((text for text, item in ACTIONS if item == value), ACTIONS[0][0])


def parse_action(text: str) -> str:
    return dict(ACTIONS).get(text, rules.UPSCALE)


LEVELS: tuple[tuple[str, bool | None], ...] = (("default", None), ("on", True), ("off", False))

SIZE_HELP = (
    "Sizes accept 1920, 1920p, a range 1600-1920, an open end 1985- or -1250, "
    "or any. A rule that names a size always beats a rule that says any, "
    "wherever the two sit in the table."
)


class RulesModel(QAbstractTableModel):
    """Rows of :class:`~janai.core.rules.Rule`, presented for the table."""

    edited = Signal()

    def __init__(
        self,
        palette: Palette,
        items: Sequence[rules.Rule] = (),
        gray_on: bool = True,
        installed: Sequence[str] = (),
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._palette = palette
        self._rules: list[rules.Rule] = list(items)
        self._gray_on = bool(gray_on)
        self._installed: list[str] = list(installed)

    # ---- python side -------------------------------------------------- #
    @property
    def rules(self) -> list[rules.Rule]:
        return self._rules

    def set_rules(self, items: Sequence[rules.Rule]) -> None:
        self.beginResetModel()
        self._rules = list(items)
        self.endResetModel()

    def set_gray(self, gray_on: bool) -> None:
        """Detection decides which half of the table can fire, so recolour."""
        self._gray_on = bool(gray_on)
        self._repaint()

    def set_installed(self, installed: Sequence[str]) -> None:
        self._installed = list(installed)
        self._repaint()

    def set_palette(self, palette: Palette) -> None:
        self._palette = palette
        self._repaint()

    def _repaint(self) -> None:
        if not self._rules:
            return
        top = self.index(0, 0)
        bottom = self.index(len(self._rules) - 1, len(HEADERS) - 1)
        self.dataChanged.emit(top, bottom)

    def rule_at(self, row: int) -> rules.Rule | None:
        if 0 <= row < len(self._rules):
            return self._rules[row]
        return None

    def idle(self, rule: rules.Rule) -> bool:
        """True when the rule can never fire as things stand."""
        return rule.kind == rules.GRAYSCALE and not self._gray_on

    # ---- Qt side ------------------------------------------------------ #
    def rowCount(self, parent: QModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(self._rules)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        if parent is not None and parent.isValid():
            return 0
        return len(HEADERS)

    def headerData(
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ) -> Any:
        if orientation == Qt.Orientation.Vertical:
            # Row numbers, so a warning that names "row 17" can be found by
            # looking instead of counting.
            if role == Qt.ItemDataRole.DisplayRole:
                return str(section + 1)
            if role == Qt.ItemDataRole.TextAlignmentRole:
                return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            return None
        if role == Qt.ItemDataRole.DisplayRole and 0 <= section < len(HEADERS):
            return HEADERS[section]
        if role == Qt.ItemDataRole.TextAlignmentRole:
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        base = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if index.column() == 0:
            return base | Qt.ItemFlag.ItemIsUserCheckable
        return base

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        rule = self.rule_at(index.row())
        if rule is None:
            return None
        column = index.column()

        if column == 0:
            if role == Qt.ItemDataRole.CheckStateRole:
                return Qt.CheckState.Checked if rule.enabled else Qt.CheckState.Unchecked
            if role == Qt.ItemDataRole.ToolTipRole:
                return "Switch this rule off without deleting it."
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return rule.columns()[column - 1]

        if role == Qt.ItemDataRole.ForegroundRole:
            if not rule.enabled or self.idle(rule):
                return QColor(self._palette.muted)
            if column == 3 and rules.problems(rule, self._installed):
                return QColor(self._palette.warn)
            return None

        if role == Qt.ItemDataRole.TextAlignmentRole and column == 4:
            return int(Qt.AlignmentFlag.AlignCenter)

        if role == Qt.ItemDataRole.ToolTipRole:
            return self._tooltip(rule)

        return None

    def _tooltip(self, rule: rules.Rule) -> str:
        lines = [rule.describe()]
        if not rule.enabled:
            lines.append("This rule is off and is skipped.")
        elif self.idle(rule):
            lines.append("Idle: grayscale detection is off, so this rule cannot fire.")
        lines.extend(rules.problems(rule, self._installed))
        return "\n".join(lines)

    def setData(
        self,
        index: QModelIndex,
        value: Any,
        role: int = Qt.ItemDataRole.EditRole,
    ) -> bool:
        if not index.isValid() or index.column() != 0:
            return False
        if role != Qt.ItemDataRole.CheckStateRole:
            return False
        row = index.row()
        rule = self.rule_at(row)
        if rule is None:
            return False
        self._rules[row] = rules.with_enabled(rule, Qt.CheckState(value) == Qt.CheckState.Checked)
        left = self.index(row, 0)
        right = self.index(row, len(HEADERS) - 1)
        self.dataChanged.emit(left, right)
        self.edited.emit()
        return True

    def toggle(self, row: int) -> None:
        """Flip one row, from a button or the space bar."""
        rule = self.rule_at(row)
        if rule is None:
            return
        state = Qt.CheckState.Unchecked if rule.enabled else Qt.CheckState.Checked
        self.setData(self.index(row, 0), state, Qt.ItemDataRole.CheckStateRole)


class RulesTable(QTableView):
    """The view: rows, no grid, and the model column takes the slack."""

    def __init__(self, model: RulesModel, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setModel(model)
        self._model = model
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setWordWrap(False)
        self.setCornerButtonEnabled(False)
        self.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setTabKeyNavigation(False)

        head = self.horizontalHeader()
        head.setHighlightSections(False)
        head.setSectionsClickable(False)
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        head.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        head.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        head.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(0, 34)
        self.setColumnWidth(4, 96)

        rows = self.verticalHeader()
        rows.setVisible(True)  # the warnings under the table cite row numbers
        rows.setHighlightSections(False)
        rows.setFixedWidth(36)
        rows.setDefaultSectionSize(28)
        rows.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.setMinimumHeight(28 * 6 + 34)

    def current_row(self) -> int:
        index = self.currentIndex()
        return index.row() if index.isValid() else -1

    def select_row(self, row: int) -> None:
        if 0 <= row < self._model.rowCount():
            self.selectRow(row)
            self.setCurrentIndex(self._model.index(row, 1))

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Space toggles the selected rule wherever the cursor sits."""
        if event.key() == Qt.Key.Key_Space:
            row = self.current_row()
            if row >= 0:
                self._model.toggle(row)
                event.accept()
                return
        super().keyPressEvent(event)


class RuleDialog(QDialog):
    """A small modal editor for one rule."""

    def __init__(
        self,
        parent: QWidget | None,
        title: str,
        draft: rules.Rule,
        models: Sequence[str],
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self._draft = draft

        box = QVBoxLayout(self)
        box.setContentsMargins(18, 16, 18, 16)
        box.setSpacing(14)
        form = FieldGrid(self)

        self.cb_kind = combo(
            list(rules.KINDS),
            draft.kind,
            width=170,
            tip=(
                "Which pages this rule may claim: ones detected as grayscale, ones "
                "detected as colour, or any page. Grayscale rules never fire while "
                "Grayscale detection is off."
            ),
        )
        form.field("Page kind", "", self.cb_kind)

        self.cb_scale = combo(
            [name for name, _ in SCALES],
            scale_text(draft.scale),
            width=170,
            tip=(
                "Restricts the rule to one output factor, so 2x and 4x rows can live "
                "in the same table. \u201cany\u201d fires whatever the target is, which "
                "is what width, height and fit targets need since their factor changes "
                "per page. The box is editable: type 3x - or 1.5x, 8x - for a model "
                "whose factor is not one of the presets."
            ),
        )
        self.cb_scale.setEditable(True)
        self.cb_scale.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        form.field("Target scale", "", self.cb_scale)

        self.cb_action = combo(
            [text for text, _ in ACTIONS],
            action_text(draft.action),
            width=260,
            tip=(
                "What happens to a page this rule matches. Upscaling is the usual "
                "case. \u201cSkip the model\u201d makes the row a size exclusion: the "
                "page is only re-encoded into the chosen format. Pair it with a page "
                "size - height from 20000, say - to pass webtoon mega-strips through "
                "instead of spending minutes upscaling them."
            ),
        )
        form.field("Action", "", self.cb_action)

        self.ed_width = line_edit(
            draft.width,
            placeholder="any",
            width=180,
            tip=(
                "Matched against the source page width in pixels, before upscaling. "
                "Leave it on any unless you need to separate double spreads from "
                "single pages."
            ),
        )
        form.field("Page width", "", self.ed_width)

        self.ed_height = line_edit(
            draft.height,
            placeholder="any",
            width=180,
            tip=(
                "Matched against the source page height in pixels, before upscaling. "
                "This is the one that matters for manga: the MangaJaNai models are "
                "trained per page height."
            ),
        )
        form.field("Page height", "", self.ed_height)

        choices = list(models)
        if draft.model and draft.model not in choices:
            choices.append(draft.model)
        self.cb_model = combo(
            choices,
            draft.model,
            width=420,
            tip=(
                "The weights this rule runs. Only installed models are listed, and the "
                "factor in the name (1x, 2x, 4x) is what the model was trained for - "
                "matching it to your target avoids a resample."
            ),
        )
        form.field("Model", "", self.cb_model)

        levels_label = next(name for name, value in LEVELS if value is draft.auto_levels)
        self.cb_levels = combo(
            [name for name, _ in LEVELS],
            levels_label,
            width=170,
            tip=(
                "Grayscale pages only. \u201cdefault\u201d follows the Auto levels "
                "checkbox in the Upscale card; \u201con\u201d and \u201coff\u201d "
                "override it for the pages this rule claims."
            ),
        )
        form.field("Auto levels", "", self.cb_levels)
        box.addWidget(form)

        box.addWidget(label(SIZE_HELP, "hint", wrap=True))
        self.chk_on = checkbox(
            "Rule is on",
            draft.enabled,
            tip=(
                "A rule that is off stays in the table and is skipped. The table shows "
                "it as a cleared checkbox in the first column."
            ),
        )
        box.addWidget(self.chk_on)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setProperty("variant", "accent")
        save.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        box.addWidget(buttons)
        self.setMinimumWidth(560)

    def rule(self) -> rules.Rule:
        """The edited rule, normalised by :meth:`Rule.from_dict`."""
        levels = dict(LEVELS)
        return rules.Rule.from_dict(
            {
                "kind": self.cb_kind.currentText(),
                "scale": parse_scale(self.cb_scale.currentText()),
                "action": parse_action(self.cb_action.currentText()),
                "width": self.ed_width.text(),
                "height": self.ed_height.text(),
                "model": self.cb_model.currentText(),
                "auto_levels": levels[self.cb_levels.currentText()],
                "enabled": self.chk_on.isChecked(),
                "note": self._draft.note,
            }
        )

    @staticmethod
    def edit(
        parent: QWidget | None,
        title: str,
        draft: rules.Rule,
        models: Sequence[str],
    ) -> rules.Rule | None:
        """Show the editor; return the new rule, or None if it was cancelled."""
        dialog = RuleDialog(parent, title, draft, models)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.rule()
        return None
