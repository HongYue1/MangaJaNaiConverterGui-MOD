"""The small widget kit the window is assembled from.

Everything here is a thin wrapper over a real Qt widget: cards, a collapsible
panel, a segmented control, a drop target, a log view and the field factories.
The old build had to draw these by hand on a canvas; the point of this module
is that it no longer does, so each class is mostly layout and naming.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from html import escape
from typing import Any

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QFont, QMouseEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


def repolish(widget: QWidget) -> None:
    """Re-read the stylesheet after a dynamic property changed."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()


def clear_layout(layout: QLayout) -> None:
    """Delete every child of ``layout``, rows and spacers included."""
    while layout.count():
        item = layout.takeAt(0)
        child = item.widget()
        if child is not None:
            child.setParent(None)
            child.deleteLater()
        sub = item.layout()
        if sub is not None:
            clear_layout(sub)


class WheelGuard(QObject):
    """Let the page scroll even when the pointer rests on a control.

    Qt delivers the wheel to whatever sits under the pointer, so scrolling the
    page with the cursor over a dropdown would silently change its value. An
    unfocused control hands the gesture back, which the scroll area then takes;
    click or tab into the control and the wheel adjusts it as usual.
    """

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if (
            event.type() == QEvent.Type.Wheel
            and isinstance(watched, QWidget)
            and not watched.hasFocus()
        ):
            # Left unaccepted, so Qt passes the gesture up to the scroll area.
            event.ignore()
            return True
        return False


def guard_wheel(widget: QWidget) -> QWidget:
    """Install :class:`WheelGuard` on ``widget`` and hand it back."""
    widget.installEventFilter(WheelGuard(widget))
    if isinstance(widget, QComboBox | QAbstractSpinBox):
        widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    return widget


# --------------------------------------------------------------------------- #
# factories
# --------------------------------------------------------------------------- #
def label(text: str, role: str = "", wrap: bool = False, tip: str = "") -> QLabel:
    out = QLabel(text)
    if role:
        out.setProperty("role", role)
    out.setWordWrap(wrap)
    if tip:
        out.setToolTip(tip)
    return out


def button(
    text: str,
    on_click: Callable[[], None] | None = None,
    variant: str = "",
    tip: str = "",
) -> QPushButton:
    out = QPushButton(text)
    if variant:
        out.setProperty("variant", variant)
    if tip:
        out.setToolTip(tip)
    if on_click is not None:
        # clicked(bool) carries a checked flag that none of these want.
        out.clicked.connect(lambda *_: on_click())
    out.setCursor(Qt.CursorShape.PointingHandCursor)
    return out


def checkbox(
    text: str,
    checked: bool = False,
    on_change: Callable[[], None] | None = None,
    tip: str = "",
) -> QCheckBox:
    out = QCheckBox(text)
    out.setChecked(bool(checked))
    if tip:
        out.setToolTip(tip)
    if on_change is not None:
        out.toggled.connect(lambda _checked: on_change())
    return out


def combo(
    items: Sequence[str],
    value: str = "",
    on_change: Callable[[], None] | None = None,
    width: int = 0,
    tip: str = "",
) -> QComboBox:
    out = QComboBox()
    out.addItems(list(items))
    if value:
        set_combo(out, value)
    if width:
        out.setMinimumWidth(width)
    if tip:
        out.setToolTip(tip)
    if on_change is not None:
        out.currentIndexChanged.connect(lambda _index: on_change())
    out.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContentsOnFirstShow)
    guard_wheel(out)
    return out


def set_combo(box: QComboBox, value: str) -> None:
    """Select ``value``, adding it if the list does not offer it yet."""
    index = box.findText(value)
    if index < 0 and value:
        box.addItem(value)
        index = box.count() - 1
    if index >= 0:
        box.setCurrentIndex(index)


def spin_int(
    lo: int,
    hi: int,
    value: int,
    step: int = 1,
    on_change: Callable[[], None] | None = None,
    suffix: str = "",
    special: str = "",
    tip: str = "",
) -> QSpinBox:
    out = QSpinBox()
    out.setRange(lo, hi)
    out.setSingleStep(step)
    out.setValue(int(value))
    out.setKeyboardTracking(False)
    if suffix:
        out.setSuffix(suffix)
    if special:
        out.setSpecialValueText(special)
    if tip:
        out.setToolTip(tip)
    if on_change is not None:
        out.valueChanged.connect(lambda _value: on_change())
    guard_wheel(out)
    return out


def spin_float(
    lo: float,
    hi: float,
    value: float,
    step: float = 0.25,
    decimals: int = 2,
    on_change: Callable[[], None] | None = None,
    suffix: str = "",
    tip: str = "",
) -> QDoubleSpinBox:
    out = QDoubleSpinBox()
    out.setRange(lo, hi)
    out.setSingleStep(step)
    out.setDecimals(decimals)
    out.setValue(float(value))
    out.setKeyboardTracking(False)
    if suffix:
        out.setSuffix(suffix)
    if tip:
        out.setToolTip(tip)
    if on_change is not None:
        out.valueChanged.connect(lambda _value: on_change())
    guard_wheel(out)
    return out


def line_edit(
    value: str = "",
    placeholder: str = "",
    on_change: Callable[[], None] | None = None,
    width: int = 0,
    tip: str = "",
) -> QLineEdit:
    out = QLineEdit(value)
    if placeholder:
        out.setPlaceholderText(placeholder)
    if width:
        out.setMaximumWidth(width)
    if tip:
        out.setToolTip(tip)
    if on_change is not None:
        out.textChanged.connect(lambda _text: on_change())
    return out


def separator() -> QFrame:
    out = QFrame()
    out.setProperty("role", "sep")
    out.setFrameShape(QFrame.Shape.NoFrame)
    out.setFixedHeight(1)
    return out


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #
class FieldGrid(QWidget):
    """A two-column body: a label with its explanation, then the control."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setHorizontalSpacing(16)
        self.grid.setVerticalSpacing(9)
        self.grid.setColumnStretch(0, 0)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnMinimumWidth(0, 132)
        self._row = 0

    def reset(self) -> None:
        """Empty the grid so it can be rebuilt from scratch."""
        clear_layout(self.grid)
        self._row = 0

    def field(self, title: str, hint: str, widget: QWidget, tip: str = "") -> QWidget:
        """Add one labelled row and return the control.

        ``hint`` is not drawn any more. A line of explanation under every label
        crowds the page and pushes the controls apart, so it becomes the hover
        tooltip of both the label and the control instead. A control that
        already carries its own tooltip keeps it: that text is the more
        specific of the two.
        """
        explain = tip or hint
        head = label(title, "field", tip=explain)
        if explain and not widget.toolTip():
            widget.setToolTip(explain)
        self.grid.addWidget(head, self._row, 0, Qt.AlignmentFlag.AlignVCenter)
        self.grid.addWidget(widget, self._row, 1)
        self._row += 1
        return widget

    def full(self, widget: QWidget) -> QWidget:
        """Add a row that spans both columns."""
        self.grid.addWidget(widget, self._row, 0, 1, 2)
        self._row += 1
        return widget

    def control(self, widget: QWidget) -> QWidget:
        """Add a row in the control column only, with no label beside it."""
        self.grid.addWidget(widget, self._row, 1)
        self._row += 1
        return widget

    def rule(self) -> None:
        self.full(separator())


def row(*widgets: QWidget, spacing: int = 8, stretch: bool = True) -> QWidget:
    """Lay widgets out left to right in a transparent container."""
    out = QWidget()
    box = QHBoxLayout(out)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(spacing)
    for widget in widgets:
        box.addWidget(widget)
    if stretch:
        box.addStretch(1)
    return out


class Card(QFrame):
    """A titled panel. Fields go into :attr:`body`."""

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        badge: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 14, 16, 16)
        outer.setSpacing(12)

        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        head.setSpacing(10)
        titles = QVBoxLayout()
        titles.setContentsMargins(0, 0, 0, 0)
        titles.setSpacing(2)
        titles.addWidget(label(title, "title", tip=subtitle))
        # The card's description is a tooltip on its title, not a second line
        # of prose: it repeats what the fields already say, and it costs every
        # card a row of height.
        self.lbl_subtitle = label(subtitle, "muted", wrap=True)
        self.lbl_subtitle.setVisible(False)
        titles.addWidget(self.lbl_subtitle)
        head.addLayout(titles, 1)
        self.lbl_badge = label(badge, "badge")
        self.lbl_badge.setVisible(bool(badge))
        head.addWidget(self.lbl_badge, 0, Qt.AlignmentFlag.AlignTop)
        outer.addLayout(head)

        self.body = FieldGrid(self)
        outer.addWidget(self.body)

    def set_badge(self, text: str) -> None:
        self.lbl_badge.setText(text)
        self.lbl_badge.setVisible(bool(text))


class ClickFrame(QFrame):
    """A frame that reports clicks, so a whole header row is a hit target."""

    clicked = Signal()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class Collapsible(QFrame):
    """A card that folds away, for the settings most runs never touch."""

    toggled = Signal(bool)

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        expanded: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 8, 10, 10)
        outer.setSpacing(10)

        self.head = ClickFrame(self)
        self.head.setObjectName("cardhead")
        self.head.setCursor(Qt.CursorShape.PointingHandCursor)
        self.head.clicked.connect(self.toggle)
        head = QHBoxLayout(self.head)
        head.setContentsMargins(6, 6, 8, 6)
        head.setSpacing(9)
        self.chevron = label("", "muted")
        head.addWidget(self.chevron)
        head.addWidget(label(title, "title"))
        self.lbl_hint = label(subtitle, "muted", wrap=False)
        head.addWidget(self.lbl_hint, 1)
        outer.addWidget(self.head)

        self.body = FieldGrid(self)
        wrapper = QWidget(self)
        inner = QVBoxLayout(wrapper)
        inner.setContentsMargins(6, 0, 6, 2)
        inner.setSpacing(0)
        inner.addWidget(self.body)
        self._wrapper = wrapper
        outer.addWidget(wrapper)

        self._open = bool(expanded)
        self._render()

    def toggle(self) -> None:
        self.set_open(not self._open)

    def _render(self) -> None:
        self.chevron.setText("\u25be" if self._open else "\u25b8")
        self._wrapper.setVisible(self._open)

    def set_open(self, open_: bool) -> None:
        self._open = bool(open_)
        self._render()
        self.toggled.emit(self._open)

    def is_open(self) -> bool:
        return self._open

    def set_hint(self, text: str) -> None:
        self.lbl_hint.setText(text)


class Segmented(QWidget):
    """Two to five exclusive choices, shown side by side."""

    changed = Signal(object)

    def __init__(
        self,
        options: Sequence[tuple[str, Any]],
        value: Any = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        box = QHBoxLayout(self)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: list[tuple[Any, QPushButton]] = []
        for text, option in options:
            btn = button(text, variant="seg")
            btn.setCheckable(True)
            btn.setChecked(option == value)
            self._group.addButton(btn)
            box.addWidget(btn)
            self._buttons.append((option, btn))
            btn.clicked.connect(lambda _checked=False, opt=option: self._pick(opt))
        box.addStretch(1)
        if value is None and self._buttons:
            self._buttons[0][1].setChecked(True)

    def _pick(self, option: Any) -> None:
        self.set_value(option)
        self.changed.emit(option)

    def value(self) -> Any:
        for option, btn in self._buttons:
            if btn.isChecked():
                return option
        return self._buttons[0][0] if self._buttons else None

    def set_value(self, value: Any) -> None:
        for option, btn in self._buttons:
            btn.setChecked(option == value)

    def set_option_enabled(self, value: Any, enabled: bool) -> None:
        for option, btn in self._buttons:
            if option == value:
                btn.setEnabled(bool(enabled))


class DropZone(QFrame):
    """The input target: drop files or folders anywhere on it.

    Qt gives drag and drop on every platform, so this replaces the Windows-only
    ctypes shim the old build needed.
    """

    dropped = Signal(list)

    def __init__(self, hint: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("drop")
        self.setAcceptDrops(True)
        self.setProperty("active", "false")
        box = QVBoxLayout(self)
        box.setContentsMargins(16, 18, 16, 18)
        box.setSpacing(6)
        self.lbl_hint = label(hint, "", wrap=True)
        self.lbl_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_path = label("No input selected", "muted", wrap=True)
        self.lbl_path.setAlignment(Qt.AlignmentFlag.AlignCenter)
        box.addWidget(self.lbl_hint)
        box.addWidget(self.lbl_path)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def set_hint(self, text: str) -> None:
        self.lbl_hint.setText(text)

    def set_path(self, text: str) -> None:
        self.lbl_path.setText(text or "No input selected")

    def _highlight(self, on: bool) -> None:
        self.setProperty("active", "true" if on else "false")
        repolish(self)

    def dragEnterEvent(self, event: Any) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self._highlight(True)

    def dragLeaveEvent(self, event: Any) -> None:
        self._highlight(False)
        event.accept()

    def dropEvent(self, event: Any) -> None:
        self._highlight(False)
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            event.acceptProposedAction()
            self.dropped.emit(paths)
        else:
            event.ignore()


class LogView(QPlainTextEdit):
    """The run log: colour per level, capped length, and it follows the tail
    only while the reader is already at the bottom."""

    def __init__(self, font: QFont, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setReadOnly(True)
        self.setUndoRedoEnabled(False)
        self.setMaximumBlockCount(4000)
        self.setFont(font)
        self.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.setFrameShape(QFrame.Shape.NoFrame)

    def at_bottom(self) -> bool:
        bar = self.verticalScrollBar()
        return bar.value() >= bar.maximum() - 4

    def add_line(self, text: str, colour: str) -> None:
        follow = self.at_bottom()
        body = escape(text).replace("  ", "&nbsp;&nbsp;")
        self.appendHtml(f'<span style="color:{colour}">{body}</span>')
        if follow:
            bar = self.verticalScrollBar()
            bar.setValue(bar.maximum())

    def set_wrap(self, wrap: bool) -> None:
        self.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.WidgetWidth if wrap else QPlainTextEdit.LineWrapMode.NoWrap
        )


class Banner(QFrame):
    """One line of bad news across the top of the page, with a dismiss button."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("banner")
        box = QHBoxLayout(self)
        box.setContentsMargins(12, 9, 8, 9)
        box.setSpacing(10)
        self.lbl = label("", "err", wrap=True)
        box.addWidget(self.lbl, 1)
        self.btn_close = button("\u2715", self.hide, variant="ghost", tip="Dismiss")
        self.btn_close.setFixedWidth(30)
        box.addWidget(self.btn_close, 0, Qt.AlignmentFlag.AlignTop)
        self.hide()

    def show_text(self, text: str) -> None:
        self.lbl.setText(text)
        self.setVisible(bool(text))
