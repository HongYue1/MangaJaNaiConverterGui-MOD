#!/usr/bin/env python3
"""Interface geometry harness for the Qt window.

Builds the real window and asserts the things that actually went wrong in the
Tk build: cards that did not lay out, a page that did not scroll, wheel
gestures eaten by combo boxes and spin boxes, rules columns that did not fit
their viewport, and a type ramp that drifted from the one theme.py declares.

The window is built against a throwaway folder, so running this never touches
your settings.json, your logs, or your models.

    python scripts/uicheck.py                      # offscreen, nothing appears
    JANAI_UICHECK_SHOW=1 python scripts/uicheck.py  # on the real display

Exits non-zero if any check fails, so CI can run it.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Offscreen by default, so this runs in CI and over SSH with no display.
if not os.environ.get("JANAI_UICHECK_SHOW"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QFontMetrics, QWheelEvent
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QHeaderView,
    QScrollArea,
    QWidget,
)

from janai.app.rules_table import RulesTable
from janai.app.theme import BASE_SIZES, Theme
from janai.app.widgets import Card, Collapsible, DropZone, LogView
from janai.app.window import MainWindow

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


def pump(app: QApplication, rounds: int = 3) -> None:
    """Let Qt finish laying out before anything is measured."""
    for _ in range(rounds):
        app.processEvents()


def wheel_event(widget: QWidget, notches: int = -1) -> QWheelEvent | None:
    """One wheel notch over a widget, or None if Qt will not build the event."""
    centre = widget.rect().center()
    try:
        return QWheelEvent(
            QPointF(centre),
            QPointF(widget.mapToGlobal(centre)),
            QPoint(0, notches * 40),
            QPoint(0, notches * 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
    except Exception as exc:
        print(f"  skip  synthetic wheel events unavailable ({exc})")
        return None


def send_wheel(app: QApplication, widget: QWidget, notches: int = -1) -> bool:
    """Post a wheel notch the way the application would. False if unavailable."""
    event = wheel_event(widget, notches)
    if event is None:
        return False
    QApplication.sendEvent(widget, event)
    pump(app)
    return True


def dispatch_wheel(
    app: QApplication,
    widget: QWidget,
    notches: int = -1,
    accepted: bool = False,
) -> bool | None:
    """Hand a wheel notch to a widget's own handler and report that handler's verdict.

    Posting through the application cannot answer what a widget decided: an
    ignored event travels on to the ancestors, so the accepted flag ends up
    reporting whoever took it last. Calling the handler directly keeps the answer
    local, and presetting the flag to the opposite of the expected outcome means a
    handler that quietly does nothing cannot pass. None means Qt would not build
    the event, so the caller should skip rather than fail.
    """
    event = wheel_event(widget, notches)
    if event is None:
        return None
    if accepted:
        event.accept()
    else:
        event.ignore()
    widget.wheelEvent(event)
    pump(app)
    return event.isAccepted()


def check_fonts(theme: Theme) -> None:
    print("type ramp")
    check(
        "the ramp has exactly the declared roles",
        set(theme.fonts) == set(BASE_SIZES),
        ", ".join(sorted(theme.fonts)),
    )
    for key, size in sorted(BASE_SIZES.items()):
        font = theme.fonts.get(key)
        if font is None:
            check(f"{key} font exists", False)
            continue
        check(
            f"{key} is {size}pt",
            font.pointSize() == size,
            f"{font.family()!r} {font.pointSize()}pt",
        )


def check_cards(window: MainWindow) -> None:
    print("cards")
    cards = [c for c in window.findChildren(Card) if c.isVisible()]
    # Input, Upscale and Output are cards; Performance and Size exclusions are
    # collapsible panels that start folded.
    check("the three cards are laid out", len(cards) == 3, f"{len(cards)} visible")
    panels = [p for p in window.findChildren(Collapsible) if p.isVisible()]
    check("both collapsible panels are laid out", len(panels) >= 2, f"{len(panels)} visible")
    narrow = [c for c in cards if c.width() < 320]
    check("no card collapsed narrower than 320px", not narrow, f"{len(narrow)} too narrow")
    short = [c for c in cards if c.height() < 48]
    check("no card collapsed shorter than 48px", not short, f"{len(short)} too short")
    zones = window.findChildren(DropZone)
    check("the drop zone accepts drops", bool(zones) and zones[0].acceptDrops())


def check_scrolling(window: MainWindow, app: QApplication) -> None:
    print("scrolling")
    areas = window.findChildren(QScrollArea)
    if not check("the page has a scroll area", bool(areas)):
        return
    bar = areas[0].verticalScrollBar()
    pump(app)
    check("the page is taller than its viewport", bar.maximum() > 0, f"max {bar.maximum()}")
    bar.setValue(bar.maximum())
    pump(app)
    check("the page scrolls", bar.value() > 0, f"at {bar.value()}")
    bar.setValue(0)
    pump(app)


def check_wheel_guard(window: MainWindow, app: QApplication) -> None:
    """A wheel gesture must scroll the page, not silently retune a control."""
    print("wheel guard")
    combos = [w for w in window.findChildren(QComboBox) if w.isVisible() and w.count() > 1]
    if check("a combo box is on screen", bool(combos)):
        combo = combos[0]
        combo.clearFocus()
        before = combo.currentIndex()
        if send_wheel(app, combo):
            check(
                "an unfocused combo box ignores the wheel",
                combo.currentIndex() == before,
                f"index {before} -> {combo.currentIndex()}",
            )
    spins = [w for w in window.findChildren(QAbstractSpinBox) if w.isVisible()]
    if check("a spin box is on screen", bool(spins)):
        spin = spins[0]
        spin.clearFocus()
        before_text = spin.text()
        if send_wheel(app, spin):
            check(
                "an unfocused spin box ignores the wheel",
                spin.text() == before_text,
                f"{before_text!r} -> {spin.text()!r}",
            )


def check_rules_table(window: MainWindow, theme: Theme, app: QApplication) -> None:
    print("rules table")
    # Named, not searched for: the exclusions table is a RulesTable too and it
    # starts folded away, so findChildren order can hand back a hidden widget
    # whose column geometry means nothing.
    table = window.rules_view
    if not check("the rules table is present", isinstance(table, RulesTable)):
        return
    pump(app)
    model = table.model()
    columns = model.columnCount() if model is not None else 0
    check("five columns", columns == 5, f"{columns} columns")
    header = table.horizontalHeader()
    stretching = [
        c for c in range(columns) if header.sectionResizeMode(c) == QHeaderView.ResizeMode.Stretch
    ]
    check("the Model column takes the slack", stretching == [3], f"stretching {stretching}")
    used = sum(header.sectionSize(c) for c in range(columns))
    room = table.viewport().width()
    check("every column fits the viewport", used <= room + 2, f"{used}px in {room}px")
    line = QFontMetrics(theme.fonts["body"]).height()
    row_h = table.verticalHeader().defaultSectionSize()
    check("row height clears the body font", row_h >= line + 4, f"{row_h}px row, {line}px text")


def check_log_panel(window: MainWindow, app: QApplication) -> None:
    print("log panel")
    window.show_log(True)
    pump(app)
    check("the log panel opens", window.log_panel.isVisible())
    sizes = window.splitter.sizes()
    check("the splitter gives both halves room", all(s > 0 for s in sizes), str(sizes))
    window.log("uicheck reached the log", "info")
    pump(app)
    views = window.findChildren(LogView)
    text = views[0].toPlainText() if views else ""
    check("a line reaches the log view", "uicheck reached the log" in text)
    window.show_log(False)
    pump(app)


def check_theme_toggle(window: MainWindow, app: QApplication) -> None:
    print("theme")
    was = window.theme.p.name
    before = app.styleSheet()
    window.toggle_theme()
    pump(app)
    check(
        "toggling swaps the palette", window.theme.p.name != was, f"{was} -> {window.theme.p.name}"
    )
    check("toggling restyles the whole application", app.styleSheet() != before)
    window.toggle_theme()
    pump(app)
    check("toggling back restores it", window.theme.p.name == was)


def check_text_entry(window: MainWindow, app: QApplication) -> None:
    """Typing has to reach the file-name field - a dead text box was reported."""
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    print()
    print("text entry")
    field = window.ed_pattern
    check(
        "the name pattern defaults to {name}_JaNai",
        field.text() == "{name}_JaNai",
        field.text(),
    )
    check("the field is editable", field.isEnabled() and not field.isReadOnly())
    check("the field can take focus", field.focusPolicy() != Qt.FocusPolicy.NoFocus)
    field.setFocus(Qt.FocusReason.MouseFocusReason)
    pump(app)
    check("focus lands on the field", field.hasFocus())
    was = field.text()
    field.selectAll()
    for char in "ZQ":
        for kind in (QEvent.Type.KeyPress, QEvent.Type.KeyRelease):
            app.sendEvent(
                field, QKeyEvent(kind, Qt.Key.Key_Z, Qt.KeyboardModifier.NoModifier, char)
            )
    pump(app)
    check("typed keys reach the field", field.text() == "ZQ", f"{was!r} -> {field.text()!r}")
    field.setText(was)
    pump(app)


def check_page_kind(window: MainWindow, app: QApplication) -> None:
    """The page kind is one exclusive choice, and it drives the table and summary."""
    print()
    print("page kind")
    check("three exclusive page kinds", window.cb_pagekind.count() == 3)
    was = window.page_kind()
    window.set_page_kind("colour")
    window.on_pagekind_change()
    pump(app)
    check("declaring colour reads back", window.page_kind() == "colour")
    check("declaring colour stands down the grayscale rules", not window.gray_rules_live())
    check(
        "the hint says the grayscale rules are idle",
        "every page is colour" in window.lbl_rules_hint.text(),
        window.lbl_rules_hint.text(),
    )
    # Detection only decides how a page is measured, so once every page is
    # declared these numbers cannot change anything and must not look live.
    check(
        "declaring colour greys out the detection settings",
        not window.sp_threshold.isEnabled() and not window.sp_colour.isEnabled(),
        f"threshold {window.sp_threshold.isEnabled()}, colour {window.sp_colour.isEnabled()}",
    )
    window.set_page_kind("grayscale")
    window.on_pagekind_change()
    pump(app)
    check("declaring grayscale keeps the grayscale rules live", window.gray_rules_live())
    check("declaring grayscale also greys out detection", not window.sp_threshold.isEnabled())
    window.set_page_kind("detect")
    window.on_pagekind_change()
    pump(app)
    check(
        "detect per page brings the detection settings back",
        window.sp_threshold.isEnabled() and window.sp_colour.isEnabled(),
    )
    window.set_page_kind(was)
    window.on_pagekind_change()
    pump(app)


def check_row_numbers(window: MainWindow) -> None:
    """Warnings name row numbers, so the table has to show them."""
    from PySide6.QtCore import Qt

    print()
    print("rule rows")
    table = window.rules_view
    check("row numbers are shown", table.verticalHeader().isVisible())
    model = table.model()
    first = model.headerData(0, Qt.Orientation.Vertical, Qt.ItemDataRole.DisplayRole)
    check("the first row is numbered 1", str(first) == "1", repr(first))
    notes = [r.note for r in window.rules]
    left = [n for n in notes if "catch-all" in n]
    check("no unsized catch-all rows survive", not left, f"{len(left)} left")
    check("no scale warnings on the shipped table", not window.scale_mismatches())


def check_table_wheel(window: MainWindow, app: QApplication) -> None:
    """The table keeps the wheel while it can scroll, and frees it when it cannot.

    Reaching an end used to hand the wheel to the page, which then jumped under
    the pointer, so holding on to it at the end is the fix and not a side effect.
    A table with nothing to scroll is the one case that should pass it on.
    """
    print()
    print("table wheel")
    table = window.rules_view
    areas = window.findChildren(QScrollArea)
    page = areas[0].verticalScrollBar() if areas else None
    table.setMaximumHeight(70)  # force a scrollbar without touching any data
    pump(app)
    bar = table.verticalScrollBar()
    if bar.maximum() <= 0:
        print("  skip  the table will not scroll at this size")
        table.setMaximumHeight(16777215)
        pump(app)
        return
    if page is not None:
        page.setValue(0)
    bar.setValue(0)
    pump(app)

    behind = page.value() if page is not None else 0
    verdict = dispatch_wheel(app, table)
    if verdict is None:
        table.setMaximumHeight(16777215)
        pump(app)
        return
    now = page.value() if page is not None else 0
    check(
        "the wheel scrolls the table, not the page behind it",
        verdict and bar.value() > 0 and now == behind,
        f"table 0 -> {bar.value()}, page {behind} -> {now}",
    )

    bar.setValue(bar.maximum())
    pump(app)
    end = bar.value()
    behind = page.value() if page is not None else 0
    verdict = dispatch_wheel(app, table)
    now = page.value() if page is not None else 0
    check(
        "at its end it keeps the wheel rather than scrolling the page",
        bool(verdict) and bar.value() == end and now == behind,
        f"page {behind} -> {now}, table still at {bar.value()}",
    )

    empty = window.excl_view
    if empty.verticalScrollBar().maximum() <= 0:
        passed_on = dispatch_wheel(app, empty, accepted=True)
        if passed_on is not None:
            check(
                "a table with nothing to scroll hands the wheel on",
                not passed_on,
                "the empty exclusions table",
            )

    table.setMaximumHeight(16777215)
    bar.setValue(0)
    if page is not None:
        page.setValue(0)
    pump(app)


def check_panels(window: MainWindow, app: QApplication) -> None:
    """Settings most runs never touch start folded, and exclusions get own table."""
    print()
    print("panels")
    check("the performance panel starts collapsed", not window.panel_perf.is_open())
    check("the size-exclusion panel starts collapsed", not window.panel_excl.is_open())
    check(
        "exclusions are a table of their own",
        isinstance(window.excl_view, RulesTable) and window.excl_view is not window.rules_view,
    )
    check("each table has its own model", window.excl_model is not window.rules_model)
    upscale = window.rules_model.rowCount()
    excluded = window.excl_model.rowCount()
    check(
        "the two tables together hold every rule",
        upscale + excluded == len(window.rules),
        f"{upscale} upscale + {excluded} excluded, {len(window.rules)} rules",
    )
    window.panel_excl.set_open(True)
    pump(app)
    check("opening the panel shows its table", window.excl_view.isVisible())
    header = window.excl_view.horizontalHeader()
    used = sum(header.sectionSize(c) for c in range(window.excl_model.columnCount()))
    room = window.excl_view.viewport().width()
    check("its columns fit the viewport", used <= room + 2, f"{used}px in {room}px")
    window.panel_excl.set_open(False)
    pump(app)
    check("it folds away again", not window.panel_excl.is_open())


def check_tile_ladder(window: MainWindow) -> None:
    """The fixed tile sizes were too coarse to land on the one that fits."""
    print()
    print("tile sizes")
    items = [window.cb_tile.itemText(i) for i in range(window.cb_tile.count())]
    check("the ladder offers a full range", len(items) >= 19, f"{len(items)} entries")
    missing = [w for w in ("640 px", "896 px", "1024 px", "1152 px", "2048 px") if w not in items]
    check("the sizes worth trying are offered", not missing, f"missing {missing}")
    check(
        "no tiling says plainly that it can fail",
        any("fails if it will not fit" in t for t in items),
        next((t for t in items if t.lower().startswith("no tiling")), "missing"),
    )


def check_geometry_clamp(window: MainWindow, app: QApplication) -> None:
    """A saved size larger than the desktop must be clamped, not restored as-is."""
    print()
    print("geometry")
    was = window._geometry_text()
    screen = app.primaryScreen()
    area = screen.availableGeometry() if screen is not None else None

    def centred(label: str) -> None:
        """Centred, clamped exactly the way _apply_geometry clamps it."""
        if area is None:
            return
        want_x = area.x() + max(0, (area.width() - window.width()) // 2)
        want_y = area.y() + max(0, (area.height() - window.height()) // 2)
        check(
            label,
            abs(window.x() - want_x) <= 4 and abs(window.y() - want_y) <= 4,
            f"at +{window.x()}+{window.y()}, centred is +{want_x}+{want_y}",
        )

    window._apply_geometry("4000x3000+0+0")
    pump(app)
    check(
        "an oversized saved size is clamped",
        window.width() < 4000 and window.height() < 3000,
        f"{window.width()}x{window.height()}",
    )
    if area is not None and area.width() >= 1040 and area.height() >= 760:
        check(
            "the clamped window leaves desktop around it",
            window.width() < area.width() and window.height() < area.height(),
            f"{window.width()}x{window.height()} in {area.width()}x{area.height()}",
        )
    centred("a size trimmed to fit is re-centred, not left in the corner")
    window._apply_geometry("")
    pump(app)
    check("an unsaved window still opens workably", window.width() >= 920, f"{window.width()}px")
    centred("a first run opens centred on the desktop")
    window._apply_geometry(was)
    pump(app)


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)

    # Qt can still hold a handle inside the folder after the window goes, and
    # Windows will not delete a folder that is in use. Every check is done by
    # then, so a leftover temp folder must not fail the run.
    with tempfile.TemporaryDirectory(prefix="janai-uicheck-", ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        saved = ROOT / "settings.json"
        if saved.exists():  # same layout as the real thing, none of the writes
            shutil.copy2(saved, root / "settings.json")

        theme = Theme("dark")
        theme.apply(app)
        window = MainWindow(root, theme, app)
        window.resize(1180, 900)
        window.show()
        pump(app, 6)

        screen = app.primaryScreen()
        size = screen.geometry() if screen is not None else None
        print(
            f"platform {app.platformName()!r}  "
            f"screen {size.width() if size else 0}x{size.height() if size else 0}  "
            f"ratio {window.devicePixelRatio():.2f}  "
            f"window {window.width()}x{window.height()}"
        )

        check_fonts(theme)
        check_cards(window)
        check_scrolling(window, app)
        check_wheel_guard(window, app)
        check_rules_table(window, theme, app)
        check_table_wheel(window, app)
        check_panels(window, app)
        check_tile_ladder(window)
        check_log_panel(window, app)
        check_theme_toggle(window, app)
        check_text_entry(window, app)
        check_page_kind(window, app)
        check_row_numbers(window)
        check_geometry_clamp(window, app)

        window.close()
        pump(app)

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: " + ", ".join(FAILURES))
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
