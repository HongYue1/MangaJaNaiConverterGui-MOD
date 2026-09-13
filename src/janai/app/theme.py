"""Palette, type ramp and stylesheet for the JaNai Upscaler interface.

Qt does the pixel work the old Tk theme had to fake by hand: per-monitor DPI
scaling, real controls, and one stylesheet for the whole window. Switching
theme is therefore a single string swap instead of a walk over every widget.

Colours live in :class:`Palette`. :meth:`Theme.apply` pushes them into the
application twice over: as a QPalette, so the parts Qt draws itself (checkbox
ticks, dropdown arrows, text cursors) follow the theme, and as targeted QSS
for cards, tables, buttons and typography.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtGui import QColor, QFont, QFontDatabase, QPalette
from PySide6.QtWidgets import QApplication

#: Small SVGs for the one part of a control a stylesheet has to redraw itself:
#: the checkbox tick. Qt wants a URL, so the path is POSIX-style on Windows too.
ASSETS = Path(__file__).resolve().parent / "assets"


def asset_url(name: str) -> str:
    return ASSETS.joinpath(name).as_posix()


#: Point sizes for the type ramp. Qt multiplies points by the display scale
#: itself, so these are written once and never touched again - the double
#: scaling that made the old build's text twice too big cannot happen here.
BASE_SIZES: dict[str, int] = {"display": 16, "title": 11, "body": 9, "small": 8, "mono": 9}

#: Preferred families, best first. Qt falls back per glyph, so a missing face
#: degrades to the next one instead of drawing empty boxes.
SANS: tuple[str, ...] = (
    "Segoe UI Variable Text",
    "Segoe UI",
    "Inter",
    "Noto Sans",
    "DejaVu Sans",
    "Helvetica Neue",
)
MONO: tuple[str, ...] = (
    "Cascadia Mono",
    "Consolas",
    "JetBrains Mono",
    "Menlo",
    "DejaVu Sans Mono",
    "Courier New",
)


@dataclass(frozen=True)
class Palette:
    """One theme's colours. Everything drawn comes from these fields."""

    name: str
    bg: str  # window behind the cards
    surface: str  # card face
    surface2: str  # inputs, headers, raised chips
    line: str  # borders and separators
    text: str
    muted: str  # secondary text, disabled rows
    accent: str
    accent_text: str  # text drawn on top of the accent
    accent_hi: str  # accent, hovered
    ok: str
    warn: str
    err: str
    sel: str  # selected table row
    drop: str  # the drop target's face
    row: str  # alternating table row


DARK = Palette(
    name="dark",
    bg="#0f1116",
    surface="#161a21",
    surface2="#1d222b",
    line="#272e3a",
    text="#e7eaf0",
    muted="#98a2b3",
    accent="#4c8dff",
    accent_text="#08101f",
    accent_hi="#6ba2ff",
    ok="#48d597",
    warn="#f3c14b",
    err="#ff6b6b",
    sel="#24334d",
    drop="#12161d",
    row="#1a1f27",
)

LIGHT = Palette(
    name="light",
    bg="#f4f6f9",
    surface="#ffffff",
    surface2="#eef1f5",
    line="#dce1e9",
    text="#161a22",
    muted="#5b6577",
    accent="#2563eb",
    accent_text="#ffffff",
    accent_hi="#1d4ed8",
    ok="#0f7b4f",
    warn="#a4650a",
    err="#c02626",
    sel="#dce8ff",
    drop="#f7f9fc",
    row="#f8fafc",
)

PALETTES: dict[str, Palette] = {DARK.name: DARK, LIGHT.name: LIGHT}


def _family(preferred: tuple[str, ...], fixed: bool = False) -> str:
    """The first installed family from ``preferred``, else the system default."""
    try:
        installed = set(QFontDatabase.families())
    except Exception:  # no QGuiApplication yet
        installed = set()
    for name in preferred:
        if name in installed:
            return name
    role = QFontDatabase.SystemFont.FixedFont if fixed else QFontDatabase.SystemFont.GeneralFont
    try:
        return QFontDatabase.systemFont(role).family()
    except Exception:
        return "monospace" if fixed else "sans-serif"


def _fonts() -> dict[str, QFont]:
    """The type ramp as ready-made fonts."""
    sans, mono = _family(SANS), _family(MONO, fixed=True)
    out: dict[str, QFont] = {}
    for key, size in BASE_SIZES.items():
        font = QFont(mono if key == "mono" else sans)
        font.setPointSize(size)
        if key == "display" or key == "title":
            font.setWeight(QFont.Weight.DemiBold)
        out[key] = font
    return out


class Theme:
    """The active palette, and the means to put it on screen."""

    def __init__(self, mode: str = "dark") -> None:
        self.p = PALETTES.get(str(mode).lower().strip(), DARK)
        self.fonts = _fonts()

    @property
    def mode(self) -> str:
        return self.p.name

    def apply(self, app: QApplication) -> None:
        """Dress the whole application in the current palette."""
        app.setStyle("Fusion")  # one predictable base on every platform
        app.setFont(self.fonts["body"])
        app.setPalette(self.qpalette())
        app.setStyleSheet(self.qss())

    def toggle(self, app: QApplication) -> str:
        """Swap dark and light, repaint everything, and report the new mode."""
        self.p = LIGHT if self.p.name == DARK.name else DARK
        self.apply(app)
        return self.p.name

    # ------------------------------------------------------------------ #
    def qpalette(self) -> QPalette:
        """Colours for everything Qt draws without asking the stylesheet."""
        p = self.p
        pal = QPalette()
        role = QPalette.ColorRole
        group = QPalette.ColorGroup
        pal.setColor(role.Window, QColor(p.bg))
        pal.setColor(role.WindowText, QColor(p.text))
        pal.setColor(role.Base, QColor(p.surface2))
        pal.setColor(role.AlternateBase, QColor(p.row))
        pal.setColor(role.ToolTipBase, QColor(p.surface2))
        pal.setColor(role.ToolTipText, QColor(p.text))
        pal.setColor(role.Text, QColor(p.text))
        pal.setColor(role.PlaceholderText, QColor(p.muted))
        pal.setColor(role.Button, QColor(p.surface2))
        pal.setColor(role.ButtonText, QColor(p.text))
        pal.setColor(role.BrightText, QColor(p.err))
        pal.setColor(role.Link, QColor(p.accent))
        pal.setColor(role.Highlight, QColor(p.accent))
        pal.setColor(role.HighlightedText, QColor(p.accent_text))
        pal.setColor(role.Mid, QColor(p.line))
        pal.setColor(role.Midlight, QColor(p.surface2))
        pal.setColor(role.Dark, QColor(p.line))
        pal.setColor(role.Shadow, QColor(p.bg))
        for disabled in (role.WindowText, role.Text, role.ButtonText):
            pal.setColor(group.Disabled, disabled, QColor(p.muted))
        return pal

    def qss(self) -> str:
        """The stylesheet for the parts worth styling by hand."""
        p = self.p
        body = BASE_SIZES["body"]
        check = asset_url("check.svg")
        return f"""
* {{ outline: 0; }}

/* Only the window, its dialogs and the scrolling page paint a background.
   Every container inside a card stays transparent, so a field can never draw
   a darker rectangle onto the card it sits on. */
QWidget {{ color: {p.text}; }}
QMainWindow, QDialog {{ background: {p.bg}; }}
QToolTip {{
    background: {p.surface2};
    color: {p.text};
    border: 1px solid {p.line};
    border-radius: 6px;
    padding: 6px 8px;
}}

/* ---- typography ------------------------------------------------- */
QLabel {{ background: transparent; }}
QLabel[role="display"] {{ font-size: {BASE_SIZES["display"]}pt; font-weight: 600; }}
QLabel[role="title"] {{ font-size: {BASE_SIZES["title"]}pt; font-weight: 600; }}
QLabel[role="field"] {{ font-weight: 600; }}
QLabel[role="muted"] {{ color: {p.muted}; }}
QLabel[role="hint"] {{ color: {p.muted}; font-size: {BASE_SIZES["small"]}pt; }}
QLabel[role="ok"] {{ color: {p.ok}; }}
QLabel[role="warn"] {{ color: {p.warn}; }}
QLabel[role="err"] {{ color: {p.err}; }}
QLabel[role="badge"] {{
    background: {p.surface2};
    color: {p.muted};
    border: 1px solid {p.line};
    border-radius: 9px;
    padding: 2px 9px;
    font-size: {BASE_SIZES["small"]}pt;
}}

/* ---- cards ------------------------------------------------------- */
QFrame#card {{
    background: {p.surface};
    border: 1px solid {p.line};
    border-radius: 10px;
}}
QFrame#cardhead {{ background: transparent; border: 0; border-radius: 8px; }}
QFrame#cardhead:hover {{ background: {p.surface2}; }}
QFrame#drop {{
    background: {p.drop};
    border: 1px dashed {p.line};
    border-radius: 9px;
}}
QFrame#drop[active="true"] {{ border: 1px dashed {p.accent}; background: {p.sel}; }}
QFrame#banner {{
    background: {p.surface2};
    border: 1px solid {p.err};
    border-left: 3px solid {p.err};
    border-radius: 8px;
}}
QFrame[role="sep"] {{ background: {p.line}; border: 0; max-height: 1px; min-height: 1px; }}

/* ---- buttons ----------------------------------------------------- */
QPushButton {{
    background: {p.surface2};
    color: {p.text};
    border: 1px solid {p.line};
    border-radius: 7px;
    padding: 6px 13px;
    min-height: 18px;
}}
QPushButton:hover {{ border-color: {p.muted}; }}
QPushButton:pressed {{ background: {p.line}; }}
QPushButton:disabled {{ color: {p.muted}; border-color: {p.line}; background: transparent; }}
QPushButton[variant="accent"] {{
    background: {p.accent};
    color: {p.accent_text};
    border: 1px solid {p.accent};
    font-weight: 600;
}}
QPushButton[variant="accent"]:hover {{ background: {p.accent_hi}; border-color: {p.accent_hi}; }}
QPushButton[variant="accent"]:disabled {{
    background: transparent;
    color: {p.muted};
    border: 1px solid {p.line};
}}
QPushButton[variant="ghost"] {{ background: transparent; border-color: transparent; }}
QPushButton[variant="ghost"]:hover {{ background: {p.surface2}; border-color: {p.line}; }}
/* The weight is set once, for every state. A font that changes on :checked
   widens the label but not the button, which is what clipped the text of the
   selected option. */
QPushButton[variant="seg"] {{
    background: transparent;
    border: 1px solid {p.line};
    border-radius: 7px;
    padding: 5px 13px;
    color: {p.muted};
    font-weight: 600;
}}
QPushButton[variant="seg"]:hover {{ color: {p.text}; }}
QPushButton[variant="seg"]:checked {{
    background: {p.accent};
    border-color: {p.accent};
    color: {p.accent_text};
}}
QPushButton[variant="seg"]:disabled {{ color: {p.muted}; border-color: {p.line}; }}
QToolButton {{
    background: transparent;
    border: 0;
    color: {p.text};
    padding: 2px 4px;
}}

/* ---- inputs ------------------------------------------------------ */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {p.surface2};
    color: {p.text};
    border: 1px solid {p.line};
    border-radius: 7px;
    padding: 5px 8px;
    min-height: 18px;
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {p.accent};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {p.muted};
    background: transparent;
}}
QComboBox::drop-down {{ border: 0; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {p.surface2};
    border: 1px solid {p.line};
    border-radius: 8px;
    padding: 4px;
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{ width: 15px; border: 0; }}
QCheckBox {{ background: transparent; spacing: 8px; padding: 2px 0; }}
QCheckBox:disabled {{ color: {p.muted}; }}
/* An empty box still has to read as a box. Fusion draws a cleared indicator
   in the base colour, which on a card face is all but invisible. */
QCheckBox::indicator, QTableView::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {p.muted};
    border-radius: 4px;
    background: {p.surface2};
}}
QCheckBox::indicator:hover, QTableView::indicator:hover {{ border-color: {p.accent}; }}
QCheckBox::indicator:checked, QTableView::indicator:checked {{
    background: {p.accent};
    border-color: {p.accent};
    image: url("{check}");
}}
QCheckBox::indicator:disabled {{ border-color: {p.line}; background: transparent; }}
QCheckBox::indicator:checked:disabled {{
    background: {p.line};
    border-color: {p.line};
    image: url("{check}");
}}

/* ---- table ------------------------------------------------------- */
QTableView {{
    background: {p.surface};
    alternate-background-color: {p.row};
    border: 1px solid {p.line};
    border-radius: 8px;
    gridline-color: transparent;
    selection-background-color: {p.sel};
    selection-color: {p.text};
    font-size: {body}pt;
}}
QTableView::item {{ border: 0; padding: 2px 6px; }}
QTableView::item:focus {{ border: 0; }}
QHeaderView {{ background: transparent; border: 0; }}
QHeaderView::section {{
    background: {p.surface2};
    color: {p.muted};
    border: 0;
    border-bottom: 1px solid {p.line};
    padding: 6px 7px;
    font-weight: 600;
}}

/* ---- log --------------------------------------------------------- */
QPlainTextEdit {{
    background: {p.surface};
    color: {p.text};
    border: 1px solid {p.line};
    border-radius: 8px;
    padding: 6px;
    selection-background-color: {p.accent};
    selection-color: {p.accent_text};
}}

/* ---- progress ---------------------------------------------------- */
QProgressBar {{
    background: {p.surface2};
    border: 0;
    border-radius: 4px;
    max-height: 7px;
    min-height: 7px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{ background: {p.accent}; border-radius: 4px; }}

/* ---- scrolling --------------------------------------------------- */
QScrollArea {{ background: {p.bg}; border: 0; }}
QScrollArea > QWidget > QWidget {{ background: {p.bg}; }}
QScrollBar:vertical {{ background: transparent; width: 11px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 11px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {p.line}; border-radius: 5px; min-height: 30px; }}
QScrollBar::handle:horizontal {{ background: {p.line}; border-radius: 5px; min-width: 30px; }}
QScrollBar::handle:hover {{ background: {p.muted}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; border: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QSplitter::handle {{ background: transparent; height: 8px; }}

/* ---- menus ------------------------------------------------------- */
QMenu {{
    background: {p.surface2};
    border: 1px solid {p.line};
    border-radius: 8px;
    padding: 5px;
}}
QMenu::item {{ padding: 6px 18px 6px 12px; border-radius: 5px; }}
QMenu::item:selected {{ background: {p.accent}; color: {p.accent_text}; }}
QMenu::separator {{ height: 1px; background: {p.line}; margin: 5px 6px; }}
"""
