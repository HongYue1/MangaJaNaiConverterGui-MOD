"""Dark/light theme for ttk. Pure stdlib, no third-party dependencies.

The look is deliberately minimal: one background, one card surface, one inset
surface for controls, a single accent, and a 1px border instead of shadows or
gradients. Everything is expressed as ttk styles so the widgets stay native -
real focus rings, real DPI scaling, real keyboard behaviour.

Point sizes are a plain fixed ramp. Tk already multiplies every point size by
the display's own scaling (see ``main.enable_dpi_awareness`` and the ``tk
scaling`` call next to it), so a second factor applied here made text come out
roughly twice too large on a scaled display.

One part is less obvious than it looks: **check boxes are drawn here.** clam's
own indicator is a bevelled box that loses its outline once borders are
flattened, which left the boxes hard to tell apart from their background. They
are painted as small images instead, one set per surface, sized from the body
font's measured line height and repainted whenever the palette changes.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from itertools import pairwise
from tkinter import font as tkfont, ttk


@dataclass(frozen=True)
class Palette:
    name: str
    bg: str
    surface: str
    surface2: str
    border: str
    text: str
    muted: str
    accent: str
    accent_hi: str
    accent_text: str
    ok: str
    warn: str
    err: str
    sel: str


DARK = Palette(
    name="dark",
    bg="#0f1115",
    surface="#161920",
    surface2="#1e222b",
    border="#272c37",
    text="#e6e9ef",
    muted="#8b93a3",
    accent="#6a8cff",
    accent_hi="#88a3ff",
    accent_text="#ffffff",
    ok="#4ec38a",
    warn="#e0b341",
    err="#f36d64",
    sel="#243050",
)

LIGHT = Palette(
    name="light",
    bg="#f6f7f9",
    surface="#ffffff",
    surface2="#f0f2f6",
    border="#e1e4ea",
    text="#14171d",
    muted="#5f6673",
    accent="#3b6cf6",
    accent_hi="#2a5be0",
    accent_text="#ffffff",
    ok="#147d45",
    warn="#8a6200",
    err="#c92a2a",
    sel="#dde6ff",
)

PALETTES = {"dark": DARK, "light": LIGHT}


# Preference order for the interface face and the log face. Both lists end in
# faces that ship with mainstream Linux desktops, so the app reads the same way
# there as it does on Windows instead of falling back to a bitmap font.
UI_FAMILIES = (
    "Segoe UI Variable Text",
    "Segoe UI",
    "Inter",
    "SF Pro Text",
    "Noto Sans",
    "Ubuntu",
    "Cantarell",
    "DejaVu Sans",
    "Arial",
)
MONO_FAMILIES = (
    "Cascadia Mono",
    "Cascadia Code",
    "Consolas",
    "JetBrains Mono",
    "SF Mono",
    "Liberation Mono",
    "DejaVu Sans Mono",
    "Menlo",
    "Courier New",
)


def pick_family(candidates: tuple[str, ...] = UI_FAMILIES, fallback: str = "TkDefaultFont") -> str:
    """First installed family from ``candidates``, else ``fallback``."""
    families = set(tkfont.families())
    for name in candidates:
        if name in families:
            return name
    return fallback


#: One ramp used everywhere: 9 for supporting text, 10 for body and the log,
#: 11 and 16 for the two heading levels. Tk applies the display's scaling to
#: these on its own - nothing multiplies them a second time.
BASE_SIZES = {
    "body": 10,
    "bold": 10,
    "small": 9,
    "tiny": 9,
    "title": 16,
    "card": 11,
    "mono": 10,
    "mono_bold": 10,
}
BOLD_KEYS = frozenset({"bold", "title", "card", "mono_bold"})


def _stamp_tick(rows: list[list[str]], box: int, colour: str, thick: int) -> None:
    """Draw a check mark inside an already-painted box."""
    weight = 1 if box < 22 else 2
    corners = ((0.26, 0.54), (0.44, 0.71), (0.76, 0.31))
    for (x0, y0), (x1, y1) in pairwise(corners):
        for step in range(box + 1):
            t = step / box
            cx = round((x0 + (x1 - x0) * t) * box)
            cy = round((y0 + (y1 - y0) * t) * box)
            for dy in range(-weight, weight + 1):
                for dx in range(-weight, weight + 1):
                    x, y = cx + dx, cy + dy
                    if thick <= x < box - thick and thick <= y < box - thick:
                        rows[y][x] = colour


def _check_matrix(
    box: int,
    gap: int,
    *,
    fill: str,
    edge: str,
    mark: str | None,
    outside: str,
    thick: int,
) -> str:
    """Photo-image data for one check-box state.

    The box is opaque and the gap before the label is painted in the surface
    behind it, so a single image works on any card without transparency.
    Corner pixels take the surface colour, which reads as a rounded box.
    """
    rows: list[list[str]] = []
    last = box - 1
    for y in range(box):
        row: list[str] = []
        for x in range(box):
            at_x = x < thick or x > last - thick
            at_y = y < thick or y > last - thick
            if at_x and at_y:
                row.append(outside)
            elif at_x or at_y:
                row.append(edge)
            else:
                row.append(fill)
        row.extend([outside] * gap)
        rows.append(row)
    if mark:
        _stamp_tick(rows, box, mark, thick)
    return " ".join("{" + " ".join(r) + "}" for r in rows)


class Theme:
    """Applies a palette to a root window and remembers it for redraws."""

    #: Which surface each check-button style sits on, so the pixels around the
    #: box match the background behind it exactly.
    CHECK_STYLES = (
        ("TCheckbutton", "surface"),
        ("Inset.TCheckbutton", "surface2"),
        ("Bg.TCheckbutton", "bg"),
    )

    def __init__(self, root: tk.Misc, mode: str = "dark") -> None:
        self.root = root
        self.style = ttk.Style(root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.family = pick_family()
        self.mono_family = pick_family(MONO_FAMILIES, "TkFixedFont")
        self.fonts = {
            key: tkfont.Font(
                family=self.mono_family if key.startswith("mono") else self.family,
                size=BASE_SIZES[key],
                weight="bold" if key in BOLD_KEYS else "normal",
            )
            for key in BASE_SIZES
        }
        self._sync_named_fonts()
        self._images: dict[str, tk.PhotoImage] = {}
        self._indicators: dict[str, str] = {}
        self.p = PALETTES.get(mode, DARK)
        self.apply(mode)

    # -- sizing --------------------------------------------------------- #
    def line_height(self, key: str = "body") -> int:
        """Measured line height of one of the shared fonts."""
        try:
            return int(self.fonts[key].metrics("linespace"))
        except (tk.TclError, KeyError):
            return 17

    def row_height(self) -> int:
        """The original 26px table row, widened only if the font needs it."""
        return max(26, self.line_height("small") + 6)

    def _sync_named_fonts(self) -> None:
        """Point Tk's named fonts at the same faces.

        Plain tk widgets - the log text area, tooltips, menus - read the named
        fonts rather than a ttk style.
        """
        for named, key in (
            ("TkDefaultFont", "body"),
            ("TkTextFont", "body"),
            ("TkMenuFont", "body"),
            ("TkHeadingFont", "bold"),
            ("TkTooltipFont", "small"),
            ("TkFixedFont", "mono"),
        ):
            try:
                target = tkfont.nametofont(named, root=self.root)
            except tk.TclError:
                continue
            src = self.fonts[key]
            target.configure(
                family=src.cget("family"), size=src.cget("size"), weight=src.cget("weight")
            )

    # -- check-box indicators ------------------------------------------- #
    def _indicator_size(self) -> tuple[int, int]:
        box = max(14, min(28, round(self.line_height("body") * 0.92)))
        return box, max(6, round(box * 0.45))

    def _paint_indicators(self) -> None:
        """(Re)draw every indicator image for the current palette and size."""
        p = self.p
        box, gap = self._indicator_size()
        thick = 2 if box >= 20 else 1
        for _style_name, surface in self.CHECK_STYLES:
            outside = getattr(p, surface)
            empty = p.bg if surface == "surface2" else p.surface2
            states = {
                "off": {"fill": empty, "edge": p.border, "mark": None},
                "hover": {"fill": empty, "edge": p.accent, "mark": None},
                "on": {"fill": p.accent, "edge": p.accent, "mark": p.accent_text},
                "hover_on": {"fill": p.accent_hi, "edge": p.accent_hi, "mark": p.accent_text},
                "off_off": {"fill": outside, "edge": p.border, "mark": None},
                "on_off": {"fill": p.border, "edge": p.border, "mark": p.muted},
            }
            for state, kw in states.items():
                key = f"{surface}:{state}"
                img = self._images.get(key)
                if img is None:
                    img = tk.PhotoImage(master=self.root, width=box + gap, height=box)
                    self._images[key] = img
                elif img.width() != box + gap or img.height() != box:
                    img.configure(width=box + gap, height=box)
                img.blank()
                img.put(_check_matrix(box, gap, outside=outside, thick=thick, **kw), to=(0, 0))

    def _ensure_indicator_elements(self) -> None:
        """Register the image elements and layouts once per interpreter.

        ttk element names are permanent, so the elements are created a single
        time and the *same* images are repainted afterwards. If a Tk build
        refuses the element, clam's own indicator stays in place.
        """
        if self._indicators:
            return
        for style_name, surface in self.CHECK_STYLES:
            element = f"Janai{surface.capitalize()}.Checkbutton.indicator"
            img = self._images
            try:
                self.style.element_create(
                    element,
                    "image",
                    img[f"{surface}:off"],
                    ("disabled", "selected", img[f"{surface}:on_off"]),
                    ("disabled", img[f"{surface}:off_off"]),
                    ("pressed", "selected", img[f"{surface}:hover_on"]),
                    ("active", "selected", img[f"{surface}:hover_on"]),
                    ("selected", img[f"{surface}:on"]),
                    ("active", img[f"{surface}:hover"]),
                    sticky="",
                )
            except (tk.TclError, KeyError):
                continue
            self.style.layout(
                style_name,
                [
                    (
                        "Checkbutton.padding",
                        {
                            "sticky": "nswe",
                            "children": [
                                (element, {"side": "left", "sticky": ""}),
                                (
                                    "Checkbutton.focus",
                                    {
                                        "side": "left",
                                        "sticky": "w",
                                        "children": [("Checkbutton.label", {"sticky": "nswe"})],
                                    },
                                ),
                            ],
                        },
                    )
                ],
            )
            self._indicators[style_name] = element

    # ------------------------------------------------------------------ #
    def apply(self, mode: str) -> None:
        self.p = p = PALETTES.get(mode, DARK)
        s = self.style
        f = self.fonts

        self.root.configure(background=p.bg)
        for opt, val in (
            ("*Toplevel.background", p.bg),
            ("*TCombobox*Listbox.background", p.surface2),
            ("*TCombobox*Listbox.foreground", p.text),
            ("*TCombobox*Listbox.selectBackground", p.accent),
            ("*TCombobox*Listbox.selectForeground", p.accent_text),
            ("*TCombobox*Listbox.font", f["body"]),
            ("*TCombobox*Listbox.borderWidth", "0"),
            ("*Menu.background", p.surface2),
            ("*Menu.foreground", p.text),
            ("*Menu.activeBackground", p.accent),
            ("*Menu.activeForeground", p.accent_text),
            ("*Menu.relief", "flat"),
        ):
            self.root.option_add(opt, val)

        s.configure(
            ".",
            background=p.bg,
            foreground=p.text,
            font=f["body"],
            borderwidth=0,
            focuscolor=p.accent,
        )
        s.configure("TFrame", background=p.bg)
        s.configure("Surface.TFrame", background=p.surface)
        # The card shell is a flat surface plus a hairline border - no shadow,
        # no bevel. Card.TFrame and Plain.TFrame are the borderless surface used
        # for everything nested inside a card, so nesting never draws lines.
        s.configure(
            "CardShell.TFrame",
            background=p.surface,
            bordercolor=p.border,
            lightcolor=p.border,
            darkcolor=p.border,
            borderwidth=1,
            relief="solid",
        )
        s.configure("Card.TFrame", background=p.surface, borderwidth=0, relief="flat")
        s.configure("Plain.TFrame", background=p.surface, borderwidth=0, relief="flat")
        s.configure("Inset.TFrame", background=p.surface2)
        s.configure(
            "Drop.TFrame",
            background=p.surface2,
            bordercolor=p.border,
            lightcolor=p.border,
            darkcolor=p.border,
            borderwidth=1,
            relief="solid",
        )
        s.configure("DropActive.TFrame", background=p.sel)

        s.configure("TLabel", background=p.bg, foreground=p.text)
        s.configure("Card.TLabel", background=p.surface, foreground=p.text)
        s.configure("CardTitle.TLabel", background=p.surface, foreground=p.text, font=f["card"])
        s.configure("Field.TLabel", background=p.surface, foreground=p.text, font=f["body"])
        s.configure("Muted.TLabel", background=p.surface, foreground=p.muted, font=f["small"])
        s.configure("MutedBg.TLabel", background=p.bg, foreground=p.muted, font=f["small"])
        s.configure("Inset.TLabel", background=p.surface2, foreground=p.text)
        s.configure("InsetMuted.TLabel", background=p.surface2, foreground=p.muted, font=f["small"])
        s.configure("Title.TLabel", background=p.bg, foreground=p.text, font=f["title"])
        s.configure("Ok.TLabel", background=p.surface, foreground=p.ok, font=f["small"])
        s.configure("Warn.TLabel", background=p.surface, foreground=p.warn, font=f["small"])
        s.configure("Err.TLabel", background=p.surface, foreground=p.err, font=f["small"])
        s.configure(
            "Chip.TLabel", background=p.surface2, foreground=p.muted, font=f["tiny"], padding=(8, 3)
        )

        s.configure(
            "TButton",
            background=p.surface2,
            foreground=p.text,
            padding=(13, 7),
            borderwidth=0,
            relief="flat",
            anchor="center",
        )
        s.map(
            "TButton",
            background=[("disabled", p.surface), ("pressed", p.border), ("active", p.border)],
            foreground=[("disabled", p.muted)],
        )
        s.configure(
            "Accent.TButton",
            background=p.accent,
            foreground=p.accent_text,
            padding=(20, 8),
            font=f["bold"],
        )
        s.map(
            "Accent.TButton",
            background=[("disabled", p.surface2), ("pressed", p.accent), ("active", p.accent_hi)],
            foreground=[("disabled", p.muted)],
        )
        s.configure("Ghost.TButton", background=p.surface, foreground=p.muted, padding=(10, 5))
        s.map(
            "Ghost.TButton",
            background=[("disabled", p.surface), ("active", p.surface2)],
            foreground=[("disabled", p.border), ("active", p.text)],
        )
        # Same as Ghost, for toolbars that sit on the window background.
        s.configure("GhostBg.TButton", background=p.bg, foreground=p.muted, padding=(10, 5))
        s.map(
            "GhostBg.TButton",
            background=[("disabled", p.bg), ("active", p.surface2)],
            foreground=[("disabled", p.border), ("active", p.text)],
        )
        s.configure(
            "Link.TButton", background=p.bg, foreground=p.accent, padding=(4, 2), font=f["small"]
        )
        s.map("Link.TButton", background=[("active", p.bg)], foreground=[("active", p.accent_hi)])

        # segmented control: radiobuttons drawn as flat toggle buttons
        s.configure(
            "Seg.Toolbutton",
            background=p.surface2,
            foreground=p.muted,
            padding=(14, 6),
            anchor="center",
            font=f["small"],
            relief="flat",
            borderwidth=0,
        )
        s.map(
            "Seg.Toolbutton",
            background=[("selected", p.accent), ("active", p.border)],
            foreground=[("selected", p.accent_text), ("active", p.text)],
        )

        # The box itself is an image (see _paint_indicators); indicatorcolor is
        # kept as the fallback for a Tk that refuses the custom element.
        for style_name, surface in self.CHECK_STYLES:
            background = getattr(p, surface)
            s.configure(
                style_name,
                background=background,
                foreground=p.text,
                indicatorcolor=p.surface2 if surface != "surface2" else p.bg,
                indicatorforeground=p.accent_text,
                bordercolor=p.border,
                focuscolor=p.accent,
                padding=(0, 4),
            )
            s.map(
                style_name,
                background=[("active", background)],
                indicatorcolor=[("selected", p.accent), ("pressed", p.accent_hi)],
                foreground=[("disabled", p.muted)],
            )
        s.configure(
            "TRadiobutton",
            background=p.surface,
            foreground=p.text,
            indicatorcolor=p.surface2,
            bordercolor=p.border,
            padding=(0, 4),
        )
        s.map(
            "TRadiobutton",
            background=[("active", p.surface)],
            indicatorcolor=[("selected", p.accent)],
            foreground=[("disabled", p.muted)],
        )

        s.configure(
            "TEntry",
            fieldbackground=p.surface2,
            foreground=p.text,
            bordercolor=p.border,
            lightcolor=p.surface2,
            darkcolor=p.surface2,
            insertcolor=p.text,
            padding=(8, 6),
        )
        s.map(
            "TEntry",
            bordercolor=[("focus", p.accent)],
            fieldbackground=[("disabled", p.surface)],
            foreground=[("disabled", p.muted)],
        )

        s.configure(
            "TSpinbox",
            fieldbackground=p.surface2,
            foreground=p.text,
            bordercolor=p.border,
            arrowcolor=p.muted,
            arrowsize=12,
            insertcolor=p.text,
            lightcolor=p.surface2,
            darkcolor=p.surface2,
            padding=(8, 5),
        )
        s.map(
            "TSpinbox",
            bordercolor=[("focus", p.accent)],
            arrowcolor=[("active", p.text), ("disabled", p.border)],
            foreground=[("disabled", p.muted)],
        )

        s.configure(
            "TCombobox",
            fieldbackground=p.surface2,
            background=p.surface2,
            foreground=p.text,
            bordercolor=p.border,
            arrowcolor=p.muted,
            arrowsize=13,
            lightcolor=p.surface2,
            darkcolor=p.surface2,
            padding=(8, 5),
            selectbackground=p.surface2,
            selectforeground=p.text,
        )
        s.map(
            "TCombobox",
            bordercolor=[("focus", p.accent)],
            arrowcolor=[("active", p.text), ("disabled", p.border)],
            fieldbackground=[("readonly", p.surface2), ("disabled", p.surface)],
            foreground=[("disabled", p.muted)],
        )

        # the rules table: flat, no indent column, selection in the accent tint
        s.configure(
            "Rules.Treeview",
            background=p.surface2,
            fieldbackground=p.surface2,
            foreground=p.text,
            bordercolor=p.border,
            borderwidth=0,
            relief="flat",
            rowheight=self.row_height(),
            font=f["small"],
        )
        s.map(
            "Rules.Treeview",
            background=[("selected", p.sel), ("disabled", p.surface)],
            foreground=[("selected", p.text), ("disabled", p.muted)],
        )
        s.configure(
            "Rules.Treeview.Heading",
            background=p.surface,
            foreground=p.muted,
            font=f["small"],
            relief="flat",
            padding=(8, 6),
            borderwidth=0,
        )
        s.map("Rules.Treeview.Heading", background=[("active", p.surface2)])
        s.layout("Rules.Treeview", [("Treeview.treearea", {"sticky": "nswe"})])

        s.configure(
            "TProgressbar",
            background=p.accent,
            troughcolor=p.surface2,
            bordercolor=p.surface2,
            lightcolor=p.accent,
            darkcolor=p.accent,
            thickness=5,
        )
        s.configure("TSeparator", background=p.border)
        s.configure("TScale", background=p.surface, troughcolor=p.surface2)
        for orient in ("Vertical", "Horizontal"):
            s.configure(
                f"{orient}.TScrollbar",
                background=p.border,
                troughcolor=p.bg,
                bordercolor=p.bg,
                arrowcolor=p.muted,
                darkcolor=p.border,
                lightcolor=p.border,
                arrowsize=12,
                relief="flat",
            )
            s.map(
                f"{orient}.TScrollbar",
                background=[("active", p.muted), ("disabled", p.surface2)],
                arrowcolor=[("active", p.text)],
            )
        # Inside a card the scrollbar trough should read as the card, not as
        # the window behind it.
        s.configure(
            "Card.Vertical.TScrollbar",
            background=p.border,
            troughcolor=p.surface2,
            bordercolor=p.surface2,
            darkcolor=p.border,
            lightcolor=p.border,
            arrowcolor=p.muted,
            arrowsize=12,
            relief="flat",
        )
        s.map(
            "Card.Vertical.TScrollbar",
            background=[("active", p.muted), ("disabled", p.surface2)],
            arrowcolor=[("active", p.text)],
        )

        self._paint_indicators()
        self._ensure_indicator_elements()

    # ------------------------------------------------------------------ #
    def toggle(self) -> str:
        mode = "light" if self.p.name == "dark" else "dark"
        self.apply(mode)
        return mode
