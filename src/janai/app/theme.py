"""Dark/light theme for ttk. Pure stdlib, no third-party dependencies.

The look is deliberately minimal: one background, one card surface, one inset
surface for controls, a single accent, and a 1px border instead of shadows or
gradients. Everything is expressed as ttk styles so the widgets stay native -
real focus rings, real DPI scaling, real keyboard behaviour.
"""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
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


class Theme:
    """Applies a palette to a root window and remembers it for redraws."""

    def __init__(self, root: tk.Misc, mode: str = "dark") -> None:
        self.root = root
        self.style = ttk.Style(root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass
        self.family = pick_family()
        self.mono_family = pick_family(MONO_FAMILIES, "TkFixedFont")
        # One ramp used everywhere: 9 for supporting text, 10 for body and the
        # log, 11 and 16 for the two heading levels. Nothing sits at 8 any more,
        # which was too small to read on a scaled display.
        self.fonts = {
            "body": tkfont.Font(family=self.family, size=10),
            "bold": tkfont.Font(family=self.family, size=10, weight="bold"),
            "small": tkfont.Font(family=self.family, size=9),
            "tiny": tkfont.Font(family=self.family, size=9),
            "title": tkfont.Font(family=self.family, size=16, weight="bold"),
            "card": tkfont.Font(family=self.family, size=11, weight="bold"),
            "mono": tkfont.Font(family=self.mono_family, size=10),
            "mono_bold": tkfont.Font(family=self.mono_family, size=10, weight="bold"),
        }
        # Plain tk widgets (the log text area, tooltips, menus) read the named
        # fonts rather than a ttk style, so point those at the same faces.
        for named, key in (
            ("TkDefaultFont", "body"),
            ("TkTextFont", "body"),
            ("TkMenuFont", "body"),
            ("TkHeadingFont", "bold"),
            ("TkTooltipFont", "small"),
            ("TkFixedFont", "mono"),
        ):
            try:
                target = tkfont.nametofont(named, root=root)
            except tk.TclError:
                continue
            src = self.fonts[key]
            target.configure(
                family=src.cget("family"), size=src.cget("size"), weight=src.cget("weight")
            )
        self.p = PALETTES.get(mode, DARK)
        self.apply(mode)

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

        s.configure(
            "TCheckbutton",
            background=p.surface,
            foreground=p.text,
            indicatorcolor=p.surface2,
            bordercolor=p.border,
            padding=(0, 4),
        )
        s.map(
            "TCheckbutton",
            background=[("active", p.surface)],
            indicatorcolor=[("selected", p.accent), ("pressed", p.accent_hi)],
            foreground=[("disabled", p.muted)],
        )
        s.configure(
            "Inset.TCheckbutton", background=p.surface2, foreground=p.text, indicatorcolor=p.surface
        )
        s.map(
            "Inset.TCheckbutton",
            background=[("active", p.surface2)],
            indicatorcolor=[("selected", p.accent)],
        )
        s.configure(
            "Bg.TCheckbutton", background=p.bg, foreground=p.text, indicatorcolor=p.surface2
        )
        s.map(
            "Bg.TCheckbutton",
            background=[("active", p.bg)],
            indicatorcolor=[("selected", p.accent)],
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
            rowheight=26,
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

    # ------------------------------------------------------------------ #
    def toggle(self) -> str:
        mode = "light" if self.p.name == "dark" else "dark"
        self.apply(mode)
        return mode
