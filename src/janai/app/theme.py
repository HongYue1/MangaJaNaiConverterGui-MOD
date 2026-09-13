"""Dark/light theme for ttk. Pure stdlib, no third-party dependencies."""

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
    bg="#15171c",
    surface="#1c1f26",
    surface2="#23272f",
    border="#2f3542",
    text="#e7eaf0",
    muted="#98a1b0",
    accent="#4f7cff",
    accent_hi="#6a92ff",
    accent_text="#ffffff",
    ok="#3fb950",
    warn="#d9a129",
    err="#f2544b",
    sel="#2b3242",
)

LIGHT = Palette(
    name="light",
    bg="#f3f4f7",
    surface="#ffffff",
    surface2="#eceef3",
    border="#d6dae2",
    text="#1a1d24",
    muted="#5d6675",
    accent="#2f6bff",
    accent_hi="#1f5af0",
    accent_text="#ffffff",
    ok="#1a7f37",
    warn="#9a6700",
    err="#cf222e",
    sel="#dbe5ff",
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
        # log, 11 and 15 for the two heading levels. Nothing sits at 8 any more,
        # which was too small to read on a scaled display.
        self.fonts = {
            "body": tkfont.Font(family=self.family, size=10),
            "bold": tkfont.Font(family=self.family, size=10, weight="bold"),
            "small": tkfont.Font(family=self.family, size=9),
            "tiny": tkfont.Font(family=self.family, size=9),
            "title": tkfont.Font(family=self.family, size=15, weight="bold"),
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
        s.configure("Card.TFrame", background=p.surface, relief="flat")
        s.configure("Inset.TFrame", background=p.surface2)
        s.configure("Drop.TFrame", background=p.surface2)
        s.configure("DropActive.TFrame", background=p.sel)

        s.configure("TLabel", background=p.bg, foreground=p.text)
        s.configure("Card.TLabel", background=p.surface, foreground=p.text)
        s.configure("CardTitle.TLabel", background=p.surface, foreground=p.text, font=f["card"])
        s.configure("Muted.TLabel", background=p.surface, foreground=p.muted, font=f["small"])
        s.configure("MutedBg.TLabel", background=p.bg, foreground=p.muted, font=f["small"])
        s.configure("Inset.TLabel", background=p.surface2, foreground=p.text)
        s.configure("InsetMuted.TLabel", background=p.surface2, foreground=p.muted, font=f["small"])
        s.configure("Title.TLabel", background=p.bg, foreground=p.text, font=f["title"])
        s.configure("Ok.TLabel", background=p.surface, foreground=p.ok, font=f["small"])
        s.configure("Warn.TLabel", background=p.surface, foreground=p.warn, font=f["small"])
        s.configure("Err.TLabel", background=p.surface, foreground=p.err, font=f["small"])
        s.configure(
            "Chip.TLabel", background=p.surface2, foreground=p.muted, font=f["tiny"], padding=(6, 2)
        )

        s.configure(
            "TButton",
            background=p.surface2,
            foreground=p.text,
            padding=(12, 6),
            borderwidth=0,
            relief="flat",
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
            padding=(18, 8),
            font=f["bold"],
        )
        s.map(
            "Accent.TButton",
            background=[("disabled", p.surface2), ("pressed", p.accent), ("active", p.accent_hi)],
            foreground=[("disabled", p.muted)],
        )
        s.configure("Ghost.TButton", background=p.surface, foreground=p.muted, padding=(8, 4))
        s.map("Ghost.TButton", background=[("active", p.surface2)], foreground=[("active", p.text)])
        s.configure(
            "Link.TButton", background=p.bg, foreground=p.accent, padding=(4, 2), font=f["small"]
        )
        s.map("Link.TButton", background=[("active", p.bg)], foreground=[("active", p.accent_hi)])

        # segmented control: radiobuttons drawn as flat toggle buttons
        s.configure(
            "Seg.Toolbutton",
            background=p.surface2,
            foreground=p.muted,
            padding=(12, 5),
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
            padding=(0, 3),
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
            "TEntry",
            fieldbackground=p.surface2,
            foreground=p.text,
            bordercolor=p.border,
            lightcolor=p.surface2,
            darkcolor=p.surface2,
            insertcolor=p.text,
            padding=(6, 5),
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
            insertcolor=p.text,
            lightcolor=p.surface2,
            darkcolor=p.surface2,
            padding=(6, 4),
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
            lightcolor=p.surface2,
            darkcolor=p.surface2,
            padding=(6, 4),
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
            rowheight=24,
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
            padding=(6, 4),
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
            thickness=6,
        )
        s.configure("TSeparator", background=p.border)
        s.configure("TScale", background=p.surface, troughcolor=p.surface2)
        s.configure(
            "Vertical.TScrollbar",
            background=p.surface2,
            troughcolor=p.bg,
            bordercolor=p.bg,
            arrowcolor=p.muted,
            darkcolor=p.surface2,
            lightcolor=p.surface2,
        )
        s.map("Vertical.TScrollbar", background=[("active", p.border)])
        s.configure(
            "Horizontal.TScrollbar",
            background=p.surface2,
            troughcolor=p.bg,
            bordercolor=p.bg,
            arrowcolor=p.muted,
            darkcolor=p.surface2,
            lightcolor=p.surface2,
        )

    # ------------------------------------------------------------------ #
    def toggle(self) -> str:
        mode = "light" if self.p.name == "dark" else "dark"
        self.apply(mode)
        return mode
