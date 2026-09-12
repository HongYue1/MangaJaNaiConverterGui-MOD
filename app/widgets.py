"""Small reusable ttk widgets: cards, segmented controls, tooltips, scroll area."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Iterable, Sequence


class ScrollArea(ttk.Frame):
    """Vertically scrollable container that stretches its inner frame."""

    def __init__(self, master: tk.Misc, theme: Any, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.theme = theme
        self.canvas = tk.Canvas(self, highlightthickness=0, bd=0,
                                background=theme.p.bg, takefocus=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self._on_scroll)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vbar.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.body = ttk.Frame(self.canvas)
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._on_body)
        self.canvas.bind("<Configure>", self._on_canvas)
        self.canvas.bind("<Enter>", lambda _e: self._bind_wheel(True))
        self.canvas.bind("<Leave>", lambda _e: self._bind_wheel(False))

    def _on_scroll(self, first: str, last: str) -> None:
        self.vbar.set(first, last)
        hidden = float(first) <= 0.0 and float(last) >= 1.0
        self.vbar.grid_remove() if hidden else self.vbar.grid()

    def _on_body(self, _e: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, e: tk.Event) -> None:
        self.canvas.itemconfigure(self._win, width=e.width)

    def _bind_wheel(self, on: bool) -> None:
        if on:
            self.canvas.bind_all("<MouseWheel>", self._wheel)
        else:
            self.canvas.unbind_all("<MouseWheel>")

    def _wheel(self, e: tk.Event) -> None:
        first, last = self.canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return
        self.canvas.yview_scroll(-1 * int(e.delta / 120) or (-1 if e.delta > 0 else 1), "units")

    def restyle(self) -> None:
        self.canvas.configure(background=self.theme.p.bg)


class Card(ttk.Frame):
    """Titled surface panel. Content goes into ``.body``."""

    def __init__(self, master: tk.Misc, title: str, subtitle: str = "",
                 badge: str = "", **kw: Any) -> None:
        super().__init__(master, style="Card.TFrame", padding=(16, 14, 16, 16), **kw)
        self.columnconfigure(0, weight=1)
        head = ttk.Frame(self, style="Card.TFrame")
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(1, weight=1)
        ttk.Label(head, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.badge = ttk.Label(head, text=badge, style="Chip.TLabel")
        if badge:
            self.badge.grid(row=0, column=2, sticky="e")
        self.subtitle = ttk.Label(head, text=subtitle, style="Muted.TLabel")
        if subtitle:
            self.subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))
        self.body = ttk.Frame(self, style="Card.TFrame")
        self.body.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        self.body.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

    def set_badge(self, text: str) -> None:
        self.badge.configure(text=text)
        if text:
            self.badge.grid(row=0, column=2, sticky="e")
        else:
            self.badge.grid_remove()

    def set_subtitle(self, text: str) -> None:
        self.subtitle.configure(text=text)
        if text:
            self.subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(2, 0))
        else:
            self.subtitle.grid_remove()


class Segmented(ttk.Frame):
    """Row of mutually exclusive flat buttons bound to one variable."""

    def __init__(self, master: tk.Misc, variable: tk.Variable,
                 options: Sequence[tuple[str, Any]],
                 command: Callable[[Any], None] | None = None, **kw: Any) -> None:
        super().__init__(master, style="Inset.TFrame", padding=2, **kw)
        self.var = variable
        self.buttons: dict[Any, ttk.Radiobutton] = {}
        for i, (label, value) in enumerate(options):
            rb = ttk.Radiobutton(self, text=label, value=value, variable=variable,
                                 style="Seg.Toolbutton",
                                 command=(lambda v=value: command(v)) if command else None)
            rb.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 2, 0))
            self.columnconfigure(i, weight=1)
            self.buttons[value] = rb

    def set_enabled(self, value: Any, enabled: bool) -> None:
        btn = self.buttons.get(value)
        if btn is not None:
            btn.state(["!disabled"] if enabled else ["disabled"])


class Collapsible(ttk.Frame):
    """Disclosure panel with a clickable header. Content goes into ``.body``."""

    def __init__(self, master: tk.Misc, title: str, expanded: bool = False,
                 subtitle: str = "", **kw: Any) -> None:
        super().__init__(master, style="Card.TFrame", padding=(16, 12, 16, 12), **kw)
        self.columnconfigure(0, weight=1)
        self._open = tk.BooleanVar(value=expanded)
        self.head = ttk.Frame(self, style="Card.TFrame")
        self.head.grid(row=0, column=0, sticky="ew")
        self.head.columnconfigure(1, weight=1)
        self.arrow = ttk.Label(self.head, text="\u25be" if expanded else "\u25b8",
                               style="Card.TLabel", width=2)
        self.arrow.grid(row=0, column=0, sticky="w")
        self.title = ttk.Label(self.head, text=title, style="CardTitle.TLabel")
        self.title.grid(row=0, column=1, sticky="w")
        self.hint = ttk.Label(self.head, text=subtitle, style="Muted.TLabel")
        self.hint.grid(row=0, column=2, sticky="e")
        self.body = ttk.Frame(self, style="Card.TFrame")
        self.body.columnconfigure(1, weight=1)
        if expanded:
            self.body.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        for w in (self.head, self.arrow, self.title, self.hint):
            w.bind("<Button-1>", lambda _e: self.toggle())
            w.configure(cursor="hand2")

    def toggle(self) -> None:
        self.set_open(not self._open.get())

    def set_open(self, value: bool) -> None:
        self._open.set(value)
        self.arrow.configure(text="\u25be" if value else "\u25b8")
        if value:
            self.body.grid(row=1, column=0, sticky="nsew", pady=(12, 0))
        else:
            self.body.grid_remove()

    def is_open(self) -> bool:
        return bool(self._open.get())

    def set_hint(self, text: str) -> None:
        self.hint.configure(text=text)


class Tooltip:
    """Lightweight hover tooltip."""

    def __init__(self, widget: tk.Misc, text: str, theme: Any, delay: int = 450) -> None:
        self.widget = widget
        self.text = text
        self.theme = theme
        self.delay = delay
        self._after: str | None = None
        self._tip: tk.Toplevel | None = None
        widget.bind("<Enter>", self._enter, add="+")
        widget.bind("<Leave>", self._leave, add="+")
        widget.bind("<ButtonPress>", self._leave, add="+")

    def set_text(self, text: str) -> None:
        self.text = text

    def _enter(self, _e: tk.Event) -> None:
        if not self.text:
            return
        self._after = self.widget.after(self.delay, self._show)

    def _leave(self, _e: tk.Event) -> None:
        if self._after:
            try:
                self.widget.after_cancel(self._after)
            except Exception:
                pass
            self._after = None
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None

    def _show(self) -> None:
        if self._tip is not None or not self.text:
            return
        p = self.theme.p
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tip.configure(background=p.border)
        tk.Label(tip, text=self.text, background=p.surface2, foreground=p.text,
                 font=self.theme.fonts["small"], justify="left", wraplength=320,
                 padx=8, pady=6).pack(padx=1, pady=1)
        try:
            tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        self._tip = tip


def row_label(parent: tk.Misc, row: int, text: str, hint: str = "",
              theme: Any = None, style: str = "Card.TLabel") -> ttk.Label:
    """Grid a right-aligned field label in column 0."""
    lbl = ttk.Label(parent, text=text, style=style)
    lbl.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=4)
    if hint and theme is not None:
        Tooltip(lbl, hint, theme)
    return lbl


def hint_label(parent: tk.Misc, row: int, text: str, style: str = "Muted.TLabel",
               column: int = 1, columnspan: int = 1) -> ttk.Label:
    lbl = ttk.Label(parent, text=text, style=style, wraplength=520, justify="left")
    lbl.grid(row=row, column=column, columnspan=columnspan, sticky="w", pady=(0, 6))
    return lbl


def int_spin(parent: tk.Misc, variable: tk.Variable, lo: float, hi: float,
             step: float = 1, width: int = 7, on_change: Callable[[], None] | None = None):
    sp = ttk.Spinbox(parent, from_=lo, to=hi, increment=step, textvariable=variable,
                     width=width, justify="right")
    if on_change is not None:
        sp.configure(command=on_change)
        sp.bind("<FocusOut>", lambda _e: on_change(), add="+")
        sp.bind("<Return>", lambda _e: on_change(), add="+")
    return sp


def combo(parent: tk.Misc, variable: tk.Variable, values: Iterable[str], width: int = 28,
          on_change: Callable[[], None] | None = None) -> ttk.Combobox:
    cb = ttk.Combobox(parent, textvariable=variable, values=list(values), width=width,
                      state="readonly")
    if on_change is not None:
        cb.bind("<<ComboboxSelected>>", lambda _e: on_change(), add="+")
    return cb


def clear(container: tk.Misc) -> None:
    for child in list(container.winfo_children()):
        child.destroy()
