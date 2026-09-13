"""Small reusable ttk widgets: cards, segmented controls, tables, tooltips.

Two things in here are load-bearing beyond looking tidy:

* **Wheel routing.** Tk's own class bindings make the wheel *change the value*
  of a combobox or spinbox. Inside a scrolling page that means a flick of the
  wheel silently edits a setting. Every control built here swallows the wheel
  and scrolls the page instead (see ``wheel_guard``).
* **Tables scroll themselves.** ``Table`` owns a scrollbar that hides when it
  is not needed, and the wheel over it moves the rows, handing the gesture back
  to the page only once the rows are at the end.
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Iterable, Sequence
from tkinter import ttk
from typing import Any, NamedTuple

WHEEL_EVENTS = ("<MouseWheel>", "<Button-4>", "<Button-5>")
#: Most a single wheel event may scroll. A fast flick reports several notches
#: at once, and a forty-line jump reads as a broken window rather than speed.
WHEEL_MAX_UNITS = 5


def _event_int(value: Any) -> int:
    """Read a Tk event field that is not guaranteed to hold a number.

    Tk fills in every field for every event, so a Windows ``<MouseWheel>``
    arrives with ``num == '??'``. ``int('??')`` raises, and an exception inside
    a wheel handler costs twice: the scroll is lost *and* the ``break`` that
    stops Tk from editing the control underneath never gets returned. That one
    conversion is what stopped the whole page scrolling and let the wheel start
    changing dropdowns again - so nothing in here may raise.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def wheel_units(event: tk.Event) -> int:
    """Scroll units for a wheel event: negative up, positive down.

    Windows and macOS report ``delta`` (120 per notch, and as little as 1 per
    event on precision trackpads); X11 reports buttons 4 and 5 instead.
    """
    num = _event_int(getattr(event, "num", 0))
    if num == 4:
        return -1
    if num == 5:
        return 1
    delta = _event_int(getattr(event, "delta", 0))
    if delta == 0:
        return 0
    if abs(delta) >= 120:
        notches = int(delta / 120)
        return -max(-WHEEL_MAX_UNITS, min(WHEEL_MAX_UNITS, notches))
    return -1 if delta > 0 else 1


def _is_inside(parent: tk.Misc, widget: Any) -> bool:
    """True when ``widget`` is ``parent`` or one of its descendants.

    Compares Tk path names, so it works even when the event carries a widget
    that has already been destroyed.
    """
    if widget is None:
        return False
    top = str(parent)
    name = str(widget)
    return name == top or name.startswith(top + ".")


def find_scroll_area(widget: tk.Misc | None) -> ScrollArea | None:
    """The nearest scrolling page above ``widget``, if any."""
    node: tk.Misc | None = widget
    while node is not None:
        if isinstance(node, ScrollArea):
            return node
        node = getattr(node, "master", None)
    return None


def _inner_scroller(widget: Any, stop: tk.Misc) -> Table | None:
    """A widget between ``widget`` and ``stop`` that scrolls itself.

    The rules table has its own rows to move, so a wheel gesture over it
    belongs to the table and only falls through to the page at either end.
    """
    node: Any = widget
    while node is not None and node is not stop:
        if isinstance(node, Table):
            return node
        node = getattr(node, "master", None)
    return None


def wheel_guard(widget: tk.Misc) -> tk.Misc:
    """Stop the wheel from editing a control; scroll the page instead.

    The binding is installed on the widget itself, which runs before Tk's class
    binding, and returns ``break`` so the class binding (the one that would
    change the value) never runs.
    """

    def handler(event: tk.Event) -> str:
        # Nothing in here may escape: that "break" is the only thing between a
        # stray flick of the wheel and a silently edited setting.
        try:
            area = find_scroll_area(widget)
            if area is not None:
                area.route_wheel(event)
        except tk.TclError:
            pass
        return "break"

    for seq in WHEEL_EVENTS:
        widget.bind(seq, handler, add="+")
    return widget


class ScrollArea(ttk.Frame):
    """Vertically scrollable container that stretches its inner frame."""

    def __init__(self, master: tk.Misc, theme: Any, **kw: Any) -> None:
        super().__init__(master, **kw)
        self.theme = theme
        self.canvas = tk.Canvas(
            self, highlightthickness=0, bd=0, background=theme.p.bg, takefocus=0
        )
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
        # One global wheel binding, filtered by "is the pointer over us". The
        # old Enter/Leave pair broke as soon as the pointer moved onto a child
        # widget, which is most of the page.
        for seq in WHEEL_EVENTS:
            self.bind_all(seq, self._wheel, add="+")

    def _on_scroll(self, first: str, last: str) -> None:
        self.vbar.set(first, last)
        if float(first) <= 0.0 and float(last) >= 1.0:
            self.vbar.grid_remove()
        else:
            self.vbar.grid()

    def _on_body(self, _e: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas(self, e: tk.Event) -> None:
        self.canvas.itemconfigure(self._win, width=e.width)

    def _pointer_widget(self, e: tk.Event) -> Any:
        """The widget under the pointer, falling back to the event's own.

        Tk 8.6 on Windows delivers ``<MouseWheel>`` to the *focused* widget
        rather than the one under the pointer, so trusting ``event.widget``
        alone makes the page refuse to scroll whenever focus happens to sit in
        the log or on a footer button. X11 is already pointer-based, and its
        button events keep working through the fallback.
        """
        try:
            found = self.winfo_containing(e.x_root, e.y_root)
        except (tk.TclError, AttributeError):
            found = None
        return found if found is not None else getattr(e, "widget", None)

    def _wheel(self, e: tk.Event) -> None:
        target = self._pointer_widget(e)
        if not _is_inside(self, target):
            return  # another toplevel (a dialog) owns this gesture
        self._route(target, wheel_units(e))

    def route_wheel(self, e: tk.Event) -> None:
        """Scroll on behalf of a control that swallowed the wheel itself."""
        target = self._pointer_widget(e)
        self._route(target if _is_inside(self, target) else None, wheel_units(e))

    def _route(self, target: Any, units: int) -> None:
        if not units:
            return
        inner = _inner_scroller(target, self) if target is not None else None
        if inner is not None and inner.wheel_scroll(units):
            return
        self.scroll_by(units)

    def scroll_by(self, units: int) -> None:
        """Scroll the page, unless everything already fits."""
        if not units:
            return
        first, last = self.canvas.yview()
        if first <= 0.0 and last >= 1.0:
            return
        self.canvas.yview_scroll(units, "units")

    def restyle(self) -> None:
        self.canvas.configure(background=self.theme.p.bg)


class Card(ttk.Frame):
    """Titled surface panel. Content goes into ``.body``."""

    def __init__(
        self, master: tk.Misc, title: str, subtitle: str = "", badge: str = "", **kw: Any
    ) -> None:
        super().__init__(master, style="CardShell.TFrame", padding=(18, 15, 18, 18), **kw)
        self.columnconfigure(0, weight=1)
        head = ttk.Frame(self, style="Plain.TFrame")
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(1, weight=1)
        ttk.Label(head, text=title, style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.badge = ttk.Label(head, text=badge, style="Chip.TLabel")
        if badge:
            self.badge.grid(row=0, column=2, sticky="e")
        self.subtitle = ttk.Label(head, text=subtitle, style="Muted.TLabel")
        if subtitle:
            self.subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(3, 0))
        self.body = ttk.Frame(self, style="Plain.TFrame")
        self.body.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
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
            self.subtitle.grid(row=1, column=0, columnspan=3, sticky="w", pady=(3, 0))
        else:
            self.subtitle.grid_remove()


class Segmented(ttk.Frame):
    """Row of mutually exclusive flat buttons bound to one variable."""

    def __init__(
        self,
        master: tk.Misc,
        variable: tk.Variable,
        options: Sequence[tuple[str, Any]],
        command: Callable[[Any], None] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(master, style="Inset.TFrame", padding=2, **kw)
        self.var = variable
        self.buttons: dict[Any, ttk.Radiobutton] = {}
        for i, (label, value) in enumerate(options):
            rb = ttk.Radiobutton(
                self,
                text=label,
                value=value,
                variable=variable,
                style="Seg.Toolbutton",
                command=(lambda v=value: command(v)) if command else None,
            )
            rb.grid(row=0, column=i, sticky="ew", padx=(0 if i == 0 else 2, 0))
            self.columnconfigure(i, weight=1)
            self.buttons[value] = rb
            wheel_guard(rb)

    def set_enabled(self, value: Any, enabled: bool) -> None:
        btn = self.buttons.get(value)
        if btn is not None:
            btn.state(["!disabled"] if enabled else ["disabled"])


class Collapsible(ttk.Frame):
    """Disclosure panel with a clickable header. Content goes into ``.body``."""

    def __init__(
        self, master: tk.Misc, title: str, expanded: bool = False, subtitle: str = "", **kw: Any
    ) -> None:
        super().__init__(master, style="CardShell.TFrame", padding=(18, 13, 18, 13), **kw)
        self.columnconfigure(0, weight=1)
        self._open = tk.BooleanVar(value=expanded)
        self.head = ttk.Frame(self, style="Plain.TFrame")
        self.head.grid(row=0, column=0, sticky="ew")
        self.head.columnconfigure(1, weight=1)
        self.arrow = ttk.Label(
            self.head, text="\u25be" if expanded else "\u25b8", style="Card.TLabel", width=2
        )
        self.arrow.grid(row=0, column=0, sticky="w")
        self.title = ttk.Label(self.head, text=title, style="CardTitle.TLabel")
        self.title.grid(row=0, column=1, sticky="w")
        self.hint = ttk.Label(self.head, text=subtitle, style="Muted.TLabel")
        self.hint.grid(row=0, column=2, sticky="e")
        self.body = ttk.Frame(self, style="Plain.TFrame")
        self.body.columnconfigure(1, weight=1)
        if expanded:
            self.body.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        for w in (self.head, self.arrow, self.title, self.hint):
            w.bind("<Button-1>", lambda _e: self.toggle())
            w.configure(cursor="hand2")

    def toggle(self) -> None:
        self.set_open(not self._open.get())

    def set_open(self, value: bool) -> None:
        self._open.set(value)
        self.arrow.configure(text="\u25be" if value else "\u25b8")
        if value:
            self.body.grid(row=1, column=0, sticky="nsew", pady=(14, 0))
        else:
            self.body.grid_remove()

    def is_open(self) -> bool:
        return bool(self._open.get())

    def set_hint(self, text: str) -> None:
        self.hint.configure(text=text)


class Column(NamedTuple):
    """One table column. ``minwidth`` is what it may shrink to, never past."""

    key: str
    heading: str
    width: int
    anchor: str
    stretch: bool
    minwidth: int


def _column(spec: Sequence[Any]) -> Column:
    key, heading, width, anchor, stretch = spec[:5]
    floor = int(spec[5]) if len(spec) > 5 else max(32, min(int(width), 90))
    return Column(str(key), str(heading), int(width), str(anchor), bool(stretch), floor)


class Table(ttk.Frame):
    """A Treeview that scrolls itself and keeps its columns inside the frame.

    ``columns`` is a sequence of ``(key, heading, width, anchor, stretch)``,
    with an optional sixth ``minwidth``. The tree is exposed as ``.tree`` so
    callers keep the full Treeview API.

    A Treeview never shrinks a column to fit its widget: it keeps the widths it
    was handed and clips whatever hangs off the right edge, which is how the
    last heading ended up sliced in half against the scrollbar. So the widths
    are recomputed on every resize (:meth:`fit_columns`) to add up to exactly
    the space available.
    """

    #: A hair of slack so the rightmost cell border stays inside the treearea.
    TRIM = 2

    def __init__(
        self,
        master: tk.Misc,
        columns: Sequence[Sequence[Any]],
        height: int = 8,
        style: str = "Rules.Treeview",
        **kw: Any,
    ) -> None:
        super().__init__(master, style="Plain.TFrame", **kw)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.cols: list[Column] = [_column(spec) for spec in columns]
        self.tree = ttk.Treeview(
            self,
            columns=[c.key for c in self.cols],
            show="headings",
            height=height,
            selectmode="browse",
            style=style,
        )
        for col in self.cols:
            self.tree.heading(col.key, text=col.heading)
            # stretch is honoured by fit_columns, not by Tk: Tk's own version
            # only ever grows a column, and never below its requested width.
            self.tree.column(
                col.key,
                width=col.width,
                minwidth=col.minwidth,
                anchor=col.anchor,
                stretch=False,
            )
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.vbar = ttk.Scrollbar(
            self, orient="vertical", style="Card.Vertical.TScrollbar", command=self.tree.yview
        )
        self.vbar.grid(row=0, column=1, sticky="ns", padx=(3, 0))
        self.tree.configure(yscrollcommand=self._on_scroll)
        self.tree.bind("<Configure>", self._on_resize, add="+")
        for seq in WHEEL_EVENTS:
            self.tree.bind(seq, self._wheel, add="+")

    def _on_scroll(self, first: str, last: str) -> None:
        self.vbar.set(first, last)
        if float(first) <= 0.0 and float(last) >= 1.0:
            self.vbar.grid_remove()
        else:
            self.vbar.grid()

    def _on_resize(self, e: tk.Event) -> None:
        self.fit_columns(_event_int(getattr(e, "width", 0)))

    def fit_columns(self, width: int = 0) -> None:
        """Make the columns add up to exactly the visible width.

        Slack goes to the flexible columns; when the frame is too narrow for
        the requested widths, every column gives space back - flexible ones
        first, then the rest from the right - down to its ``minwidth``, so the
        last column stays inside the table instead of falling off the end.
        """
        avail = (width or self.tree.winfo_width()) - self.TRIM
        if avail <= 0 or not self.cols:
            return
        widths = [c.width for c in self.cols]
        slack = avail - sum(widths)
        if slack > 0:
            flexible = [i for i, c in enumerate(self.cols) if c.stretch] or [len(widths) - 1]
            share, extra = divmod(slack, len(flexible))
            for i in flexible:
                widths[i] += share
            widths[flexible[0]] += extra
        elif slack < 0:
            debt = -slack
            order = [i for i, c in enumerate(self.cols) if c.stretch]
            order += [i for i, c in enumerate(self.cols) if not c.stretch][::-1]
            for i in order:
                if debt <= 0:
                    break
                give = min(debt, widths[i] - self.cols[i].minwidth)
                widths[i] -= give
                debt -= give
        for col, w in zip(self.cols, widths, strict=True):
            self.tree.column(col.key, width=w)

    def _wheel(self, e: tk.Event) -> str:
        units = wheel_units(e)
        if not self.wheel_scroll(units):
            # Nothing left to scroll here - let the page keep moving.
            area = find_scroll_area(self)
            if area is not None:
                area.scroll_by(units)
        return "break"

    def wheel_scroll(self, units: int) -> bool:
        """Move the rows. ``False`` means "at the end, the page takes it"."""
        if not units:
            return False
        first, last = self.tree.yview()
        if (first <= 0.0 and last >= 1.0) or (units < 0 and first <= 0.0):
            return False
        if units > 0 and last >= 1.0:
            return False
        self.tree.yview_scroll(units, "units")
        return True


class Tooltip:
    """Lightweight hover tooltip."""

    def __init__(self, widget: tk.Misc, text: str, theme: Any, delay: int = 400) -> None:
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
        x = self.widget.winfo_rootx() + 10
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 7
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        tip.configure(background=p.border)
        tk.Label(
            tip,
            text=self.text,
            background=p.surface2,
            foreground=p.text,
            font=self.theme.fonts["small"],
            justify="left",
            wraplength=380,
            padx=10,
            pady=8,
        ).pack(padx=1, pady=1)
        try:
            tip.wm_attributes("-topmost", True)
        except Exception:
            pass
        self._tip = tip


def row_label(
    parent: tk.Misc,
    row: int,
    text: str,
    hint: str = "",
    theme: Any = None,
    style: str = "Field.TLabel",
) -> ttk.Label:
    """Grid a field label in column 0, with its explanation on hover."""
    lbl = ttk.Label(parent, text=text, style=style)
    lbl.grid(row=row, column=0, sticky="w", padx=(0, 14), pady=5)
    if hint and theme is not None:
        Tooltip(lbl, hint, theme)
    return lbl


def hint_label(
    parent: tk.Misc,
    row: int,
    text: str,
    style: str = "Muted.TLabel",
    column: int = 1,
    columnspan: int = 1,
) -> ttk.Label:
    lbl = ttk.Label(parent, text=text, style=style, wraplength=560, justify="left")
    lbl.grid(row=row, column=column, columnspan=columnspan, sticky="w", pady=(0, 6))
    return lbl


def int_spin(
    parent: tk.Misc,
    variable: tk.Variable,
    lo: float,
    hi: float,
    step: float = 1,
    width: int = 7,
    on_change: Callable[[], None] | None = None,
):
    """Spinbox for a numeric field.

    A fractional step gets an explicit format, so the arrows produce 0.25
    instead of 0.30000000000000004. The wheel is disarmed: scrolling past a
    spinbox scrolls the page instead of editing the number underneath.
    """
    extra: dict[str, Any] = {}
    if float(step) != int(float(step)):
        decimals = len(f"{float(step):.6f}".rstrip("0").split(".")[1]) or 2
        extra["format"] = f"%.{decimals}f"
    sp = ttk.Spinbox(
        parent,
        from_=lo,
        to=hi,
        increment=step,
        textvariable=variable,
        width=width,
        justify="right",
        **extra,
    )
    if on_change is not None:
        sp.configure(command=on_change)
        sp.bind("<FocusOut>", lambda _e: on_change(), add="+")
        sp.bind("<Return>", lambda _e: on_change(), add="+")
    wheel_guard(sp)
    return sp


def combo(
    parent: tk.Misc,
    variable: tk.Variable,
    values: Iterable[str],
    width: int = 28,
    on_change: Callable[[], None] | None = None,
) -> ttk.Combobox:
    """Read-only dropdown. The wheel scrolls the page, it never picks a value."""
    cb = ttk.Combobox(
        parent, textvariable=variable, values=list(values), width=width, state="readonly"
    )
    if on_change is not None:
        cb.bind("<<ComboboxSelected>>", lambda _e: on_change(), add="+")
    wheel_guard(cb)
    return cb


def entry(
    parent: tk.Misc,
    variable: tk.Variable,
    width: int | None = None,
    on_change: Callable[[], None] | None = None,
) -> ttk.Entry:
    """Text field that reports edits and does not react to the wheel."""
    kw: dict[str, Any] = {"textvariable": variable}
    if width is not None:
        kw["width"] = width
    e = ttk.Entry(parent, **kw)
    if on_change is not None:
        e.bind("<KeyRelease>", lambda _e: on_change(), add="+")
    wheel_guard(e)
    return e


def clear(container: tk.Misc) -> None:
    for child in list(container.winfo_children()):
        child.destroy()
