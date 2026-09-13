"""Interface geometry harness: page scrolling, table columns, indicator styling.

These three things are invisible to ``compileall`` and to the smoke test, and
all three have regressed at least once:

* the wheel must scroll the page from anywhere on it, including from on top of
  a dropdown or a number field, while the rules table scrolls its own rows
  first and hands the page the gesture once it reaches either end;
* every rules column must be fully inside the table, with the scrollbar
  outside the cells rather than on top of the last one;
* checkbuttons must draw a real indicator rather than a missing-glyph box, and
  the fonts must come straight from the ramp - Tk already multiplies every
  point size by the display's scaling, so a second factor in the theme made
  the text roughly twice too big.

Run it with the bundled interpreter:

    backend\\python\\python.exe scripts\\uicheck.py

It needs a display; it never touches the GPU, the backend or your settings.
"""

from __future__ import annotations

import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("JANAI_SMOKE", "1")

from janai.app.theme import BASE_SIZES, Theme
from janai.app.ui import App
from janai.app.widgets import WHEEL_EVENTS, ScrollArea, Table

failures: list[str] = []
notes: list[str] = []


def check(ok: bool, label: str, detail: str = "") -> None:
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        failures.append(label)


def wheel(widget: tk.Misc, delta: int = -120) -> None:
    """Send one wheel notch to ``widget`` the way Windows does."""
    widget.event_generate("<MouseWheel>", delta=delta, x=5, y=5)
    widget.update_idletasks()


def find(root: tk.Misc, cls: type) -> list[tk.Misc]:
    out: list[tk.Misc] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if isinstance(node, cls):
            out.append(node)
        stack.extend(node.winfo_children())
    return out


def main() -> int:
    root = tk.Tk()
    root.withdraw()
    app = App(root, ROOT)
    root.geometry("1040x700")
    root.deiconify()
    root.update()
    root.update_idletasks()

    print("\nscroll area")
    area = app.scroll
    canvas = area.canvas
    print(
        f"  canvas {canvas.winfo_width()}x{canvas.winfo_height()} "
        f"body req {area.body.winfo_reqheight()} "
        f"region {canvas.cget('scrollregion')!r} yview {canvas.yview()}"
    )
    first, last = canvas.yview()
    check(last - first < 0.999, "page has something to scroll", f"yview {first:.3f}-{last:.3f}")

    # The wheel has to work from every kind of widget on the page, not just
    # from the bare canvas: in practice the pointer is always over a child.
    targets: list[tuple[str, tk.Misc]] = [("canvas", canvas)]
    for label, cls in (("label", ttk.Label), ("combobox", ttk.Combobox), ("spinbox", ttk.Spinbox)):
        found = [w for w in find(area, cls) if w.winfo_ismapped()]
        if found:
            targets.append((label, found[0]))
    for label, widget in targets:
        canvas.yview_moveto(0.0)
        root.update_idletasks()
        before = canvas.yview()[0]
        wheel(widget)
        after = canvas.yview()[0]
        check(
            after > before, f"wheel over {label} scrolls the page", f"{before:.3f} -> {after:.3f}"
        )

    # The rules table has rows of its own, so it takes the gesture first and
    # only hands it to the page once it runs out - assert that nested
    # behaviour instead of demanding the page move from on top of it.
    tables = find(area, Table)
    if tables:
        rows_tree = tables[0].tree
        canvas.yview_moveto(0.0)
        rows_tree.yview_moveto(0.0)
        root.update_idletasks()
        span = rows_tree.yview()
        print(f"  table rows yview {span[0]:.3f}-{span[1]:.3f}")
        page_before, rows_before = canvas.yview()[0], rows_tree.yview()[0]
        wheel(rows_tree)
        page_after, rows_after = canvas.yview()[0], rows_tree.yview()[0]
        check(
            rows_after > rows_before or page_after > page_before,
            "wheel over the table moves something",
            f"rows {rows_before:.3f}->{rows_after:.3f} page {page_before:.3f}->{page_after:.3f}",
        )
        rows_tree.yview_moveto(1.0)
        root.update_idletasks()
        page_before = canvas.yview()[0]
        wheel(rows_tree)
        check(
            canvas.yview()[0] > page_before,
            "table hands the page the wheel at its end",
            f"{page_before:.3f} -> {canvas.yview()[0]:.3f}",
        )

    canvas.yview_moveto(1.0)
    root.update_idletasks()
    bottom = canvas.yview()[0]
    wheel(canvas, delta=120)
    check(canvas.yview()[0] < bottom, "wheel scrolls back up")

    # A dropdown must not change value when the wheel passes over it.
    combos = [w for w in find(area, ttk.Combobox) if w.winfo_ismapped()]
    if combos:
        cb = combos[0]
        values = list(cb.cget("values") or [])
        if len(values) > 1:
            cb.set(values[0])
            wheel(cb)
            check(cb.get() == values[0], "wheel does not change a dropdown", cb.get())

    print("\nrules table")
    table = app.rules_table
    tree = table.tree
    inner = tree.winfo_width()
    cols = list(tree.cget("columns"))
    widths = [int(tree.column(c, "width")) for c in cols]
    total = sum(widths)
    print(f"  tree width {inner} · columns {dict(zip(cols, widths, strict=True))} · total {total}")
    check(inner > 1, "table has been laid out", str(inner))
    check(total <= inner, "columns fit inside the table", f"{total} <= {inner}")
    check(inner - total <= 2, "columns fill the table", f"gap {inner - total}")

    # bbox of the last column tells us whether it is really on screen.
    if tree.get_children():
        item = tree.get_children()[0]
        box = tree.bbox(item, cols[-1])
        if box:
            x, _y, w, _h = box
            check(x + w <= inner, "last column ends inside the table", f"{x + w} <= {inner}")
        heading = tree.heading(cols[-1], "text")
        check(heading == "Auto levels", "last heading is intact", heading)
    else:
        notes.append("no rules installed, skipped the last-column bbox check")

    check(
        str(table.vbar.winfo_manager()) == "grid" or not table.vbar.winfo_ismapped(),
        "table scrollbar is managed",
    )
    if table.vbar.winfo_ismapped():
        check(
            table.vbar.winfo_x() >= tree.winfo_x() + inner,
            "scrollbar sits outside the cells",
            f"bar x {table.vbar.winfo_x()} vs tree end {tree.winfo_x() + inner}",
        )

    print("\nstyling")
    theme: Theme = app.theme
    style = theme.style
    layout = str(style.layout("TCheckbutton"))
    print(f"  checkbutton layout {layout}")
    check("indicator" in layout, "checkbutton has an indicator element")
    scaling = float(root.tk.call("tk", "scaling"))
    body = theme.fonts["body"]
    print(
        f"  tk scaling {scaling:.3f} · body {body.cget('family')!r} {body.cget('size')}pt "
        f"= {body.metrics('linespace')}px · rowheight {style.lookup('Rules.Treeview', 'rowheight')}"
    )
    check(
        body.cget("size") == BASE_SIZES["body"],
        "body font is the ramp, not scaled twice",
        f"{body.cget('size')}pt vs ramp {BASE_SIZES['body']}pt",
    )
    check(
        body.metrics("linespace") >= 15, "body text is readable", f"{body.metrics('linespace')}px"
    )
    rowheight = int(style.lookup("Rules.Treeview", "rowheight") or 0)
    check(
        rowheight >= body.metrics("linespace") + 6,
        "table rows clear the font",
        f"{rowheight} vs {body.metrics('linespace')}",
    )

    # Every wheel-guarded control should still be reachable by the page.
    guarded = 0
    for cls in (ttk.Combobox, ttk.Spinbox, ttk.Entry):
        for widget in find(area, cls):
            binds = [seq for seq in WHEEL_EVENTS if widget.bind(seq)]
            if binds:
                guarded += 1
    print(f"  wheel-guarded controls: {guarded}")
    check(guarded > 0, "controls carry a wheel guard")
    check(isinstance(area, ScrollArea), "page is a scroll area")

    root.destroy()

    print()
    for note in notes:
        print(f"  note: {note}")
    if failures:
        print(f"FAILED: {len(failures)} check(s): " + ", ".join(failures))
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
