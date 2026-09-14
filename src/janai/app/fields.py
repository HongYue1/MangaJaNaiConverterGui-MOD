"""The option ladders the controls offer, and the label/value conversions.

A combo box shows text while the settings file stores a value, so every ladder
needs both halves and one pair of conversions between them. They live here
rather than in ``window.py`` because three separate places read them - the
builders that populate the controls, the preset reload that re-selects an
entry, and the job builder that hands the value to the worker - and a module
that knows only about option tuples can be imported by all three without
pulling the window in.
"""

from __future__ import annotations

from typing import Any

from janai.core.formats import Opt

# Fixed sizes in 128px steps up to 1024 and 256px steps above it: the useful
# range is wide, and the difference between neighbouring sizes is worth having
# because the fastest tile is usually the largest one that still fits.
TILE_CHOICES: tuple[tuple[str, str], ...] = (
    ("Auto (adaptive)", "auto"),
    ("Maximum", "maximum"),
    ("No tiling (fails if it will not fit)", "none"),
    ("128 px", "128"),
    ("192 px", "192"),
    ("256 px", "256"),
    ("384 px", "384"),
    ("512 px", "512"),
    ("640 px", "640"),
    ("768 px", "768"),
    ("896 px", "896"),
    ("1024 px", "1024"),
    ("1152 px", "1152"),
    ("1280 px", "1280"),
    ("1536 px", "1536"),
    ("1792 px", "1792"),
    ("2048 px", "2048"),
    ("2560 px", "2560"),
    ("3072 px", "3072"),
)

MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Scale", "scale"),
    ("Width", "width"),
    ("Height", "height"),
    ("Fit", "fit"),
)


def choice_label(opt: Opt, value: Any) -> str:
    for text, val in opt.choices:
        if val == value:
            return text
    return opt.choices[0][0] if opt.choices else str(value)


def choice_value(opt: Opt, text: str) -> Any:
    for lbl, val in opt.choices:
        if lbl == text:
            return val
    return opt.default


def tile_label(value: str) -> str:
    for text, val in TILE_CHOICES:
        if val == value:
            return text
    return TILE_CHOICES[0][0]


def tile_value(text: str) -> str:
    for lbl, val in TILE_CHOICES:
        if lbl == text:
            return val
    return "auto"
