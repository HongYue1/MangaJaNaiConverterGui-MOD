"""Display presets for the "Fit" target mode.

The original fork shipped a device list (``DisplayDevice`` plus a portrait /
landscape switch) so you could upscale a chapter to exactly what an e-reader
or tablet shows. This is the same idea in one small table: pick a device, get
its panel size, flip the orientation if you read in landscape.

Sizes are the panel resolution in portrait orientation (width x height).
"""

from __future__ import annotations

from typing import NamedTuple


class Display(NamedTuple):
    id: str
    label: str
    width: int
    height: int


CUSTOM = "custom"

#: Portrait-orientation panel sizes. Keep the list short and recognisable;
#: anything else is what "Custom" is for.
DISPLAYS: tuple[Display, ...] = (
    Display("kindle_pw", "Kindle Paperwhite / Signature", 1236, 1648),
    Display("kindle_oasis", "Kindle Oasis", 1264, 1680),
    Display("kindle_scribe", "Kindle Scribe", 1860, 2480),
    Display("kobo_clara", "Kobo Clara HD / Colour", 1072, 1448),
    Display("kobo_libra", "Kobo Libra 2 / Sage", 1264, 1680),
    Display("kobo_elipsa", "Kobo Elipsa 2E", 1404, 1872),
    Display("remarkable2", "reMarkable 2", 1404, 1872),
    Display("boox_note", "Boox Note Air / Tab Ultra", 1404, 1872),
    Display("ipad_109", "iPad 10.9\u2033", 1640, 2360),
    Display("ipad_pro_11", "iPad Pro 11\u2033", 1668, 2388),
    Display("ipad_pro_13", "iPad Pro 13\u2033", 2064, 2752),
    Display("tablet_1440", "Android tablet 1600\u00d72560", 1600, 2560),
    Display("screen_1080", "Monitor 1080p", 1080, 1920),
    Display("screen_1440", "Monitor 1440p", 1440, 2560),
    Display("screen_4k", "Monitor 4K", 2160, 3840),
)

BY_ID = {d.id: d for d in DISPLAYS}


def size(display_id: str, portrait: bool = True) -> tuple[int, int] | None:
    """Target (width, height) for a preset, or None for custom/unknown."""
    d = BY_ID.get(display_id)
    if d is None:
        return None
    return (d.width, d.height) if portrait else (d.height, d.width)


def match(width: int, height: int) -> str:
    """Reverse lookup: which preset (if any) these dimensions belong to."""
    for d in DISPLAYS:
        if (width, height) in ((d.width, d.height), (d.height, d.width)):
            return d.id
    return CUSTOM


def labels() -> list[str]:
    return ["Custom"] + [d.label for d in DISPLAYS]


def id_for_label(label: str) -> str:
    for d in DISPLAYS:
        if d.label == label:
            return d.id
    return CUSTOM


def label_for_id(display_id: str) -> str:
    d = BY_ID.get(display_id)
    return d.label if d else "Custom"
