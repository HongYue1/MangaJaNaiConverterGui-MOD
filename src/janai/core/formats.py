"""Output format catalogue, shared by the GUI and the worker.

Every option here maps onto a real libvips save argument (``vips`` field), so
anything the GUI exposes is something the encoder actually honours. Options
with ``vips=None`` are control switches interpreted by the worker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Opt:
    key: str
    label: str
    kind: str  # "int" | "float" | "bool" | "choice"
    default: Any
    vips: str | None = None
    lo: float = 0.0
    hi: float = 100.0
    step: float = 1.0
    choices: tuple[tuple[str, Any], ...] = ()
    hint: str = ""
    advanced: bool = False
    # `object`, not `Any`: a condition value is only ever compared, never used,
    # and one `Any` operand is enough to make `==` leak Any out of `_matches`.
    only_if: tuple[str, object] | None = None
    not_if: tuple[str, object] | None = None


@dataclass(frozen=True)
class Fmt:
    id: str
    label: str
    ext: str
    suffix: str
    opts: tuple[Opt, ...]
    hint: str = ""
    probe: str = ""  # libvips operation that must exist for this format


_PNG = (
    Opt(
        "compression",
        "Compression",
        "int",
        6,
        vips="compression",
        lo=0,
        hi=9,
        hint="zlib level. 6 is a good balance, 9 is slow for ~1% smaller files.",
    ),
    Opt(
        "effort",
        "Palette effort",
        "int",
        7,
        vips="effort",
        lo=1,
        hi=10,
        advanced=True,
        only_if=("palette", True),
        hint="Quantisation effort when writing a palette PNG.",
    ),
    Opt(
        "interlace",
        "Interlaced (Adam7)",
        "bool",
        False,
        vips="interlace",
        advanced=True,
        hint="Progressive display, slightly larger files.",
    ),
    Opt(
        "palette",
        "Indexed palette",
        "bool",
        False,
        vips="palette",
        hint="Quantise to a palette. Much smaller for flat line art, lossy.",
    ),
    Opt("Q", "Palette quality", "int", 100, vips="Q", lo=1, hi=100, only_if=("palette", True)),
    Opt(
        "bitdepth",
        "Palette bit depth",
        "choice",
        8,
        vips="bitdepth",
        advanced=True,
        only_if=("palette", True),
        choices=(("8 bit (256)", 8), ("4 bit (16)", 4), ("2 bit (4)", 2), ("1 bit (2)", 1)),
    ),
    Opt(
        "dither",
        "Dither",
        "float",
        1.0,
        vips="dither",
        lo=0,
        hi=1,
        step=0.1,
        advanced=True,
        only_if=("palette", True),
    ),
)

_JPEG = (
    Opt("Q", "Quality", "int", 92, vips="Q", lo=1, hi=100),
    Opt(
        "subsample_mode",
        "Chroma subsampling",
        "choice",
        "off",
        vips="subsample_mode",
        choices=(
            ("4:4:4 (no subsampling)", "off"),
            ("Auto (4:2:0 below Q90)", "auto"),
            ("4:2:0 (always)", "on"),
        ),
        hint="4:4:4 keeps coloured text and screentones clean.",
    ),
    Opt(
        "optimize_coding",
        "Optimize Huffman tables",
        "bool",
        True,
        vips="optimize_coding",
        hint="Smaller files, no quality change.",
    ),
    Opt("interlace", "Progressive", "bool", True, vips="interlace"),
    Opt(
        "trellis_quant",
        "Trellis quantisation",
        "bool",
        False,
        vips="trellis_quant",
        advanced=True,
        hint="mozjpeg feature: smaller files, slower encode.",
    ),
    Opt(
        "overshoot_deringing",
        "Overshoot deringing",
        "bool",
        False,
        vips="overshoot_deringing",
        advanced=True,
        hint="Reduces ringing around hard black edges.",
    ),
    Opt(
        "optimize_scans",
        "Optimize progressive scans",
        "bool",
        False,
        vips="optimize_scans",
        advanced=True,
        only_if=("interlace", True),
    ),
    Opt(
        "quant_table",
        "Quant table",
        "choice",
        3,
        vips="quant_table",
        advanced=True,
        choices=(
            ("0 - JPEG Annex K", 0),
            ("1 - Flat", 1),
            ("2 - Custom MSSIM", 2),
            ("3 - ImageMagick", 3),
            ("4 - Custom PSNR-HVS", 4),
            ("5 - Klein", 5),
            ("6 - Watson", 6),
            ("7 - Ahumada", 7),
            ("8 - Peterson", 8),
        ),
    ),
)

_WEBP = (
    Opt("Q", "Quality", "int", 90, vips="Q", lo=1, hi=100, not_if=("lossless", True)),
    Opt("lossless", "Lossless", "bool", False, vips="lossless"),
    Opt(
        "effort",
        "Effort",
        "int",
        4,
        vips="effort",
        lo=0,
        hi=6,
        hint="0 = fastest, 6 = smallest. Encode time grows quickly.",
    ),
    Opt(
        "preset",
        "Preset",
        "choice",
        "drawing",
        vips="preset",
        choices=(
            ("Default", "default"),
            ("Picture", "picture"),
            ("Photo", "photo"),
            ("Drawing", "drawing"),
            ("Icon", "icon"),
            ("Text", "text"),
        ),
        hint="Drawing/Text suit manga and line art.",
    ),
    Opt(
        "smart_subsample",
        "Smart subsampling",
        "bool",
        True,
        vips="smart_subsample",
        not_if=("lossless", True),
    ),
    Opt(
        "near_lossless",
        "Near-lossless",
        "bool",
        False,
        vips="near_lossless",
        advanced=True,
        only_if=("lossless", True),
        hint="Lossless container, Q used as pre-processing strength.",
    ),
    Opt("alpha_q", "Alpha quality", "int", 100, vips="alpha_q", lo=0, hi=100, advanced=True),
    Opt(
        "min_size",
        "Minimise size",
        "bool",
        False,
        vips="min_size",
        advanced=True,
        hint="Slow exhaustive search for the smallest file.",
    ),
)

_AVIF = (
    Opt(
        "Q",
        "Quality",
        "int",
        63,
        vips="Q",
        lo=1,
        hi=100,
        not_if=("lossless", True),
        hint="AVIF quality is not comparable to JPEG: 55-70 is usually visually clean.",
    ),
    Opt("lossless", "Lossless", "bool", False, vips="lossless"),
    Opt(
        "effort",
        "Effort",
        "int",
        4,
        vips="effort",
        lo=0,
        hi=9,
        hint="0 = fastest, 9 = smallest. AVIF encoding is CPU heavy.",
    ),
    Opt(
        "bitdepth",
        "Bit depth",
        "choice",
        8,
        vips="bitdepth",
        choices=(("8 bit", 8), ("10 bit", 10), ("12 bit", 12)),
        advanced=True,
    ),
    Opt(
        "subsample_mode",
        "Chroma subsampling",
        "choice",
        "off",
        vips="subsample_mode",
        choices=(("4:4:4 (no subsampling)", "off"), ("Auto", "auto"), ("4:2:0", "on")),
        advanced=True,
    ),
    Opt(
        "encoder",
        "Encoder",
        "choice",
        "auto",
        vips="encoder",
        advanced=True,
        choices=(("Auto", "auto"), ("aom", "aom"), ("rav1e", "rav1e"), ("svt", "svt")),
    ),
)

_JXL = (
    Opt(
        "rate_mode",
        "Rate control",
        "choice",
        "quality",
        vips=None,
        choices=(("Quality (0-100)", "quality"), ("Butteraugli distance", "distance")),
        not_if=("lossless", True),
    ),
    Opt(
        "Q",
        "Quality",
        "int",
        90,
        vips="Q",
        lo=1,
        hi=100,
        only_if=("rate_mode", "quality"),
        not_if=("lossless", True),
    ),
    Opt(
        "distance",
        "Distance",
        "float",
        1.0,
        vips="distance",
        lo=0.1,
        hi=15.0,
        step=0.1,
        only_if=("rate_mode", "distance"),
        not_if=("lossless", True),
        hint="Butteraugli target. 1.0 is visually lossless, lower is better quality.",
    ),
    Opt(
        "lossless",
        "Lossless",
        "bool",
        False,
        vips="lossless",
        hint="Modular lossless mode. Excellent on grayscale manga, larger files.",
    ),
    Opt(
        "effort",
        "Effort",
        "int",
        7,
        vips="effort",
        lo=3,
        hi=9,
        hint="3 = fast, 7 = default, 9 = slowest and smallest.",
    ),
    Opt(
        "tier",
        "Decode speed tier",
        "int",
        0,
        vips="tier",
        lo=0,
        hi=4,
        advanced=True,
        hint="Higher decodes faster at the cost of density.",
    ),
)

FORMATS: dict[str, Fmt] = {
    "png": Fmt(
        "png",
        "PNG",
        ".png",
        ".png",
        _PNG,
        probe="pngsave",
        hint="Lossless, universally supported, biggest files.",
    ),
    "jpeg": Fmt(
        "jpeg",
        "JPEG",
        ".jpg",
        ".jpg",
        _JPEG,
        probe="jpegsave",
        hint="Lossy 8-bit, fastest to encode and read anywhere.",
    ),
    "webp": Fmt(
        "webp",
        "WEBP",
        ".webp",
        ".webp",
        _WEBP,
        probe="webpsave",
        hint="Good size/quality, wide reader support.",
    ),
    "avif": Fmt(
        "avif",
        "AVIF",
        ".avif",
        ".avif",
        _AVIF,
        probe="heifsave",
        hint="Smallest lossy files, slow encode, newer readers only.",
    ),
    "jxl": Fmt(
        "jxl",
        "JPEG XL",
        ".jxl",
        ".jxl",
        _JXL,
        probe="jxlsave",
        hint="Best quality per byte and true lossless, limited reader support.",
    ),
}

FORMAT_IDS = tuple(FORMATS)


# --------------------------------------------------------------------------- #
# output containers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Container:
    id: str
    label: str
    ext: str
    hint: str = ""


#: How encoded pages are delivered. The page encoder (``Fmt``) is chosen
#: independently, so a folder of JPEGs can come out as a .cbz full of JPEG XL.
CONTAINERS: dict[str, Container] = {
    "files": Container(
        "files",
        "Loose files",
        "",
        "One output image per input image. Comic archives are still re-packed as .cbz.",
    ),
    "cbz": Container(
        "cbz",
        "CBZ per folder",
        ".cbz",
        "Every folder of images becomes one .cbz, so one folder per chapter gives "
        "one .cbz per chapter. Archives stay archives.",
    ),
    "cbz_single": Container(
        "cbz_single",
        "One CBZ",
        ".cbz",
        "Everything found in the input is packed into a single .cbz.",
    ),
}

CONTAINER_IDS = tuple(CONTAINERS)


def container(cid: str) -> Container:
    return CONTAINERS.get(cid) or CONTAINERS["files"]


def packs_archive(cid: str) -> bool:
    """True when loose input images are packed into an archive on the way out."""
    return container(cid).id in ("cbz", "cbz_single")


def fmt(fid: str) -> Fmt:
    return FORMATS[fid]


def defaults(fid: str) -> dict[str, Any]:
    return {o.key: o.default for o in FORMATS[fid].opts}


def all_defaults() -> dict[str, dict[str, Any]]:
    return {fid: defaults(fid) for fid in FORMATS}


def _matches(cond: tuple[str, object], values: dict[str, Any]) -> bool:
    key, want = cond
    # Both operands must be `object`, never `Any`: these are opaque option
    # values compared only by `==`/`in`, and a single `Any` operand makes `==`
    # return Any, which then leaks out of this `-> bool` function.
    cur: object = values.get(key)
    if isinstance(want, (tuple, list, set)):
        return cur in want
    return cur == want


def is_active(opt: Opt, values: dict[str, Any]) -> bool:
    """False when another option makes this one irrelevant."""
    if opt.only_if is not None and not _matches(opt.only_if, values):
        return False
    return not (opt.not_if is not None and _matches(opt.not_if, values))


def merged(fid: str, values: dict[str, Any] | None) -> dict[str, Any]:
    out = defaults(fid)
    for k, v in (values or {}).items():
        if k in out:
            out[k] = v
    return out


def save_kwargs(fid: str, values: dict[str, Any] | None) -> dict[str, Any]:
    """libvips save arguments for the given format and user values."""
    vals = merged(fid, values)
    kwargs: dict[str, Any] = {}
    for o in FORMATS[fid].opts:
        if o.vips is None or not is_active(o, vals):
            continue
        v = vals[o.key]
        if o.kind == "int":
            v = int(v)
        elif o.kind == "float":
            v = float(v)
        elif o.kind == "bool":
            v = bool(v)
        kwargs[o.vips] = v
    return kwargs


def summary(fid: str, values: dict[str, Any] | None) -> str:
    """Short one-line description of the current encoder settings."""
    vals = merged(fid, values)
    bits: list[str] = []
    if vals.get("lossless"):
        bits.append("lossless")
    elif fid == "jxl" and vals.get("rate_mode") == "distance":
        bits.append(f"d{vals['distance']:g}")
    elif "Q" in vals and is_active(FORMATS[fid].opts[0], vals):
        bits.append(f"Q{int(vals['Q'])}")
    if "effort" in vals:
        bits.append(f"effort {int(vals['effort'])}")
    if fid == "png":
        bits = [f"zlib {int(vals['compression'])}"] + (["palette"] if vals.get("palette") else [])
    return f"{FORMATS[fid].label} · " + ", ".join(b for b in bits if b)
