"""Rules-based model selection.

The table is the only thing that chooses a model. Every page is matched
against an ordered list of rules; a rule is a set of conditions on the left
and the outcome on the right::

    on  when            page size           model                       levels
    *   grayscale 2x    h 1551-1760         2x_MangaJaNai_1600p_...     on
    *   grayscale 2x    h 1761-1984         2x_MangaJaNai_1920p_...     on
    *   colour 4x       any                 4x_IllustrationJaNai_...    -
    *   colour          any                 2x_IllustrationJaNai_...    -

Matching rules:

* A rule that names a size always beats a rule that says "any", wherever the
  two sit in the list - that is the "specific dimensions take priority"
  behaviour. An exact size beats a range, and a range beats "any".
* After specificity, the list order decides, so moving a rule up still means
  something for equally specific rules.
* A rule can be switched off without deleting it (``enabled``); the table shows
  that state in its first column.
* ``levels`` only applies to grayscale pages. ``None`` means "use the Auto
  levels checkbox", ``True``/``False`` override it for the pages this rule
  claims.

``model = auto`` is no longer a choice a user can make: the shipped table
names real files, and :func:`materialise` upgrades any ``auto`` left in an
older settings file to the file it would have resolved to. If a rule still
carries it (a hand-edited JSON, a model that was uninstalled), the worker
falls back to its built-in picker and the table flags the row.

The module is pure standard library so the interface and the worker can both
import it, and matching is a handful of integer comparisons over a compiled
list, with a per-page-size cache on top: negligible next to a 20-second page.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

# --------------------------------------------------------------------------- #
# vocabulary
# --------------------------------------------------------------------------- #
ANY = "any"
AUTO = "auto"
COLOUR = "colour"
GRAYSCALE = "grayscale"
KINDS: tuple[str, ...] = (ANY, COLOUR, GRAYSCALE)

# Upstream's default workflow shipped one MangaJaNai chain per page-height
# band, in 2x and 4x flavours. These are those bands, copied from the original
# default_cli_configuration.json (MaxResolution "0x1250", "0x1350", ...), as
# (inclusive height limit, model bucket) pairs. They are the seed for the
# default working set below, and the fallback the worker still uses when no
# rule matches.
GRAY_HEIGHT_BANDS: tuple[tuple[int, int], ...] = (
    (1250, 1200),
    (1350, 1300),
    (1450, 1400),
    (1550, 1500),
    (1760, 1600),
    (1984, 1920),
)
GRAY_TOP_BUCKET = 2048  # anything taller than the last band

# What "auto" means for colour pages, per target scale.
COLOUR_DEFAULTS: dict[int, str] = {
    2: "2x_IllustrationJaNai_V3denoise_FDAT_M_unshuffle_30k_fp16.safetensors",
    4: "4x_IllustrationJaNai_V3denoise_FDAT_M_47k_fp16.safetensors",
}


# --------------------------------------------------------------------------- #
# sizes
# --------------------------------------------------------------------------- #
_RANGE_RE = re.compile(r"^(\d*)\s*(?:-|\u2013|\.\.)\s*(\d*)$")
_DIGITS_RE = re.compile(r"\d+")


def parse_dim(spec: object) -> tuple[int, int]:
    """Parse a size condition into ``(low, high)`` pixels; 0 means unbounded.

    Accepts ``any``/blank, an exact ``1920`` (or ``1920p``), a closed range
    ``1600-1920``, an open range ``1985-`` / ``-1250``, and ``>=1600`` /
    ``<=1250``.
    """
    if spec is None:
        return (0, 0)
    text = str(spec).strip().lower().replace(" ", "")
    if not text or text in {ANY, "*", "0", "-"}:
        return (0, 0)
    if text.startswith(">="):
        return (_first_int(text), 0)
    if text.startswith("<="):
        return (0, _first_int(text))
    if text.startswith(">"):
        return (_first_int(text) + 1, 0)
    if text.startswith("<"):
        return (0, max(0, _first_int(text) - 1))
    match = _RANGE_RE.match(text)
    if match:
        low = int(match.group(1) or 0)
        high = int(match.group(2) or 0)
        if low and high and low > high:
            low, high = high, low
        return (low, high)
    exact = _first_int(text)
    return (exact, exact) if exact else (0, 0)


def _first_int(text: str) -> int:
    found = _DIGITS_RE.search(text)
    return int(found.group()) if found else 0


def dim_spec(low: int, high: int) -> str:
    """The canonical text for a ``(low, high)`` pair, ready to store or show."""
    if not low and not high:
        return ANY
    if low and high:
        return str(low) if low == high else f"{low}-{high}"
    return f"{low}-" if low else f"-{high}"


def dim_score(low: int, high: int) -> int:
    """How specific a size condition is: exact 2, range 1, any 0."""
    if not low and not high:
        return 0
    return 2 if low == high else 1


def bucket_scale(scale: float) -> int:
    """Map any target factor onto the model factors that actually exist.

    Height, width and fit targets produce factors like 1.8 or 3.7, so both the
    page's factor and the rule's are bucketed before they are compared.
    """
    value = float(scale or 0)
    if value <= 0:
        return 0
    if value <= 1.5:
        return 1
    return 2 if value <= 3.0 else 4


# --------------------------------------------------------------------------- #
# model names
# --------------------------------------------------------------------------- #
_MODEL_SCALE_RE = re.compile(r"(?:^|[^0-9a-z])([1-9])\s*x(?:[^0-9a-z]|$)", re.IGNORECASE)
_MODEL_HEIGHT_RE = re.compile(r"(\d{3,4})\s*p(?:[^0-9a-z]|$)", re.IGNORECASE)


def model_scale(name: str) -> int:
    """The factor a model's filename claims (``4x_...`` -> 4), or 0."""
    match = _MODEL_SCALE_RE.search(str(name or ""))
    return int(match.group(1)) if match else 0


def model_height(name: str) -> int:
    """The page height a MangaJaNai model is tuned for (``1920p`` -> 1920)."""
    match = _MODEL_HEIGHT_RE.search(str(name or ""))
    return int(match.group(1)) if match else 0


def is_manga_model(name: str) -> bool:
    return "mangajanai" in str(name or "").lower()


# --------------------------------------------------------------------------- #
# the rule
# --------------------------------------------------------------------------- #
#: What a matching rule does with the page. PASSTHROUGH is the size-exclusion
#: case: the page skips the model entirely and is only re-encoded, which is what
#: the old hardcoded long-strip switch did with numbers nobody could see.
UPSCALE = "upscale"
PASSTHROUGH = "passthrough"
ACTIONS = (UPSCALE, PASSTHROUGH)


@dataclass(frozen=True, slots=True)
class Rule:
    """One row of the table: conditions on the left, outcomes on the right."""

    kind: str = ANY  # any | colour | grayscale
    scale: float = 0.0  # 0 = any target factor
    width: str = ANY
    height: str = ANY
    model: str = AUTO
    action: str = UPSCALE  # upscale | passthrough (size exclusion)
    auto_levels: bool | None = None  # grayscale only; None = inherit
    enabled: bool = True
    note: str = field(default="", compare=False)

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "kind": self.kind,
            "width": self.width,
            "height": self.height,
            "model": self.model,
        }
        if self.scale:
            data["scale"] = self.scale
        if self.action != UPSCALE:
            data["action"] = self.action
        if self.auto_levels is not None:
            data["auto_levels"] = bool(self.auto_levels)
        if not self.enabled:
            data["enabled"] = False
        if self.note:
            data["note"] = self.note
        return data

    @classmethod
    def from_dict(cls, data: object) -> Rule:
        raw = data if isinstance(data, dict) else {}
        kind = str(raw.get("kind") or ANY).strip().lower()
        if kind in {"gray", "grey", "greyscale", "manga"}:
            kind = GRAYSCALE
        elif kind in {"color", "colour", "illustration"}:
            kind = COLOUR
        elif kind not in KINDS:
            kind = ANY
        levels = raw.get("auto_levels")
        return cls(
            kind=kind,
            scale=_as_float(raw.get("scale")),
            width=dim_spec(*parse_dim(raw.get("width"))),
            height=dim_spec(*parse_dim(raw.get("height"))),
            model=str(raw.get("model") or AUTO).strip() or AUTO,
            action=_as_action(raw.get("action")),
            auto_levels=None if levels is None else bool(levels),
            enabled=bool(raw.get("enabled", True)),
            note=str(raw.get("note") or ""),
        )

    # -- presentation ------------------------------------------------------ #
    @property
    def is_auto(self) -> bool:
        return self.model.strip().lower() in {"", AUTO}

    def when_label(self) -> str:
        kind = {COLOUR: "colour", GRAYSCALE: "grayscale"}.get(self.kind, "any page")
        return kind if not self.scale else f"{kind} @ {self.scale:g}x"

    def size_label(self) -> str:
        width = dim_spec(*parse_dim(self.width))
        height = dim_spec(*parse_dim(self.height))
        if width == ANY and height == ANY:
            return "any size"
        parts = []
        if width != ANY:
            parts.append(f"w {width}")
        if height != ANY:
            parts.append(f"h {height}")
        return "  ".join(parts)

    def levels_label(self) -> str:
        """What the Levels cell shows: never applies, follows the setting, or set."""
        if self.kind == COLOUR:
            return "\u2014"
        if self.auto_levels is None:
            return "default"
        return "on" if self.auto_levels else "off"

    def enabled_mark(self) -> str:
        """The on/off dot the table shows in its first column."""
        return "\u25cf" if self.enabled else "\u25cb"

    def model_label(self) -> str:
        """What the Model cell shows: the file, or that the page is excluded."""
        if self.action == PASSTHROUGH:
            return "no upscale \u2014 re-encode only"
        return self.model

    def columns(self) -> tuple[str, str, str, str]:
        """The four content cells the interface shows for this rule."""
        return (self.when_label(), self.size_label(), self.model_label(), self.levels_label())

    def cells(self) -> tuple[str, str, str, str, str]:
        """Every cell of the table row, including the enabled indicator."""
        return (self.enabled_mark(), *self.columns())

    def describe(self) -> str:
        return f"{self.when_label()} / {self.size_label()} -> {self.model_label()}"

    def specificity(self) -> tuple[int, int, int]:
        """Higher sorts first: size, then factor, then colour/grayscale."""
        sizes = dim_score(*parse_dim(self.width)) + dim_score(*parse_dim(self.height))
        return (sizes, 1 if self.scale else 0, 0 if self.kind == ANY else 1)


def _as_float(value: object) -> float:
    try:
        return max(0.0, float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _as_action(value: object) -> str:
    action = str(value or UPSCALE).strip().lower()
    if action in {"skip", "exclude", "excluded", "passthrough", "pass-through", "re-encode"}:
        return PASSTHROUGH
    return action if action in ACTIONS else UPSCALE


# --------------------------------------------------------------------------- #
# the compiled set
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class _Compiled:
    rule: Rule
    kind: str
    scale: int
    w_low: int
    w_high: int
    h_low: int
    h_high: int

    def hits(self, kind: str, width: int, height: int, scale: int) -> bool:
        if self.kind != ANY and self.kind != kind:
            return False
        if self.scale and scale and self.scale != scale:
            return False
        if self.w_low and width < self.w_low:
            return False
        if self.w_high and width > self.w_high:
            return False
        if self.h_low and height < self.h_low:
            return False
        return not (self.h_high and height > self.h_high)


def _compile(rule: Rule) -> _Compiled:
    """Parse a rule's text conditions once, so matching is integer work."""
    w_low, w_high = parse_dim(rule.width)
    h_low, h_high = parse_dim(rule.height)
    return _Compiled(
        rule=rule,
        kind=rule.kind,
        scale=bucket_scale(rule.scale),
        w_low=w_low,
        w_high=w_high,
        h_low=h_low,
        h_high=h_high,
    )


class RuleSet:
    """An ordered set of rules, compiled once and matched per page."""

    __slots__ = ("_cache", "_compiled", "rules")

    def __init__(self, rules: Iterable[Rule] = ()) -> None:
        self.rules: list[Rule] = [r for r in rules if isinstance(r, Rule)]
        # Sort by specificity but keep the author's order for ties, so "move
        # up" still means something between equally specific rules.
        order = sorted(
            ((position, rule) for position, rule in enumerate(self.rules) if rule.enabled),
            key=lambda item: (tuple(-n for n in item[1].specificity()), item[0]),
        )
        self._compiled: tuple[_Compiled, ...] = tuple(_compile(rule) for _, rule in order)
        self._cache: dict[tuple[bool, int, int, int], Rule | None] = {}

    # -- construction ------------------------------------------------------ #
    @classmethod
    def from_dicts(cls, data: object) -> RuleSet:
        if not isinstance(data, (list, tuple)):
            return cls(())
        return cls(Rule.from_dict(item) for item in data)

    def to_dicts(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.rules]

    def __len__(self) -> int:
        return len(self.rules)

    def __bool__(self) -> bool:
        return bool(self._compiled)

    def __iter__(self) -> Iterator[Rule]:
        return iter(self.rules)

    # -- matching ---------------------------------------------------------- #
    def match(self, *, gray: bool, width: int, height: int, scale: float) -> Rule | None:
        """The winning rule for one page, or None if nothing matches."""
        if not self._compiled:
            return None
        bucket = bucket_scale(scale)
        key = (bool(gray), int(width), int(height), bucket)
        hit = self._cache.get(key, _MISS)
        if hit is not _MISS:
            return hit  # type: ignore[return-value]
        kind = GRAYSCALE if gray else COLOUR
        found: Rule | None = None
        for entry in self._compiled:
            if entry.hits(kind, int(width), int(height), bucket):
                found = entry.rule
                break
        self._cache[key] = found
        return found

    def ordered(self) -> list[Rule]:
        """The enabled rules in the order they are actually evaluated."""
        return [entry.rule for entry in self._compiled]


_MISS = object()


# --------------------------------------------------------------------------- #
# editing helpers (used by the interface)
# --------------------------------------------------------------------------- #
def move(rules: Sequence[Rule], index: int, delta: int) -> list[Rule]:
    """Return the list with one rule moved, clamped to the ends."""
    items = list(rules)
    target = index + delta
    if not (0 <= index < len(items)) or not (0 <= target < len(items)):
        return items
    items[index], items[target] = items[target], items[index]
    return items


def with_enabled(rule: Rule, enabled: bool) -> Rule:
    return replace(rule, enabled=bool(enabled))


def problems(rule: Rule, installed: Sequence[str] = ()) -> list[str]:
    """Human-readable warnings for one rule: never fatal, just honest."""
    out: list[str] = []
    names = {str(n) for n in installed}
    # An exclusion runs no model, so none of the model checks apply to it:
    # warning that its model is missing would be warning about a field the
    # row never uses.
    if rule.action != PASSTHROUGH:
        if rule.is_auto:
            out.append("no model chosen - open the row and pick one")
        if not rule.is_auto and names and rule.model not in names:
            out.append(f"{rule.model} is not installed")
        factor = model_scale(rule.model) if not rule.is_auto else 0
        if factor and rule.scale and bucket_scale(factor) != bucket_scale(rule.scale):
            out.append(f"{rule.model} is x{factor} but the rule targets {rule.scale:g}x")
        if rule.kind == COLOUR and rule.auto_levels:
            out.append("auto-levels only applies to grayscale pages")
    elif rule.width in (ANY, "") and rule.height in (ANY, ""):
        out.append("this exclusion has no page size, so it would skip every page")
    w_low, w_high = parse_dim(rule.width)
    h_low, h_high = parse_dim(rule.height)
    if w_low and w_high and w_low > w_high:
        out.append("the width range is inverted")
    if h_low and h_high and h_low > h_high:
        out.append("the height range is inverted")
    return out


# --------------------------------------------------------------------------- #
# the default working set
# --------------------------------------------------------------------------- #
def bands() -> list[tuple[int, int, int]]:
    """The upstream height bands as ``(low, high, bucket)``, 0 = unbounded."""
    out: list[tuple[int, int, int]] = []
    low = 0
    for limit, bucket in GRAY_HEIGHT_BANDS:
        out.append((low, limit, bucket))
        low = limit + 1
    out.append((low, 0, GRAY_TOP_BUCKET))
    return out


def nearest_height_model(names: Sequence[str], bucket: int) -> str:
    """The installed MangaJaNai model closest to a height bucket."""
    tagged = [n for n in names if model_height(n)]
    if not tagged:
        return ""
    return min(tagged, key=lambda n: (abs(model_height(n) - bucket), model_height(n), n))


def gray_bucket(height: int) -> int:
    """The MangaJaNai page-height bucket (1200p, 1300p, ...) for a page height."""
    for limit, bucket in GRAY_HEIGHT_BANDS:
        if height <= limit:
            return bucket
    return GRAY_TOP_BUCKET


def gray_model(installed: Sequence[str], scale: int, bucket: int) -> str:
    """The best installed grayscale model for a factor and a height band."""
    pool = [str(n) for n in installed if model_scale(n) == scale]
    manga = [n for n in pool if is_manga_model(n) and model_height(n)]
    if manga:
        return nearest_height_model(manga, bucket)
    rest = [n for n in pool if is_manga_model(n)] or pool
    return min(rest) if rest else ""


def colour_model(installed: Sequence[str], scale: int) -> str:
    """The best installed colour model for a factor."""
    names = [str(n) for n in installed]
    wanted = COLOUR_DEFAULTS.get(scale, "")
    if wanted and wanted in names:
        return wanted
    pool = [n for n in names if model_scale(n) == scale]
    illustration = [n for n in pool if not is_manga_model(n)] or pool
    denoise = [n for n in illustration if "denoise" in n.lower()] or illustration
    return min(denoise) if denoise else ""


def default_working_set(
    installed: Sequence[str] = (), scales: Sequence[int] = (2, 4)
) -> list[Rule]:
    """The shipped table: what the old hidden "auto" did, written out in full.

    Every row names a real file from the models actually installed - the table
    is the only thing that picks a model, so it must never ship a placeholder.
    Adjacent height bands that resolve to the same file are merged into one row
    to keep it readable. Every row names its factor: a target that is neither 2x
    nor 4x falls into the nearest bucket, and a model with some other factor -
    3x, say - is a row the user adds, with that factor typed into the rule.
    """
    names = [str(n) for n in installed]
    out: list[Rule] = []
    for scale in scales:
        rows: list[tuple[int, int, str]] = []
        for low, high, bucket in bands():
            pick = gray_model(names, scale, bucket)
            if not pick:
                continue
            if rows and rows[-1][2] == pick:
                # Same file, so widen the previous band. A tuple rather than a
                # mutable row because the three columns have three types, and
                # `list[int | str]` would defeat `dim_spec` and `model=` below.
                rows[-1] = (rows[-1][0], high, pick)
            else:
                rows.append((low, high, pick))
        for low, high, pick in rows:
            out.append(
                Rule(
                    kind=GRAYSCALE,
                    scale=float(scale),
                    height=dim_spec(low, high),
                    model=pick,
                    auto_levels=True,
                    note="default working set",
                )
            )
        colour = colour_model(names, scale)
        if colour:
            out.append(
                Rule(kind=COLOUR, scale=float(scale), model=colour, note="default working set")
            )

    # No unsized catch-all rows: every shipped model is 2x or 4x, and a factor
    # in between (1.8 from a width target, say) buckets to one of them. A row
    # that matched any target could only repeat a decision already made, while
    # warning that its model does not match the target it never chose.
    return out


def materialise(
    items: Iterable[Rule], installed: Sequence[str] = ()
) -> tuple[list[Rule], list[str]]:
    """Turn legacy ``auto`` rows into real files. Returns (rules, notes).

    Settings written before the table became the single source of truth could
    say ``auto``; that is resolved here once, against the installed models, so
    what the table shows is what will actually run.
    """
    names = [str(n) for n in installed]
    out: list[Rule] = []
    notes: list[str] = []
    for rule in items:
        if not rule.is_auto:
            out.append(rule)
            continue
        scale = bucket_scale(rule.scale) or 2
        if scale < 2:
            scale = 2
        if rule.kind == GRAYSCALE:
            low, high = parse_dim(rule.height)
            pick = gray_model(names, scale, gray_bucket(high or low or GRAY_TOP_BUCKET))
        else:
            pick = colour_model(names, scale)
        if not pick:
            notes.append(f"dropped {rule.describe()} (nothing installed to run it)")
            continue
        out.append(replace(rule, model=pick, note=rule.note or "resolved from auto"))
        notes.append(f"{rule.when_label()} / {rule.size_label()} -> {pick}")
    return out, notes


def default_dicts(installed: Sequence[str] = ()) -> list[dict[str, Any]]:
    return [r.to_dict() for r in default_working_set(installed)]
