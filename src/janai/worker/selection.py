"""Which model a page gets, whether it is levelled, and whether it is skipped.

The rule table in :mod:`janai.core.rules` decides; the built-in height-band
picker in :mod:`janai.worker.models` is the fallback for a page no rule claims,
which is exactly what the shipped table's catch-all rows do explicitly.

This lives apart from :mod:`janai.worker.job` because the dry run and the real
run have to answer these questions *identically*. The dry run is handed
:meth:`PagePolicy.model` and :meth:`PagePolicy.excluded` as callables, so there
is one implementation and a predicted plan cannot drift from the run it
predicts.
"""

from dataclasses import dataclass, field

from janai.core.rules import AUTO, PASSTHROUGH, Rule, RuleSet
from janai.worker.events import log
from janai.worker.models import choose_model


@dataclass(frozen=True, slots=True)
class PagePolicy:
    """Per-page model and levels decisions for one job.

    Every field is a job-wide setting except ``_logged``, which is the only
    mutable state here: a rule hit is announced once per *distinct rule* for
    the whole job, not once per page, or a 200-page chapter would repeat the
    same line 200 times. That memory of "already said that" has to outlive a
    single page, which is why this is an object with a job lifetime rather than
    a few free functions. The dataclass is frozen so the settings cannot be
    reassigned mid-job; only the log-once set mutates.
    """

    models: list[dict]
    rule_set: RuleSet
    model_colour: str
    model_gray: str
    mode: str
    t_scale: float
    t_w: int
    t_h: int
    force_gray: bool
    do_gray: bool
    do_levels: bool
    _logged: set[str] = field(default_factory=set)

    def factor(self, oh: int, ow: int) -> float:
        """The factor this page will actually be upscaled by."""
        if self.mode == "height" and self.t_h:
            return self.t_h / max(1, oh)
        if self.mode == "width" and self.t_w:
            return self.t_w / max(1, ow)
        if self.mode == "fit" and self.t_w and self.t_h:
            return min(self.t_w / max(1, ow), self.t_h / max(1, oh))
        return self.t_scale

    def gray_page(self, gray: bool) -> bool:
        """Whether this page counts as grayscale for model and rule purposes.

        "Force grayscale" wins outright; otherwise a page only counts as gray
        when it measured gray *and* grayscale handling is switched on.
        """
        return self.force_gray or (gray and self.do_gray)

    def resolve(self, wanted: str, gray: bool, oh: int, factor: float) -> dict | None:
        """Turn a model name (or ``auto``) into an installed model."""
        name = (wanted or "").strip()
        if name.lower() in ("", AUTO):
            return choose_model(self.models, gray, oh, factor)
        found = next((m for m in self.models if m["name"] == name or m["path"] == name), None)
        if found is None:
            found = choose_model(self.models, gray, oh, factor)
            if found:
                log(f"model {name} not found, using {found['name']}", "warn")
        return found

    def _note(self, hit: Rule) -> None:
        """Say which rule fired, once per distinct rule, not once per page."""
        text = hit.describe()
        if text not in self._logged:
            self._logged.add(text)
            log(f"rule: {text}")

    def plan(self, gray: bool, oh: int, ow: int) -> tuple[dict | None, bool]:
        """What happens to one page: which model, and whether to auto-level.

        A matching rule decides. A page no rule claims falls back to the
        built-in picker.
        """
        is_gray = self.gray_page(gray)
        levels = self.do_levels
        if not self.models:
            return None, levels
        factor = self.factor(oh, ow)
        hit = self.rule_set.match(gray=is_gray, width=ow, height=oh, scale=factor)
        if hit is None:
            wanted = self.model_gray if is_gray else self.model_colour
        else:
            self._note(hit)
            wanted = hit.model
            if hit.auto_levels is not None:
                levels = bool(hit.auto_levels)
        return self.resolve(wanted, is_gray, oh, factor), levels

    def model(self, gray: bool, oh: int, ow: int) -> dict | None:
        """Model only; the dry run reports models without touching levels."""
        return self.plan(gray, oh, ow)[0]

    def excluded(self, gray: bool, oh: int, ow: int) -> bool:
        """A rule can exclude a page from the model instead of choosing one.

        The row's page-size condition decides which pages skip upscaling and
        are only re-encoded - the same outcome as the old long-strip switch,
        with the sizes visible and editable instead of hardcoded.
        """
        hit = self.rule_set.match(
            gray=self.gray_page(gray),
            width=ow,
            height=oh,
            scale=self.factor(oh, ow),
        )
        # Rule.action is a typed str field on a frozen slots dataclass, so the
        # getattr() default this replaced could never have fired.
        return hit is not None and hit.action == PASSTHROUGH
