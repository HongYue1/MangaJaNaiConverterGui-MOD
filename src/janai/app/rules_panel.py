"""Both rule tables: what runs on a page, and what skips the model entirely.

The window shows them apart - model rules inside the upscale card, size
exclusions in their own collapsed panel - because they answer different
questions. They are one feature all the same: one editor dialog, one saved
list, one warning line, and an edit to either table re-renders and re-saves
both.
"""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget

from janai.app.rules_table import EXCLUSION_HEADERS, RuleDialog, RulesModel, RulesTable
from janai.app.widgets import Collapsible, button, label, row
from janai.core import rules


class RulesPanelMixin:
    """The two rule tables and every edit made to them.

    Mixed into :class:`janai.app.window.MainWindow`, so ``self`` is the window.
    The page-kind control and the scale spinner this reads belong to the
    upscale card, and ``_confirm`` stays on the window because the full reset
    shares it.
    """

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _rule_buttons(self, which: str) -> QWidget:
        """The column of buttons beside one of the two rule tables."""
        side = QWidget()
        sbox = QVBoxLayout(side)
        sbox.setContentsMargins(0, 0, 0, 0)
        sbox.setSpacing(6)
        actions: tuple[tuple[str, Callable[[], None], str], ...] = (
            ("Add", lambda: self.rule_add(which), "Add a row below the selected one."),
            (
                "Edit",
                lambda: self.rule_edit(which),
                "Edit the selected row (or double-click it).",
            ),
            (
                "Toggle",
                lambda: self.rule_toggle(which),
                (
                    "Switch the selected row off without deleting it. The checkbox "
                    "in the first column shows the state, and the space bar does the "
                    "same thing."
                ),
            ),
            ("Remove", lambda: self.rule_remove(which), "Delete the selected row."),
            (
                "Up",
                lambda: self.rule_move(-1, which),
                (
                    "Move the row up (Alt+Up). Order only decides between rows that "
                    "are equally specific."
                ),
            ),
            ("Down", lambda: self.rule_move(1, which), "Move the row down (Alt+Down)."),
        )
        if which == "rules":
            actions = (
                *actions,
                (
                    "Defaults",
                    self.rules_reset,
                    (
                        "Rewrite this table as the shipped set: the MangaJaNai height "
                        "bands for grayscale pages and the IllustrationJaNai denoise "
                        "models for colour, built from the models you have installed. "
                        "Only this table is touched - your exclusions and the rest of "
                        "your settings are left alone."
                    ),
                ),
            )
        for text, action, hint in actions:
            btn = button(text, action, variant="ghost", tip=hint)
            btn.setMinimumWidth(96)
            sbox.addWidget(btn)
        sbox.addStretch(1)
        return side

    def _table_row(self, view: RulesTable, side: QWidget) -> QWidget:
        """A table with its buttons beside it."""
        holder = QWidget()
        tbox = QHBoxLayout(holder)
        tbox.setContentsMargins(0, 0, 0, 0)
        tbox.setSpacing(10)
        tbox.addWidget(view, 1)
        tbox.addWidget(side, 0)
        return holder

    def _build_exclusions_block(self, seeded: list[rules.Rule]) -> QWidget:
        """Size exclusions, in their own panel, closed until they are wanted.

        They answer a different question from the model rules - which pages
        should not go through a model at all - and most runs never touch them,
        so they sit in a panel that starts collapsed rather than taking up half
        of the card.
        """
        panel = Collapsible(
            "Size exclusions",
            "pages that skip the model",
            expanded=False,
        )
        self.panel_excl = panel
        panel.setToolTip(
            "Pages that skip the model and are only re-encoded: the long webtoon "
            "strips the old build skipped with numbers nobody could see. An "
            "exclusion names a page size, and a sized row always beats a model "
            "rule that says \u201cany\u201d, so those pages pass through untouched."
        )
        body = panel.body

        self.excl_model = RulesModel(
            self.theme.p,
            [r for r in seeded if r.action == rules.PASSTHROUGH],
            bool(self.settings.data["upscale"].get("grayscale_convert", True)),
            self.model_names(),
            parent=self,
            headers=EXCLUSION_HEADERS,
        )
        self.excl_model.edited.connect(self.on_rule_checked)
        self.excl_view = RulesTable(self.excl_model, rows=3)
        self.excl_view.doubleClicked.connect(lambda _index: self.rule_edit("excl"))
        body.full(self._table_row(self.excl_view, self._rule_buttons("excl")))

        self.lbl_excl_hint = label("", "hint", wrap=True)
        body.full(self.lbl_excl_hint)
        return panel

    def _build_rules_block(self, seeded: list[rules.Rule]) -> QWidget:
        """The model-rules table, its side buttons, and the lines beneath it."""
        block = QWidget()
        box = QVBoxLayout(block)
        box.setContentsMargins(0, 4, 0, 0)
        box.setSpacing(8)

        how = (
            "page kind + size decide the model \u00b7 a sized row always beats "
            "an \u201cany\u201d row"
        )
        tip = (
            "Every page is matched against this table. The first row whose "
            "conditions fit decides which model runs, and for grayscale pages "
            "whether auto levels is applied. A row that names a page size wins "
            "over a row that says \u201cany\u201d wherever the two sit, so a "
            "catch-all at the top cannot swallow everything by accident. "
            "Double-click a row to edit it, or clear its checkbox to switch it off."
        )
        box.addWidget(row(label("Model rules", "field", tip=tip), label(how, "hint", tip=tip)))

        self.rules_model = RulesModel(
            self.theme.p,
            [r for r in seeded if r.action != rules.PASSTHROUGH],
            bool(self.settings.data["upscale"].get("grayscale_convert", True)),
            self.model_names(),
            parent=self,
        )
        self.rules_model.edited.connect(self.on_rule_checked)
        self.rules_view = RulesTable(self.rules_model, rows=6)
        self.rules_view.doubleClicked.connect(lambda _index: self.rule_edit("rules"))
        box.addWidget(self._table_row(self.rules_view, self._rule_buttons("rules")))

        self.lbl_rules_hint = label("", "hint", wrap=True)
        box.addWidget(self.lbl_rules_hint)
        self.lbl_rules_warn = label("", "warn", wrap=True)
        self.lbl_rules_warn.setVisible(False)
        box.addWidget(self.lbl_rules_warn)
        return block

    # ------------------------------------------------------------------ #
    # which rows can fire
    # ------------------------------------------------------------------ #
    def gray_rules_live(self) -> bool:
        """Grayscale rows can only fire while some page can be grayscale."""
        return self.page_kind() != "colour"

    def set_gray_rows(self, live: bool) -> None:
        """Both tables grey out their grayscale rows together."""
        self.rules_model.set_gray(live)
        self.excl_model.set_gray(live)

    # ------------------------------------------------------------------ #
    # the rule set
    # ------------------------------------------------------------------ #
    @property
    def rules(self) -> list[rules.Rule]:
        """Every rule from both tables: exclusions first, then the model rules.

        The interface splits them - model rules in the card, size exclusions in
        their own panel - but matching, saving and the job payload all want the
        whole set, and exclusions come first so that reading the saved file top
        to bottom follows the order a page is decided in.
        """
        return [*self.excl_model.rules, *self.rules_model.rules]

    def set_all_rules(self, items: list[rules.Rule]) -> None:
        """Deal one saved list into the two tables, keeping relative order."""
        self.rules_model.set_rules([r for r in items if r.action != rules.PASSTHROUGH])
        self.excl_model.set_rules([r for r in items if r.action == rules.PASSTHROUGH])

    def _table(self, which: str = "rules") -> tuple[RulesModel, RulesTable]:
        if which == "excl":
            return self.excl_model, self.excl_view
        return self.rules_model, self.rules_view

    def _focused_table(self) -> str:
        """Which table a keyboard shortcut should act on."""
        return "excl" if self.excl_view.hasFocus() else "rules"

    def model_names(self) -> list[str]:
        return [str(m.get("name")) for m in self.models if m.get("name")]

    def seed_rules(self) -> None:
        """Make sure the table is filled in and names real files, once.

        Both cases need the probe to have reported the installed models: a
        first run gets the shipped working set written out in full, and a
        settings file from an older build gets its legacy "auto" rows resolved
        to the file they would have picked.
        """
        if self._rules_seeded or not self.models:
            return
        installed = self.model_names()
        current = list(self.rules)
        if not current:
            current = rules.default_working_set(installed)
        elif any(r.is_auto for r in current):
            current, notes = rules.materialise(current, installed)
            for note in notes:
                self.log(f"rule resolved: {note}", "debug")
            if not current:
                current = rules.default_working_set(installed)
        self.set_all_rules(current)
        self._rules_seeded = True
        self.save_rules()

    # ------------------------------------------------------------------ #
    # rendering
    # ------------------------------------------------------------------ #
    def render_rules(self) -> None:
        """Refresh the lines under the two tables. The models paint the rows."""
        names = self.model_names()
        self.rules_model.set_installed(names)
        self.excl_model.set_installed(names)
        kind = self.page_kind()
        items = self.rules_model.rules
        if not items:
            hint = (
                "The table is empty, so nothing can run. \u201cDefaults\u201d fills it "
                "with the shipped set, built from the models you have installed."
            )
        else:
            active = sum(1 for r in items if r.enabled)
            hint = f"{active} of {len(items)} rules on"
            if kind == "colour":
                hint += "  \u00b7  grayscale rules are idle: every page is colour"
            elif kind == "grayscale":
                hint += "  \u00b7  colour rules are idle: every page is grayscale"
            hint += "  \u00b7  space toggles a row, double-click edits it"
        self.lbl_rules_hint.setText(hint)

        excluded = self.excl_model.rules
        live = sum(1 for r in excluded if r.enabled)
        if not excluded:
            self.lbl_excl_hint.setText(
                "Nothing is excluded, so every page goes through a model. Add a row "
                "to let pages of a given size skip the model and only be re-encoded."
            )
            self.panel_excl.set_hint("none")
        else:
            self.lbl_excl_hint.setText(
                f"{live} of {len(excluded)} exclusions on  \u00b7  matching pages skip "
                "the model and are only re-encoded"
            )
            self.panel_excl.set_hint(f"{live} on" if live else f"{len(excluded)} off")
        self.refresh_rule_warnings()

    def refresh_rule_warnings(self) -> None:
        """Everything wrong with either table, on one line under the rules.

        Each table numbers its own rows, so a warning has to say which table it
        points at or the number would send you to the wrong row.
        """
        installed = self.model_names()
        notes: list[str] = []
        for what, items in (
            ("row", self.rules_model.rules),
            ("exclusion", self.excl_model.rules),
        ):
            for index, rule in enumerate(items):
                if not rule.enabled:
                    continue
                notes.extend(
                    f"{what} {index + 1}: {note}" for note in rules.problems(rule, installed)
                )
        notes.extend(self.scale_mismatches())
        self.lbl_rules_warn.setText("\u26a0  " + "; ".join(notes[:4]) if notes else "")
        self.lbl_rules_warn.setVisible(bool(notes))

    def scale_mismatches(self) -> list[str]:
        """Rows whose model name advertises a factor the target will not use."""
        if str(self.seg_mode.value()) != "scale":
            return []
        want = float(self.sp_scale.value())
        out: list[str] = []
        for index, rule in enumerate(self.rules_model.rules):
            if not rule.enabled or rule.is_auto:
                continue
            if rule.scale and abs(rule.scale - want) > 0.01:
                continue
            found = rules.model_scale(rule.model)
            if found and abs(found - want) > 0.01:
                out.append(
                    f"row {index + 1} runs a {found}\u00d7 model but the target is "
                    f"{want:g}\u00d7, so the result gets resampled"
                )
        return out

    # ------------------------------------------------------------------ #
    # editing
    # ------------------------------------------------------------------ #
    def save_rules(self) -> None:
        """Rules are remembered as they are edited, not only on Start."""
        self.settings.set("upscale", "rules", [r.to_dict() for r in self.rules])
        self.settings.save()

    def on_rule_checked(self) -> None:
        """The checkbox column changed a row in place."""
        self.render_rules()
        self.save_rules()
        self.update_summary()

    def rules_changed(self, which: str = "rules", select: int = -1) -> None:
        model, view = self._table(which)
        if 0 <= select < model.rowCount():
            view.select_row(select)
        self.render_rules()
        self.save_rules()
        self.update_summary()

    def default_rule_model(self) -> str:
        """A sensible model to open a new rule with - never a placeholder."""
        names = self.model_names()
        if not names:
            return ""
        scale = max(1, round(float(self.sp_scale.value())))
        return rules.gray_model(names, scale, rules.GRAY_TOP_BUCKET) or names[0]

    def rule_add(self, which: str = "rules") -> None:
        exclusion = which == "excl"
        draft = rules.Rule(
            kind=rules.ANY if exclusion else rules.GRAYSCALE,
            scale=0.0 if exclusion else float(self.sp_scale.value()),
            # A new exclusion opens on the case it exists for: pages far taller
            # than a page, which is what the old hardcoded switch matched.
            height=rules.dim_spec(3000, 0) if exclusion else rules.ANY,
            auto_levels=None if exclusion else True,
            model=self.default_rule_model(),
            action=rules.PASSTHROUGH if exclusion else rules.UPSCALE,
        )
        title = "Add exclusion" if exclusion else "Add rule"
        made = RuleDialog.edit(self, title, draft, self.model_names())
        if made is None:
            return
        self._insert_rule(which, made)

    def _insert_rule(self, which: str, made: rules.Rule) -> None:
        """File an edited rule in whichever table its action belongs to."""
        target = "excl" if made.action == rules.PASSTHROUGH else "rules"
        model, _view = self._table(target)
        items = list(model.rules)
        index = self._table(which)[1].current_row() if target == which else -1
        at = len(items) if index < 0 else index + 1
        items.insert(at, made)
        model.set_rules(items)
        self.rules_changed(target, at)

    def rule_edit(self, which: str = "rules") -> None:
        model, view = self._table(which)
        index = view.current_row()
        if index < 0:
            return
        title = "Edit exclusion" if which == "excl" else "Edit rule"
        made = RuleDialog.edit(self, title, model.rules[index], self.model_names())
        if made is None:
            return
        items = list(model.rules)
        if (made.action == rules.PASSTHROUGH) != (which == "excl"):
            # Its action changed, so the row now belongs in the other table.
            del items[index]
            model.set_rules(items)
            self._insert_rule(which, made)
            return
        items[index] = made
        model.set_rules(items)
        self.rules_changed(which, index)

    def rule_remove(self, which: str = "rules") -> None:
        model, view = self._table(which)
        index = view.current_row()
        if index < 0:
            return
        items = list(model.rules)
        del items[index]
        model.set_rules(items)
        self.rules_changed(which, min(index, len(items) - 1))

    def rule_move(self, delta: int, which: str = "rules") -> None:
        model, view = self._table(which)
        index = view.current_row()
        if index < 0:
            return
        items = rules.move(model.rules, index, delta)
        model.set_rules(items)
        self.rules_changed(which, max(0, min(index + delta, len(items) - 1)))

    def rule_toggle(self, which: str = "rules") -> None:
        model, view = self._table(which)
        index = view.current_row()
        if index >= 0:
            model.toggle(index)

    def rules_reset(self) -> None:
        if self.rules_model.rules and not self._confirm(
            "Reset the rules table",
            "Replace every row with the shipped set, built from the models you "
            "have installed?\n\nOnly this table changes \u2014 your exclusions and "
            "the rest of your settings are left alone.",
        ):
            return
        self.rules_model.set_rules(rules.default_working_set(self.model_names()))
        self._rules_seeded = True
        self.rules_changed("rules", 0)
        self.log(f"rules reset to the shipped set ({len(self.rules_model.rules)} rows)")
