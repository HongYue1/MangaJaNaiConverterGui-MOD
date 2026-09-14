"""The composed-window surface that every panel mixin is allowed to assume.

`MainWindow` (`window.py`) is assembled from six panel mixins plus
`QMainWindow`. Each mixin builds one card and then freely reaches for the shared
collaborators (`settings`, `runner`, `log`), widgets owned by *other* panels
(`sp_scale`, `chk_fp16`), and sibling methods (`update_summary`,
`render_status`) - none of which exist on the mixin itself. That is deliberate:
the panels are views onto one window, not standalone widgets.

This Protocol writes that contract down once, so a type checker can verify the
panel bodies instead of silently treating every such access as `Any`. Mixins
inherit it **for the type checker only**:

    if TYPE_CHECKING:
        from janai.app.surface import WindowSurface as _Base
    else:  # the surface is a type-only view; at runtime a mixin is a mixin
        _Base = object

    class RunPanelMixin(_Base):
        ...

WHY type-check-only: at runtime the base collapses to `object`, which is
already in every mixin's MRO. The linearisation `MainWindow` depends on (the
six mixins first, `QMainWindow` last) and Qt's Shiboken metaclass are therefore
untouched - a declaration here cannot change behaviour. WHY a Protocol: it is a
description that is never instantiated, so empty bodies are honest;
`raise NotImplementedError` would imply a call path that does not exist.

Keep it in step with `window.py` and the panels. A member renamed or retyped
there but not here turns a checked access back into a lie, and the checker
cannot warn us: `PySide6` has no stubs in the checker's environment, so
`MainWindow`'s Qt base is `Any` and the window's own attributes are
unverifiable. The panels are what this buys us.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
)

from janai.app.runlog import RunLog
from janai.app.runner import Runner
from janai.app.state import Settings
from janai.app.theme import Theme
from janai.app.widgets import Banner, Card, Collapsible, Segmented
from janai.core.rules import Rule


class WindowSurface(Protocol):
    """What the composed `MainWindow` provides to its panel mixins."""

    # Collaborators, constructed once in MainWindow.__init__.
    app: QApplication
    theme: Theme
    dir: Path
    settings: Settings
    runner: Runner
    runlog: RunLog

    # Hardware and model facts, filled in by the worker's probe.
    probe: dict
    caps: dict
    models: list
    devices: list
    profile: dict
    _profiling: bool
    _profile_offered: bool

    # Run accounting. `render_status` is the single writer of the status label,
    # and these counters are what it reports; nothing else may write it.
    total: int
    completed: int
    failed: int
    skipped: int
    started_index: int
    started_at: float
    progress_sub: str
    dry: bool
    last_out_dir: Path | None
    scan_text: str
    fmt_values: dict[str, dict[str, Any]]
    _in_path: str
    _saved_device: str
    _log_visible: bool
    _rules_seeded: bool

    # Chrome and the Upscale card, owned by window.py.
    banner: Banner
    splitter: QSplitter
    scroll: QScrollArea
    page: QVBoxLayout
    lbl_env: QLabel
    btn_refresh: QPushButton
    seg_mode: Segmented
    seg_orient: Segmented
    sp_scale: QDoubleSpinBox
    sp_width: QSpinBox
    sp_height: QSpinBox
    cb_display: QComboBox
    chk_levels: QCheckBox
    sp_threshold: QSpinBox
    sp_colour: QDoubleSpinBox
    sp_pre_h: QSpinBox

    # input_panel.py
    card_input: Card
    chk_recursive: QCheckBox
    chk_archives: QCheckBox

    # output_panel.py
    seg_fmt: Segmented
    ed_sub: QLineEdit
    ed_out: QLineEdit
    ed_pattern: QLineEdit
    chk_adv: QCheckBox
    chk_same: QCheckBox
    chk_overwrite: QCheckBox
    chk_keep_tree: QCheckBox

    # perf_panel.py
    panel_perf: Collapsible
    cb_tile: QComboBox
    btn_profile: QPushButton
    chk_fp16: QCheckBox
    chk_cudnn: QCheckBox
    chk_tf32: QCheckBox
    chk_wipe: QCheckBox
    chk_wake: QCheckBox
    sp_budget: QSpinBox
    sp_threads: QSpinBox
    sp_io: QSpinBox
    sp_vips: QSpinBox

    # log_panel.py
    lbl_log_file: QLabel
    chk_wrap: QCheckBox
    chk_debug: QCheckBox

    # run_panel.py
    btn_log: QPushButton

    # ---------------------------------------------------------------- #
    # Methods reached across panel boundaries, grouped by owning module.
    # ---------------------------------------------------------------- #

    def page_kind(self) -> str:
        """Return the selected page-kind id. Owned by window.py."""

    def update_summary(self) -> None:
        """Recompute the one-line job summary. Owned by window.py."""

    def show_banner(self, text: str) -> None:
        """Show the notice banner above the panels. Owned by window.py."""

    def hide_banner(self) -> None:
        """Hide the notice banner. Owned by window.py."""

    def _geometry_text(self) -> str:
        """Return the window geometry string persisted in settings. Owned by window.py."""

    def _confirm(self, title: str, text: str) -> bool:
        """Ask a modal yes/no question and return the answer. Owned by window.py."""

    def log(self, message: str, level: str = "info") -> None:
        """Append one line to the in-app log and the run log. Owned by log_panel.py."""

    def toggle_log(self) -> None:
        """Show or hide the log pane. Owned by log_panel.py."""

    def render_format_options(self) -> None:
        """Rebuild the per-format option rows. Owned by output_panel.py."""

    def format_values(self, fid: str) -> dict:
        """Return the current option values for one output format. Owned by output_panel.py."""

    def container_value(self) -> str:
        """Return the selected container id. Owned by output_panel.py."""

    def resolved_out_dir(self) -> Path | None:
        """Return the output directory the job will write to. Owned by output_panel.py."""

    def device_value(self) -> str:
        """Return the selected device id. Owned by perf_panel.py."""

    def update_wake_lock(self) -> None:
        """Apply the keep-awake setting for the current run state. Owned by perf_panel.py."""

    def profile_for_run(self) -> dict:
        """Return the tuning profile to hand the worker. Owned by perf_panel.py."""

    def render_profile(self) -> None:
        """Redraw the profiling readout. Owned by perf_panel.py."""

    def apply_probe(self, probe: dict, cached: bool = False) -> None:
        """Adopt a hardware probe result and refresh dependent fields. Owned by perf_panel.py."""

    def on_hold(self, event: dict) -> None:
        """Handle a worker `hold` event. Owned by perf_panel.py."""

    def on_profile(self, event: dict) -> None:
        """Handle a finished `profile` event. Owned by perf_panel.py."""

    def on_profile_progress(self, event: dict) -> None:
        """Handle a `profile_progress` event. Owned by perf_panel.py."""

    def on_probe_error(self, event: dict) -> None:
        """Report a failed hardware probe. Owned by perf_panel.py."""

    @property
    def rules(self) -> list[Rule]:
        """Return the rules currently in the table. Owned by rules_panel.py."""

    def model_names(self) -> list[str]:
        """Return the installed model file names. Owned by rules_panel.py."""

    def seed_rules(self) -> None:
        """Populate the rules table with a default working set. Owned by rules_panel.py."""

    def render_rules(self) -> None:
        """Redraw the rules table and its hints. Owned by rules_panel.py."""

    def update_start_state(self) -> None:
        """Enable or disable Start for the current inputs. Owned by run_panel.py."""

    def render_status(self, total: int = 0) -> None:
        """Write the status label. The ONLY writer of it. Owned by run_panel.py."""

    def sync_settings(self) -> None:
        """Copy every widget value back into `settings`. Owned by run_panel.py."""
