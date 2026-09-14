"""The run-log panel: its widgets, and everything that writes to them.

Construction and behaviour deliberately live together. ``_build_log`` creates
``log_view``, ``chk_debug``, ``chk_wrap`` and ``lbl_log_file``, and the methods
below are the only code that reads them, so splitting the builder away from its
handlers would put every use of a widget in a different file from the line that
created it.

Mixed into :class:`janai.app.window.MainWindow`, so ``self`` is the window: the
panel is a child of the window's ``splitter`` (that is what lets the user drag
the log taller), and ``log()`` also feeds ``self.runlog``, the on-disk run log
in :mod:`janai.app.runlog`.
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QWidget

from janai.app.runner import open_in_explorer
from janai.app.widgets import LogView, button, checkbox, label

#: Levels the panel can colour; anything else is rendered as "info".
LOG_TAGS = ("info", "debug", "warn", "error", "ok", "skip", "dry")


class LogPanelMixin:
    """Run-log panel: the widgets, the log sink, and the copy/save actions."""

    def _log_dir(self) -> Path:
        # Called from MainWindow.__init__ to place RunLog, i.e. before any of
        # the widgets below exist. It must stay free of widget access.
        raw = str((self.settings.data.get("log") or {}).get("dir") or "logs")
        path = Path(raw).expanduser()
        return path if path.is_absolute() else self.dir / path

    def _build_log(self) -> None:
        panel = QWidget(self.splitter)
        box = QVBoxLayout(panel)
        box.setContentsMargins(18, 0, 18, 0)
        box.setSpacing(8)

        bar = QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(10)
        bar.addWidget(label("Run log", "title"))
        self.lbl_log_file = label("", "hint")
        bar.addWidget(self.lbl_log_file, 1)

        lg = self.settings.data.get("log") or {}
        self.chk_wrap = checkbox("Wrap", bool(lg.get("wrap", False)), self.apply_log_wrap)
        self.chk_debug = checkbox(
            "Debug",
            bool(lg.get("show_debug", False)),
            tip=(
                "Show the noisy lines (library warnings, tracebacks). They are always "
                "written to the run file either way."
            ),
        )
        bar.addWidget(self.chk_wrap)
        bar.addWidget(self.chk_debug)
        for text, action in (
            ("Copy", self.copy_log),
            ("Save as\u2026", self.save_log_as),
            ("Log folder", self.open_log_folder),
            ("Clear", self.clear_log),
            ("Hide", self.toggle_log),
        ):
            bar.addWidget(button(text, action, variant="ghost"))
        box.addLayout(bar)

        self.log_view = LogView(self.theme.fonts["mono"], panel)
        self.log_view.set_wrap(bool(lg.get("wrap", False)))
        box.addWidget(self.log_view, 1)
        self.log_panel = panel

    def log_colour(self, level: str) -> str:
        p = self.theme.p
        return {
            "info": p.text,
            "debug": p.muted,
            "skip": p.muted,
            "ok": p.ok,
            "dry": p.accent,
            "warn": p.warn,
            "error": p.err,
        }.get(level, p.text)

    def log(self, message: str, level: str = "info") -> None:
        """One line in the panel and, unless it is noise, in the run file."""
        if not message:
            return
        stamp = time.strftime("%H:%M:%S")
        tag = level if level in LOG_TAGS else "info"
        self.runlog.write(stamp, message)
        if tag == "debug" and not self.chk_debug.isChecked():
            return
        # LogView caps itself at 4000 blocks, so nothing is trimmed by hand.
        self.log_view.add_line(f"{stamp}  {message}", self.log_colour(tag))

    def log_text(self) -> str:
        return self.log_view.toPlainText()

    def show_log(self, visible: bool) -> None:
        """The log shares a splitter with the cards, so it can be dragged."""
        self._log_visible = bool(visible)
        self.log_panel.setVisible(self._log_visible)
        self.btn_log.setText("Hide log" if self._log_visible else "Log")
        if self._log_visible:
            weight = max(1, int(self.settings.data["ui"].get("log_weight", 2) or 2))
            total = max(self.splitter.height(), 520)
            share = min(max(total * weight // (weight + 4), 180), total - 240)
            self.splitter.setSizes([total - share, share])

    def toggle_log(self) -> None:
        self.show_log(not self._log_visible)

    def apply_log_wrap(self) -> None:
        self.log_view.set_wrap(self.chk_wrap.isChecked())

    def copy_log(self) -> None:
        text = self.log_text()
        clipboard = QGuiApplication.clipboard()
        if text and clipboard is not None:
            clipboard.setText(text)

    def save_log_as(self) -> None:
        path, _chosen = QFileDialog.getSaveFileName(
            self,
            "Save log",
            str(self.dir / f"Run_{time.strftime('%Y%m%d-%H%M%S')}.log"),
            "Log files (*.log);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self.log_text() + "\n", encoding="utf-8")
        except OSError as exc:
            self.log(f"could not save the log: {exc}", "error")
            return
        self.log(f"log saved to {path}")

    def open_log_folder(self) -> None:
        target = self._log_dir()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        if target.exists():
            open_in_explorer(target)

    def clear_log(self) -> None:
        self.log_view.clear()
