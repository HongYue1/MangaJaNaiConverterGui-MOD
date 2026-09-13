"""JaNai Upscaler - GUI entry point.

Run via JaNaiUpscaler.cmd, or directly:
    backend\\python\\Scripts\\pythonw.exe -m janai

The interface is Qt (PySide6). Nothing heavier is imported here, and nothing
heavier is imported by the window either: torch and friends only ever live
inside the worker process, which is started on demand.

Everything in this file is work that has to happen before any widget exists -
the display-scaling policy, the identity Windows uses for the taskbar, and a
readable message if Qt is not installed yet.

Environment:
    JANAI_SMOKE=1   build the whole window, then close it (used for self-tests)
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve()
SRC = _HERE.parents[2]  # <app folder>/src, the import root
ROOT = _HERE.parents[3]  # the app folder itself

# Running this file directly puts src/janai/app on sys.path, not src, so the
# package would not be importable. Fix that before importing anything of ours.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

ICON_NAMES = ("logo.png", "logo.ico", "assets/logo.png")

NO_QT = (
    "JaNai Upscaler needs Qt (PySide6) and this interpreter does not have it.\n\n"
    "Run setup.cmd (Windows) or ./setup.sh (Linux) in the app folder, or install\n"
    "it by hand:\n\n"
    "    python -m pip install PySide6-Essentials\n"
)


def fatal(message: str) -> None:
    """Say it on stderr, and in a dialog if Qt got far enough to show one."""
    sys.stderr.write(message + "\n")
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        existing = QApplication.instance()
        app = existing or QApplication(sys.argv)
        QMessageBox.critical(None, "JaNai Upscaler", message)
        if existing is None:
            app.quit()
    except Exception:
        pass


def declare_windows_identity() -> None:
    """Give Windows an app id, so the taskbar groups and pins this window."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("JaNai.Upscaler")
    except Exception:
        pass


def find_icon() -> Path | None:
    for name in ICON_NAMES:
        candidate = ROOT / name
        if candidate.exists():
            return candidate
    return None


def saved_theme() -> str:
    """The palette the window was last left in, read before it is built."""
    try:
        from janai.app.state import Settings

        return str(Settings(ROOT / "settings.json").load().data.get("theme") or "dark")
    except Exception:
        return "dark"


def main() -> int:
    declare_windows_identity()

    try:
        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtGui import QIcon
        from PySide6.QtWidgets import QApplication
    except Exception as exc:  # pragma: no cover - depends on the install
        sys.stderr.write(f"{NO_QT}\n{exc}\n")
        return 2

    # Fractional display scales (125%, 150%, 175%) are passed through rather
    # than rounded, so a 150% display gets 150% widgets and text stays crisp.
    # Qt does the scaling itself, which is the thing the Tk build could never
    # do; it is set explicitly here because it matters enough to be stated.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("JaNai Upscaler")
    app.setApplicationDisplayName("JaNai Upscaler")
    app.setOrganizationName("JaNai")
    app.setDesktopFileName("janai-upscaler")  # how Wayland/X11 find the icon

    icon = find_icon()
    if icon is not None:
        app.setWindowIcon(QIcon(str(icon)))

    try:
        from janai.app.theme import Theme
        from janai.app.window import MainWindow
    except Exception:
        fatal("Could not load the interface:\n\n" + traceback.format_exc())
        return 2

    theme = Theme(saved_theme())
    theme.apply(app)  # one stylesheet for the whole application

    try:
        window = MainWindow(ROOT, theme, app)
    except Exception:
        fatal("JaNai Upscaler failed to start:\n\n" + traceback.format_exc())
        return 1

    window.show()
    window.raise_()
    window.activateWindow()

    if os.environ.get("JANAI_SMOKE"):
        QTimer.singleShot(900, app.quit)

    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
