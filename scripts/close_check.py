"""Does closing the window actually stop a worker that ignores 'cancel'?

Run offscreen with the embedded interpreter:

    QT_QPA_PLATFORM=offscreen backend/python/python.exe scripts/close_check.py

The scenario is the real one: a live child process that never reads stdin, so
the cooperative cancel cannot reach it and only the kill backstop can end it.
Each mode drives a real QApplication event loop through a real close.

  current  replicates window.closeEvent as written: cancel, then
           QTimer.singleShot(400, runner.kill), then accept.
  fixed    calls Runner.shutdown(), the bounded wait this check exists to justify.

Expected: 'fixed' leaves nothing behind, while 'current' provably orphans the
child -- that measurement is what justifies Runner.shutdown(). A FAIL under
'fixed' means closing the window orphans a process holding VRAM. A FAIL under
'current' means the Qt premise changed and this check needs revisiting.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QWidget

from janai.app.runner import Runner

STUBBORN = Path(__file__).resolve().parent / "stubborn_child.py"
START_TIMEOUT = 10.0
SETTLE = 1.2  # generous: the 400 ms backstop has long since been due

checks: list[tuple[bool, str]] = []


def check(ok: bool, label: str) -> None:
    checks.append((bool(ok), label))
    print(f"{'PASS' if ok else 'FAIL'}  {label}")


class StubbornRunner(Runner):
    """The real Runner, pointed at a worker that ignores every control word."""

    def python_exe(self) -> Path:
        return Path(sys.executable)

    def worker_script(self) -> Path:
        return STUBBORN


def wait_running(runner: Runner) -> bool:
    deadline = time.monotonic() + START_TIMEOUT
    while time.monotonic() < deadline:
        if runner.running:
            return True
        time.sleep(0.05)
    return False


def scenario(app: QApplication, mode: str) -> None:
    runner = StubbornRunner(ROOT)
    started = runner.start({"stub": True})
    check(started and wait_running(runner), f"[{mode}] stubborn worker is running")
    pid = runner.proc.pid if runner.proc is not None else 0
    fired = {"backstop": False}

    def backstop() -> None:
        fired["backstop"] = True
        runner.kill()

    class Window(QWidget):
        def closeEvent(self, event) -> None:  # noqa: N802 - Qt's name, not ours
            runner.hold_stop()
            if mode == "current":
                if runner.running:
                    runner.cancel()
                    QTimer.singleShot(400, backstop)
            else:
                runner.shutdown()
            event.accept()

    window = Window()
    window.show()
    QTimer.singleShot(150, window.close)
    app.exec()  # returns when the last window closes

    time.sleep(SETTLE)
    alive = runner.proc is not None and runner.proc.poll() is None
    if mode == "current":
        # Both of these passing is the bug report: the deferred kill is dead
        # code, so the pattern it belongs to cannot stop a stubborn worker.
        check(
            not fired["backstop"],
            "[current] the 400 ms kill backstop never fires (event loop is gone)",
        )
        check(alive, f"[current] worker pid {pid} is therefore orphaned by the close")
    else:
        check(not alive, f"[{mode}] worker pid {pid} is gone once the window has closed")

    runner.kill()  # never leave a stray process behind, whatever the verdict
    time.sleep(0.2)


def main() -> int:
    if not STUBBORN.exists():
        print(f"missing fixture: {STUBBORN}")
        return 2
    app = QApplication.instance() or QApplication([])
    scenario(app, "current")
    if hasattr(Runner, "shutdown"):
        scenario(app, "fixed")
    else:
        print("SKIP  [fixed] Runner.shutdown() does not exist yet")
    failed = [label for ok, label in checks if not ok]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} PASS")
    for label in failed:
        print(f"  failed: {label}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
