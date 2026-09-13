"""Cancel and pause, fed by the interface as bare words on stdin.

The abort path is cooperative and that is deliberate. ``Control`` only sets
flags and mirrors them into the backend's progress token; the pipeline notices
at its own gates and raises ``Cancelled``. Nothing here kills a thread, because
a torch forward pass or a libvips write cannot be interrupted safely halfway
through.

Two properties of ``gate()`` matter:

* it returns immediately once cancelled, so "Cancel" still works while the job
  is paused -- waiting on the pause event instead would make cancel mean "resume
  first, then stop";
* it polls rather than blocks, so no stage can be parked in an uninterruptible
  wait with no abort path.

The stdin reader is a daemon thread: the process must be able to exit even if
the interface never closes the pipe.
"""

from __future__ import annotations

import sys
import threading
import time

PAUSE_POLL_SECONDS = 0.05
"""How often a paused stage re-checks the flags. Short enough that resume and
cancel feel instant, long enough not to spin a core while paused."""


class Control:
    """Cancel/pause flags fed by stdin, mirrored into the backend progress token."""

    def __init__(self) -> None:
        self._cancel = threading.Event()
        self._pause = threading.Event()
        self.progress = None  # backend ProgressController, attached later

    def start(self) -> None:
        threading.Thread(target=self._pump, name="stdin", daemon=True).start()

    def _pump(self) -> None:
        try:
            for raw in sys.stdin:
                cmd = raw.strip().lower()
                if not cmd:
                    continue
                if cmd in ("cancel", "abort", "stop"):
                    self.cancel()
                elif cmd == "pause":
                    self.pause()
                elif cmd == "resume":
                    self.resume()
        except Exception:
            pass

    def cancel(self) -> None:
        self._cancel.set()
        self._pause.clear()
        self._call("abort")
        self._call("resume")

    def pause(self) -> None:
        self._pause.set()
        self._call("pause")

    def resume(self) -> None:
        self._pause.clear()
        self._call("resume")

    def _call(self, name: str) -> None:
        fn = getattr(self.progress, name, None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def paused(self) -> bool:
        return self._pause.is_set()

    def gate(self) -> None:
        """Block while paused; returns immediately when cancelled."""
        while self._pause.is_set() and not self._cancel.is_set():
            time.sleep(PAUSE_POLL_SECONDS)


CTRL = Control()
"""Process-wide flags: one worker process serves one job, so the pipeline reads
the state from here instead of threading a handle through every stage."""


class Cancelled(Exception):
    """Raised at a pipeline gate once ``CTRL`` has been cancelled."""
