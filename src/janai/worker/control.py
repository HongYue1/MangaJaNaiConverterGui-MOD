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

from janai.worker.events import log

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
        """Read control words until the interface closes the pipe.

        A dead reader is reported rather than swallowed: losing this thread
        means Cancel and Pause quietly stop working for the rest of the job,
        with nothing in the run log to say why. Retrying is not an option --
        a text stream that raised mid-line reports EOF on every later read --
        so the only useful response is to say the abort path is gone.

        The decode error that could reach here is already prevented upstream:
        ``worker.main()`` reconfigures stdin with ``errors="replace"`` before
        the job starts. This handler is what happens if that ever fails.
        """
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
        except Exception as exc:
            log(
                "control reader stopped; cancel and pause are no longer available "
                f"({type(exc).__name__}: {exc})",
                "error",
            )

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
        if not callable(fn):
            return
        try:
            fn()
        except Exception as exc:
            # The local flags are already set, so the job still aborts at its
            # own gates; this only means the backend token never heard about
            # it, which is worth seeing when an abort looks slow.
            log(f"progress token {name}() failed: {type(exc).__name__}: {exc}", "warn")

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
