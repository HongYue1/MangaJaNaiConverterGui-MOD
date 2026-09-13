"""The worker's half of the wire protocol: one JSON object per line of stdout.

``emit`` is the only writer of this process's stdout and it holds a lock across
the write and the flush. Stdout here is a wire format, not a console: the
interface treats every line starting with ``{`` as an event
(``janai/app/runner.py``), so two threads writing directly would interleave into
lines it cannot decode, and a stray ``print`` would arrive as a protocol
violation. Diagnostics go through ``log``, which is just an ``emit`` of a ``log``
event.

The event catalogue itself is documented in the module docstring of
``worker.py``, next to the CLI that produces it.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any

_stdout_lock = threading.Lock()


def emit(kind: str, **payload: Any) -> None:
    payload["type"] = kind
    try:
        line = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:  # pragma: no cover - defensive
        line = json.dumps({"type": "log", "level": "warn", "message": "unserialisable event"})
    with _stdout_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def log(message: Any, level: str = "info") -> None:
    emit("log", level=level, message=str(message))
