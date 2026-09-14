"""Does one undecodable stdin byte kill Control._pump?

Run it twice from the repo root, with and without --reconfigure, feeding an
undecodable byte followed by "cancel":

    printf '\\x81\\ncancel\\n' | backend/python/python.exe scripts/stdin_check.py
    printf '\\x81\\ncancel\\n' | backend/python/python.exe scripts/stdin_check.py --reconfigure

--reconfigure reproduces what worker.main() does before run_job() starts the
pump, so it is the production configuration. Gitignored scratch, not a test.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from janai.worker.control import CTRL

if "--reconfigure" in sys.argv:
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

print(f"stdin encoding={sys.stdin.encoding} errors={sys.stdin.errors}", flush=True)

CTRL.start()
deadline = time.time() + 5.0
while time.time() < deadline and not CTRL.cancelled:
    time.sleep(0.05)

pump_alive = any(t.name == "stdin" and t.is_alive() for t in threading.enumerate())
print(f"CANCELLED: {CTRL.cancelled}", flush=True)
print(f"PUMP_ALIVE: {pump_alive}", flush=True)
