"""Where this install keeps its runtime, models, backend source and tools.

Importing this module has side effects, deliberately. It resolves the layout
once, inserts `PATHS.import_paths()` at the front of `sys.path` so the vendored
backend under backend/src is importable, and puts the bundled command line
tools ahead of anything already on PATH. Every consumer of the heavy stack
depends on that having happened, so it runs at import time rather than on
demand.

This exists as its own module because `PATHS` has consumers all over the worker
(tool discovery, ICC profiles, the probe payload, the models directory). While
it lived in `worker.py`, anything needing it had to import `worker.py` back,
which is the import cycle the Phase 1 split exists to avoid.

`worker.py` still has to insert the import root itself before it can import
this module: run as a script, `sys.path[0]` is `src/janai/worker`, so `janai`
is not importable at all until `src` is in front. That inline block is why
`worker.py` carries the ruff E402 per-file ignore.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from janai.core import paths as _paths

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1]  # <app folder>/src, the import root
ROOT = SRC.parent  # the app folder itself

# The runtime, the models, the backend source and the ICC profiles all live in
# backend/, unless janai.config.json points somewhere else. janai.core.paths
# works that out once, here.
PATHS = _paths.resolve(ROOT)
MODELS_DIR = PATHS.models_dir or (ROOT / "backend" / "models")

for _p in reversed(PATHS.import_paths()):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Bundled command line tools (cjxl, djxl) win over anything already on PATH.
if PATHS.tools_dir:
    os.environ["PATH"] = f"{PATHS.tools_dir}{os.pathsep}{os.environ.get('PATH', '')}"
