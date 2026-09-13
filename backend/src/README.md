# backend/src — vendored chaiNNer-derived backend

These modules are vendored (chaiNNer, via MangaJaNaiConverterGui) and are
imported by the upscaler worker, `src/janai/worker/worker.py`. `setup.sh`
refuses to run when this directory is missing, so it is not optional. They are
excluded from the project's ruff config on purpose: vendored code is not held to
this app's style, so future upstream syncs stay reviewable.

## There is no CLI in here any more

This file used to document `run_upscale.py` (`python run_upscale.py -f ...`,
`python run_upscale.py --settings appstate2.json`). That script was the
chaiNNer-era entry point and had **no caller left**: the app, the launchers,
`setup.*` and CI all drive `src/janai/worker/worker.py` instead, and CI only
byte-compiled it. It was deleted rather than kept as a misleading second copy of
the pipeline; recover it from git history if the old behaviour is ever needed.

To run the upscaler without the GUI, use the worker's own CLI (embedded
interpreter, from the repo root):

```bash
backend/python/python.exe src/janai/worker/worker.py --probe
backend/python/python.exe src/janai/worker/worker.py --job job.json
backend/python/python.exe src/janai/worker/worker.py --job job.json --dry-run
```

Models live in `backend/models`. See `AGENTS.md` at the repo root for the
architecture map, the JSON Lines event protocol the worker emits, and the
verification commands.
