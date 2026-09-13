"""What this install can actually do: libvips operations and bundled tools.

These answers describe the installation rather than any particular image, which
is why they live apart from image IO: the probe reports them once and the GUI
uses them to decide which output formats to offer at all.

`find_tool` deliberately prefers the bundled tools folder over PATH, so a
system cjxl/djxl of some other version cannot silently change our output.

`vips_has` swallows every exception on purpose. It is also called before the
heavy stack has been loaded, when the pyvips handle is still None; the answer
then is "no", which is correct for a capability question.
"""

from __future__ import annotations

import shutil

from janai.worker import runtime
from janai.worker.environment import PATHS


def vips_has(op: str) -> bool:
    try:
        return bool(runtime.pyvips.type_find("VipsOperation", op))
    except Exception:
        return False


def find_tool(name: str) -> str:
    """A tool bundled in the tools folder, else whatever PATH offers."""
    if PATHS.tools_dir:
        for cand in (PATHS.tools_dir / f"{name}.exe", PATHS.tools_dir / name):
            if cand.is_file():
                return str(cand)
    return shutil.which(name) or ""


def find_cjxl() -> str:
    return find_tool("cjxl")


def find_djxl() -> str:
    return find_tool("djxl")


def pillow_jxl_available() -> bool:
    try:
        import pillow_jxl  # noqa: F401

        return True
    except Exception:
        return False
