"""Handing Windows a path it will actually accept.

Windows refuses a path longer than ``MAX_PATH`` unless the caller opts in, and
the only opt-in a portable app can rely on is the extended-length ``\\\\?\\``
prefix: CPython passes such a path straight to the wide Win32 APIs, so it works
whether or not ``HKLM\\...\\Control\\FileSystem\\LongPathsEnabled`` is set on the
machine. That switch is off by default, and ``keep_structure=True`` mirrors a
deep source tree underneath the output folder, so a real manga library reaches
260 characters without trying.

Two rules keep this from causing more trouble than it cures:

* **Prefix only at the file system boundary.** ``\\\\?\\C:\\...`` is a valid path
  but an ugly one, and :mod:`janai.worker.planning` de-dupes output names by
  comparing path *text*. A prefixed string must never reach an event payload,
  the run log or ``path_key``, or the interface shows the prefix to the user and
  a name this run already claimed stops matching itself.
* **Prefix only a path that needs it.** The prefix also switches *off* the
  normalisation Windows usually applies: no ``..`` collapsing, no ``/`` to
  ``\\`` translation, no trailing dot or space trimming. The output folder
  arrives from a JSON job file that a human may have typed, so below the
  threshold the path is returned untouched and an ordinary run behaves exactly
  as it always did.

Nothing here raises. ``io_path`` is used at ``exists()`` sites too, where an
exception would abandon the whole run rather than the one unit that is too
long. The one limit no prefix can lift -- a single name longer than the file
system allows, which is not a ``MAX_PATH`` problem -- is reported by
``path_too_long`` so the caller can fail that unit through its existing error
path, with a message that names the culprit.

Standard library only: this sits in ``core`` because it is about paths rather
than about upscaling, and ``core`` may not import the worker's dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path

LONG_PATH_PREFIX = "\\\\?\\"
"""Extended-length prefix: tells Win32 to skip parsing and the MAX_PATH check."""

UNC_LONG_PATH_PREFIX = "\\\\?\\UNC"
"""The same opt-in for a network path: ``\\\\server\\share`` becomes
``\\\\?\\UNC\\server\\share``, because the prefix replaces the leading slash."""

DEVICE_PREFIX = "\\\\.\\"
"""Device namespace. Already unparsed, so it is left exactly as it arrived."""

MAX_PATH = 260
"""Windows' classic limit for a whole path, counting the terminating NUL."""

PREFIX_ABOVE = 248
"""Prefix a path at or above this length.

``CreateDirectory`` stops twelve characters short of ``MAX_PATH`` because it
reserves room for an 8.3 name, so the directory limit bites before the file one.
Using the stricter of the two for both means a folder we are about to create is
covered by the length of the file inside it.
"""

MAX_COMPONENT = 255
"""Longest single file or folder name NTFS accepts.

Measured, not assumed: with the prefix applied, a 300-character name still
fails with ``OSError: [Errno 22] Invalid argument``. This is the file system's
own limit, so it is the residue the prefix cannot rescue.
"""


def io_path(path: Path) -> Path:
    """The form of *path* to hand to the file system. Never raises.

    Returns the path unchanged on anything but Windows, on a path that is
    already unparsed, and on any path short enough not to need help -- which is
    every path in an ordinary run.
    """
    if os.name != "nt":
        return path
    text = str(path)
    if text.startswith((LONG_PATH_PREFIX, DEVICE_PREFIX)):
        return path
    if len(text) < PREFIX_ABOVE:
        return path
    # Normalise BEFORE prefixing: the prefix freezes the path exactly as given,
    # so a "C:/deep/../deep" typed into the job file has to be resolved here or
    # it reaches the API literally and fails.
    #
    # abspath rather than Path.resolve: resolve() touches the disk -- it follows
    # symlinks and asks Windows for the final name -- and this path does not
    # exist yet and may already be past MAX_PATH, which is the very thing being
    # worked around. The normalisation here has to stay purely lexical.
    absolute = os.path.abspath(text)  # noqa: PTH100
    if absolute.startswith("\\\\"):
        return Path(UNC_LONG_PATH_PREFIX + absolute[1:])
    return Path(LONG_PATH_PREFIX + absolute)


def path_too_long(path: Path) -> str | None:
    """A message naming the component no prefix can save, or ``None``.

    Callers use this to fail one unit with something a user can act on, rather
    than letting a bare ``[Errno 22] Invalid argument`` stand in for "this
    chapter's name is too long".
    """
    if os.name != "nt":
        return None
    # Lexical normalisation only, for the reason given in io_path.
    for part in Path(os.path.abspath(str(path))).parts:  # noqa: PTH100
        if len(part) > MAX_COMPONENT:
            return (
                f"{part[:40]}... is {len(part)} characters long; Windows allows at most "
                f"{MAX_COMPONENT} in a single file or folder name"
            )
    return None
