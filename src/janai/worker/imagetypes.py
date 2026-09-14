"""One name for a decoded page, so every stage spells it the same way.

An ``ImageArray`` is a numpy array in HWC order -- uint8 or uint16, one or three
channels -- and it is the value that flows from the decoders through the
transforms into the encoders.

It lives in a module of its own rather than in whichever stage happened to need
it first, because the stages that pass pages around sit on both sides of the
dependency direction: the concurrency primitives treat a page as an opaque
payload, while the I/O and transform stages actually touch the pixels. Naming it
here means neither has to import the other to say "page".

Two things about it are deliberate:

* **The checker resolves it to ``Any``, by configuration rather than by
  accident.** numpy is installed only in backend/python, which mypy does not
  use (see ``ignore_missing_imports`` in pyproject.toml). So the alias buys a
  reader clarity about which parameters are page-shaped; it does not buy a
  proof. Do not "fix" that by spelling ``Any`` at every call site instead.
* **numpy is never imported at runtime just to name a type.** The import stays
  inside ``TYPE_CHECKING`` because a dry run and the ``--probe`` path must never
  pay for numpy, which is the same reason the worker imports it inside functions
  (the ``PLC0415`` ignore in pyproject.toml records that decision).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    import numpy as np

    ImageArray: TypeAlias = np.ndarray
else:
    ImageArray: TypeAlias = Any
