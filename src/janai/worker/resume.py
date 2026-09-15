"""Resume state: which sources an output folder has already finished.

A run cancelled half way through a hundred chapters used to leave nothing
behind that said *which* chapters were finished. The next run had to infer it
from ``dest.exists()``, which cannot tell a chapter this job published from an
unrelated file sitting at that path -- and says nothing at all when the user
has overwrite on, so the whole folder was converted again. This module writes
that fact down instead.

Exactly two things are recorded:

* ``units`` -- a source this output folder has **finished**, keyed by
  :func:`unit_key`. Written only after the atomic ``replace()`` that publishes
  the output, so a crash can lose a completed unit's record but can never
  claim an unfinished one. Losing a record costs one redundant re-convert;
  claiming one loses a chapter, so the asymmetry is deliberate.
* ``partials`` -- the entry names already packed into a surviving
  ``.cbz.part``, in order. This is what lets "cancelled inside chapter 11"
  resume inside chapter 11 instead of at its first page.

The manifest is only trusted when its fingerprint matches the current job's
(see :func:`fingerprint`): a record saying "done" under different settings
would silently skip exactly the work the user had just changed.

This module is a leaf. It writes the one file it owns and nothing else writes
that file -- the same single-writer rule the GUI status label follows, for the
same reason: two writers of one piece of state disagree eventually.
"""

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from janai.core.fspath import io_path
from janai.worker.events import log
from janai.worker.planning import path_key

#: Lives in the output root, beside the files it describes, so moving the
#: output folder moves its resume state with it.
MANIFEST_NAME = ".janai-resume.json"

#: Bumped only when the on-disk shape changes incompatibly. A manifest with an
#: unknown version is ignored rather than guessed at.
MANIFEST_VERSION = 1

#: Enough of the digest to make an accidental collision irrelevant while
#: keeping the file readable by a human deciding whether to delete it.
FINGERPRINT_CHARS = 16

#: Keys that say *where* output goes rather than *what* it contains. They must
#: not enter the fingerprint, or resuming into the same folder by a different
#: spelling of its path would discard perfectly good progress.
LOCATION_KEYS = frozenset({"dir"})

#: How many deferred records may wait before one flush covers them all.
#: Measured rather than guessed: every flush rewrites the *whole* manifest, so
#: a caller that records once per page costs O(n^2) bytes -- 4000 pages spent
#: 13.9 s and rewrote 1412 MiB on bookkeeping alone, and doubling the pages
#: quadrupled both (`.tmp/f39b_flush_cost.py`, embedded interpreter). Batching
#: turns that into tens of MiB while risking at most this many redundant
#: re-converts after a hard kill -- the cheap side of the asymmetry above.
FLUSH_EVERY_RECORDS = 64

#: ...and no deferred record waits longer than this, so a slow job still
#: leaves usable progress behind instead of holding a batch open for minutes.
FLUSH_INTERVAL_SECONDS = 10.0


def unit_key(src: Path, base: Path) -> str:
    """Stable identity of one source file inside this job's input root.

    Relative to ``base`` so moving or renaming the input root does not orphan
    every record, and absolute only when the source lies outside it.
    """
    try:
        rel = src.relative_to(base)
    except ValueError:
        rel = src
    # `path_key` owns the case rule -- Windows treats A.cbz and a.cbz as one
    # file -- and as_posix keeps the key separator-stable inside the JSON, so a
    # manifest stays readable and comparable regardless of who wrote it.
    return path_key(Path(rel.as_posix())).replace("\\", "/")


def planned_entries(names: list[str], ext: str) -> list[str]:
    """The entry name this run would give each page, without decoding one.

    Mirrors the naming and de-dup inside `orchestrate.pack_archive`'s loop on
    purpose: its only job is to decide whether a surviving ``.part`` holds a
    prefix of *this* plan. It is a prediction, not the authority. The live loop
    names a page only once it has decoded, so a previous run that lost a page
    wrote a shorter list than this predicts, the prefix check fails, and that
    archive starts over. Restarting an archive costs time; appending to a file
    whose page order no longer matches the plan costs correctness.

    It lives in this leaf rather than beside the loop it mirrors so the
    dependency-free gate can test it: importing :mod:`janai.worker.orchestrate`
    pulls in the decoder and torch, which `scripts/smoke.py` must never do.
    """
    claimed: set[str] = set()
    planned: list[str] = []
    for k, name in enumerate(names, 1):
        entry = str(Path(name).with_suffix(ext).as_posix())
        while entry.lower() in claimed:
            entry = f"{entry[: -len(ext)]}_{k}{ext}"
        claimed.add(entry.lower())
        planned.append(entry)
    return planned


def fingerprint(job: Mapping[str, Any]) -> str:
    """Digest of the settings that decide what the output *contains*.

    ``perf`` is excluded on purpose: threads, tiling and device choice change
    how long a page takes, not which pages exist or what they hold. Output
    location is excluded for the reason given on :data:`LOCATION_KEYS`.
    Everything else in ``format``, ``upscale`` and ``output`` is included,
    because each of them can change the bytes on disk.
    """
    payload = {
        "format": job.get("format") or {},
        "upscale": job.get("upscale") or {},
        "output": {
            key: value
            for key, value in (job.get("output") or {}).items()
            if key not in LOCATION_KEYS
        },
    }
    # sort_keys so a settings dict that merely reordered is still the same job;
    # default=str so an exotic value degrades to a stable string instead of
    # raising and taking the run with it.
    text = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:FINGERPRINT_CHARS]


class ResumeLog:
    """The one writer of the resume manifest.

    Locked because its writers are not all on one thread: a finished archive
    reports from the dispatch thread, a loose page from a write-pool thread and
    a bundle from the bundle thread -- the same reason :class:`Counters` is
    locked.

    Every disk operation is best effort. A manifest is an optimisation for the
    *next* run; failing the current job because a progress note could not be
    written would trade a real conversion for a bookkeeping detail.
    """

    def __init__(self, path: Path, fp: str, *, enabled: bool = True) -> None:
        self.path = path
        self.fingerprint = fp
        self.enabled = enabled
        self._lock = threading.Lock()
        self._units: dict[str, dict[str, Any]] = {}
        self._partials: dict[str, dict[str, Any]] = {}
        self._warned = False
        self._deferred = 0
        self._last_write = time.monotonic()

    @classmethod
    def load(cls, out_dir: Path, fp: str, *, enabled: bool = True) -> "ResumeLog":
        """Read the manifest in ``out_dir``, or start an empty one.

        A manifest from another version or another fingerprint is dropped
        rather than merged: partial trust is worse than none, because it skips
        work while looking like it resumed.
        """
        state = cls(out_dir / MANIFEST_NAME, fp, enabled=enabled)
        if not enabled:
            return state
        try:
            raw = json.loads(io_path(state.path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return state
        except (OSError, ValueError) as exc:
            log(f"ignoring unreadable resume manifest: {exc}", "debug")
            return state
        if not isinstance(raw, dict) or raw.get("version") != MANIFEST_VERSION:
            log("ignoring resume manifest written by another version", "debug")
            return state
        if raw.get("fingerprint") != fp:
            log("settings changed since the last run, converting everything again", "info")
            return state
        units = raw.get("units")
        partials = raw.get("partials")
        state._units = dict(units) if isinstance(units, dict) else {}
        state._partials = dict(partials) if isinstance(partials, dict) else {}
        return state

    def is_done(self, key: str) -> bool:
        """Has this source already been finished under the current settings?"""
        if not self.enabled:
            return False
        with self._lock:
            return key in self._units

    def finished_count(self) -> int:
        """How many sources the manifest already accounts for."""
        with self._lock:
            return len(self._units)

    def partial_entries(self, key: str) -> list[str]:
        """Entry names already packed into this source's surviving ``.part``."""
        if not self.enabled:
            return []
        with self._lock:
            record = self._partials.get(key) or {}
        entries = record.get("entries")
        return [str(name) for name in entries] if isinstance(entries, list) else []

    def mark_done(self, key: str, out: Path, entries: int = 0, *, defer: bool = False) -> None:
        """Record a finished source. Call this *after* the publishing rename.

        ``defer`` batches the disk write for callers that fire once per *page*
        instead of once per chapter -- see :data:`FLUSH_EVERY_RECORDS` for the
        measurement that made batching necessary. Deferring can only delay a
        record, never claim one early, so it stays on the safe side of the
        asymmetry in this module's docstring; the price is that the job must
        call :meth:`flush` before it exits or the tail of the batch is lost.
        Chapters do not defer: they are rare and each one is worth minutes.
        """
        if not self.enabled:
            return
        with self._lock:
            self._units[key] = {
                "out": str(out),
                "entries": int(entries),
                "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._partials.pop(key, None)
            if not defer:
                self._write()
                return
            self._deferred += 1
            if (
                self._deferred >= FLUSH_EVERY_RECORDS
                or time.monotonic() - self._last_write >= FLUSH_INTERVAL_SECONDS
            ):
                self._write()

    def mark_partial(self, key: str, tmp: Path, entries: Sequence[str]) -> None:
        """Record how far an unfinished archive got, and where its ``.part`` is."""
        if not self.enabled:
            return
        with self._lock:
            self._partials[key] = {"tmp": str(tmp), "entries": [str(name) for name in entries]}
            self._write()

    def drop(self, key: str) -> None:
        """Forget a source entirely, so the next run treats it as untouched."""
        if not self.enabled:
            return
        with self._lock:
            changed = self._units.pop(key, None) is not None
            changed = self._partials.pop(key, None) is not None or changed
            if changed:
                self._write()

    def flush(self) -> None:
        """Write out any deferred records.

        Called on every exit path of the job, normal or cancelled, because a
        deferred record that never reaches disk is work the next run repeats.
        Idempotent, so belt-and-braces calls are free.
        """
        if not self.enabled:
            return
        with self._lock:
            if self._deferred:
                self._write()

    def _write(self) -> None:
        """Flush the manifest. Caller holds the lock.

        Written through a ``.part`` and an atomic ``replace()`` -- the same rule
        the chapters themselves follow -- so a crash mid-flush leaves the
        previous manifest intact instead of a truncated one that would be
        thrown away on the next read.
        """
        payload = {
            "version": MANIFEST_VERSION,
            "fingerprint": self.fingerprint,
            "units": self._units,
            "partials": self._partials,
        }
        tmp = self.path.with_name(self.path.name + ".part")
        # Reset before attempting the write, not after it succeeds: a
        # read-only output folder would otherwise make every subsequent
        # deferred record retry the failing write in the hot page loop.
        self._deferred = 0
        self._last_write = time.monotonic()
        try:
            target, staging = io_path(self.path), io_path(tmp)
            target.parent.mkdir(parents=True, exist_ok=True)
            staging.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            staging.replace(target)
        except OSError as exc:
            # Once per run: a read-only output folder would otherwise repeat
            # this line for every single unit and bury the real log.
            if not self._warned:
                self._warned = True
                log(f"could not write the resume manifest: {exc}", "warn")
