"""Hardware identity for the tile profile.

A profile is a set of measurements taken on one machine, so it is only worth
trusting while that machine is still the same one. This module turns the
worker's probe - which the app already holds at startup, and which costs no
GPU work - into a short key, so "has the hardware changed?" is a string
comparison rather than another benchmark.

Deliberately ignores everything that changes without the hardware changing:
driver build numbers, how much VRAM happens to be free at the moment of
asking, and the index a device was handed this session.

No torch, no Qt: the app imports this during startup.
"""

from __future__ import annotations

import hashlib
from typing import Any

#: Bumped when a measurement's meaning changes, so a stored profile written by
#: an older build is re-taken rather than misread.
PROFILE_VERSION = 1


def _devices(probe: Any) -> list[dict[str, Any]]:
    if not isinstance(probe, dict):
        return []
    return [d for d in (probe.get("devices") or []) if isinstance(d, dict)]


def accelerators(probe: Any) -> list[dict[str, Any]]:
    """The compute devices that are not the CPU, in the order probed."""
    return [d for d in _devices(probe) if str(d.get("value") or "") != "cpu"]


def device_name(device: dict[str, Any]) -> str:
    """The card's name without the index the session gave it.

    Labels arrive as "NVIDIA GeForce RTX 3060 (cuda:0)". A card that merely
    moved index is the same card, so the suffix is dropped.
    """
    label = str(device.get("label") or device.get("value") or "?")
    return label.split(" (", maxsplit=1)[0].strip() or "?"


def fingerprint(probe: Any) -> str:
    """A short key that changes when the GPUs or the torch build change.

    Returns "" when the probe says nothing useful. An empty key never matches
    and never mismatches, so a missing probe cannot be mistaken for a hardware
    change and cannot invalidate a good profile.
    """
    gpus = accelerators(probe)
    if not gpus:
        return ""
    parts = [f"v{PROFILE_VERSION}"]
    parts.extend(f"{device_name(d)}|{int(d.get('vram') or 0)}" for d in gpus)
    if isinstance(probe, dict):
        # A torch or CUDA upgrade changes both memory use and speed, so it
        # invalidates the measurements even on identical silicon.
        parts.append(f"torch={probe.get('torch') or '?'}")
        parts.append(f"cuda={probe.get('cuda') or ''}")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def describe(probe: Any) -> str:
    """The hardware in one short phrase, for a status line."""
    gpus = accelerators(probe)
    if not gpus:
        return "CPU only"
    extra = f" +{len(gpus) - 1}" if len(gpus) > 1 else ""
    return f"{device_name(gpus[0])}{extra}"


def profile_models(profile: Any) -> dict[str, Any]:
    """The per-model measurements in a stored profile, or an empty dict."""
    if not isinstance(profile, dict):
        return {}
    models = profile.get("models")
    return models if isinstance(models, dict) else {}


def profile_is_current(profile: Any, probe: Any) -> bool:
    """Whether `profile` was measured on the machine `probe` describes.

    A profile that does not match is not deleted: it is simply not used, and
    the app offers to measure again. Keeping it means a card swapped back
    still has its numbers.
    """
    if not profile_models(profile):
        return False
    if int(profile.get("version") or 0) != PROFILE_VERSION:
        return False
    key = fingerprint(probe)
    if not key:
        # Nothing to compare against yet: trust what was measured rather than
        # discarding it because the probe has not arrived.
        return True
    return str(profile.get("fingerprint") or "") == key


def profile_for_run(profile: Any, probe: Any, fp16: bool) -> dict[str, Any]:
    """The measurements a run may use, or an empty dict.

    FP16 and FP32 have different per-pixel costs - the element size is the
    whole difference - so measurements taken in one precision are not handed
    to a run using the other.
    """
    if not profile_is_current(profile, probe):
        return {}
    if bool(profile.get("fp16", fp16)) != bool(fp16):
        return {}
    return profile_models(profile)
