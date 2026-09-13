"""Run log: one aligned, readable line per event, mirrored to a file.

The worker speaks JSONL; this turns each event into a fixed-width line so a
run reads like a table instead of a paragraph, and appends the same text to
``logs/Run_<timestamp>.log`` so there is always a record after the window is
closed.

Formatting lives here rather than in the window so the file and the on-screen
log can never drift apart.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

NAME_WIDTH = 32
TAG_WIDTH = 4


def fmt_secs(seconds: float) -> str:
    s = int(max(0, seconds))
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


def fmt_bytes(n: int) -> str:
    val = float(n or 0)
    for unit in ("B", "KB", "MB", "GB"):
        if val < 1024 or unit == "GB":
            return f"{val:.0f} {unit}" if unit == "B" else f"{val:.1f} {unit}"
        val /= 1024
    return f"{val:.1f} GB"


def fmt_ms(ms: Any) -> str:
    try:
        value = int(ms)
    except Exception:
        return ""
    if value <= 0:
        return ""
    if value < 1000:
        return f"{value}ms"
    if value < 60_000:
        return f"{value / 1000:.1f}s"
    return fmt_secs(value / 1000)


def elide(text: str, width: int = NAME_WIDTH) -> str:
    """Pad or shorten to exactly `width`, keeping the tail (the extension)."""
    text = str(text or "")
    if len(text) <= width:
        return text.ljust(width)
    tail = min(10, max(4, width // 3))
    head = width - tail - 1
    return f"{text[:head]}\u2026{text[-tail:]}"


def counter(index: Any, total: Any) -> str:
    try:
        i, n = int(index), int(total)
    except Exception:
        return ""
    if n <= 0:
        return ""
    width = len(str(n))
    return f"{i:>{width}}/{n}"


def _fields(ev: dict) -> list[str]:
    """The trailing, optional columns of a per-file line."""
    out: list[str] = []
    w, h = ev.get("w"), ev.get("h")
    if w and h:
        out.append(f"{int(w)}\u00d7{int(h)}")
    if ev.get("gray") is not None:
        out.append("gray" if ev.get("gray") else "colour")
    if ev.get("bytes"):
        out.append(fmt_bytes(int(ev["bytes"])))
    took = fmt_ms(ev.get("ms"))
    if took:
        out.append(took)
    tile = ev.get("tile")
    if isinstance(tile, int):
        if tile > 0:
            out.append(f"tile {tile}")
        elif tile == -1:
            out.append("no tiling")
        elif tile == -2:
            out.append("tile max")
    entries = ev.get("entries")
    if entries:
        out.append(f"{int(entries)} pages")
    return out


def format_file(ev: dict) -> tuple[str, str]:
    """(line, level) for a `file` event."""
    name = Path(str(ev.get("path") or "")).name or str(ev.get("out") or "")
    count = counter(ev.get("i"), ev.get("total"))
    error = str(ev.get("error") or "")
    dry = bool(ev.get("dry"))
    if error:
        skipped = "skip" in error.lower() or "exists" in error.lower()
        tag, level = ("skip", "skip") if skipped else ("fail", "error")
        body = error
    else:
        tag, level = ("plan", "dry") if dry else ("ok", "ok")
        body = "  ".join(_fields(ev))
    parts = [tag.ljust(TAG_WIDTH), count.rjust(9), elide(name), body]
    entry = str(ev.get("entry") or "")
    if entry and not error:
        parts.append(f"\u2192 {entry}")
    model = str(ev.get("model") or "")
    if model and not error:
        parts.append(f"[{model}]")
    return "  ".join(p for p in parts if p).rstrip(), level


def format_bundle(ev: dict) -> tuple[str, str]:
    """(line, level) for a `bundle` event: one archive finished or planned."""
    name = Path(str(ev.get("out") or "")).name
    planned = bool(ev.get("planned"))
    tag = "pack" if not planned else "plan"
    bits = [f"{int(ev.get('entries') or 0)} pages"]
    if ev.get("bytes"):
        bits.append(fmt_bytes(int(ev["bytes"])))
    took = fmt_ms(ev.get("ms"))
    if took:
        bits.append(took)
    if ev.get("failed"):
        bits.append(f"{int(ev['failed'])} failed")
    line = "  ".join([tag.ljust(TAG_WIDTH), "".ljust(9), elide(name), "  ".join(bits)])
    return line.rstrip(), ("dry" if planned else "ok")


def format_start(ev: dict) -> list[tuple[str, str]]:
    """Header lines for a `start` event."""
    dry = bool(ev.get("dry"))
    bits = [
        f"{int(ev.get('total') or 0)} item(s)",
        str(ev.get("format") or "").upper(),
        str(ev.get("container") or "files"),
        str(ev.get("device") or "auto"),
        "FP16" if ev.get("fp16") else "FP32",
        f"tile {str(ev.get('tile') or 'auto').lower()}",
    ]
    if ev.get("bundles"):
        bits.append(f"{int(ev['bundles'])} archive(s)")
    head = "dry run" if dry else "run"
    lines = [(f"{head}: " + "  \u00b7  ".join(b for b in bits if b), "info")]
    if ev.get("out_dir"):
        lines.append((f"    output: {ev['out_dir']}", "info"))
    return lines


def format_done(ev: dict) -> tuple[str, str]:
    processed = int(ev.get("processed") or 0)
    failed = int(ev.get("failed") or 0)
    skipped = int(ev.get("skipped") or 0)
    bits = [f"{processed} done"]
    if skipped:
        bits.append(f"{skipped} skipped")
    if failed:
        bits.append(f"{failed} failed")
    bits.append(fmt_secs(float(ev.get("elapsed") or 0)))
    head = (
        "cancelled"
        if ev.get("cancelled")
        else ("dry run finished" if ev.get("dry") else "finished")
    )
    level = "error" if failed or not ev.get("ok") else ("dry" if ev.get("dry") else "ok")
    if ev.get("cancelled"):
        level = "warn"
    text = f"{head}: " + "  \u00b7  ".join(bits)
    if ev.get("error"):
        text += f"  \u2014 {ev['error']}"
    return text, level


class RunLog:
    """Mirrors the on-screen log into ``logs/Run_<timestamp>.log``.

    Opened when a run starts and closed when it ends, so every run leaves
    exactly one file behind. Failures to write are swallowed: a log that
    cannot be saved must never take a job down with it.
    """

    def __init__(self, log_dir: Path, keep: int = 30, enabled: bool = True) -> None:
        self.dir = Path(log_dir)
        self.keep = max(0, int(keep or 0))
        self.enabled = bool(enabled)
        self.path: Path | None = None
        self._fh: Any = None

    # -- lifecycle --------------------------------------------------------- #
    def begin(self, header: Iterable[str] = (), dry: bool = False) -> Path | None:
        self.end()
        if not self.enabled:
            return None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            name = f"Run_{stamp}{'_dry' if dry else ''}.log"
            path = self.dir / name
            n = 2
            while path.exists():  # two runs inside one second
                path = self.dir / f"Run_{stamp}_{n}{'_dry' if dry else ''}.log"
                n += 1
            self._fh = path.open("w", encoding="utf-8", newline="\n")
            self.path = path
            for line in header:
                self.raw(line)
            self.raw("-" * 78)
            self.prune()
            return path
        except Exception:
            self._fh = None
            self.path = None
            return None

    def end(self, footer: Iterable[str] = ()) -> None:
        if self._fh is None:
            return
        try:
            for line in footer:
                self.raw(line)
            self._fh.flush()
            self._fh.close()
        except Exception:
            pass
        finally:
            self._fh = None

    # -- writing ----------------------------------------------------------- #
    def raw(self, text: str) -> None:
        if self._fh is None:
            return
        try:
            self._fh.write(f"{text}\n")
        except Exception:
            pass

    def write(self, stamp: str, text: str) -> None:
        self.raw(f"{stamp}  {text}")
        if self._fh is not None:
            try:
                self._fh.flush()  # a crash mid-run should still leave the log
            except Exception:
                pass

    # -- housekeeping ------------------------------------------------------ #
    def prune(self) -> None:
        if not self.keep:
            return
        try:
            runs = sorted(self.dir.glob("Run_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            for old in runs[self.keep :]:
                old.unlink(missing_ok=True)
        except Exception:
            pass
