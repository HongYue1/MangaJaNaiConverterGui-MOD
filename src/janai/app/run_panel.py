"""The footer, the worker's event stream, and everything Start touches.

This is one feature, not three. Pressing Start collects every card's widgets
into a job, resets the counters and spawns the worker; the worker's JSONL events
then land here and are rendered into the very widgets this module creates.
Splitting the footer from the handlers would put ``render_status`` in a different
file from the label it writes and the counters it reads - which is how that
label came to have two writers fighting over it in the first place.

Invariant: ``render_status`` is the only writer of the *progress figure* on
``lbl_status``. The other methods here write terminal states ("Done", "Failed",
"Paused") and never a count, so the running number has exactly one source. Its
docstring records why that number is derived from the finished counters rather
than taken from the worker's index.
"""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtWidgets import QHBoxLayout, QProgressBar, QVBoxLayout, QWidget

from janai.app.fields import tile_value
from janai.app.runlog import (
    fmt_secs,
    format_bundle,
    format_done,
    format_file,
    format_start,
)
from janai.app.runner import open_in_explorer
from janai.app.widgets import button, label
from janai.core import displays
from janai.core.formats import CONTAINERS, FORMATS


class RunPanelMixin:
    """The run lifecycle: the footer, the worker's events, and the run log."""

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _build_footer(self) -> QWidget:
        foot = QWidget(self)
        box = QVBoxLayout(foot)
        box.setContentsMargins(18, 8, 18, 12)
        box.setSpacing(8)

        self.bar = QProgressBar(foot)
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        box.addWidget(self.bar)

        line = QHBoxLayout()
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(10)
        status = QVBoxLayout()
        status.setContentsMargins(0, 0, 0, 0)
        status.setSpacing(1)
        self.lbl_status = label("Ready", "field")
        self.lbl_detail = label("", "hint")
        status.addWidget(self.lbl_status)
        status.addWidget(self.lbl_detail)
        line.addLayout(status, 1)

        self.btn_log = button("Log", self.toggle_log, variant="ghost", tip="Ctrl+L")
        self.btn_open = button(
            "Open output", self.open_output, variant="ghost", tip="Show the output folder."
        )
        self.btn_dry = button(
            "Dry run",
            self.start_dry,
            variant="ghost",
            tip=(
                "Walks the whole job and reports every file it would write, the size, "
                "the model and the archives it would build \u2014 without touching "
                "the disk or the GPU.  (Ctrl+Shift+Enter)"
            ),
        )
        self.btn_pause = button("Pause", self.toggle_pause)
        self.btn_start = button("Start", self.on_start_clicked, variant="accent", tip="Ctrl+Enter")
        self.btn_start.setMinimumWidth(110)
        for btn in (self.btn_log, self.btn_open, self.btn_dry, self.btn_pause, self.btn_start):
            line.addWidget(btn)
        box.addLayout(line)
        return foot

    # ------------------------------------------------------------------ #
    # what the footer allows right now
    # ------------------------------------------------------------------ #
    def update_start_state(self) -> None:
        ok = bool(self._in_path.strip()) and self.resolved_out_dir() is not None
        # An empty table means no model would run, so there is nothing to start.
        ok = ok and any(r.enabled for r in self.rules)
        running = self.runner.running
        if running:
            self.btn_start.setText("Cancel")
            self.btn_start.setProperty("variant", "")
            self.btn_start.setEnabled(True)
            self.btn_pause.setEnabled(not self.dry)
            self.btn_dry.setEnabled(False)
        else:
            self.btn_start.setText("Start")
            self.btn_start.setProperty("variant", "accent")
            self.btn_start.setEnabled(ok)
            self.btn_dry.setEnabled(ok)
            self.btn_pause.setEnabled(False)
            self.btn_pause.setText("Pause")
        style = self.btn_start.style()
        style.unpolish(self.btn_start)
        style.polish(self.btn_start)
        self.btn_open.setEnabled(self.last_out_dir is not None)

    # ------------------------------------------------------------------ #
    # worker events
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        """Drain whatever the worker said since the last frame."""
        self.runner.drain(self.on_event)

    def on_event(self, event: dict) -> None:
        kind = str(event.get("type") or "")
        if kind == "scan":
            self.on_scan(event)
        elif kind == "probe":
            self.apply_probe(event)
        elif kind == "probe_error":
            self.on_probe_error(event)
        elif kind == "start":
            self.on_job_start(event)
        elif kind == "progress":
            self.on_progress(event)
        elif kind == "file":
            self.on_file(event)
        elif kind == "bundle":
            text, level = format_bundle(event)
            self.log(text, level)
        elif kind == "profile":
            self.on_profile(event)
        elif kind == "profile_progress":
            self.on_profile_progress(event)
        elif kind == "log":
            self.log(str(event.get("message", "")), str(event.get("level", "info")))
        elif kind == "done":
            self.on_done(event)
        elif kind == "hold":
            self.on_hold(event)
        elif kind == "exit":
            self.on_exit(event)

    def on_scan(self, event: dict) -> None:
        if str(event.get("path") or "") != self._in_path:
            return
        kind = str(event.get("kind") or "")
        images = int(event.get("images") or 0)
        archives = int(event.get("archives") or 0)
        folders = int(event.get("folders") or 0)
        if kind == "missing":
            self.card_input.set_badge("not found")
            self.scan_text = ""
            self.update_start_state()
            return
        parts = []
        if images:
            parts.append(f"{images} image{'s' if images != 1 else ''}")
        if archives:
            parts.append(f"{archives} archive{'s' if archives != 1 else ''}")
        if folders > 1:
            parts.append(f"{folders} folders")
        self.scan_text = ", ".join(parts) or "nothing to do"
        head = "Single" if kind == "single" else "Bulk"
        self.card_input.set_badge(f"{head} \u00b7 {self.scan_text}")
        self.update_summary()

    def on_job_start(self, event: dict) -> None:
        self.total = int(event.get("total") or 0)
        self.dry = bool(event.get("dry")) or self.dry
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.started_index = 0
        self.progress_sub = ""
        self.render_status()
        self.lbl_detail.setText(
            f"{event.get('device')} \u00b7 "
            f"{'FP16' if event.get('fp16') else 'FP32'} \u00b7 "
            f"tile {event.get('tile')}"
        )
        for text, level in format_start(event):
            self.log(text, level)

    def render_status(self, total: int = 0) -> None:
        """Write the status line from one place.

        Two counters used to fight over this label: the worker's progress
        index, which counts the files it has *started*, and the finished count.
        The worker reads, upscales and writes on separate threads, so by the
        time a file is finished it has already announced the next one or two -
        which made the number jump forward and then fall back a moment later.
        Show the oldest file still in flight instead (finished + 1, never past
        what the worker has actually started): that only ever moves forward.
        """
        total = total or self.total or 0
        finished = self.completed + self.failed + self.skipped
        current = min(finished + 1, self.started_index) if self.started_index else finished
        current = max(current, finished)
        if total:
            current = min(current, total)
        elapsed = max(0.001, time.time() - self.started_at)
        rate = self.completed / elapsed if self.completed else 0.0
        left = max(0, total - finished)
        eta = f" \u00b7 ETA {fmt_secs(left / rate)}" if rate > 0 and left > 0 else ""
        head = "Planning" if self.dry else "Upscaling"
        self.lbl_status.setText(f"{head} {current}/{total}{self.progress_sub}{eta}")

    def on_progress(self, event: dict) -> None:
        name = Path(str(event.get("path", ""))).name
        index = int(event.get("i") or 0)
        total = int(event.get("total") or 0)
        if total:
            self.total = total
        # keep the furthest file the worker has started, but let render_status
        # decide which number to show, so read-ahead cannot reach the label
        self.started_index = max(self.started_index, index)
        self.progress_sub = (
            f" (page {event.get('sub_i')}/{event.get('sub_n')})" if event.get("sub_n") else ""
        )
        self.render_status(total)
        self.lbl_detail.setText(name)

    def on_file(self, event: dict) -> None:
        total = int(event.get("total") or self.total or 1)
        error = str(event.get("error") or "")
        if error:
            if "skip" in error.lower() or "exists" in error.lower():
                self.skipped += 1
            else:
                self.failed += 1
        else:
            self.completed += 1
        text, level = format_file(event)
        self.log(text, level)
        finished = self.completed + self.failed + self.skipped
        self.bar.setValue(int(min(100.0, 100.0 * finished / max(1, total))))
        self.render_status(total)

    def on_done(self, event: dict) -> None:
        elapsed = float(event.get("elapsed") or (time.time() - self.started_at))
        processed = int(event.get("processed") or self.completed)
        failed = int(event.get("failed") or self.failed)
        skipped = int(event.get("skipped") or self.skipped)
        dry = bool(event.get("dry")) or self.dry
        text, level = format_done(event)
        self.log(text, level)
        self.bar.setRange(0, 100)
        if event.get("error"):
            self.lbl_status.setText("Failed")
            self.lbl_detail.setText(str(event["error"]))
        elif event.get("cancelled"):
            self.lbl_status.setText(f"Cancelled after {processed} file(s)")
            self.lbl_detail.setText(fmt_secs(elapsed))
        else:
            bits = [f"{processed} file(s) in {fmt_secs(elapsed)}"]
            if failed:
                bits.append(f"{failed} failed")
            if skipped:
                bits.append(f"{skipped} skipped")
            self.lbl_status.setText("Dry run complete" if dry else "Done")
            self.lbl_detail.setText(" \u00b7 ".join(bits))
            self.bar.setValue(100)
        self.update_start_state()
        self.settings.save()

    def on_exit(self, event: dict) -> None:
        self.update_start_state()
        self.update_wake_lock()
        if self._profiling:
            # The worker left without reporting. Whatever went wrong, the
            # control must not be left dead.
            self._profiling = False
            self.btn_profile.setEnabled(True)
            self.render_profile()
        if int(event.get("code") or 0) not in (0, 1, 2):
            self.lbl_status.setText("Worker stopped unexpectedly")
            self.log(f"worker exited with code {event.get('code')}", "error")
        self.end_run_log()

    # ------------------------------------------------------------------ #
    # settings and jobs
    # ------------------------------------------------------------------ #
    def sync_settings(self) -> None:
        """Pull every widget back into the settings dict, ready to save."""
        d = self.settings.data
        d["theme"] = self.theme.p.name
        d["input"] = {
            "path": self._in_path,
            "recursive": self.chk_recursive.isChecked(),
            "include_archives": self.chk_archives.isChecked(),
        }
        d["upscale"] = {
            "mode": str(self.seg_mode.value()),
            "scale": float(self.sp_scale.value()),
            "width": int(self.sp_width.value()),
            "height": int(self.sp_height.value()),
            "display": displays.id_for_label(self.cb_display.currentText()),
            "display_portrait": bool(self.seg_orient.value()),
            "auto_levels": self.chk_levels.isChecked(),
            "page_kind": self.page_kind(),
            # Kept in step with page_kind so the worker, and any settings file
            # read by an older build, still sees the flag it understands.
            "grayscale_convert": self.page_kind() != "colour",
            "grayscale_threshold": int(self.sp_threshold.value()),
            "grayscale_colour_percent": float(self.sp_colour.value()),
            "pre_downscale_height": int(self.sp_pre_h.value()),
            "rules": [r.to_dict() for r in self.rules],
        }
        d["format"]["id"] = str(self.seg_fmt.value())
        for fid in FORMATS:
            d["format"]["options"][fid] = self.format_values(fid)
        d["output"] = {
            "dir": self.ed_out.text(),
            "same_as_input": self.chk_same.isChecked(),
            "subfolder": self.ed_sub.text(),
            "container": self.container_value(),
            "pattern": self.ed_pattern.text(),
            "overwrite": self.chk_overwrite.isChecked(),
            "keep_structure": self.chk_keep_tree.isChecked(),
        }
        d["perf"] = {
            "device": self.device_value(),
            "use_fp16": self.chk_fp16.isChecked(),
            "tile": tile_value(self.cb_tile.currentText()),
            "budget_limit": int(self.sp_budget.value()),
            "force_cache_wipe": self.chk_wipe.isChecked(),
            "torch_threads": int(self.sp_threads.value()),
            "io_workers": max(1, int(self.sp_io.value())),
            "vips_concurrency": int(self.sp_vips.value()),
            "cudnn_benchmark": self.chk_cudnn.isChecked(),
            "allow_tf32": self.chk_tf32.isChecked(),
            "gpu_wake_lock": self.chk_wake.isChecked(),
        }
        log_cfg = dict(d.get("log") or {})
        log_cfg.update(
            {"wrap": self.chk_wrap.isChecked(), "show_debug": self.chk_debug.isChecked()}
        )
        d["log"] = log_cfg
        # Only keys the settings file already knows: unknown ones are dropped
        # by the merge on load, and the geometry keeps the old "WxH+X+Y" form
        # so a settings file written by either build still opens correctly.
        d["ui"] = dict(
            d.get("ui") or {},
            advanced_format=self.chk_adv.isChecked(),
            perf_open=self.panel_perf.is_open(),
            log_open=self._log_visible,
            geometry=self._geometry_text(),
        )

    def build_job(self, dry: bool = False) -> dict | None:
        """The job payload for the worker, or None with the reason on screen."""
        self.sync_settings()
        d = self.settings.data
        src = Path(d["input"]["path"])
        if not src.exists():
            self.show_banner(f"Input not found: {src}")
            return None
        out_dir = self.resolved_out_dir()
        if out_dir is None:
            self.show_banner("Choose an output folder.")
            return None
        fid = d["format"]["id"]
        if self.caps and not self.caps.get(fid, {}).get("ok", False):
            self.show_banner(f"{FORMATS[fid].label} cannot be written in this install.")
            return None
        if not [r for r in d["upscale"]["rules"] if r.get("enabled", True)]:
            self.show_banner(
                "The rules table has no rows switched on, so no model would run. "
                "Add a rule, or press Defaults beside the table."
            )
            return None
        self.hide_banner()
        job = {
            "input": {
                "path": str(src),
                "mode": "single" if src.is_file() else "bulk",
                "recursive": d["input"]["recursive"],
                "include_archives": d["input"]["include_archives"],
            },
            "output": {
                "dir": str(out_dir),
                "container": d["output"]["container"],
                "pattern": d["output"]["pattern"],
                "overwrite": d["output"]["overwrite"],
                "keep_structure": d["output"]["keep_structure"],
            },
            "format": {"id": fid, "options": d["format"]["options"][fid]},
            "upscale": dict(d["upscale"], models_dir=str(self.runner.paths().models_dir or "")),
            # Measurements taken on this machine, in this precision - or {},
            # which leaves the worker on its cautious first-page path.
            "perf": dict(d["perf"], profile=self.profile_for_run()),
        }
        if dry:
            job["dry_run"] = True
        return job

    # ------------------------------------------------------------------ #
    # starting, cancelling, pausing
    # ------------------------------------------------------------------ #
    def on_start_clicked(self) -> None:
        """The primary button is Start, and Cancel while a job is running."""
        if self.runner.running:
            self.cancel()
        else:
            self.start()

    def start(self, dry: bool = False) -> None:
        if self.runner.running:
            return
        job = self.build_job(dry=dry)
        if job is None:
            return
        self.settings.save()
        self.dry = dry
        self.total = 0
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.started_index = 0
        self.progress_sub = ""
        self.started_at = time.time()
        self.bar.setValue(0)
        self.last_out_dir = Path(job["output"]["dir"])
        self.lbl_status.setText("Planning\u2026" if dry else "Starting worker\u2026")
        self.lbl_detail.setText("")
        self.begin_run_log(job, dry)
        if not dry and not self.runner.has_portable_python():
            self.log("no backend runtime found, using the interpreter running the GUI", "warn")
        self.runner.hold_stop()  # the job gets the whole card to itself
        if self.runner.start(job):
            self.update_start_state()

    def start_dry(self) -> None:
        self.start(dry=True)

    def cancel(self) -> None:
        if self.runner.running:
            self.runner.cancel()
            self.lbl_status.setText("Cancelling\u2026")

    def toggle_pause(self) -> None:
        if not self.runner.running:
            return
        if self.runner.paused:
            self.runner.resume()
            self.btn_pause.setText("Pause")
            self.lbl_status.setText("Resumed")
        else:
            self.runner.pause()
            self.btn_pause.setText("Resume")
            self.lbl_status.setText("Paused")

    def open_output(self) -> None:
        target = self.last_out_dir or self.resolved_out_dir()
        if target is not None and Path(target).exists():
            open_in_explorer(Path(target))

    # ------------------------------------------------------------------ #
    # run log file
    # ------------------------------------------------------------------ #
    def begin_run_log(self, job: dict, dry: bool) -> None:
        u, o, p = job["upscale"], job["output"], job["perf"]
        fid = job["format"]["id"]
        rows = [r for r in (u.get("rules") or []) if r.get("enabled", True)]
        used = sorted({str(r.get("model") or "") for r in rows})
        models = f"{len(rows)} rule(s)"
        if used:
            models += "  \u00b7  " + ", ".join(used[:3])
            if len(used) > 3:
                models += f", +{len(used) - 3} more"
        header = [
            (
                f"JaNai Upscaler \u2014 {'dry run' if dry else 'run'} "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}"
            ),
            f"input    {job['input']['path']}",
            f"output   {o['dir']}",
            f"format   {FORMATS[fid].label}  \u00b7  package {CONTAINERS[o['container']].label}",
            f"rules    {models}",
            (
                f"device   {p.get('device') or 'auto'}  \u00b7  "
                f"{'FP16' if p.get('use_fp16') else 'FP32'}  \u00b7  tile {p.get('tile')}"
            ),
        ]
        path = self.runlog.begin(header, dry=dry)
        self.lbl_log_file.setText(str(path) if path else "not saved")
        self.log(
            f"{'dry run' if dry else 'job'}: {job['input']['path']} \u2192 {o['dir']} "
            f"[{FORMATS[fid].label}]"
        )
        if path:
            self.log(f"log: {path}", "debug")

    def end_run_log(self) -> None:
        if self.runlog.path is None:
            return
        self.runlog.end(
            [
                "-" * 78,
                (
                    f"ended {time.strftime('%Y-%m-%d %H:%M:%S')}  \u00b7  "
                    f"{self.completed} done, {self.failed} failed, "
                    f"{self.skipped} skipped"
                ),
            ]
        )
