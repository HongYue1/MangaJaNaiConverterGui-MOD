"""The Performance card: device, precision, tiling, threads - and the probe.

Detection, device choice and the hardware profile look like three features but
are one. The probe reports which devices exist and what the encoders can do;
the device choice decides which stored measurements still apply; and all of it
renders into this card. Splitting them would put ``_refresh_device_list`` in a
different file from the combo box it refills.

The GPU wake lock lives here too, because whether to hold a context is entirely
a function of this card's device choice and its "Keep GPU awake" checkbox.

Two rules worth keeping:

- ``device_value`` never silently falls back to the CPU. Until the probe lands
  the saved choice is returned as-is, so a slow backend cannot quietly move a
  run onto the CPU.
- A profile is only used on the machine it was measured on (see
  :mod:`janai.core.hardware`); numbers from another GPU are worse than none.

Mixed into :class:`janai.app.window.MainWindow`, so ``self`` is the window.
"""

from __future__ import annotations

import re
import time
from typing import Any

from janai.app.fields import TILE_CHOICES, tile_label
from janai.app.runlog import fmt_bytes, fmt_secs
from janai.app.widgets import (
    Collapsible,
    button,
    checkbox,
    combo,
    label,
    row,
    set_combo,
    spin_int,
)
from janai.core import hardware, rules
from janai.core.formats import FORMAT_IDS, FORMATS

#: An empty device string means "let the worker pick the best one".
AUTO_DEVICE = "Auto (best available)"
DEVICE_RE = re.compile(r"^(cpu|cuda|xpu|mps|dml|privateuseone)(:\d+)?$")


class PerfPanelMixin:
    """The Performance card, the device picker, the probe and the profile."""

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def _build_perf(self) -> None:
        panel = Collapsible(
            "Performance",
            "device, precision, tiling, threads",
            # Always closed on startup. It is the panel of last resort, and a
            # window that opens with it expanded buries the settings that are
            # actually used on every run.
            expanded=False,
        )
        self.panel_perf = panel
        body = panel.body
        p = self.settings.data["perf"]

        self.cb_device = combo(
            self._device_labels(),
            self._device_label(self._saved_device),
            self.on_device_change,
            width=330,
        )
        body.field("Device", "Auto picks the fastest GPU it can find.", self.cb_device)

        self.chk_fp16 = checkbox(
            "FP16 (half precision)",
            bool(p.get("use_fp16", True)),
            self.update_summary,
            tip=(
                "On by default: roughly twice as fast and half the VRAM on supported "
                "GPUs. The worker falls back to FP32 by itself on hardware or models "
                "that cannot do it."
            ),
        )
        body.control(self.chk_fp16)
        self.lbl_fp16 = label("", "hint")
        body.control(self.lbl_fp16)

        self.cb_tile = combo(
            [text for text, _ in TILE_CHOICES],
            tile_label(str(p.get("tile", "auto"))),
            self.update_summary,
            width=210,
            tip=(
                "Auto measures what the model actually costs on the first tiles and "
                "grows to the largest tile that fits the free VRAM \u2014 fewer, "
                "bigger tiles means fewer seams and less overhead."
            ),
        )
        body.field("Tile size", "Splits large images so they fit in VRAM.", self.cb_tile)

        self.btn_profile = button(
            "Measure this machine",
            self.on_profile_clicked,
            tip=(
                "Runs each installed model over a ladder of tile sizes and records what "
                "it costs and how fast it is. Auto then sizes the first page from "
                "measurements instead of holding it at 1024px, and a tile this card "
                "has already refused is never planned again. Writes no images, and is "
                "only needed once per machine."
            ),
        )
        body.field(
            "Hardware profile",
            "Measured once per machine, then reused.",
            row(self.btn_profile),
        )
        self.lbl_profile = label("", "hint", wrap=True)
        body.control(self.lbl_profile)

        self.sp_budget = spin_int(
            0,
            128,
            int(p.get("budget_limit", 0)),
            1,
            self.update_summary,
            suffix=" GiB",
            special="no cap",
        )
        body.field("VRAM budget", "An upper bound on what a job may reserve.", self.sp_budget)

        self.sp_threads = spin_int(
            0, 256, int(p.get("torch_threads", 0)), 1, self.update_summary, special="auto"
        )
        body.field("CPU threads", "Leave on auto to let torch decide.", self.sp_threads)

        self.sp_io = spin_int(1, 16, max(1, int(p.get("io_workers", 2))), 1, self.update_summary)
        body.field("I/O workers", "Threads that decode and encode while the GPU works.", self.sp_io)

        self.sp_vips = spin_int(
            0, 64, int(p.get("vips_concurrency", 0)), 1, self.update_summary, special="default"
        )
        body.field("libvips concurrency", "Leave on default unless tuning.", self.sp_vips)

        self.chk_cudnn = checkbox(
            "cuDNN autotune",
            bool(p.get("cudnn_benchmark", True)),
            tip=(
                "Benchmarks convolution algorithms once per shape. Faster for long "
                "runs of same-sized pages."
            ),
        )
        self.chk_tf32 = checkbox(
            "TF32 matmuls",
            bool(p.get("allow_tf32", False)),
            tip=(
                "Ampere and newer: faster matmuls at slightly reduced precision. Off "
                "by default because the measured gain on a single 17 s image is "
                "0.1-0.2 s, which is inside the noise - not worth trading precision "
                "for unasked. Worth turning on for long batches, where it compounds."
            ),
        )
        self.chk_wipe = checkbox(
            "Wipe cache between images",
            bool(p.get("force_cache_wipe", False)),
            tip=("Frees VRAM after every image. Slower, but avoids fragmentation on small GPUs."),
        )
        body.control(row(self.chk_cudnn, self.chk_tf32, self.chk_wipe, spacing=18))

        self.chk_wake = checkbox(
            "Keep GPU awake",
            bool(p.get("gpu_wake_lock", True)),
            self.on_wake_toggle,
            tip=(
                "Holds a tiny context on the GPU while this window is open, so the "
                "driver keeps the card powered (and a laptop dGPU does not park) and "
                "the first run starts at full speed. It holds a CUDA context, which "
                "costs about 120 MB of VRAM while the app sits idle, and it is "
                "released automatically while a job runs. Turn it off to leave the "
                "GPU completely alone."
            ),
        )
        body.control(self.chk_wake)
        self.page.addWidget(panel)

    # ------------------------------------------------------------------ #
    # devices
    # ------------------------------------------------------------------ #
    def _known_devices(self) -> list[dict]:
        """Probed devices, falling back to the cached probe from last launch."""
        if self.devices:
            return [d for d in self.devices if isinstance(d, dict)]
        cached = (self.settings.data.get("probe") or {}).get("devices") or []
        return [d for d in cached if isinstance(d, dict)]

    def _device_label(self, value: str) -> str:
        value = (value or "").strip()
        if not value:
            return AUTO_DEVICE
        for device in self._known_devices():
            if str(device.get("value")) == value:
                return str(device.get("label"))
        return value

    def _device_labels(self) -> list[str]:
        labels = [AUTO_DEVICE] + [str(d.get("label")) for d in self._known_devices()]
        current = self.cb_device.currentText() if hasattr(self, "cb_device") else ""
        if current and current not in labels:
            labels.append(current)
        return labels

    def device_value(self) -> str:
        """The device string for the job. Empty means auto, never a silent CPU
        fallback: before the probe lands the saved choice is kept."""
        text = self.cb_device.currentText().strip()
        if not text or text == AUTO_DEVICE:
            return ""
        for device in self._known_devices():
            if str(device.get("label")) == text:
                return str(device.get("value"))
        if DEVICE_RE.match(text):
            return text
        return self._saved_device

    def on_device_change(self) -> None:
        """Keep the user's FP16 preference; only report what the device can do."""
        value = self.device_value()
        self._saved_device = value
        known = self._known_devices()
        if value:
            device = next((d for d in known if str(d.get("value")) == value), None)
        else:
            device = next((d for d in known if str(d.get("value")) != "cpu"), None)
        note = ""
        if device is not None:
            supported = bool(device.get("fp16")) and str(device.get("value")) != "cpu"
            self.chk_fp16.setEnabled(supported)
            if not supported and self.chk_fp16.isChecked():
                note = (
                    "this device runs FP32 \u2014 the preference is kept for GPUs that support it"
                )
        self.lbl_fp16.setText(note)
        # The line exists only to warn. When FP16 simply works, the checkbox
        # already says so, so nothing is added and the line disappears.
        self.lbl_fp16.setVisible(bool(note))
        self.update_wake_lock()
        self.update_summary()

    # ------------------------------------------------------------------ #
    # GPU wake lock
    # ------------------------------------------------------------------ #
    def wake_lock_device(self) -> str | None:
        """Device to hold awake: "" for auto, None when it should not be held."""
        if not self.chk_wake.isChecked():
            return None
        value = self.device_value()
        if value == "cpu":
            return None
        if value:
            return value
        known = self._known_devices()
        if known and not any(str(d.get("value")) != "cpu" for d in known):
            return None
        return ""

    def update_wake_lock(self) -> None:
        """Start or release the idle hold. Never held while a job is running,
        so the worker gets the whole card to itself."""
        want = self.wake_lock_device()
        if self.runner.running or want is None:
            if self.runner.holding:
                self.runner.hold_stop()
            return
        if self.runner.holding and self.runner.hold_device == want:
            return
        self.runner.hold_start(want)

    def on_wake_toggle(self) -> None:
        if not self.chk_wake.isChecked() and self.runner.holding:
            self.log("GPU wake lock off")
        self.update_wake_lock()
        self.update_summary()

    def on_hold(self, event: dict) -> None:
        if event.get("released"):
            return
        if event.get("ok"):
            held = int(event.get("reserved") or 0)
            where = str(event.get("name") or event.get("device") or "GPU")
            extra = f" ({fmt_bytes(held)} reserved)" if held else ""
            self.log(f"GPU wake lock on {where}{extra}", "debug")
        else:
            self.log(f"GPU wake lock unavailable: {event.get('error', 'unknown reason')}", "warn")

    # ------------------------------------------------------------------ #
    # hardware profile
    # ------------------------------------------------------------------ #
    def profile_for_run(self) -> dict:
        """The measurements this run may use, or ``{}`` to plan cautiously."""
        return hardware.profile_for_run(self.profile, self.probe, self.chk_fp16.isChecked())

    def profile_model_paths(self) -> list[str]:
        """The models this table would actually run.

        What gets measured is per model, so measuring a model no rule names
        buys nothing. Measured on a real machine: picking one model per scale
        chose the file that sorts first at 4x, while the table ran a different
        4x file, so every page still fell back to the cautious first-page tile
        and the measurement bought nothing at all.

        So: one model per way a page can be routed - grayscale or colour, at
        each factor - taken from the rules that are switched on. Then one per
        factor the rules never mention, so a table of only 2x rows still
        learns something about a 4x file. Capped, because every model costs a
        ladder of real passes.
        """
        by_name: dict[str, str] = {}
        scale_of: dict[str, Any] = {}
        for entry in self.models:
            if isinstance(entry, dict) and entry.get("path"):
                by_name[str(entry.get("name"))] = str(entry["path"])
                scale_of[str(entry["path"])] = entry.get("scale")
        routes: dict[tuple[str, int], str] = {}
        for rule in self.rules:
            if not rule.enabled or rule.action == rules.PASSTHROUGH or rule.is_auto:
                continue
            path = by_name.get(rule.model)
            if not path:
                continue
            factor = rules.bucket_scale(rule.scale or rules.model_scale(rule.model))
            routes.setdefault((rule.kind, factor), path)
        paths: list[str] = []
        for path in routes.values():
            if path not in paths:
                paths.append(path)
        covered = {scale_of.get(path) for path in paths}
        for entry in self.models:
            if not (isinstance(entry, dict) and entry.get("path")):
                continue
            if entry.get("scale") in covered:
                continue
            covered.add(entry.get("scale"))
            paths.append(str(entry["path"]))
        # Four is the number of routes a page can take (grayscale/colour at
        # 2x/4x); past that the wait stops being worth the measurement.
        return paths[:4]

    def build_profile_job(self) -> dict:
        """A job that measures this machine and converts nothing.

        Only ``perf`` matters here: there is no input, output or format,
        because nothing is written. The models come from the probe, and the
        fingerprint records which machine the numbers belong to so they are
        dropped rather than trusted once it changes.
        """
        self.sync_settings()
        perf = dict(self.settings.data["perf"])
        perf["profile"] = None
        return {
            "perf": perf,
            "models": self.profile_model_paths(),
            "fingerprint": hardware.fingerprint(self.probe),
        }

    def on_profile_clicked(self) -> None:
        if self.runner.running:
            self.log("something is already running, so the measurement has to wait", "warn")
            return
        job = self.build_profile_job()
        if not job["models"]:
            self.show_banner("No models are installed, so there is nothing to measure.")
            return
        self._profiling = True
        self.btn_profile.setEnabled(False)
        self.lbl_profile.setText("measuring\u2026")
        self.started_at = time.time()
        self.log("measuring this machine - no images are written")
        if not self.runner.start_profile(job):
            self._profiling = False
            self.btn_profile.setEnabled(True)
            self.render_profile()
            self.log("the worker would not start", "error")
            return
        self.update_start_state()

    def on_profile_progress(self, event: dict) -> None:
        """One line per step, in the panel rather than the log.

        The worker reports each step twice: once on the way in, which is what
        the label follows, and once on the way out carrying the result.
        """
        index = int(event.get("index") or 0)
        total = max(1, int(event.get("total") or 1))
        name = str(event.get("model") or "")
        tile = int(event.get("tile") or 0)
        if "ok" not in event:
            self.lbl_profile.setText(f"measuring {index}/{total}: {name} at {tile}px")
        elif not event.get("ok"):
            self.log(f"{name}: {tile}px did not fit, so that is the ceiling", "debug")

    def on_profile(self, event: dict) -> None:
        self._profiling = False
        self.btn_profile.setEnabled(True)
        profile = event.get("profile")
        if not event.get("ok") or not isinstance(profile, dict):
            reason = "cancelled" if event.get("cancelled") else ""
            reason = reason or str(event.get("error") or "nothing could be measured")
            self.log(f"the measurement did not finish: {reason}", "warn")
            self.render_profile()
            return
        self.profile = profile
        self.settings.data["profile"] = profile
        self.settings.save()
        count = len(hardware.profile_models(profile))
        elapsed = fmt_secs(float(event.get("elapsed") or 0.0))
        self.log(
            f"measured {count} model(s) in {elapsed} \u2014 Auto now sizes tiles from this",
            "ok",
        )
        self.render_profile()
        self.update_summary()

    def render_profile(self) -> None:
        """Say what the measurements know, and whether they still apply."""
        models = hardware.profile_models(self.profile)
        if not models:
            self.btn_profile.setText("Measure this machine")
            self.lbl_profile.setText(
                "not measured \u2014 Auto holds the first page of an unseen model at "
                "1024px until it has measured it"
            )
            return
        if not hardware.profile_is_current(self.profile, self.probe):
            self.btn_profile.setText("Measure this machine")
            was = str((self.profile.get("hardware") or {}).get("name") or "another machine")
            self.lbl_profile.setText(
                f"measured on {was}, which is not what is here now \u2014 not in use"
            )
            return
        self.btn_profile.setText("Measure again")
        best = max(
            (int(e.get("best_tile") or 0) for e in models.values() if isinstance(e, dict)),
            default=0,
        )
        bits = [f"{len(models)} model(s) measured"]
        if best:
            bits.append(f"fastest tile {best}px")
        created = str(self.profile.get("created") or "")[:10]
        if created:
            bits.append(created)
        self.lbl_profile.setText("   \u00b7   ".join(bits))

    def offer_profile(self) -> None:
        """A first run, or new hardware: say so once, and never block on it."""
        if self._profile_offered or self._profiling or not self.models:
            return
        if hardware.profile_is_current(self.profile, self.probe):
            return
        gpus = [d for d in self.devices if isinstance(d, dict) and str(d.get("value")) != "cpu"]
        if not gpus:
            # Nothing to size against: the CPU path has no VRAM budget.
            return
        self._profile_offered = True
        what = (
            "The hardware changed since the last measurement"
            if hardware.profile_models(self.profile)
            else "This machine has not been measured yet"
        )
        press = self.btn_profile.text()
        self.log(
            f"{what}. Open Performance and press \u201c{press}\u201d so Auto can size"
            " tiles from measurements instead of a cautious guess.",
            "warn",
        )
        if not (self.probe.get("errors") or []):
            self.show_banner(
                f"{what}. Performance \u203a Hardware profile \u2192 \u201c{press}\u201d"
                " measures it once, in about a minute, and writes no images."
            )

    # ------------------------------------------------------------------ #
    # probe
    # ------------------------------------------------------------------ #
    def refresh_probe(self) -> None:
        self.lbl_env.setText("detecting hardware\u2026")
        self.btn_refresh.setEnabled(False)
        self.runner.probe()

    def apply_probe(self, probe: dict, cached: bool = False) -> None:
        if not probe:
            self._refresh_device_list()
            return
        self.probe = probe
        self.caps = probe.get("formats", {}) or {}
        self.models = probe.get("models", []) or []
        self.devices = probe.get("devices", []) or []

        self._refresh_device_list()

        # The shipped table is built from the models actually installed, so it
        # can only be seeded once the probe has reported them.
        self.seed_rules()
        self.render_rules()

        for fid in FORMAT_IDS:
            self.seg_fmt.set_option_enabled(fid, bool(self.caps.get(fid, {}).get("ok", True)))
        current = str(self.seg_fmt.value())
        if not self.caps.get(current, {}).get("ok", True):
            fallback = next((f for f in FORMAT_IDS if self.caps.get(f, {}).get("ok")), "png")
            self.log(
                f"{FORMATS[current].label} is unavailable, switching to {FORMATS[fallback].label}",
                "warn",
            )
            self.seg_fmt.set_value(fallback)

        self.lbl_env.setText(self._environment_line(probe, cached))

        if not cached:
            self.settings.data["probe"] = probe
            self.btn_refresh.setEnabled(True)
            errors = probe.get("errors") or []
            if errors:
                self.show_banner("; ".join(str(e) for e in errors[:2]))
            else:
                self.hide_banner()
            self.settings.save()

        self.render_format_options()
        self.update_summary()
        self.update_start_state()
        self.update_wake_lock()
        self.render_profile()
        if not cached:
            # Only meaningful once the real probe has arrived: a cached probe
            # cannot tell whether the hardware changed underneath it.
            self.offer_profile()

    def _refresh_device_list(self) -> None:
        """Refill the device picker, keeping the current choice selected."""
        wanted = self._device_label(self._saved_device)
        self.cb_device.blockSignals(True)
        self.cb_device.clear()
        self.cb_device.addItems(self._device_labels())
        set_combo(self.cb_device, wanted)
        self.cb_device.blockSignals(False)
        self.on_device_change()

    def _environment_line(self, probe: dict, cached: bool) -> str:
        """The line under the title: what this machine can actually do."""
        devices = [d for d in (probe.get("devices") or []) if isinstance(d, dict)]
        gpu = next((d for d in devices if str(d.get("value")) != "cpu"), None)
        bits: list[str] = []
        if gpu is not None:
            name = str(gpu.get("label") or gpu.get("value"))
            vram = int(gpu.get("vram") or 0)
            bits.append(f"{name}  \u00b7  {fmt_bytes(vram)}" if vram else name)
        else:
            bits.append("CPU only")
        if self.models:
            bits.append(f"{len(self.models)} models")
        working = [f for f in FORMAT_IDS if self.caps.get(f, {}).get("ok")]
        if working:
            bits.append(f"{len(working)} encoders")
        if cached:
            bits.append("cached \u00b7 press Re-detect to refresh")
        return "   \u00b7   ".join(bits)

    def on_probe_error(self, event: dict) -> None:
        self.btn_refresh.setEnabled(True)
        self.lbl_env.setText("backend not ready")
        message = str(event.get("message") or "").strip()
        self.show_banner(
            (
                "The Python backend is not ready. Run setup.cmd in this folder "
                f"to create it.\n{message}"
            ).strip()
        )
