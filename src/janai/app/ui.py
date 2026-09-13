"""JaNai Upscaler main window.

Four cards, read top to bottom: what goes in, how far to upscale it, what
comes out, and (folded away) how hard to push the hardware. Everything heavy
happens in the worker process; this file collects settings, renders events and
keeps a log.
"""

from __future__ import annotations

import re
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Any

from janai import __version__
from janai.app import dnd
from janai.app.runlog import (
    RunLog,
    fmt_bytes,
    fmt_secs,
    format_bundle,
    format_done,
    format_file,
    format_start,
)
from janai.app.runner import Runner, open_in_explorer
from janai.app.state import Settings, defaults
from janai.app.theme import Theme
from janai.app.widgets import (
    Card,
    Collapsible,
    ScrollArea,
    Segmented,
    Table,
    Tooltip,
    clear,
    combo,
    entry,
    int_spin,
    row_label,
)
from janai.core import displays, presets, rules
from janai.core.formats import (
    CONTAINER_IDS,
    CONTAINERS,
    FORMAT_IDS,
    FORMATS,
    Opt,
    is_active,
    packs_archive,
    summary,
)

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".jfif",
    ".webp",
    ".avif",
    ".jxl",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
    ".heic",
    ".heif",
    ".ppm",
    ".pgm",
}
ARCHIVE_EXTS = {".zip", ".cbz", ".rar", ".cbr"}

TILE_CHOICES = [
    ("Auto (adaptive)", "auto"),
    ("Maximum", "maximum"),
    ("No tiling", "none"),
    ("128 px", "128"),
    ("192 px", "192"),
    ("256 px", "256"),
    ("384 px", "384"),
    ("512 px", "512"),
    ("768 px", "768"),
    ("1024 px", "1024"),
]

MODE_OPTIONS = [("Scale", "scale"), ("Width", "width"), ("Height", "height"), ("Fit", "fit")]

LOG_TAGS = ("info", "debug", "warn", "error", "ok", "skip", "dry")

# An empty device string means "let the worker pick the best one".
AUTO_DEVICE = "Auto (best available)"
DEVICE_RE = re.compile(r"^(cpu|cuda|xpu|mps|dml|privateuseone)(:\d+)?$")


def choice_label(opt: Opt, value: Any) -> str:
    for label, val in opt.choices:
        if val == value:
            return label
    return opt.choices[0][0] if opt.choices else str(value)


def choice_value(opt: Opt, label: str) -> Any:
    for lbl, val in opt.choices:
        if lbl == label:
            return val
    return opt.default


class App:
    def __init__(self, root: tk.Tk, root_dir: Path) -> None:
        self.root = root
        self.dir = root_dir
        self.settings = Settings(root_dir / "settings.json").load()
        self.theme = Theme(root, str(self.settings.data.get("theme", "dark")))
        self.runner = Runner(root_dir)

        log_cfg = self.settings.data.get("log") or {}
        self.runlog = RunLog(
            self._log_dir(),
            keep=int(log_cfg.get("keep", 30) or 0),
            enabled=bool(log_cfg.get("auto_save", True)),
        )

        probe = self.settings.data.get("probe") or {}
        self.probe: dict = probe if isinstance(probe, dict) else {}
        self.caps: dict = self.probe.get("formats", {}) or {}
        self.models: list = self.probe.get("models", []) or []
        self.devices: list = self.probe.get("devices", []) or []

        self.total = 0
        self.completed = 0
        self.failed = 0
        self.skipped = 0
        self.started_at = 0.0
        self.dry = False
        self.last_out_dir: Path | None = None
        self.scan_text = ""

        self._build_vars()
        self._build_ui()
        self._bind_keys()
        self.apply_probe(self.probe, cached=True)
        self.runner.probe()
        self.render_format_options()
        self.render_target()
        self.seed_rules()
        self.render_rules()
        self.update_start_state()
        self.update_wake_lock()
        self.root.after(80, self._tick)
        if self.var_in_path.get():
            self.scan_input_async(self.var_in_path.get())

    # ------------------------------------------------------------------ #
    # variables
    # ------------------------------------------------------------------ #
    def _build_vars(self) -> None:
        d = self.settings.data
        i, o, u, p, ui = d["input"], d["output"], d["upscale"], d["perf"], d["ui"]
        lg = d.get("log") or {}

        self.var_in_path = tk.StringVar(value=i.get("path", ""))
        self.var_recursive = tk.BooleanVar(value=bool(i.get("recursive", True)))
        self.var_archives = tk.BooleanVar(value=bool(i.get("include_archives", True)))

        self.var_mode = tk.StringVar(value=str(u.get("mode", "scale")))
        self.var_scale = tk.DoubleVar(value=float(u.get("scale", 2.0)))
        self.var_width = tk.IntVar(value=int(u.get("width", 2048)))
        self.var_height = tk.IntVar(value=int(u.get("height", 2160)))
        self.var_display = tk.StringVar(
            value=displays.label_for_id(str(u.get("display", displays.CUSTOM)))
        )
        self.var_portrait = tk.BooleanVar(value=bool(u.get("display_portrait", True)))
        self.var_levels = tk.BooleanVar(value=bool(u.get("auto_levels", True)))
        self.var_gray = tk.BooleanVar(value=bool(u.get("grayscale_convert", True)))
        self.var_threshold = tk.IntVar(value=int(u.get("grayscale_threshold", 12)))
        self.var_colour_pct = tk.DoubleVar(value=float(u.get("grayscale_colour_percent", 0.25)))
        self.var_pre_h = tk.IntVar(value=int(u.get("pre_downscale_height", 0)))
        self.var_skip_long = tk.BooleanVar(value=bool(u.get("skip_long_strips", False)))
        self.rules: list[rules.Rule] = [rules.Rule.from_dict(r) for r in (u.get("rules") or [])]
        self._rules_seeded = bool(self.rules)

        self.var_fmt = tk.StringVar(value=str(d["format"].get("id", "png")))
        self.var_adv = tk.BooleanVar(value=bool(ui.get("advanced_format", False)))
        self.fmt_vars: dict[str, dict[str, tk.Variable]] = {}
        for fid, spec in FORMATS.items():
            saved = self.settings.format_options(fid)
            bucket: dict[str, tk.Variable] = {}
            for opt in spec.opts:
                value = saved.get(opt.key, opt.default)
                if opt.kind == "bool":
                    bucket[opt.key] = tk.BooleanVar(value=bool(value))
                elif opt.kind == "int":
                    bucket[opt.key] = tk.IntVar(value=int(value))
                elif opt.kind == "float":
                    bucket[opt.key] = tk.DoubleVar(value=float(value))
                else:
                    bucket[opt.key] = tk.StringVar(value=choice_label(opt, value))
            self.fmt_vars[fid] = bucket

        self.var_container = tk.StringVar(value=str(o.get("container", "files")))
        self.var_same = tk.BooleanVar(value=bool(o.get("same_as_input", True)))
        self.var_subfolder = tk.StringVar(value=str(o.get("subfolder", "upscaled")))
        self.var_out_dir = tk.StringVar(value=str(o.get("dir", "")))
        self.var_pattern = tk.StringVar(value=str(o.get("pattern", "{name}")))
        self.var_overwrite = tk.BooleanVar(value=bool(o.get("overwrite", False)))
        self.var_keep_tree = tk.BooleanVar(value=bool(o.get("keep_structure", True)))

        self._saved_device = str(p.get("device", "") or "")
        self.var_device = tk.StringVar(value=self._device_label(self._saved_device))
        self.var_wake = tk.BooleanVar(value=bool(p.get("gpu_wake_lock", True)))
        self.var_fp16 = tk.BooleanVar(value=bool(p.get("use_fp16", True)))
        self.var_tile = tk.StringVar(value=self._tile_label(str(p.get("tile", "auto"))))
        self.var_budget = tk.IntVar(value=int(p.get("budget_limit", 0)))
        self.var_wipe = tk.BooleanVar(value=bool(p.get("force_cache_wipe", False)))
        self.var_threads = tk.IntVar(value=int(p.get("torch_threads", 0)))
        self.var_io = tk.IntVar(value=int(p.get("io_workers", 2)))
        self.var_vips = tk.IntVar(value=int(p.get("vips_concurrency", 0)))
        self.var_cudnn = tk.BooleanVar(value=bool(p.get("cudnn_benchmark", True)))
        self.var_tf32 = tk.BooleanVar(value=bool(p.get("allow_tf32", True)))

        self.var_log_wrap = tk.BooleanVar(value=bool(lg.get("wrap", False)))
        self.var_log_debug = tk.BooleanVar(value=bool(lg.get("show_debug", False)))
        self.var_log_file = tk.StringVar(value="")

        self.var_status = tk.StringVar(value="Ready")
        self.var_detail = tk.StringVar(value="")
        self.var_progress = tk.DoubleVar(value=0.0)

    @staticmethod
    def _tile_label(value: str) -> str:
        for label, val in TILE_CHOICES:
            if val == value:
                return label
        return TILE_CHOICES[0][0]

    @staticmethod
    def _tile_value(label: str) -> str:
        for lbl, val in TILE_CHOICES:
            if lbl == label:
                return val
        return "auto"

    def _log_dir(self) -> Path:
        raw = str((self.settings.data.get("log") or {}).get("dir") or "logs")
        p = Path(raw).expanduser()
        return p if p.is_absolute() else self.dir / p

    # ------------------------------------------------------------------ #
    # layout
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        r = self.root
        r.title("JaNai Upscaler")
        r.minsize(940, 640)
        geom = str(self.settings.data["ui"].get("geometry") or "")
        r.geometry(geom if "x" in geom else "1060x860")
        r.columnconfigure(0, weight=1)
        r.rowconfigure(1, weight=1)  # the cards
        r.protocol("WM_DELETE_WINDOW", self.on_close)

        self._build_header()

        self.scroll = ScrollArea(r, self.theme)
        self.scroll.grid(row=1, column=0, sticky="nsew", padx=(14, 6))
        body = self.scroll.body
        body.columnconfigure(0, weight=1)

        self.banner = ttk.Label(
            body, text="", style="Err.TLabel", wraplength=900, justify="left", padding=(12, 8)
        )
        self._banner_visible = False

        self._build_input(body)
        self._build_upscale(body)
        self._build_output(body)
        self._build_perf(body)
        ttk.Frame(body, height=8).grid(row=9, column=0)

        self._build_footer()
        self._build_log()

        self._log_visible = False
        self.show_log(bool(self.settings.data["ui"].get("log_open", False)))

        if dnd.enable(r, self.on_drop):
            self.drop_hint.configure(text="Drop images, a folder or a .cbz here")
        else:
            self.drop_hint.configure(text="Choose a file or folder to start")

    def _build_header(self) -> None:
        head = ttk.Frame(self.root, padding=(18, 14, 18, 8))
        head.grid(row=0, column=0, sticky="ew")
        head.columnconfigure(1, weight=1)
        ttk.Label(head, text="JaNai Upscaler", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        self.lbl_env = ttk.Label(head, text="detecting hardware\u2026", style="MutedBg.TLabel")
        self.lbl_env.grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 0))
        btns = ttk.Frame(head)
        btns.grid(row=0, column=2, rowspan=2, sticky="e")
        btn_theme = ttk.Button(
            btns, text="Theme", style="GhostBg.TButton", command=self.toggle_theme
        )
        btn_theme.grid(row=0, column=0, padx=(0, 6))
        Tooltip(btn_theme, "Switch between the dark and light palette.", self.theme)
        self.btn_refresh = ttk.Button(
            btns, text="Re-detect", style="GhostBg.TButton", command=self.refresh_probe
        )
        self.btn_refresh.grid(row=0, column=1)
        Tooltip(
            self.btn_refresh,
            "Ask the backend again which GPU, encoders and models are available. "
            "Use it after installing models or changing drivers \u2014 the answer "
            "is cached between runs so the window can open instantly.",
            self.theme,
        )
        self.btn_presets = ttk.Button(
            btns, text="Presets", style="GhostBg.TButton", command=self.preset_menu
        )
        self.btn_presets.grid(row=0, column=2, padx=(6, 0))
        Tooltip(
            self.btn_presets,
            "Save the current settings as a preset, load one from a file, or "
            "switch to one you already saved. A preset carries the target, the "
            "rules table, the format, the output layout and the performance "
            "options \u2014 never your folders or your device, so someone "
            "else's preset cannot redirect your output.",
            self.theme,
        )
        self.btn_reset = ttk.Button(
            btns, text="Reset all", style="GhostBg.TButton", command=self.reset_all
        )
        self.btn_reset.grid(row=0, column=3, padx=(6, 0))
        Tooltip(
            self.btn_reset,
            "Put every setting back to the shipped defaults: target, rules table, "
            "output format and layout, performance options and log view. Your "
            "input and output folders are kept, and you get a confirmation "
            "prompt first.",
            self.theme,
        )

    def _build_input(self, body: tk.Misc) -> None:
        card = Card(
            body,
            "Input",
            subtitle="A folder, an archive, or single images. Drop them here.",
            badge="",
        )
        card.grid(row=1, column=0, sticky="ew", pady=(6, 10))
        self.card_input = card
        b = card.body
        b.columnconfigure(0, weight=1)

        drop = ttk.Frame(b, style="Drop.TFrame", padding=(16, 18))
        drop.grid(row=0, column=0, columnspan=3, sticky="ew")
        drop.columnconfigure(0, weight=1)
        self.drop_hint = ttk.Label(drop, text="", style="Inset.TLabel", anchor="center")
        self.drop_hint.grid(row=0, column=0, sticky="ew")
        self.lbl_path = ttk.Label(
            drop,
            text="No input selected",
            style="InsetMuted.TLabel",
            anchor="center",
            wraplength=820,
            justify="center",
        )
        self.lbl_path.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        row = ttk.Frame(b, style="Card.TFrame")
        row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 0))
        ttk.Button(row, text="Choose file\u2026", command=self.choose_file).grid(row=0, column=0)
        ttk.Button(row, text="Choose folder\u2026", command=self.choose_folder).grid(
            row=0, column=1, padx=8
        )
        ttk.Button(row, text="Clear", style="Ghost.TButton", command=self.clear_input).grid(
            row=0, column=2
        )

        opts = ttk.Frame(b, style="Card.TFrame")
        opts.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        c1 = ttk.Checkbutton(
            opts,
            text="Include subfolders",
            variable=self.var_recursive,
            command=self.on_input_options,
        )
        c1.grid(row=0, column=0, sticky="w")
        Tooltip(
            c1,
            "Walk the whole tree. With a CBZ package this is what turns each "
            "chapter folder into its own archive.",
            self.theme,
        )
        c2 = ttk.Checkbutton(
            opts,
            text="Include archives (cbz/zip/cbr/rar)",
            variable=self.var_archives,
            command=self.on_input_options,
        )
        c2.grid(row=0, column=1, sticky="w", padx=(18, 0))
        Tooltip(
            c2,
            "Comic archives found in the input are re-packed as .cbz with every page upscaled.",
            self.theme,
        )

    def _build_upscale(self, body: tk.Misc) -> None:
        card = Card(
            body,
            "Upscale",
            subtitle="How big the result is, and which model each page gets.",
        )
        card.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        self.card_upscale = card
        b = card.body

        row_label(b, 0, "Target", "How large the result should be.", self.theme)
        Segmented(b, self.var_mode, MODE_OPTIONS, command=lambda _v: self.render_target()).grid(
            row=0, column=1, sticky="w", pady=4
        )

        self.target_box = ttk.Frame(b, style="Card.TFrame")
        self.target_box.grid(row=1, column=1, sticky="w", pady=(2, 6))
        self.f_scale = ttk.Frame(self.target_box, style="Card.TFrame")
        int_spin(self.f_scale, self.var_scale, 1.0, 8.0, 0.25, 7, self.update_summary).grid(
            row=0, column=0
        )
        ttk.Label(self.f_scale, text="\u00d7 original size", style="Muted.TLabel").grid(
            row=0, column=1, padx=(8, 0)
        )
        self.f_width = ttk.Frame(self.target_box, style="Card.TFrame")
        ttk.Label(self.f_width, text="Width", style="Muted.TLabel").grid(
            row=0, column=0, padx=(0, 6)
        )
        int_spin(self.f_width, self.var_width, 64, 30000, 16, 8, self.update_summary).grid(
            row=0, column=1
        )
        ttk.Label(self.f_width, text="px", style="Muted.TLabel").grid(row=0, column=2, padx=(6, 0))
        self.f_height = ttk.Frame(self.target_box, style="Card.TFrame")
        ttk.Label(self.f_height, text="Height", style="Muted.TLabel").grid(
            row=0, column=0, padx=(12, 6)
        )
        int_spin(self.f_height, self.var_height, 64, 30000, 16, 8, self.update_summary).grid(
            row=0, column=1
        )
        ttk.Label(self.f_height, text="px", style="Muted.TLabel").grid(row=0, column=2, padx=(6, 0))

        # Fit mode also offers the original fork's display-device list, so a
        # chapter can be sized for a specific reader in one click.
        self.f_fit = ttk.Frame(self.target_box, style="Card.TFrame")
        ttk.Label(self.f_fit, text="Device", style="Muted.TLabel").grid(
            row=0, column=0, padx=(0, 6)
        )
        self.cb_display = combo(
            self.f_fit,
            self.var_display,
            displays.labels(),
            width=30,
            on_change=self.on_display_change,
        )
        self.cb_display.grid(row=0, column=1)
        Segmented(
            self.f_fit,
            self.var_portrait,
            [("Portrait", True), ("Landscape", False)],
            command=lambda _v: self.on_display_change(),
        ).grid(row=0, column=2, padx=(10, 0))

        # Page kind comes before the table, because it decides which rules can
        # ever fire: with detection off every page is treated as colour.
        row_label(
            b,
            2,
            "Pages",
            "Whether each page is judged on its own or the whole run is treated as colour.",
            self.theme,
        )
        kinds = ttk.Frame(b, style="Plain.TFrame")
        kinds.grid(row=2, column=1, sticky="w", pady=4)
        c1 = ttk.Checkbutton(
            kinds, text="Grayscale detection", variable=self.var_gray, command=self.on_gray_toggle
        )
        c1.grid(row=0, column=0, sticky="w")
        Tooltip(
            c1,
            "Measures each page and decides whether it is really grayscale. A "
            "grayscale page is stored as one channel (smaller files, faster), "
            "downscaled with the dot-gain aware filter, and matched against the "
            "grayscale rules in the table. Turn this off and every page is "
            "treated as colour, so only the colour rules can fire - useful for "
            "an all-colour artbook.",
            self.theme,
        )
        c2 = ttk.Checkbutton(
            kinds, text="Auto levels", variable=self.var_levels, command=self.on_levels_toggle
        )
        c2.grid(row=0, column=1, sticky="w", padx=(18, 0))
        Tooltip(
            c2,
            "Stretches the black and white points of grayscale pages before "
            "upscaling, which lifts washed-out scans. Colour pages are never "
            "touched. This is the setting a rule follows when its Levels cell "
            "says \u201cdefault\u201d; a rule can override it per page size.",
            self.theme,
        )

        # The table is the only thing that chooses a model. There is no picker
        # beside it to contradict it, and no "auto" row to hide the choice.
        self.rules_box = ttk.Frame(b, style="Plain.TFrame")
        self.rules_box.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(16, 0))
        self.rules_box.columnconfigure(0, weight=1)

        rules_head = ttk.Frame(self.rules_box, style="Plain.TFrame")
        rules_head.grid(row=0, column=0, columnspan=2, sticky="ew")
        rules_head.columnconfigure(1, weight=1)
        lbl_rules = ttk.Label(rules_head, text="Model rules", style="Field.TLabel")
        lbl_rules.grid(row=0, column=0, sticky="w")
        lbl_rules_how = ttk.Label(
            rules_head,
            text="page kind + size decide the model \u00b7 a sized row always "
            "beats an \u201cany\u201d row",
            style="Muted.TLabel",
        )
        lbl_rules_how.grid(row=0, column=1, sticky="w", padx=(12, 0))
        for w in (lbl_rules, lbl_rules_how):
            Tooltip(
                w,
                "Every page is matched against this table top to bottom. The "
                "first row whose conditions fit decides which model runs, and "
                "for grayscale pages whether auto levels is applied. A row that "
                "names a page size wins over a row that says \u201cany\u201d "
                "wherever the two sit, so a catch-all at the top cannot swallow "
                "everything by accident. Double-click a row to edit it, or click "
                "its dot to switch it off without deleting it.",
                self.theme,
            )

        self.rules_table = Table(
            self.rules_box,
            (
                # key, heading, width, anchor, takes slack, never narrower than
                ("on", "On", 40, "center", False, 38),
                ("when", "When", 150, "w", False, 86),
                ("size", "Page size", 120, "w", False, 78),
                ("model", "Model", 420, "w", True, 160),
                ("levels", "Auto levels", 96, "center", False, 92),
            ),
            height=8,
        )
        self.rules_table.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.tree_rules = self.rules_table.tree
        self.tree_rules.bind("<Double-1>", self.on_rule_double_click)
        self.tree_rules.bind("<Button-1>", self.on_rule_click, add="+")
        self.tree_rules.bind("<space>", lambda _e: self.rule_toggle(), add="+")

        side = ttk.Frame(self.rules_box, style="Plain.TFrame")
        side.grid(row=1, column=1, sticky="n", padx=(10, 0), pady=(8, 0))
        for index, (text, action, tip) in enumerate(
            (
                ("Add", self.rule_add, "Add a rule below the selected one."),
                ("Edit", self.rule_edit, "Edit the selected rule (or double-click it)."),
                (
                    "Toggle",
                    self.rule_toggle,
                    (
                        "Switch the selected rule off without deleting it. The dot "
                        "in the first column shows the state; clicking the dot does "
                        "the same thing."
                    ),
                ),
                ("Remove", self.rule_remove, "Delete the selected rule."),
                (
                    "Up",
                    lambda: self.rule_move(-1),
                    "Move the rule up. Order only decides between rules that are equally specific.",
                ),
                ("Down", lambda: self.rule_move(1), "Move the rule down."),
                (
                    "Defaults",
                    self.rules_reset,
                    (
                        "Rewrite this table as the shipped set: the MangaJaNai "
                        "height bands for grayscale pages and the IllustrationJaNai "
                        "denoise models for colour, built from the models you have "
                        "installed. Only the table is touched, nothing else."
                    ),
                ),
            )
        ):
            button = ttk.Button(side, text=text, style="Ghost.TButton", command=action)
            button.grid(row=index, column=0, sticky="ew", pady=(0 if index == 0 else 4, 0))
            Tooltip(button, tip, self.theme)

        self.lbl_rules_hint = ttk.Label(
            self.rules_box, text="", style="Muted.TLabel", wraplength=760, justify="left"
        )
        self.lbl_rules_hint.grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        self.lbl_rules_warn = ttk.Label(
            self.rules_box, text="", style="Warn.TLabel", wraplength=760, justify="left"
        )

        adv = ttk.Frame(b, style="Plain.TFrame")
        adv.grid(row=4, column=1, sticky="w", pady=(14, 0))
        ttk.Label(adv, text="Gray threshold", style="Muted.TLabel").grid(row=0, column=0)
        sp = int_spin(adv, self.var_threshold, 0, 64, 1, 5, self.update_summary)
        sp.grid(row=0, column=1, padx=(8, 16))
        Tooltip(
            sp,
            "How far a pixel's red, green and blue may drift apart (0-255) before "
            "it counts as coloured. 12 matches the original app. Raise it to send "
            "yellowed or sepia scans to the grayscale model anyway; lower it if "
            "faintly tinted pages should be treated as colour.",
            self.theme,
        )
        ttk.Label(adv, text="Colour pixels %", style="Muted.TLabel").grid(row=0, column=2)
        sp3 = int_spin(adv, self.var_colour_pct, 0.0, 25.0, 0.05, 6, self.update_summary)
        sp3.grid(row=0, column=3, padx=(8, 16))
        Tooltip(
            sp3,
            "The second test, for pages whose average still looks gray: once this "
            "share of pixels is clearly coloured, the page is treated as colour. "
            "0.25% catches a coloured title or one spot-colour panel on an "
            "otherwise black-and-white page. Set it to 0 to judge by the "
            "threshold alone.",
            self.theme,
        )
        ttk.Label(adv, text="Pre-downscale height", style="Muted.TLabel").grid(row=0, column=4)
        sp2 = int_spin(adv, self.var_pre_h, 0, 20000, 100, 7, self.update_summary)
        sp2.grid(row=0, column=5, padx=(8, 0))
        Tooltip(
            sp2,
            "0 = off. Shrinks an oversized page to this height first and then "
            "upscales it as usual. Worth using when raws are far larger than the "
            "model was trained for - a 3000px scan through a 1600p model - since "
            "the model sees fewer pixels, which is both faster and often cleaner. "
            "The page is still upscaled: this is not the long-strip option below.",
            self.theme,
        )
        c3 = ttk.Checkbutton(
            adv,
            text="Convert huge long strips without upscaling",
            variable=self.var_skip_long,
            command=self.update_summary,
        )
        c3.grid(row=1, column=0, columnspan=6, sticky="w", pady=(10, 0))
        Tooltip(
            c3,
            "For webtoon mega-strips: very tall, very many pixels. Those pages "
            "skip the model completely and are only re-encoded into the chosen "
            "output format, because upscaling a 20000px strip costs minutes and "
            "gains little. Off by default - the adaptive tiler copes with them. "
            "This is the opposite of Pre-downscale height, which shrinks a page "
            "and still upscales it.",
            self.theme,
        )

        self.lbl_upscale_sum = ttk.Label(
            b, text="", style="Muted.TLabel", wraplength=620, justify="left"
        )
        self.lbl_upscale_sum.grid(row=5, column=1, sticky="w", pady=(12, 0))

    def _build_output(self, body: tk.Misc) -> None:
        card = Card(
            body,
            "Output",
            subtitle="The encoder, how pages are packaged, and where they land.",
        )
        card.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        self.card_output = card
        b = card.body

        row_label(b, 0, "Format", "The encoder used for every page.", self.theme)
        self.seg_fmt = Segmented(
            b,
            self.var_fmt,
            [(FORMATS[fid].label, fid) for fid in FORMAT_IDS],
            command=lambda _v: self.on_format_change(),
        )
        self.seg_fmt.grid(row=0, column=1, sticky="w", pady=4)

        self.lbl_fmt_hint = ttk.Label(
            b, text="", style="Muted.TLabel", wraplength=560, justify="left"
        )
        self.lbl_fmt_hint.grid(row=1, column=1, sticky="w", pady=(0, 6))

        self.opt_frame = ttk.Frame(b, style="Card.TFrame")
        self.opt_frame.grid(row=2, column=1, sticky="ew")
        self.opt_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            b,
            text="Show advanced encoder options",
            variable=self.var_adv,
            command=self.render_format_options,
        ).grid(row=3, column=1, sticky="w", pady=(6, 10))

        ttk.Separator(b).grid(row=4, column=0, columnspan=2, sticky="ew", pady=(2, 10))

        row_label(
            b, 5, "Package", "Loose files, or pack the pages into comic archives.", self.theme
        )
        self.seg_container = Segmented(
            b,
            self.var_container,
            [(CONTAINERS[cid].label, cid) for cid in CONTAINER_IDS],
            command=lambda _v: self.on_container_change(),
        )
        self.seg_container.grid(row=5, column=1, sticky="w", pady=4)
        self.lbl_container_hint = ttk.Label(
            b, text="", style="Muted.TLabel", wraplength=620, justify="left"
        )
        self.lbl_container_hint.grid(row=6, column=1, sticky="w", pady=(0, 8))

        ttk.Separator(b).grid(row=7, column=0, columnspan=2, sticky="ew", pady=(2, 10))

        row_label(b, 8, "Destination", "", self.theme)
        dest = ttk.Frame(b, style="Card.TFrame")
        dest.grid(row=8, column=1, sticky="ew", pady=2)
        dest.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            dest,
            text="Next to the input, in subfolder",
            variable=self.var_same,
            command=self.on_dest_change,
        ).grid(row=0, column=0, sticky="w")
        self.e_sub = entry(dest, self.var_subfolder, width=18)
        self.e_sub.grid(row=0, column=1, sticky="w", padx=(8, 0))
        self.e_sub.bind("<KeyRelease>", lambda _e: self.update_summary(), add="+")

        self.dest_custom = ttk.Frame(b, style="Card.TFrame")
        self.dest_custom.grid(row=9, column=1, sticky="ew", pady=(4, 0))
        self.dest_custom.columnconfigure(0, weight=1)
        self.e_out = entry(self.dest_custom, self.var_out_dir)
        self.e_out.grid(row=0, column=0, sticky="ew")
        self.e_out.bind("<KeyRelease>", lambda _e: self.update_summary(), add="+")
        ttk.Button(
            self.dest_custom,
            text="Browse\u2026",
            style="Ghost.TButton",
            command=self.choose_out_dir,
        ).grid(row=0, column=1, padx=(8, 0))

        self.lbl_names = row_label(b, 10, "File names", "", self.theme)
        names = ttk.Frame(b, style="Card.TFrame")
        names.grid(row=10, column=1, sticky="ew", pady=(6, 0))
        names.columnconfigure(0, weight=1)
        e_pat = entry(names, self.var_pattern)
        e_pat.grid(row=0, column=0, sticky="ew")
        e_pat.bind("<KeyRelease>", lambda _e: self.update_summary(), add="+")
        Tooltip(
            e_pat,
            "Tokens: {name} original name, {parent} folder name, "
            "{index} position, {index0} zero-padded position. With a CBZ "
            "package this names the pages inside the archive.",
            self.theme,
        )
        flags = ttk.Frame(b, style="Card.TFrame")
        flags.grid(row=11, column=1, sticky="w", pady=(8, 0))
        ttk.Checkbutton(
            flags,
            text="Overwrite existing",
            variable=self.var_overwrite,
            command=self.update_summary,
        ).grid(row=0, column=0, sticky="w")
        c = ttk.Checkbutton(
            flags,
            text="Mirror folder structure",
            variable=self.var_keep_tree,
            command=self.update_summary,
        )
        c.grid(row=0, column=1, sticky="w", padx=(18, 0))
        Tooltip(c, "Recreate the input's subfolder layout inside the output folder.", self.theme)

        self.lbl_out_sum = ttk.Label(
            b, text="", style="Muted.TLabel", wraplength=620, justify="left"
        )
        self.lbl_out_sum.grid(row=12, column=1, sticky="w", pady=(10, 0))

    def _build_perf(self, body: tk.Misc) -> None:
        panel = Collapsible(
            body,
            "4 \u00b7 Performance",
            expanded=bool(self.settings.data["ui"].get("perf_open", False)),
            subtitle="device, precision, tiling, threads",
        )
        panel.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.panel_perf = panel
        b = panel.body

        row_label(b, 0, "Device", "Auto picks the fastest GPU it can find.", self.theme)
        self.cb_device = combo(
            b, self.var_device, self._device_labels(), width=42, on_change=self.on_device_change
        )
        self.cb_device.grid(row=0, column=1, sticky="w", pady=4)

        self.chk_fp16 = ttk.Checkbutton(
            b, text="FP16 (half precision)", variable=self.var_fp16, command=self.update_summary
        )
        self.chk_fp16.grid(row=1, column=1, sticky="w", pady=(2, 0))
        Tooltip(
            self.chk_fp16,
            "On by default: roughly twice as fast and half the VRAM on "
            "supported GPUs. The worker falls back to FP32 by itself on "
            "hardware or models that cannot do it.",
            self.theme,
        )
        self.lbl_fp16 = ttk.Label(b, text="", style="Muted.TLabel")
        self.lbl_fp16.grid(row=2, column=1, sticky="w", pady=(0, 6))

        row_label(b, 3, "Tile size", "Splits large images so they fit in VRAM.", self.theme)
        cb_tile = combo(
            b,
            self.var_tile,
            [label for label, _ in TILE_CHOICES],
            width=24,
            on_change=self.update_summary,
        )
        cb_tile.grid(row=3, column=1, sticky="w", pady=4)
        Tooltip(
            cb_tile,
            "Auto measures what the model actually costs on the first tiles and "
            "grows to the largest tile that fits the free VRAM \u2014 fewer, "
            "bigger tiles means fewer seams and less overhead.",
            self.theme,
        )

        row_label(b, 4, "VRAM budget (GiB)", "0 = no cap.", self.theme)
        int_spin(b, self.var_budget, 0, 128, 1, 6, self.update_summary).grid(
            row=4, column=1, sticky="w", pady=4
        )

        row_label(b, 5, "CPU threads", "0 = let torch decide.", self.theme)
        int_spin(b, self.var_threads, 0, 256, 1, 6, self.update_summary).grid(
            row=5, column=1, sticky="w", pady=4
        )

        row_label(
            b, 6, "I/O workers", "Threads that decode and encode while the GPU works.", self.theme
        )
        int_spin(b, self.var_io, 1, 16, 1, 6, self.update_summary).grid(
            row=6, column=1, sticky="w", pady=4
        )

        row_label(b, 7, "libvips concurrency", "0 = libvips default.", self.theme)
        int_spin(b, self.var_vips, 0, 64, 1, 6, self.update_summary).grid(
            row=7, column=1, sticky="w", pady=4
        )

        extra = ttk.Frame(b, style="Card.TFrame")
        extra.grid(row=8, column=1, sticky="w", pady=(8, 0))
        c1 = ttk.Checkbutton(extra, text="cuDNN autotune", variable=self.var_cudnn)
        c1.grid(row=0, column=0, sticky="w")
        Tooltip(
            c1,
            "Benchmarks convolution algorithms once per shape. Faster for long runs "
            "of same-sized pages.",
            self.theme,
        )
        c2 = ttk.Checkbutton(extra, text="TF32 matmuls", variable=self.var_tf32)
        c2.grid(row=0, column=1, sticky="w", padx=(18, 0))
        Tooltip(c2, "Ampere and newer: faster matmuls at slightly reduced precision.", self.theme)
        c3 = ttk.Checkbutton(extra, text="Wipe cache between images", variable=self.var_wipe)
        c3.grid(row=0, column=2, sticky="w", padx=(18, 0))
        Tooltip(
            c3,
            "Frees VRAM after every image. Slower, but avoids fragmentation on small GPUs.",
            self.theme,
        )
        c4 = ttk.Checkbutton(
            extra, text="Keep GPU awake", variable=self.var_wake, command=self.on_wake_toggle
        )
        c4.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8, 0))
        Tooltip(
            c4,
            "Holds a tiny context on the GPU while this window is open, so the "
            "driver keeps the card powered (and a laptop dGPU does not park) and "
            "the first run starts at full speed. Costs a few MB of VRAM and is "
            "released automatically while a job runs.",
            self.theme,
        )

    def _build_footer(self) -> None:
        foot = ttk.Frame(self.root, padding=(18, 8, 18, 10))
        foot.grid(row=2, column=0, sticky="ew")
        foot.columnconfigure(0, weight=1)
        self.foot = foot

        self.bar = ttk.Progressbar(foot, variable=self.var_progress, maximum=100)
        self.bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8))

        left = ttk.Frame(foot)
        left.grid(row=1, column=0, sticky="w")
        ttk.Label(left, textvariable=self.var_status, style="TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(left, textvariable=self.var_detail, style="MutedBg.TLabel").grid(
            row=1, column=0, sticky="w", pady=(2, 0)
        )

        right = ttk.Frame(foot)
        right.grid(row=1, column=1, sticky="e")
        self.btn_log = ttk.Button(right, text="Log", style="Ghost.TButton", command=self.toggle_log)
        self.btn_log.grid(row=0, column=0, padx=(0, 6))
        self.btn_open = ttk.Button(
            right, text="Open output", style="Ghost.TButton", command=self.open_output
        )
        self.btn_open.grid(row=0, column=1, padx=(0, 6))
        self.btn_dry = ttk.Button(
            right, text="Dry run", style="Ghost.TButton", command=self.start_dry
        )
        self.btn_dry.grid(row=0, column=2, padx=(0, 6))
        Tooltip(
            self.btn_dry,
            "Walks the whole job and reports every file it would write, "
            "the size, the model and the archives it would build \u2014 "
            "without touching the disk or the GPU.",
            self.theme,
        )
        self.btn_pause = ttk.Button(right, text="Pause", command=self.toggle_pause)
        self.btn_pause.grid(row=0, column=3, padx=(0, 6))
        self.btn_pause.state(["disabled"])
        self.btn_start = ttk.Button(right, text="Start", style="Accent.TButton", command=self.start)
        self.btn_start.grid(row=0, column=4)

    def _build_log(self) -> None:
        """The log lives in its own window row so it can actually grow.

        It used to be an unweighted row inside the footer with a fixed height of
        nine lines, which is why it never filled the window.
        """
        panel = ttk.Frame(self.root, padding=(18, 0, 18, 12))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        self.log_panel = panel

        bar = ttk.Frame(panel)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        bar.columnconfigure(1, weight=1)
        ttk.Label(bar, text="Run log", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(bar, textvariable=self.var_log_file, style="MutedBg.TLabel").grid(
            row=0, column=1, sticky="w", padx=(12, 12)
        )

        tools = ttk.Frame(bar)
        tools.grid(row=0, column=2, sticky="e")
        cw = ttk.Checkbutton(
            tools, text="Wrap", variable=self.var_log_wrap, command=self.apply_log_wrap
        )
        cw.grid(row=0, column=0, padx=(0, 10))
        cd = ttk.Checkbutton(
            tools, text="Debug", variable=self.var_log_debug, command=self.update_summary
        )
        cd.grid(row=0, column=1, padx=(0, 10))
        Tooltip(
            cd,
            "Show the noisy lines (library warnings, tracebacks). They are always "
            "written to the run file either way.",
            self.theme,
        )
        ttk.Button(tools, text="Copy", style="Ghost.TButton", command=self.copy_log).grid(
            row=0, column=2, padx=(0, 6)
        )
        ttk.Button(
            tools, text="Save as\u2026", style="Ghost.TButton", command=self.save_log_as
        ).grid(row=0, column=3, padx=(0, 6))
        ttk.Button(
            tools, text="Log folder", style="Ghost.TButton", command=self.open_log_folder
        ).grid(row=0, column=4, padx=(0, 6))
        ttk.Button(tools, text="Clear", style="Ghost.TButton", command=self.clear_log).grid(
            row=0, column=5, padx=(0, 6)
        )
        ttk.Button(tools, text="Hide", style="Ghost.TButton", command=self.toggle_log).grid(
            row=0, column=6
        )

        self.log_box = tk.Text(
            panel,
            height=10,
            bd=0,
            highlightthickness=0,
            wrap="word" if self.var_log_wrap.get() else "none",
            font=self.theme.fonts["mono"],
            state="disabled",
        )
        self.log_box.grid(row=1, column=0, sticky="nsew")
        self.log_scroll = ttk.Scrollbar(panel, orient="vertical", command=self.log_box.yview)
        self.log_scroll.grid(row=1, column=1, sticky="ns")
        self.log_box.configure(yscrollcommand=self.log_scroll.set)
        self.restyle_log()

    # ------------------------------------------------------------------ #
    # keyboard
    # ------------------------------------------------------------------ #
    def _bind_keys(self) -> None:
        r = self.root
        r.bind("<Control-o>", lambda _e: self.choose_file())
        r.bind("<Control-O>", lambda _e: self.choose_file())
        r.bind("<Control-Shift-o>", lambda _e: self.choose_folder())
        r.bind("<Control-Shift-O>", lambda _e: self.choose_folder())
        r.bind("<Control-Return>", lambda _e: self.start())
        r.bind("<Control-Shift-Return>", lambda _e: self.start_dry())
        r.bind("<Escape>", lambda _e: self.cancel())
        r.bind("<Control-l>", lambda _e: self.toggle_log())
        r.bind("<Control-d>", lambda _e: self.toggle_theme())
        r.bind("<F5>", lambda _e: self.refresh_probe())

    # ------------------------------------------------------------------ #
    # input handling
    # ------------------------------------------------------------------ #
    def choose_file(self) -> None:
        types = [
            (
                "Images and archives",
                (
                    "*.png *.jpg *.jpeg *.webp *.avif *.jxl *.bmp *.tif "
                    "*.tiff *.gif *.cbz *.zip *.cbr *.rar"
                ),
            ),
            ("All files", "*.*"),
        ]
        path = filedialog.askopenfilename(title="Choose an image or archive", filetypes=types)
        if path:
            self.set_input(path)

    def choose_folder(self) -> None:
        path = filedialog.askdirectory(title="Choose a folder", mustexist=True)
        if path:
            self.set_input(path)

    def choose_out_dir(self) -> None:
        path = filedialog.askdirectory(title="Choose an output folder")
        if path:
            self.var_same.set(False)
            self.var_out_dir.set(path)
            self.on_dest_change()

    def clear_input(self) -> None:
        self.var_in_path.set("")
        self.scan_text = ""
        self.lbl_path.configure(text="No input selected")
        self.card_input.set_badge("")
        self.update_start_state()
        self.update_summary()

    def on_drop(self, paths: list[str]) -> None:
        if paths:
            self.set_input(paths[0])
            if len(paths) > 1:
                self.log(f"{len(paths)} items dropped; using {Path(paths[0]).name}", "warn")

    def set_input(self, path: str) -> None:
        self.var_in_path.set(path)
        self.lbl_path.configure(text=path)
        self.scan_input_async(path)
        self.update_start_state()
        self.update_summary()

    def on_input_options(self) -> None:
        if self.var_in_path.get():
            self.scan_input_async(self.var_in_path.get())
        self.update_summary()

    def scan_input_async(self, path: str) -> None:
        recursive = bool(self.var_recursive.get())
        archives = bool(self.var_archives.get())

        def work() -> None:
            p = Path(path)
            images = arch = 0
            folders: set[str] = set()
            kind = "single"
            try:
                if p.is_file():
                    if p.suffix.lower() in ARCHIVE_EXTS:
                        arch = 1
                    else:
                        images = 1
                elif p.is_dir():
                    kind = "bulk"
                    it = p.rglob("*") if recursive else p.glob("*")
                    for q in it:
                        if not q.is_file():
                            continue
                        ext = q.suffix.lower()
                        if ext in IMAGE_EXTS:
                            images += 1
                            folders.add(str(q.parent))
                        elif archives and ext in ARCHIVE_EXTS:
                            arch += 1
                else:
                    kind = "missing"
            except Exception:
                pass
            self.runner.events.put(
                {
                    "type": "scan",
                    "kind": kind,
                    "images": images,
                    "archives": arch,
                    "folders": len(folders),
                    "path": path,
                }
            )

        threading.Thread(target=work, name="scan", daemon=True).start()

    # ------------------------------------------------------------------ #
    # dynamic rendering
    # ------------------------------------------------------------------ #
    def render_target(self) -> None:
        mode = self.var_mode.get()
        for frame in (self.f_scale, self.f_width, self.f_height, self.f_fit):
            frame.grid_remove()
        if mode == "scale":
            self.f_scale.grid(row=0, column=0, sticky="w")
        elif mode == "width":
            self.f_width.grid(row=0, column=0, sticky="w")
        elif mode == "height":
            self.f_height.grid(row=0, column=1, sticky="w")
        else:
            self.f_width.grid(row=0, column=0, sticky="w")
            self.f_height.grid(row=0, column=1, sticky="w")
            self.f_fit.grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.update_summary()

    def on_display_change(self) -> None:
        """Fill the fit box from a device preset (or leave it alone for Custom)."""
        did = displays.id_for_label(self.var_display.get())
        wh = displays.size(did, bool(self.var_portrait.get()))
        if wh is not None:
            self.var_width.set(wh[0])
            self.var_height.set(wh[1])
        self.update_summary()

    def on_gray_toggle(self) -> None:
        """Detection decides which half of the table can fire, so redraw it."""
        self.render_rules()
        self.update_summary()

    def on_levels_toggle(self) -> None:
        """Rows whose Levels cell says "default" follow this checkbox."""
        self.render_rules()
        self.update_summary()

    # ------------------------------------------------------------------ #
    # rules
    # ------------------------------------------------------------------ #
    def model_names(self) -> list[str]:
        return [str(m.get("name")) for m in self.models if m.get("name")]

    def seed_rules(self) -> None:
        """Make sure the table is filled in and names real files, once.

        Both cases need the probe to have reported the installed models: a
        first run gets the shipped working set written out in full, and a
        settings file from an older build gets its legacy "auto" rows resolved
        to the file they would have picked.
        """
        if self._rules_seeded or not self.models:
            return
        installed = self.model_names()
        if not self.rules:
            self.rules = rules.default_working_set(installed)
        elif any(r.is_auto for r in self.rules):
            self.rules, notes = rules.materialise(self.rules, installed)
            for note in notes:
                self.log(f"rule resolved: {note}", "debug")
            if not self.rules:
                self.rules = rules.default_working_set(installed)
        self._rules_seeded = True
        self.save_rules()

    def render_rules(self) -> None:
        """Redraw the table and its hint. Warnings are refreshed separately."""
        tree = self.tree_rules
        keep = self.selected_rule()
        tree.delete(*tree.get_children())
        gray_on = bool(self.var_gray.get())
        for index, rule in enumerate(self.rules):
            tags: tuple[str, ...] = ()
            if not rule.enabled:
                tags = ("off",)
            elif rule.kind == rules.GRAYSCALE and not gray_on:
                tags = ("idle",)
            tree.insert("", "end", iid=str(index), values=rule.cells(), tags=tags)
        tree.tag_configure("off", foreground=self.theme.p.muted)
        tree.tag_configure("idle", foreground=self.theme.p.muted)
        if 0 <= keep < len(self.rules):
            tree.selection_set(str(keep))

        if not self.rules:
            hint = (
                "The table is empty, so nothing can run. \u201cDefaults\u201d fills it "
                "with the shipped set, built from the models you have installed."
            )
        else:
            active = sum(1 for r in self.rules if r.enabled)
            hint = f"{active} of {len(self.rules)} rules on"
            if not gray_on:
                hint += "  \u00b7  grayscale rules are idle while detection is off"
            hint += "  \u00b7  click a dot to switch a row off, double-click a row to edit"
        self.lbl_rules_hint.configure(text=hint)
        self.refresh_rule_warnings()

    def refresh_rule_warnings(self) -> None:
        """Everything wrong with the table, on one line underneath it.

        Kept apart from :meth:`render_rules` because the target scale can change
        without the rows changing, and rebuilding the rows would drop the
        selection under the user's cursor.
        """
        installed = self.model_names()
        notes: list[str] = []
        for index, rule in enumerate(self.rules):
            if not rule.enabled:
                continue
            notes.extend(f"row {index + 1}: {note}" for note in rules.problems(rule, installed))
        notes.extend(self.scale_mismatches())
        if notes:
            self.lbl_rules_warn.configure(text="\u26a0  " + "; ".join(notes[:4]))
            self.lbl_rules_warn.grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))
        else:
            self.lbl_rules_warn.grid_remove()

    def rules_summary(self) -> str:
        """One phrase for the card summary: how much of the table is live."""
        if not self.rules:
            return "no rules \u2014 the table is empty"
        active = sum(1 for r in self.rules if r.enabled)
        return f"{active} of {len(self.rules)} rules on"

    def scale_mismatches(self) -> list[str]:
        """Rows whose model name advertises a factor the target will not use.

        Only a plain scale target has one fixed factor. In width, height and
        fit modes the factor depends on each page, so comparing a name there
        would fire on perfectly sensible setups. Rules that deliberately target
        another factor are skipped too - they simply will not match.
        """
        if self.var_mode.get() != "scale":
            return []
        want = self.safe_float(self.var_scale, 2.0)
        out: list[str] = []
        for index, rule in enumerate(self.rules):
            if not rule.enabled or rule.is_auto:
                continue
            if rule.scale and abs(rule.scale - want) > 0.01:
                continue
            found = rules.model_scale(rule.model)
            if found and abs(found - want) > 0.01:
                out.append(
                    f"row {index + 1} runs a {found}\u00d7 model but the target is "
                    f"{want:g}\u00d7, so the result gets resampled"
                )
        return out

    def save_rules(self) -> None:
        """Rules are remembered as they are edited, not only on Start."""
        self.settings.set("upscale", "rules", [r.to_dict() for r in self.rules])
        self.settings.save()

    def selected_rule(self) -> int:
        chosen = self.tree_rules.selection()
        return int(chosen[0]) if chosen else -1

    def rules_changed(self, select: int = -1) -> None:
        self.render_rules()
        if 0 <= select < len(self.rules):
            self.tree_rules.selection_set(str(select))
        self.save_rules()
        self.update_summary()

    def on_rule_click(self, event: tk.Event) -> str | None:
        """Clicking the On dot toggles that row; anywhere else just selects."""
        tree = self.tree_rules
        if tree.identify_region(event.x, event.y) != "cell":
            return None
        if tree.identify_column(event.x) != "#1":
            return None
        item = tree.identify_row(event.y)
        if not item:
            return None
        tree.selection_set(item)
        self.rule_toggle()
        return "break"

    def on_rule_double_click(self, event: tk.Event) -> str:
        """Double-click edits - except on the dot, where it would toggle twice."""
        if self.tree_rules.identify_column(event.x) != "#1":
            self.rule_edit()
        return "break"

    def rule_toggle(self) -> None:
        index = self.selected_rule()
        if index < 0:
            return
        data = self.rules[index].to_dict()
        data["enabled"] = not self.rules[index].enabled
        self.rules[index] = rules.Rule.from_dict(data)
        self.rules_changed(index)

    def default_rule_model(self) -> str:
        """A sensible model to open a new rule with - never a placeholder."""
        names = self.model_names()
        if not names:
            return ""
        scale = max(1, round(self.safe_float(self.var_scale, 2.0)))
        return rules.gray_model(names, scale, rules.GRAY_TOP_BUCKET) or names[0]

    def rule_add(self) -> None:
        draft = rules.Rule(
            kind=rules.GRAYSCALE,
            scale=self.safe_float(self.var_scale, 2.0),
            auto_levels=True,
            model=self.default_rule_model(),
        )
        made = self.rule_dialog("Add rule", draft)
        if made is not None:
            self.rules.append(made)
            self.rules_changed(len(self.rules) - 1)

    def rule_edit(self) -> None:
        index = self.selected_rule()
        if index < 0:
            return
        made = self.rule_dialog("Edit rule", self.rules[index])
        if made is not None:
            self.rules[index] = made
            self.rules_changed(index)

    def rule_remove(self) -> None:
        index = self.selected_rule()
        if index < 0:
            return
        del self.rules[index]
        self.rules_changed(min(index, len(self.rules) - 1))

    def rule_move(self, delta: int) -> None:
        index = self.selected_rule()
        if index < 0:
            return
        self.rules = rules.move(self.rules, index, delta)
        self.rules_changed(max(0, min(index + delta, len(self.rules) - 1)))

    def rules_reset(self) -> None:
        if self.rules and not messagebox.askyesno(
            "Reset the rules table",
            "Replace every row with the shipped set, built from the models you "
            "have installed?\n\nOnly this table changes \u2014 the rest of your "
            "settings are left alone.",
            parent=self.root,
        ):
            return
        self.rules = rules.default_working_set(self.model_names())
        self._rules_seeded = True
        self.rules_changed(0)
        self.log(f"rules reset to the shipped set ({len(self.rules)} rows)")

    def rule_dialog(self, title: str, draft: rules.Rule) -> rules.Rule | None:
        """A small modal editor for one rule."""
        scales = {"any": 0.0, "1x": 1.0, "2x": 2.0, "4x": 4.0}
        # "default" follows the Auto levels checkbox in the Upscale card. It
        # used to read "inherit", which never said what it inherited from.
        levels: dict[str, bool | None] = {"default": None, "on": True, "off": False}

        win = tk.Toplevel(self.root)
        win.title(title)
        win.transient(self.root)
        win.resizable(False, False)
        box = ttk.Frame(win, padding=16)
        box.grid(row=0, column=0, sticky="nsew")

        v_kind = tk.StringVar(value=draft.kind)
        v_scale = tk.StringVar(
            value=next((k for k, v in scales.items() if v == draft.scale), "any")
        )
        v_width = tk.StringVar(value=draft.width)
        v_height = tk.StringVar(value=draft.height)
        v_model = tk.StringVar(value=draft.model)
        v_levels = tk.StringVar(value=next(k for k, v in levels.items() if v is draft.auto_levels))
        v_on = tk.BooleanVar(value=draft.enabled)

        fields = (
            (
                "Page kind",
                combo(box, v_kind, list(rules.KINDS), width=16),
                (
                    "Which pages this rule may claim: ones detected as grayscale, "
                    "ones detected as colour, or any page. Grayscale rules never "
                    "fire while Grayscale detection is off."
                ),
            ),
            (
                "Target scale",
                combo(box, v_scale, list(scales), width=16),
                (
                    "Restricts the rule to one output factor, so 2x and 4x rows can "
                    "live in the same table. \u201cany\u201d fires whatever the "
                    "target is, which is what width, height and fit targets need "
                    "since their factor changes per page."
                ),
            ),
            (
                "Page width",
                entry(box, v_width, width=18),
                (
                    "Matched against the source page width in pixels, before "
                    "upscaling. Leave it on any unless you need to separate double "
                    "spreads from single pages."
                ),
            ),
            (
                "Page height",
                entry(box, v_height, width=18),
                (
                    "Matched against the source page height in pixels, before "
                    "upscaling. This is the one that matters for manga: the "
                    "MangaJaNai models are trained per page height."
                ),
            ),
            (
                "Model",
                combo(box, v_model, self.model_names(), width=54),
                (
                    "The weights this rule runs. Only installed models are listed, "
                    "and the factor in the name (1x, 2x, 4x) is what the model was "
                    "trained for - matching it to your target avoids a resample."
                ),
            ),
            (
                "Auto levels",
                combo(box, v_levels, list(levels), width=16),
                (
                    "Grayscale pages only. \u201cdefault\u201d follows the Auto "
                    "levels checkbox in the Upscale card; \u201con\u201d and "
                    "\u201coff\u201d override it for the pages this rule claims."
                ),
            ),
        )
        for row, (label, widget, tip) in enumerate(fields):
            lbl = ttk.Label(box, text=label, style="MutedBg.TLabel")
            lbl.grid(row=row, column=0, sticky="w", padx=(0, 14), pady=4)
            widget.grid(row=row, column=1, sticky="w", pady=4)
            Tooltip(lbl, tip, self.theme)
            Tooltip(widget, tip, self.theme)
        ttk.Label(
            box,
            text="Sizes accept 1920, 1920p, a range 1600-1920, an open end 1985- "
            "or -1250, or any. A rule that names a size always beats a rule that "
            "says any, wherever the two sit in the table.",
            style="MutedBg.TLabel",
            wraplength=430,
            justify="left",
        ).grid(row=len(fields), column=0, columnspan=2, sticky="w", pady=(10, 0))
        chk = ttk.Checkbutton(box, text="Rule is on", variable=v_on, style="Bg.TCheckbutton")
        chk.grid(row=len(fields) + 1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        Tooltip(
            chk,
            "A rule that is off stays in the table and is skipped. The table "
            "shows it as a hollow dot in the first column.",
            self.theme,
        )

        out: dict[str, rules.Rule] = {}

        def commit() -> None:
            out["rule"] = rules.Rule.from_dict(
                {
                    "kind": v_kind.get(),
                    "scale": scales.get(v_scale.get(), 0.0),
                    "width": v_width.get(),
                    "height": v_height.get(),
                    "model": v_model.get(),
                    "auto_levels": levels[v_levels.get()],
                    "enabled": bool(v_on.get()),
                    "note": draft.note,
                }
            )
            win.destroy()

        bar = ttk.Frame(box)
        bar.grid(row=len(fields) + 2, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(bar, text="Cancel", style="Ghost.TButton", command=win.destroy).grid(
            row=0, column=0, padx=(0, 8)
        )
        ttk.Button(bar, text="Save", style="Accent.TButton", command=commit).grid(row=0, column=1)
        win.bind("<Return>", lambda _e: commit())
        win.bind("<Escape>", lambda _e: win.destroy())

        win.update_idletasks()
        win.geometry(f"+{self.root.winfo_rootx() + 90}+{self.root.winfo_rooty() + 100}")
        win.grab_set()
        self.root.wait_window(win)
        return out.get("rule")

    # ------------------------------------------------------------------ #
    # presets
    # ------------------------------------------------------------------ #
    def preset_menu(self) -> None:
        """Save, load, or jump straight to a preset already saved."""
        p = self.theme.p
        menu = tk.Menu(
            self.root,
            tearoff=0,
            borderwidth=0,
            activeborderwidth=0,
            background=p.surface2,
            foreground=p.text,
            activebackground=p.accent,
            activeforeground=p.accent_text,
        )
        menu.add_command(label="Save current settings\u2026", command=self.preset_save)
        menu.add_command(label="Load from file\u2026", command=self.preset_load)
        saved = presets.available(self.dir)
        if saved:
            menu.add_separator()
            for name, path in saved:
                menu.add_command(label=name, command=lambda target=path: self.preset_apply(target))
        try:
            menu.tk_popup(
                self.btn_presets.winfo_rootx(),
                self.btn_presets.winfo_rooty() + self.btn_presets.winfo_height(),
            )
        finally:
            menu.grab_release()

    def preset_save(self) -> None:
        self.sync_settings()
        name = simpledialog.askstring(
            "Save preset", "Name this preset:", parent=self.root, initialvalue="My settings"
        )
        if not name:
            return
        payload = presets.build(name, self.settings.data, app_version=__version__)
        try:
            written = presets.write(presets.folder(self.dir) / presets.filename(name), payload)
        except OSError as exc:
            self.show_banner(f"Could not write the preset: {exc}")
            return
        self.settings.save()
        self.log(f"preset saved: {written.name}  \u00b7  {presets.summary(payload)}")

    def preset_load(self) -> None:
        start = presets.folder(self.dir)
        chosen = filedialog.askopenfilename(
            parent=self.root,
            title="Load preset",
            initialdir=str(start if start.is_dir() else self.dir),
            filetypes=[("JaNai preset", "*.json"), ("All files", "*.*")],
        )
        if chosen:
            self.preset_apply(Path(chosen))

    def preset_apply(self, path: Path) -> None:
        try:
            preset = presets.read(path)
        except presets.PresetError as exc:
            self.show_banner(str(exc))
            return
        self.sync_settings()  # so untouched sections survive the merge
        changed = presets.apply(self.settings.data, preset)
        self.reload_widgets()
        name = str(preset.get("name") or path.stem)
        if changed:
            self.hide_banner()
            self.log(f"preset applied: {name}  \u00b7  {', '.join(changed)}")
        else:
            self.log(f"preset {name} already matches the current settings")

    def reload_widgets(self) -> None:
        """Push the settings dict back into the widgets, after a preset.

        The device and the folders are deliberately left alone: a preset
        describes how to convert, not where this machine keeps things.
        """
        d = self.settings.data
        u, o, p = d["upscale"], d["output"], d["perf"]
        lg = d.get("log") or {}

        self.var_mode.set(str(u.get("mode", "scale")))
        self.var_scale.set(float(u.get("scale", 2.0)))
        self.var_width.set(int(u.get("width", 2048)))
        self.var_height.set(int(u.get("height", 2160)))
        self.var_display.set(displays.label_for_id(str(u.get("display", displays.CUSTOM))))
        self.var_portrait.set(bool(u.get("display_portrait", True)))
        self.var_levels.set(bool(u.get("auto_levels", True)))
        self.var_gray.set(bool(u.get("grayscale_convert", True)))
        self.var_threshold.set(int(u.get("grayscale_threshold", 12)))
        self.var_colour_pct.set(float(u.get("grayscale_colour_percent", 0.25)))
        self.var_pre_h.set(int(u.get("pre_downscale_height", 0)))
        self.var_skip_long.set(bool(u.get("skip_long_strips", False)))
        self.rules = [rules.Rule.from_dict(r) for r in (u.get("rules") or [])]
        self._rules_seeded = bool(self.rules)

        self.var_container.set(str(o.get("container", "files")))
        self.var_same.set(bool(o.get("same_as_input", True)))
        self.var_subfolder.set(str(o.get("subfolder", "upscaled")))
        self.var_pattern.set(str(o.get("pattern", "{name}")))
        self.var_overwrite.set(bool(o.get("overwrite", False)))
        self.var_keep_tree.set(bool(o.get("keep_structure", True)))

        self.var_fmt.set(str(d["format"].get("id", "png")))
        for fid, spec in FORMATS.items():
            saved = self.settings.format_options(fid)
            for opt in spec.opts:
                var = self.fmt_vars.get(fid, {}).get(opt.key)
                if var is None:
                    continue
                value = saved.get(opt.key, opt.default)
                if opt.kind == "bool":
                    var.set(bool(value))
                elif opt.kind == "int":
                    var.set(int(value))
                elif opt.kind == "float":
                    var.set(float(value))
                else:
                    var.set(choice_label(opt, value))

        self.var_fp16.set(bool(p.get("use_fp16", True)))
        self.var_tile.set(self._tile_label(str(p.get("tile", "auto"))))
        self.var_budget.set(int(p.get("budget_limit", 0)))
        self.var_wipe.set(bool(p.get("force_cache_wipe", False)))
        self.var_threads.set(int(p.get("torch_threads", 0)))
        self.var_io.set(int(p.get("io_workers", 2)))
        self.var_vips.set(int(p.get("vips_concurrency", 0)))
        self.var_cudnn.set(bool(p.get("cudnn_benchmark", False)))
        self.var_tf32.set(bool(p.get("allow_tf32", True)))
        self.var_wake.set(bool(p.get("gpu_wake_lock", True)))
        self.var_log_wrap.set(bool(lg.get("wrap", False)))
        self.var_log_debug.set(bool(lg.get("show_debug", False)))

        self.render_format_options()
        self.render_target()
        self.render_rules()
        self.apply_log_wrap()
        self.update_summary()
        self.update_start_state()
        self.update_wake_lock()
        self.settings.save()

    def on_container_change(self) -> None:
        self.update_summary()
        self.update_start_state()

    def on_format_change(self) -> None:
        self.render_format_options()

    def on_opt_change(self, rerender: bool = False) -> None:
        if rerender:
            self.root.after_idle(self.render_format_options)
        else:
            self.update_summary()

    def render_format_options(self) -> None:
        clear(self.opt_frame)
        fid = self.var_fmt.get()
        if fid not in FORMATS:
            fid = "png"
            self.var_fmt.set(fid)
        spec = FORMATS[fid]
        values = self.format_values(fid)
        show_adv = bool(self.var_adv.get())

        cap = self.caps.get(fid, {})
        hint = spec.hint
        if self.caps and not cap.get("ok", False):
            hint = f"Not available in this install \u2014 {cap.get('reason', 'unsupported')}"
        elif cap.get("via") and cap.get("via") != "libvips":
            hint = f"{spec.hint}  (encoded with {cap['via']})"
        self.lbl_fmt_hint.configure(
            text=hint,
            style="Err.TLabel" if (self.caps and not cap.get("ok", False)) else "Muted.TLabel",
        )

        row = 0
        hidden_adv = False
        for opt in spec.opts:
            if not is_active(opt, values):
                continue
            if opt.advanced and not show_adv:
                hidden_adv = True
                continue
            row_label(self.opt_frame, row, opt.label, opt.hint, self.theme)
            var = self.fmt_vars[fid][opt.key]
            if opt.kind == "bool":
                w: tk.Widget = ttk.Checkbutton(
                    self.opt_frame,
                    text="",
                    variable=var,
                    command=lambda: self.on_opt_change(rerender=True),
                )
            elif opt.kind == "choice":
                w = combo(
                    self.opt_frame,
                    var,
                    [lbl for lbl, _ in opt.choices],
                    width=26,
                    on_change=lambda: self.on_opt_change(rerender=True),
                )
            else:
                step = opt.step or (0.1 if opt.kind == "float" else 1)
                w = int_spin(self.opt_frame, var, opt.lo, opt.hi, step, 8, self.update_summary)
            w.grid(row=row, column=1, sticky="w", pady=3)
            if opt.hint:
                Tooltip(w, opt.hint, self.theme)
            row += 1
        if hidden_adv and not show_adv:
            ttk.Label(self.opt_frame, text="More options are hidden", style="Muted.TLabel").grid(
                row=row, column=1, sticky="w", pady=(4, 0)
            )
        self.update_summary()

    def format_values(self, fid: str) -> dict:
        out: dict[str, Any] = {}
        for opt in FORMATS[fid].opts:
            var = self.fmt_vars[fid][opt.key]
            try:
                raw = var.get()
            except Exception:
                raw = opt.default
            try:
                if opt.kind == "bool":
                    out[opt.key] = bool(raw)
                elif opt.kind == "int":
                    out[opt.key] = int(raw)
                elif opt.kind == "float":
                    out[opt.key] = float(raw)
                elif opt.kind == "choice":
                    out[opt.key] = choice_value(opt, str(raw))
                else:
                    out[opt.key] = raw
            except Exception:
                out[opt.key] = opt.default
        return out

    def on_dest_change(self) -> None:
        same = bool(self.var_same.get())
        self.e_sub.state(["!disabled"] if same else ["disabled"])
        if same:
            self.dest_custom.grid_remove()
        else:
            self.dest_custom.grid(row=9, column=1, sticky="ew", pady=(4, 0))
        self.update_start_state()
        self.update_summary()

    def on_device_change(self) -> None:
        """Keep the user's FP16 preference; only report what the device can do.

        The checkbox used to be forced off (and saved off) on any device that
        reported no fp16 support, which then followed the user to the next GPU.
        """
        value = self.device_value()
        self._saved_device = value
        known = self._known_devices()
        if value:
            dev = next((d for d in known if str(d.get("value")) == value), None)
        else:
            dev = next((d for d in known if str(d.get("value")) != "cpu"), None)
        note = ""
        if dev is not None:
            supported = bool(dev.get("fp16")) and str(dev.get("value")) != "cpu"
            self.chk_fp16.state(["!disabled"] if supported else ["disabled"])
            if not supported and self.var_fp16.get():
                note = (
                    "this device runs FP32 \u2014 the preference is kept for GPUs that support it"
                )
            elif supported and self.var_fp16.get():
                note = "half precision on"
        self.lbl_fp16.configure(text=note)
        self.update_wake_lock()
        self.update_summary()

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
        for d in self._known_devices():
            if str(d.get("value")) == value:
                return str(d.get("label"))
        return value

    def _device_labels(self) -> list[str]:
        labels = [AUTO_DEVICE] + [str(d.get("label")) for d in self._known_devices()]
        current = self.var_device.get()
        if current and current not in labels:
            labels.append(current)
        return labels

    def device_value(self) -> str:
        """The device string for the job. Empty means auto, never a silent CPU
        fallback: before the probe lands the saved choice is kept."""
        label = self.var_device.get().strip()
        if not label or label == AUTO_DEVICE:
            return ""
        for d in self._known_devices():
            if str(d.get("label")) == label:
                return str(d.get("value"))
        if DEVICE_RE.match(label):
            return label
        return self._saved_device

    # ------------------------------------------------------------------ #
    # summaries
    # ------------------------------------------------------------------ #
    def update_summary(self) -> None:
        mode = self.var_mode.get()
        if mode == "scale":
            target = f"{self.safe_float(self.var_scale, 2.0):g}\u00d7"
        elif mode == "width":
            target = f"{self.safe_int(self.var_width, 2048)} px wide"
        elif mode == "height":
            target = f"{self.safe_int(self.var_height, 2160)} px tall"
        else:
            w = self.safe_int(self.var_width, 2048)
            h = self.safe_int(self.var_height, 2160)
            target = f"fit {w}\u00d7{h}"
            found = displays.match(w, h)
            if displays.id_for_label(self.var_display.get()) != found:
                self.var_display.set(displays.label_for_id(found))
            if found != displays.CUSTOM:
                target += f" ({self.var_display.get()})"
        bits = [target, self.rules_summary()]
        if self.var_gray.get():
            bits.append("grayscale detection")
        if self.var_levels.get():
            bits.append("auto levels")
        if self.safe_int(self.var_pre_h, 0):
            bits.append(f"pre-downscale {self.safe_int(self.var_pre_h, 0)}px")
        if self.var_skip_long.get():
            bits.append("long strips passed through")
        self.lbl_upscale_sum.configure(text=" \u00b7 ".join(bits))
        self.refresh_rule_warnings()

        cid = self.container_value()
        self.lbl_container_hint.configure(text=CONTAINERS[cid].hint)
        self.lbl_names.configure(text="Page names" if packs_archive(cid) else "File names")

        fid = self.var_fmt.get()
        if fid in FORMATS:
            try:
                enc = summary(fid, self.format_values(fid))
            except Exception:
                enc = FORMATS[fid].label
            dest = self.resolved_out_dir()
            dest_text = str(dest) if dest else "choose a destination"
            pack = f"  \u2192  {CONTAINERS[cid].label}" if packs_archive(cid) else ""
            self.lbl_out_sum.configure(text=f"{enc}{pack}  \u2192  {dest_text}")

        dev = self.var_device.get() or AUTO_DEVICE
        fp16 = "FP16" if self.var_fp16.get() else "FP32"
        tile = self.var_tile.get()
        hint = [dev, fp16, f"tile {tile.lower()}"]
        if self.var_wake.get():
            hint.append("GPU kept awake")
        self.panel_perf.set_hint(" \u00b7 ".join(hint))

    def container_value(self) -> str:
        cid = str(self.var_container.get() or "files")
        return cid if cid in CONTAINERS else "files"

    @staticmethod
    def safe_int(var: tk.Variable, fallback: int) -> int:
        try:
            return int(float(var.get()))
        except Exception:
            return fallback

    @staticmethod
    def safe_float(var: tk.Variable, fallback: float) -> float:
        try:
            return float(var.get())
        except Exception:
            return fallback

    def resolved_out_dir(self) -> Path | None:
        src = self.var_in_path.get().strip()
        if bool(self.var_same.get()):
            if not src:
                return None
            p = Path(src)
            base = p.parent if p.is_file() else p
            sub = self.var_subfolder.get().strip() or "upscaled"
            return base / sub
        custom = self.var_out_dir.get().strip()
        return Path(custom) if custom else None

    def update_start_state(self) -> None:
        ok = bool(self.var_in_path.get().strip()) and self.resolved_out_dir() is not None
        # An empty table means no model would run, so there is nothing to start.
        ok = ok and any(r.enabled for r in self.rules)
        if self.runner.running:
            self.btn_start.configure(text="Cancel", style="TButton", command=self.cancel)
            self.btn_start.state(["!disabled"])
            self.btn_pause.state(["disabled"] if self.dry else ["!disabled"])
            self.btn_dry.state(["disabled"])
        else:
            self.btn_start.configure(text="Start", style="Accent.TButton", command=self.start)
            self.btn_start.state(["!disabled"] if ok else ["disabled"])
            self.btn_dry.state(["!disabled"] if ok else ["disabled"])
            self.btn_pause.state(["disabled"])
            self.btn_pause.configure(text="Pause")
        if self.last_out_dir is None:
            self.btn_open.state(["disabled"])
        else:
            self.btn_open.state(["!disabled"])

    # ------------------------------------------------------------------ #
    # GPU wake lock
    # ------------------------------------------------------------------ #
    def wake_lock_device(self) -> str | None:
        """Device to hold awake: "" for auto, None when it should not be held."""
        if not self.var_wake.get():
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
        if not self.var_wake.get() and self.runner.holding:
            self.log("GPU wake lock off")
        self.update_wake_lock()
        self.update_summary()

    def on_hold(self, ev: dict) -> None:
        if ev.get("released"):
            return
        if ev.get("ok"):
            held = int(ev.get("reserved") or 0)
            where = str(ev.get("name") or ev.get("device") or "GPU")
            self.log(
                f"GPU wake lock on {where}" + (f" ({fmt_bytes(held)} reserved)" if held else ""),
                "debug",
            )
        else:
            self.log(f"GPU wake lock unavailable: {ev.get('error', 'unknown reason')}", "warn")

    # ------------------------------------------------------------------ #
    # probe
    # ------------------------------------------------------------------ #
    def refresh_probe(self) -> None:
        self.lbl_env.configure(text="detecting hardware\u2026")
        self.btn_refresh.state(["disabled"])
        self.runner.probe()

    def apply_probe(self, probe: dict, cached: bool = False) -> None:
        if not probe:
            self.cb_device.configure(values=self._device_labels())
            if not self.var_device.get():
                self.var_device.set(AUTO_DEVICE)
            return
        self.probe = probe
        self.caps = probe.get("formats", {}) or {}
        self.models = probe.get("models", []) or []
        self.devices = probe.get("devices", []) or []

        labels = self._device_labels()
        if self.var_device.get() not in labels:
            match = next(
                (d for d in self.devices if str(d.get("value")) == self._saved_device), None
            )
            self.var_device.set(str(match["label"]) if match else AUTO_DEVICE)
            labels = self._device_labels()
        self.cb_device.configure(values=labels)

        # The shipped table is built from the models actually installed, so it
        # can only be seeded once the probe has reported them.
        self.seed_rules()
        self.render_rules()

        for fid in FORMAT_IDS:
            self.seg_fmt.set_enabled(fid, bool(self.caps.get(fid, {}).get("ok", True)))
        if not self.caps.get(self.var_fmt.get(), {}).get("ok", True):
            fallback = next((f for f in FORMAT_IDS if self.caps.get(f, {}).get("ok")), "png")
            self.log(
                f"{FORMATS[self.var_fmt.get()].label} is unavailable, switching to "
                f"{FORMATS[fallback].label}",
                "warn",
            )
            self.var_fmt.set(fallback)

        env_bits = []
        gpu = next((d for d in self.devices if d.get("value") != "cpu"), None)
        if gpu:
            vram = int(gpu.get("vram") or 0)
            env_bits.append(
                f"{gpu.get('label')}" + (f" \u00b7 {vram / 1024**3:.1f} GB" if vram else "")
            )
            if gpu.get("fp16"):
                env_bits.append("FP16 capable")
        else:
            env_bits.append("CPU only")
        if probe.get("torch"):
            env_bits.append(
                f"torch {probe['torch']}" + (f" cu{probe.get('cuda')}" if probe.get("cuda") else "")
            )
        if probe.get("libvips"):
            env_bits.append(f"libvips {probe['libvips']}")
        env_bits.append(f"{len(self.models)} models")
        origins = (probe.get("paths") or {}).get("origins") or {}
        if origins.get("python_dir") == "config":
            env_bits.append("external runtime")
        ok_fmts = [FORMATS[f].label for f in FORMAT_IDS if self.caps.get(f, {}).get("ok")]
        if ok_fmts:
            env_bits.append("writes " + "/".join(ok_fmts))
        self.lbl_env.configure(text="  \u00b7  ".join(env_bits) + ("  (cached)" if cached else ""))

        if not cached:
            self.settings.data["probe"] = probe
            self.btn_refresh.state(["!disabled"])
            errors = probe.get("errors") or []
            if errors:
                self.show_banner("; ".join(str(e) for e in errors))
            elif not self.models:
                where = probe.get("models_dir") or "backend/models"
                self.show_banner(
                    f"No models found in {where}. Run setup.cmd to download the model "
                    "packs, or drop .pth files there; without models images are only "
                    "resized."
                )
            else:
                self.hide_banner()
        self.on_device_change()
        self.render_format_options()
        self.update_summary()

    def show_banner(self, text: str) -> None:
        self.banner.configure(text=text)
        if not self._banner_visible:
            self.banner.grid(row=0, column=0, sticky="ew", pady=(6, 4))
            self._banner_visible = True

    def hide_banner(self) -> None:
        if self._banner_visible:
            self.banner.grid_remove()
            self._banner_visible = False

    # ------------------------------------------------------------------ #
    # job control
    # ------------------------------------------------------------------ #
    def sync_settings(self) -> None:
        d = self.settings.data
        d["theme"] = self.theme.p.name
        d["input"] = {
            "path": self.var_in_path.get(),
            "recursive": bool(self.var_recursive.get()),
            "include_archives": bool(self.var_archives.get()),
        }
        d["upscale"] = {
            "mode": self.var_mode.get(),
            "scale": self.safe_float(self.var_scale, 2.0),
            "width": self.safe_int(self.var_width, 2048),
            "height": self.safe_int(self.var_height, 2160),
            "display": displays.id_for_label(self.var_display.get()),
            "display_portrait": bool(self.var_portrait.get()),
            "auto_levels": bool(self.var_levels.get()),
            "grayscale_convert": bool(self.var_gray.get()),
            "grayscale_threshold": self.safe_int(self.var_threshold, 12),
            "grayscale_colour_percent": self.safe_float(self.var_colour_pct, 0.25),
            "pre_downscale_height": self.safe_int(self.var_pre_h, 0),
            "skip_long_strips": bool(self.var_skip_long.get()),
            "rules": [r.to_dict() for r in self.rules],
        }
        d["format"]["id"] = self.var_fmt.get()
        for fid in FORMATS:
            d["format"]["options"][fid] = self.format_values(fid)
        d["output"] = {
            "dir": self.var_out_dir.get(),
            "same_as_input": bool(self.var_same.get()),
            "subfolder": self.var_subfolder.get(),
            "container": self.container_value(),
            "pattern": self.var_pattern.get(),
            "overwrite": bool(self.var_overwrite.get()),
            "keep_structure": bool(self.var_keep_tree.get()),
        }
        d["perf"] = {
            "device": self.device_value(),
            "use_fp16": bool(self.var_fp16.get()),
            "tile": self._tile_value(self.var_tile.get()),
            "budget_limit": self.safe_int(self.var_budget, 0),
            "force_cache_wipe": bool(self.var_wipe.get()),
            "torch_threads": self.safe_int(self.var_threads, 0),
            "io_workers": max(1, self.safe_int(self.var_io, 2)),
            "vips_concurrency": self.safe_int(self.var_vips, 0),
            "cudnn_benchmark": bool(self.var_cudnn.get()),
            "allow_tf32": bool(self.var_tf32.get()),
            "gpu_wake_lock": bool(self.var_wake.get()),
        }
        log_cfg = dict(d.get("log") or {})
        log_cfg.update(
            {"wrap": bool(self.var_log_wrap.get()), "show_debug": bool(self.var_log_debug.get())}
        )
        d["log"] = log_cfg
        d["ui"] = dict(
            d.get("ui") or {},
            advanced_format=bool(self.var_adv.get()),
            perf_open=self.panel_perf.is_open(),
            log_open=self._log_visible,
            geometry=self.root.winfo_geometry(),
        )

    def build_job(self, dry: bool = False) -> dict | None:
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
            "perf": d["perf"],
        }
        if dry:
            job["dry_run"] = True
        return job

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
        self.started_at = time.time()
        self.var_progress.set(0)
        self.last_out_dir = Path(job["output"]["dir"])
        self.var_status.set("Planning\u2026" if dry else "Starting worker\u2026")
        self.var_detail.set("")
        self.begin_run_log(job, dry)
        if not dry and not self.runner.has_portable_python():
            self.log("no backend runtime found, using the interpreter running the GUI", "warn")
        self.runner.hold_stop()
        if self.runner.start(job):
            self.update_start_state()

    def start_dry(self) -> None:
        self.start(dry=True)

    def cancel(self) -> None:
        if self.runner.running:
            self.runner.cancel()
            self.var_status.set("Cancelling\u2026")

    def toggle_pause(self) -> None:
        if not self.runner.running:
            return
        if self.runner.paused:
            self.runner.resume()
            self.btn_pause.configure(text="Pause")
            self.var_status.set("Resumed")
        else:
            self.runner.pause()
            self.btn_pause.configure(text="Resume")
            self.var_status.set("Paused")

    def open_output(self) -> None:
        target = self.last_out_dir or self.resolved_out_dir()
        if target is not None and Path(target).exists():
            open_in_explorer(Path(target))

    # ------------------------------------------------------------------ #
    # events
    # ------------------------------------------------------------------ #
    def _tick(self) -> None:
        self.runner.drain(self.on_event)
        self.root.after(90, self._tick)

    def on_event(self, ev: dict) -> None:
        kind = ev.get("type")
        if kind == "scan":
            self.on_scan(ev)
        elif kind == "probe":
            self.apply_probe(ev)
        elif kind == "probe_error":
            self.btn_refresh.state(["!disabled"])
            self.lbl_env.configure(text="backend not ready")
            self.show_banner(
                "The Python backend is not ready. Run setup.cmd in this folder "
                f"to create it.\n{ev.get('message', '')}".strip()
            )
        elif kind == "start":
            self.total = int(ev.get("total") or 0)
            self.dry = bool(ev.get("dry")) or self.dry
            self.var_status.set(("Planning 0/" if self.dry else "Upscaling 0/") + str(self.total))
            self.var_detail.set(
                f"{ev.get('device')} \u00b7 "
                f"{'FP16' if ev.get('fp16') else 'FP32'} \u00b7 "
                f"tile {ev.get('tile')}"
            )
            for text, level in format_start(ev):
                self.log(text, level)
        elif kind == "progress":
            self.on_progress(ev)
        elif kind == "file":
            self.on_file(ev)
        elif kind == "bundle":
            text, level = format_bundle(ev)
            self.log(text, level)
        elif kind == "log":
            self.log(str(ev.get("message", "")), str(ev.get("level", "info")))
        elif kind == "done":
            self.on_done(ev)
        elif kind == "hold":
            self.on_hold(ev)
        elif kind == "exit":
            self.update_start_state()
            self.update_wake_lock()
            if int(ev.get("code") or 0) not in (0, 1, 2):
                self.var_status.set("Worker stopped unexpectedly")
                self.log(f"worker exited with code {ev.get('code')}", "error")
            self.end_run_log()

    def on_scan(self, ev: dict) -> None:
        if ev.get("path") != self.var_in_path.get():
            return
        kind = ev.get("kind")
        images = int(ev.get("images") or 0)
        archives = int(ev.get("archives") or 0)
        folders = int(ev.get("folders") or 0)
        if kind == "missing":
            self.card_input.set_badge("not found")
            self.scan_text = ""
            return
        parts = []
        if images:
            parts.append(f"{images} image{'s' if images != 1 else ''}")
        if archives:
            parts.append(f"{archives} archive{'s' if archives != 1 else ''}")
        if folders > 1:
            parts.append(f"{folders} folders")
        self.scan_text = ", ".join(parts) or "nothing to do"
        self.card_input.set_badge(
            ("Single" if kind == "single" else "Bulk") + " \u00b7 " + self.scan_text
        )
        self.update_summary()

    def on_progress(self, ev: dict) -> None:
        name = Path(str(ev.get("path", ""))).name
        i = int(ev.get("i") or 0)
        total = int(ev.get("total") or self.total or 0)
        sub = ""
        if ev.get("sub_n"):
            sub = f" (page {ev.get('sub_i')}/{ev.get('sub_n')})"
        head = "Planning" if self.dry else "Upscaling"
        self.var_status.set(f"{head} {i}/{total}{sub}")
        self.var_detail.set(name)

    def on_file(self, ev: dict) -> None:
        total = int(ev.get("total") or self.total or 1)
        error = str(ev.get("error") or "")
        if error:
            if "skip" in error.lower() or "exists" in error.lower():
                self.skipped += 1
            else:
                self.failed += 1
        else:
            self.completed += 1
        text, level = format_file(ev)
        self.log(text, level)
        finished = self.completed + self.failed + self.skipped
        self.var_progress.set(min(100.0, 100.0 * finished / max(1, total)))
        elapsed = max(0.001, time.time() - self.started_at)
        rate = self.completed / elapsed if self.completed else 0
        left = total - finished
        eta = f" \u00b7 ETA {fmt_secs(left / rate)}" if rate > 0 and left > 0 else ""
        head = "Planning" if self.dry else "Upscaling"
        self.var_status.set(f"{head} {finished}/{total}{eta}")

    def on_done(self, ev: dict) -> None:
        elapsed = float(ev.get("elapsed") or (time.time() - self.started_at))
        processed = int(ev.get("processed") or self.completed)
        failed = int(ev.get("failed") or self.failed)
        skipped = int(ev.get("skipped") or self.skipped)
        dry = bool(ev.get("dry")) or self.dry
        text, level = format_done(ev)
        self.log(text, level)
        if ev.get("error"):
            self.var_status.set("Failed")
            self.var_detail.set(str(ev["error"]))
        elif ev.get("cancelled"):
            self.var_status.set(f"Cancelled after {processed} file(s)")
            self.var_detail.set(fmt_secs(elapsed))
        else:
            bits = [f"{processed} file(s) in {fmt_secs(elapsed)}"]
            if failed:
                bits.append(f"{failed} failed")
            if skipped:
                bits.append(f"{skipped} skipped")
            self.var_status.set("Dry run complete" if dry else "Done")
            self.var_detail.set(" \u00b7 ".join(bits))
            self.var_progress.set(100)
        self.update_start_state()
        self.settings.save()

    # ------------------------------------------------------------------ #
    # log
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
        self.var_log_file.set(str(path) if path else "not saved")
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

    def restyle_log(self) -> None:
        p = self.theme.p
        self.log_box.configure(
            background=p.surface, foreground=p.text, insertbackground=p.text, selectbackground=p.sel
        )
        self.log_box.tag_configure("info", foreground=p.text)
        self.log_box.tag_configure("debug", foreground=p.muted)
        self.log_box.tag_configure("skip", foreground=p.muted)
        self.log_box.tag_configure("ok", foreground=p.ok)
        self.log_box.tag_configure("dry", foreground=p.accent)
        self.log_box.tag_configure("warn", foreground=p.warn)
        self.log_box.tag_configure("error", foreground=p.err)

    def apply_log_wrap(self) -> None:
        self.log_box.configure(wrap="word" if self.var_log_wrap.get() else "none")

    def show_log(self, visible: bool) -> None:
        """Give the log its own weighted window row, so it fills what is left."""
        weight = max(1, int(self.settings.data["ui"].get("log_weight", 2) or 2))
        if visible:
            self.log_panel.grid(row=3, column=0, sticky="nsew")
            self.root.rowconfigure(3, weight=weight, minsize=180)
            self.btn_log.configure(text="Hide log")
        else:
            self.log_panel.grid_remove()
            self.root.rowconfigure(3, weight=0, minsize=0)
            self.btn_log.configure(text="Log")
        self._log_visible = visible

    def toggle_log(self) -> None:
        self.show_log(not self._log_visible)

    def log(self, message: str, level: str = "info") -> None:
        if not message:
            return
        stamp = time.strftime("%H:%M:%S")
        tag = level if level in LOG_TAGS else "info"
        self.runlog.write(stamp, message)
        if tag == "debug" and not self.var_log_debug.get():
            return
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{stamp}  {message}\n", tag)
        lines = int(self.log_box.index("end-1c").split(".")[0])
        if lines > 4000:
            self.log_box.delete("1.0", f"{lines - 4000}.0")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def log_text(self) -> str:
        return self.log_box.get("1.0", "end-1c")

    def copy_log(self) -> None:
        text = self.log_text()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def save_log_as(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save log",
            defaultextension=".log",
            initialfile=f"Run_{time.strftime('%Y%m%d-%H%M%S')}.log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(self.log_text() + "\n", encoding="utf-8")
            self.log(f"log saved to {path}")
        except Exception as exc:
            self.log(f"could not save the log: {exc}", "error")

    def open_log_folder(self) -> None:
        target = self._log_dir()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if target.exists():
            open_in_explorer(target)

    def clear_log(self) -> None:
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def toggle_theme(self) -> None:
        mode = self.theme.toggle()
        self.settings.data["theme"] = mode
        self.scroll.restyle()
        self.restyle_log()
        self.render_format_options()
        self.render_rules()  # the on/off row colours come from the palette

    def reset_all(self) -> None:
        """Every setting back to the shipped defaults, folders excluded.

        Deliberately keeps the input and output paths and the cached hardware
        probe: nobody presses this wanting to retype where their manga lives or
        to wait for the backend to be detected again.
        """
        if not messagebox.askyesno(
            "Reset all settings",
            "Put every setting back to its default?\n\nThe rules table, target, "
            "output format and layout, performance options and log view are "
            "reset. Your input and output folders are kept.",
            parent=self.root,
        ):
            return
        old = self.settings.data
        fresh = defaults()
        fresh["theme"] = old.get("theme", fresh.get("theme"))
        fresh["probe"] = old.get("probe", {})
        fresh["input"]["path"] = str((old.get("input") or {}).get("path", ""))
        out = old.get("output") or {}
        fresh["output"]["dir"] = str(out.get("dir", ""))
        fresh["output"]["same_as_input"] = bool(out.get("same_as_input", True))
        fresh["ui"] = dict(
            fresh.get("ui") or {}, geometry=(old.get("ui") or {}).get("geometry", "")
        )
        self.settings.data = fresh
        self._rules_seeded = False
        self.reload_widgets()
        self.seed_rules()
        self.render_rules()
        self.update_summary()
        self.log("all settings reset to defaults")

    # ------------------------------------------------------------------ #
    def on_close(self) -> None:
        try:
            self.sync_settings()
            self.settings.save()
        except Exception:
            pass
        self.end_run_log()
        self.runner.hold_stop()
        if self.runner.running:
            self.runner.cancel()
            self.root.after(400, self.runner.kill)
            self.root.after(600, self.root.destroy)
            return
        self.root.destroy()
