"""The Input card: choosing what to convert, and counting it.

Holds the drop zone, the file/folder pickers and the two scope checkboxes,
together with the handlers that read them. ``scan_input_async`` is the only
reason this panel needs a thread: walking a large folder takes long enough to
freeze the window, so the count is produced off the GUI thread and delivered as
a synthetic ``scan`` event through ``runner.events`` - the same queue the worker
process uses, so the window has exactly one place where events arrive.

The extension sets below are the GUI's own copy on purpose. The worker keeps
its authoritative copies in :mod:`janai.worker.planning`, and importing that
from the app would drag the worker's environment bootstrap (sys.path and PATH
side effects) into the GUI process just to preview a file count.

Mixed into :class:`janai.app.window.MainWindow`, so ``self`` is the window.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QFileDialog

from janai.app.widgets import Card, DropZone, button, checkbox, row

if TYPE_CHECKING:
    from janai.app.surface import WindowSurface as _Base
else:  # type-only: at runtime the base is object, so the MRO is untouched
    _Base = object

#: What the pre-run scan counts as a page. The worker decides for real.
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

FILE_FILTER = (
    "Images and archives (*.png *.jpg *.jpeg *.jfif *.webp *.avif *.jxl *.bmp "
    "*.tif *.tiff *.gif *.heic *.heif *.cbz *.zip *.cbr *.rar);;All files (*)"
)


class InputPanelMixin(_Base):
    """The Input card plus the pickers, drop handling and the async scan."""

    # Window state this panel mutates. Re-declared even though the surface
    # already declares it: a mixin that assigns an attribute gets its own
    # inferred slot for it, and without a type here every read placed above the
    # assignment is untypable.
    _in_path: str
    scan_text: str

    def _build_input(self) -> None:
        card = Card("Input", "A folder, an archive, or single images. Drop them here.")
        self.card_input = card
        self.drop = DropZone("Drop images, a folder or a .cbz here")
        self.drop.dropped.connect(self.on_drop)
        self.drop.set_path(self._in_path)
        card.body.full(self.drop)

        card.body.full(
            row(
                button("Choose file\u2026", self.choose_file, tip="Ctrl+O"),
                button("Choose folder\u2026", self.choose_folder, tip="Ctrl+Shift+O"),
                button("Clear", self.clear_input, variant="ghost"),
            )
        )

        i = self.settings.data["input"]
        self.chk_recursive = checkbox(
            "Include subfolders",
            bool(i.get("recursive", True)),
            self.on_input_options,
            tip=(
                "Walk the whole tree. With a CBZ package this is what turns each "
                "chapter folder into its own archive."
            ),
        )
        self.chk_archives = checkbox(
            "Include archives (cbz/zip/cbr/rar)",
            bool(i.get("include_archives", True)),
            self.on_input_options,
            tip=(
                "Comic archives found in the input are re-packed as .cbz with every page upscaled."
            ),
        )
        card.body.full(row(self.chk_recursive, self.chk_archives, spacing=18))
        self.page.addWidget(card)

    def choose_file(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Choose an image or archive", self._in_path or "", FILE_FILTER
        )
        if path:
            self.set_input(path)

    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose a folder", self._in_path or "")
        if path:
            self.set_input(path)

    def clear_input(self) -> None:
        self._in_path = ""
        self.scan_text = ""
        self.drop.set_path("")
        self.card_input.set_badge("")
        self.update_start_state()
        self.update_summary()

    def on_drop(self, paths: list) -> None:
        if not paths:
            return
        self.set_input(str(paths[0]))
        if len(paths) > 1:
            self.log(f"{len(paths)} items dropped; using {Path(paths[0]).name}", "warn")

    def set_input(self, path: str) -> None:
        self._in_path = path
        self.drop.set_path(path)
        self.scan_input_async(path)
        self.update_start_state()
        self.update_summary()

    def on_input_options(self) -> None:
        if self._in_path:
            self.scan_input_async(self._in_path)
        self.update_summary()

    def scan_input_async(self, path: str) -> None:
        """Count what the input holds without blocking the window."""
        recursive = self.chk_recursive.isChecked()
        archives = self.chk_archives.isChecked()

        def work() -> None:
            target = Path(path)
            images = arch = 0
            folders: set[str] = set()
            kind = "single"
            try:
                if target.is_file():
                    if target.suffix.lower() in ARCHIVE_EXTS:
                        arch = 1
                    else:
                        images = 1
                elif target.is_dir():
                    kind = "bulk"
                    walk = target.rglob("*") if recursive else target.glob("*")
                    for item in walk:
                        if not item.is_file():
                            continue
                        ext = item.suffix.lower()
                        if ext in IMAGE_EXTS:
                            images += 1
                            folders.add(str(item.parent))
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
