"""Which files are pages, and which are containers of pages.

Both processes need this answer and neither owns it. The GUI's pre-run scan
counts pages to show "N images, M archives" before a job starts; the worker
decides for real when it walks the input and when it lists an archive's
entries. They used to keep private copies - identical only by luck, with
nothing holding them in step - so adding a suffix on one side would have made
the preview disagree with the result (F6/F11).

This lives in ``janai.core`` rather than in ``janai.worker.planning`` because
the GUI must not import the worker package: the worker's environment bootstrap
(``sys.path`` and ``PATH`` side effects) would follow it into the GUI process
just to preview a file count. Standard library only - in fact no imports at
all - so the GUI, the worker and a dry run all pay nothing for it.
"""

from __future__ import annotations

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
"""Extensions treated as a page. Also filters entries when listing an archive,
so a chapter's cover.txt or ComicInfo.xml is never fed to the model."""

ARCHIVE_EXTS = {".zip", ".cbz", ".rar", ".cbr"}
"""Extensions treated as a container of pages rather than a page."""
