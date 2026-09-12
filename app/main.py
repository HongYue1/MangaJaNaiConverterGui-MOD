"""JaNai Upscaler - GUI entry point.

Run via JaNaiUpscaler.cmd, or directly:
    backend\\python\\Scripts\\pythonw.exe app\\main.py

The GUI imports nothing heavier than the Python standard library; torch and
friends only ever live inside the worker process.

Environment:
    JANAI_SMOKE=1   build the whole window, then close it (used for self-tests)
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def enable_dpi_awareness() -> None:
    """Crisp text on high-DPI Windows displays."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
    except Exception:
        return
    try:
        user32 = ctypes.windll.user32
        if hasattr(user32, "SetProcessDpiAwarenessContext"):
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
            if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except Exception:
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def fatal(message: str) -> None:
    sys.stderr.write(message + "\n")
    try:
        import tkinter as tk
        from tkinter import messagebox

        tmp = tk.Tk()
        tmp.withdraw()
        messagebox.showerror("JaNai Upscaler", message)
        tmp.destroy()
    except Exception:
        pass


def main() -> int:
    enable_dpi_awareness()

    try:
        import tkinter as tk
    except Exception as exc:  # pragma: no cover - broken interpreter
        fatal(
            "This Python build has no Tkinter, so the interface cannot start.\n\n"
            f"{exc}\n\nRun setup.cmd to install the bundled runtime."
        )
        return 2

    try:
        from app.ui import App
    except Exception:
        fatal("Could not load the interface:\n\n" + traceback.format_exc())
        return 2

    root = tk.Tk()
    root.withdraw()
    try:
        scaling = float(root.winfo_fpixels("1i")) / 72.0
        if 0.5 < scaling < 6.0:
            root.tk.call("tk", "scaling", scaling)
    except Exception:
        pass

    try:
        App(root, ROOT)
    except Exception:
        try:
            root.destroy()
        except Exception:
            pass
        fatal("JaNai Upscaler failed to start:\n\n" + traceback.format_exc())
        return 1

    root.deiconify()
    try:
        root.lift()
        root.focus_force()
    except Exception:
        pass

    if os.environ.get("JANAI_SMOKE"):
        root.after(900, root.destroy)

    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
