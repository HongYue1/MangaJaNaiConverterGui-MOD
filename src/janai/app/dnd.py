"""Windows drag & drop for a Tk window, via ctypes only.

``enable(root, on_drop)`` returns True when WM_DROPFILES was wired up. Every
failure path is swallowed: drag & drop is a convenience, never a requirement.
"""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from collections.abc import Callable
from ctypes import wintypes

WM_DROPFILES = 0x0233
GWLP_WNDPROC = -4

_keep_alive: list = []


def enable(root: tk.Misc, on_drop: Callable[[list[str]], None]) -> bool:
    if not sys.platform.startswith("win"):
        return False
    try:
        shell32 = ctypes.windll.shell32
        user32 = ctypes.windll.user32

        try:
            hwnd = int(root.wm_frame(), 16)  # toplevel frame receives WM_DROPFILES
        except Exception:
            hwnd = root.winfo_id()

        LRESULT = ctypes.c_ssize_t
        WNDPROC = ctypes.WINFUNCTYPE(
            LRESULT, wintypes.HWND, wintypes.UINT, ctypes.c_size_t, ctypes.c_ssize_t
        )

        user32.CallWindowProcW.restype = LRESULT
        user32.CallWindowProcW.argtypes = [
            WNDPROC,
            wintypes.HWND,
            wintypes.UINT,
            ctypes.c_size_t,
            ctypes.c_ssize_t,
        ]
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            set_long = user32.SetWindowLongPtrW
            set_long.restype = LRESULT
            set_long.argtypes = [wintypes.HWND, ctypes.c_int, WNDPROC]
        else:
            set_long = user32.SetWindowLongW
            set_long.restype = LRESULT
            set_long.argtypes = [wintypes.HWND, ctypes.c_int, WNDPROC]

        shell32.DragQueryFileW.argtypes = [
            wintypes.HANDLE,
            wintypes.UINT,
            wintypes.LPWSTR,
            wintypes.UINT,
        ]
        shell32.DragQueryFileW.restype = wintypes.UINT
        shell32.DragFinish.argtypes = [wintypes.HANDLE]
        shell32.DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]

        old_proc_ref: list = []

        def handler(h, msg, wparam, lparam):
            if msg == WM_DROPFILES:
                try:
                    hdrop = wintypes.HANDLE(wparam)
                    count = shell32.DragQueryFileW(hdrop, 0xFFFFFFFF, None, 0)
                    buf = ctypes.create_unicode_buffer(32768)
                    # The query fills `buf` and returns its length, so the
                    # condition has to run before the value is read.
                    paths: list[str] = [
                        buf.value
                        for i in range(count)
                        if shell32.DragQueryFileW(hdrop, i, buf, len(buf))
                    ]
                    shell32.DragFinish(hdrop)
                    if paths:
                        root.after(0, lambda p=paths: on_drop(p))
                except Exception:
                    pass
                return 0
            return user32.CallWindowProcW(old_proc_ref[0], h, msg, wparam, lparam)

        new_proc = WNDPROC(handler)
        old_raw = set_long(hwnd, GWLP_WNDPROC, new_proc)
        if not old_raw:
            return False
        old_proc_ref.append(ctypes.cast(old_raw, WNDPROC))
        shell32.DragAcceptFiles(hwnd, True)
        _keep_alive.append((new_proc, old_proc_ref, handler))
        return True
    except Exception:
        return False
