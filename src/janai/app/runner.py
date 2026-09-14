"""Worker process supervision: launch, stream JSONL events, send control lines."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

from janai import worker as worker_pkg
from janai.core import paths


def no_window_flags() -> int:
    if os.name != "nt":
        return 0
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


def open_in_explorer(path: Path) -> None:
    try:
        if os.name == "nt":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


class Runner:
    """Owns at most one worker process and a queue of its events."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.events: queue.Queue[dict] = queue.Queue()
        self.proc: subprocess.Popen | None = None
        self._job_file: Path | None = None
        self._lock = threading.Lock()
        self.paused = False
        self.hold: subprocess.Popen | None = None
        self.hold_device = ""

    # ------------------------------------------------------------------ #
    def paths(self) -> paths.Paths:
        """Where the backend lives right now, re-read on every use so a setup
        run while the window is open is picked up by the next refresh."""
        return paths.resolve(self.root)

    def python_exe(self) -> Path:
        """The resolved backend interpreter, else the one running the GUI."""
        exe = self.paths().interpreter()
        return exe or Path(sys.executable)

    def worker_script(self) -> Path:
        """The worker module on disk, located through the package rather than a
        hardcoded folder name, so the layout can move without breaking launch."""
        return Path(worker_pkg.__file__).resolve().parent / "worker.py"

    def has_portable_python(self) -> bool:
        """Whether the environment in backend\\python was found."""
        return self.paths().interpreter() is not None

    def _env(self) -> dict:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # The worker bootstraps its own sys.path, but passing the import root
        # explicitly also covers being launched from an installed wheel.
        src = str(Path(worker_pkg.__file__).resolve().parents[2])
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{src}{os.pathsep}{existing}" if existing else src
        return env

    # ------------------------------------------------------------------ #
    def probe(self) -> None:
        """Ask the worker about devices, encoders and models (background)."""
        threading.Thread(target=self._probe, name="probe", daemon=True).start()

    def _probe(self) -> None:
        cmd = [str(self.python_exe()), str(self.worker_script()), "--probe"]
        try:
            run = subprocess.run(
                cmd,
                check=False,
                cwd=str(self.root),
                env=self._env(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=no_window_flags(),
                timeout=600,
            )
        except FileNotFoundError as exc:
            self.events.put({"type": "probe_error", "message": f"interpreter not found: {exc}"})
            return
        except Exception as exc:
            self.events.put({"type": "probe_error", "message": str(exc)})
            return
        data = None
        for line in (run.stdout or "").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") == "probe":
                data = obj
        if data is None:
            tail = ((run.stderr or "") + (run.stdout or "")).strip().splitlines()[-6:]
            self.events.put(
                {
                    "type": "probe_error",
                    "message": "\n".join(tail) or "worker produced no probe output",
                }
            )
            return
        self.events.put(data)

    # ------------------------------------------------------------------ #
    # GPU wake lock: a side process that owns a tiny device context
    # ------------------------------------------------------------------ #
    @property
    def holding(self) -> bool:
        return self.hold is not None and self.hold.poll() is None

    def hold_start(self, device: str = "") -> bool:
        """Keep `device` (or the best GPU, when empty) awake and initialised."""
        if self.holding and self.hold_device == device:
            return True
        self.hold_stop()
        cmd = [str(self.python_exe()), str(self.worker_script()), "--hold"]
        if device:
            cmd += ["--device", device]
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.root),
                env=self._env(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=no_window_flags(),
            )
        except Exception as exc:
            self.events.put(
                {
                    "type": "hold",
                    "ok": False,
                    "device": device or "auto",
                    "error": f"could not start: {exc}",
                }
            )
            return False
        self.hold = proc
        self.hold_device = device
        threading.Thread(target=self._read_hold, args=(proc,), name="hold-out", daemon=True).start()
        return True

    def _read_hold(self, proc: subprocess.Popen) -> None:
        if proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("{"):
                try:
                    self.events.put(json.loads(line))
                except Exception:
                    pass
        try:
            proc.wait()
        except Exception:
            pass

    def hold_stop(self) -> None:
        """Release the context so a job (or another app) gets the full VRAM."""
        proc, self.hold = self.hold, None
        self.hold_device = ""
        if proc is None or proc.poll() is not None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.write("stop\n")
                proc.stdin.flush()
                proc.stdin.close()
        except Exception:
            pass

        def reap() -> None:
            try:
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

        threading.Thread(target=reap, name="hold-reap", daemon=True).start()

    # ------------------------------------------------------------------ #
    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, job: dict, extra: tuple[str, ...] = ()) -> bool:
        """Run a job. ``extra`` adds worker flags, e.g. ``("--profile",)``."""
        with self._lock:
            if self.running:
                return False
            tmp = Path(tempfile.gettempdir()) / f"janai-job-{os.getpid()}.json"
            tmp.write_text(json.dumps(job, indent=2), encoding="utf-8")
            self._job_file = tmp
            cmd = [str(self.python_exe()), str(self.worker_script()), "--job", str(tmp)]
            cmd += [str(flag) for flag in extra]
            try:
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(self.root),
                    env=self._env(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    creationflags=no_window_flags(),
                )
            except Exception as exc:
                self.events.put(
                    {
                        "type": "done",
                        "ok": False,
                        "processed": 0,
                        "failed": 0,
                        "skipped": 0,
                        "cancelled": False,
                        "elapsed": 0,
                        "error": f"could not start worker: {exc}",
                    }
                )
                return False
            self.paused = False
        threading.Thread(target=self._read_stdout, name="worker-out", daemon=True).start()
        threading.Thread(target=self._read_stderr, name="worker-err", daemon=True).start()
        return True

    def start_profile(self, job: dict) -> bool:
        """Measure this machine instead of converting anything.

        Deliberately the same process slot as a real job: profiling owns the
        GPU while it runs, so Start has to be unavailable for the duration and
        Cancel has to reach it. Both fall out of ``running``.
        """
        return self.start(job, ("--profile",))

    def _read_stdout(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                try:
                    self.events.put(json.loads(line))
                    continue
                except Exception:
                    pass
            self.events.put({"type": "log", "level": "info", "message": line})
        code = proc.wait()
        self.events.put({"type": "exit", "code": code})
        self._cleanup()

    def _read_stderr(self) -> None:
        proc = self.proc
        if proc is None or proc.stderr is None:
            return
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                self.events.put({"type": "log", "level": "warn", "message": line})

    def _cleanup(self) -> None:
        if self._job_file is not None:
            try:
                self._job_file.unlink(missing_ok=True)
            except Exception:
                pass
            self._job_file = None

    # ------------------------------------------------------------------ #
    def _send(self, command: str) -> None:
        proc = self.proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return
        try:
            proc.stdin.write(command + "\n")
            proc.stdin.flush()
        except Exception:
            pass

    def cancel(self) -> None:
        self._send("cancel")

    def pause(self) -> None:
        self.paused = True
        self._send("pause")

    def resume(self) -> None:
        self.paused = False
        self._send("resume")

    def kill(self) -> None:
        proc = self.proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass

    # Shutdown has to finish synchronously. Closing the last window ends the Qt
    # event loop, so a kill deferred with QTimer (which is what closeEvent used
    # to do) never runs: measured offscreen, the timer never fired and the
    # worker outlived the GUI still holding the GPU.
    SHUTDOWN_GRACE_SECONDS = 3.0  # cancel is honoured per page, so allow one
    KILL_REAP_SECONDS = 2.0  # after kill() only OS teardown is left

    def shutdown(self, grace: float = SHUTDOWN_GRACE_SECONDS) -> None:
        """Stop the worker before this process exits, cooperatively if it can be.

        Sends ``cancel`` and waits, so a worker that reaches a gate exits having
        closed its bundle and cleaned up its temp files. Kills it once the grace
        expires, because a worker inside an uninterruptible libvips write or
        torch forward would otherwise be orphaned with VRAM still allocated and
        no UI left to stop it.
        """
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return
        self.cancel()
        if self._reap(proc, grace):
            return
        self.kill()
        self._reap(proc, self.KILL_REAP_SECONDS)

    @staticmethod
    def _reap(proc: subprocess.Popen, timeout: float) -> bool:
        """True once the process is gone. Safe to race ``_read_stdout``'s wait()."""
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return proc.poll() is not None
        return True

    # ------------------------------------------------------------------ #
    def drain(self, handler: Callable[[dict], None], limit: int = 200) -> None:
        """Dispatch pending events; call from the Qt event loop."""
        for _ in range(limit):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                return
            try:
                handler(event)
            except Exception:
                pass
