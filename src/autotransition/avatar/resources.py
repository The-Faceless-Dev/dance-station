"""Process, GPU lease, and cleanup helpers for one-at-a-time avatar jobs."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from autotransition.avatar.observability import current_event_logger


class AvatarProcessError(RuntimeError):
    def __init__(self, message: str, *, returncode: int | None = None, stdout: str = "", stderr: str = ""):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class ProcessResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float


class GpuLease:
    """Serialize GPU work without pretending multiple jobs fit in one device."""

    def __init__(self):
        self._lock = threading.Lock()

    def acquire(self, timeout: float | None = None) -> bool:
        return self._lock.acquire(timeout=-1 if timeout is None else timeout)

    def release(self) -> None:
        if self._lock.locked():
            self._lock.release()


def gpu_status() -> dict[str, object]:
    """Return a cheap, non-invasive CUDA visibility check for readiness."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return {"available": True, "source": "nvidia-smi", "devices": [line.strip() for line in result.stdout.splitlines() if line.strip()]}
        message = result.stderr.strip() or "nvidia-smi did not report a GPU"
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        message = str(exc)
    return {"available": False, "source": "nvidia-smi", "message": message}


def release_cuda_memory() -> None:
    """Release optional framework caches after every attempt."""

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError):
        pass


def terminate_process_tree(process: subprocess.Popen[str]) -> None:
    """Terminate a model process and its children on Windows or POSIX."""

    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_command(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    timeout_seconds: float,
    env: dict[str, str] | None = None,
    log: Callable[[str], None] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    component: str = "avatar-adapter",
) -> ProcessResult:
    """Run a model stage with live, durable, and structured diagnostics."""

    argv = tuple(str(value) for value in command if str(value))
    if not argv:
        raise AvatarProcessError("avatar adapter command is empty")
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    start = time.monotonic()
    event_logger = current_event_logger()
    if stdout_path is not None:
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
    if stderr_path is not None:
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
    if event_logger:
        event_logger.emit(
            "adapter_process_started",
            component=component,
            command=list(argv),
            cwd=cwd,
            timeoutSeconds=timeout_seconds,
            stdoutLog=stdout_path,
            stderrLog=stderr_path,
        )
    try:
        process = subprocess.Popen(
            argv,
            cwd=str(cwd) if cwd else None,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            universal_newlines=True,
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )
    except OSError as exc:
        if event_logger:
            event_logger.exception("adapter_process_start_failed", exc, component=component, command=list(argv))
        raise AvatarProcessError(f"could not start avatar adapter: {exc}") from exc

    stdout_tail: deque[str] = deque(maxlen=2000)
    stderr_tail: deque[str] = deque(maxlen=2000)

    def drain(stream: object, channel: str, path: Path | None, tail: deque[str]) -> None:
        handle = None
        try:
            if path is not None:
                handle = path.open("w", encoding="utf-8", newline="")
            while True:
                line = stream.readline()  # type: ignore[union-attr]
                if line == "":
                    break
                if handle is not None:
                    handle.write(line)
                    handle.flush()
                tail.append(line)
                text = line.rstrip("\r\n")
                if event_logger:
                    event_logger.emit(
                        "adapter_process_output",
                        component=component,
                        stream=channel,
                        line=text,
                    )
                if log:
                    log(text)
        finally:
            if handle is not None:
                handle.close()
            stream.close()  # type: ignore[union-attr]

    stdout_thread = threading.Thread(
        target=drain,
        args=(process.stdout, "stdout", stdout_path, stdout_tail),
        name=f"{component}-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=drain,
        args=(process.stderr, "stderr", stderr_path, stderr_tail),
        name=f"{component}-stderr",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        if event_logger:
            event_logger.emit(
                "adapter_process_timeout",
                component=component,
                command=list(argv),
                timeoutSeconds=timeout_seconds,
                elapsedSeconds=round(time.monotonic() - start, 3),
            )
        terminate_process_tree(process)
        process.wait()
        timeout_error = AvatarProcessError(
            f"avatar adapter timed out after {timeout_seconds:.0f}s",
            returncode=process.returncode,
            stdout="".join(stdout_tail),
            stderr="".join(stderr_tail),
        )
    else:
        timeout_error = None
    stdout_thread.join(timeout=10)
    stderr_thread.join(timeout=10)
    duration = time.monotonic() - start
    stdout = "".join(stdout_tail)
    stderr = "".join(stderr_tail)
    result = ProcessResult(argv, process.returncode, stdout, stderr, duration)
    if event_logger:
        event_logger.emit(
            "adapter_process_finished",
            component=component,
            command=list(argv),
            returnCode=process.returncode,
            durationSeconds=round(duration, 3),
            timedOut=timed_out,
            stdoutLog=stdout_path,
            stderrLog=stderr_path,
            stdoutTail=stdout[-4000:],
            stderrTail=stderr[-4000:],
        )
    if timeout_error is not None:
        raise timeout_error
    if process.returncode != 0:
        if event_logger:
            event_logger.emit(
                "adapter_process_failed",
                component=component,
                command=list(argv),
                returnCode=process.returncode,
                durationSeconds=round(duration, 3),
                stdoutTail=stdout[-4000:],
                stderrTail=stderr[-4000:],
            )
        raise AvatarProcessError(
            f"avatar adapter exited with code {process.returncode}",
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        )
    return result


def disk_free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free
