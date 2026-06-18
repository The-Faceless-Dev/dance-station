"""ACE-Step runtime setup/status helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from autotransition.config import RuntimeConfig


@dataclass(frozen=True)
class AceStepRuntimeStatus:
    install_dir: Path
    installed: bool
    uv_available: bool
    git_available: bool
    api_url: str
    api_running: bool
    message: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["install_dir"] = str(self.install_dir)
        return data


@dataclass(frozen=True)
class RuntimeStartResult:
    started: bool
    already_running: bool
    api_url: str
    pid: int | None
    message: str
    managed_by_current_run: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CheckStatus(StrEnum):
    OK = "ok"
    WARN = "warn"
    ERROR = "error"


@dataclass(frozen=True)
class RuntimeCheck:
    name: str
    status: CheckStatus
    message: str

    def to_dict(self) -> dict[str, str]:
        data = asdict(self)
        data["status"] = self.status.value
        return data


@dataclass(frozen=True)
class RuntimeProcess:
    pid: int
    command_line: str


def runtime_pid_path() -> Path:
    return Path("data/runtime/ace-step-api.pid")


def build_install_commands(config: RuntimeConfig = RuntimeConfig()) -> list[str]:
    install_dir = str(config.ace_step_dir)
    return [
        build_uv_install_command(),
        f"git clone https://github.com/ACE-Step/ACE-Step-1.5.git {install_dir}",
        f"cd {install_dir}",
        "uv sync",
    ]


def build_uv_install_command() -> str:
    if sys.platform == "win32":
        return 'powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"'
    return "curl -LsSf https://astral.sh/uv/install.sh | sh"


def build_start_api_command(config: RuntimeConfig = RuntimeConfig()) -> str:
    return "autotransition runtime start"


def build_debug_start_api_command(config: RuntimeConfig = RuntimeConfig()) -> str:
    uv = resolve_uv_executable() or Path("uv")
    return f"{uv} run acestep-api --host {config.api_host} --port {config.api_port}"


def resolve_uv_executable() -> Path | None:
    uv_path = shutil.which("uv")
    if uv_path:
        return Path(uv_path)

    home_uv = Path.home() / ".local" / "bin" / "uv.exe"
    if home_uv.exists():
        return home_uv

    home_uv_posix = Path.home() / ".local" / "bin" / "uv"
    if home_uv_posix.exists():
        return home_uv_posix

    return None


def build_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("AUTOTRANSITION_ALLOW_HF_TRANSFER") != "1":
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    return env


def _write_runtime_pid(pid: int) -> None:
    pid_path = runtime_pid_path()
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(pid), encoding="utf-8")


def _clear_runtime_pid(pid: int | None = None) -> None:
    pid_path = runtime_pid_path()
    if not pid_path.exists():
        return
    if pid is not None:
        try:
            current = int(pid_path.read_text(encoding="utf-8").strip())
        except ValueError:
            current = None
        if current != pid:
            return
    pid_path.unlink(missing_ok=True)


def read_runtime_pid() -> int | None:
    pid_path = runtime_pid_path()
    if not pid_path.exists():
        return None
    try:
        return int(pid_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def find_runtime_processes(config: RuntimeConfig = RuntimeConfig()) -> list[RuntimeProcess]:
    port_text = str(config.api_port)
    if sys.platform == "win32":
        command = (
            "$ErrorActionPreference='SilentlyContinue'; "
            "Get-CimInstance Win32_Process | "
            "Where-Object { $_.CommandLine -like '*acestep-api*' -and "
            f"$_.CommandLine -like '*{port_text}*' -and "
            "$_.ProcessId -ne $PID -and "
            "$_.Name -notin @('powershell.exe','pwsh.exe') } | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        try:
            completed = subprocess.run(
                ["powershell", "-NoProfile", "-Command", command],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return []
        if completed.returncode != 0 or not completed.stdout.strip():
            return []
        import json

        try:
            data = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return []
        rows = data if isinstance(data, list) else [data]
        processes: list[RuntimeProcess] = []
        for row in rows:
            try:
                processes.append(RuntimeProcess(pid=int(row["ProcessId"]), command_line=str(row["CommandLine"])))
            except (KeyError, TypeError, ValueError):
                continue
        return processes

    try:
        completed = subprocess.run(["ps", "-eo", "pid=,args="], check=False, capture_output=True, text=True, timeout=10)
    except Exception:
        return []
    processes = []
    for line in completed.stdout.splitlines():
        stripped = line.strip()
        if "acestep-api" not in stripped or port_text not in stripped:
            continue
        pid_text, _, command_line = stripped.partition(" ")
        try:
            processes.append(RuntimeProcess(pid=int(pid_text), command_line=command_line))
        except ValueError:
            continue
    return processes


def api_health(config: RuntimeConfig = RuntimeConfig()) -> bool:
    try:
        import httpx

        response = httpx.get(f"{config.api_base_url}/health", timeout=config.api_timeout_seconds)
        if response.status_code != 200:
            return False
        content_type = response.headers.get("content-type", "").lower()
        body_preview = response.text[:200].lower()
        if "text/html" in content_type or "<html" in body_preview or "<!doctype html" in body_preview:
            return False
        return True
    except Exception:
        return False


def runtime_status(config: RuntimeConfig = RuntimeConfig()) -> AceStepRuntimeStatus:
    installed = config.ace_step_dir.exists() and (config.ace_step_dir / "pyproject.toml").exists()
    uv_available = resolve_uv_executable() is not None
    git_available = shutil.which("git") is not None
    running = api_health(config)

    if running:
        message = "ACE-Step API is running."
    elif installed:
        message = "ACE-Step runtime is installed, but the API is not running."
    else:
        message = "ACE-Step runtime is not installed. Run the first-time setup command."

    return AceStepRuntimeStatus(
        install_dir=config.ace_step_dir,
        installed=installed,
        uv_available=uv_available,
        git_available=git_available,
        api_url=config.api_base_url,
        api_running=running,
        message=message,
    )


def run_install(config: RuntimeConfig = RuntimeConfig()) -> None:
    """Run first-time setup. This is called only by explicit user command/UI action."""

    uv = resolve_uv_executable()
    if uv is None:
        subprocess.run(build_install_commands(config)[0], shell=True, check=True)
        uv = resolve_uv_executable()
        if uv is None:
            raise RuntimeError("uv was installed, but the uv executable could not be found. Restart your shell and rerun setup.")

    if config.ace_step_dir.exists():
        if not (config.ace_step_dir / "pyproject.toml").exists():
            raise RuntimeError(
                f"ACE-Step runtime directory exists but does not look complete: {config.ace_step_dir}"
            )
    else:
        subprocess.run(build_install_commands(config)[1], shell=True, check=True)

    subprocess.run([str(uv), "sync"], check=True, cwd=config.ace_step_dir)


def start_api_background(config: RuntimeConfig = RuntimeConfig()) -> subprocess.Popen[bytes]:
    if not config.ace_step_dir.exists():
        raise RuntimeError(f"ACE-Step runtime is not installed: {config.ace_step_dir}")
    uv = resolve_uv_executable()
    if uv is None:
        raise RuntimeError("uv.exe was not found. Run `autotransition setup` first.")
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout = (log_dir / "ace-step-api.log").open("ab")
    stderr = (log_dir / "ace-step-api.err.log").open("ab")
    process = subprocess.Popen(
        [str(uv), "run", "acestep-api", "--host", config.api_host, "--port", str(config.api_port)],
        cwd=config.ace_step_dir,
        stdout=stdout,
        stderr=stderr,
        env=build_runtime_env(),
    )
    _write_runtime_pid(process.pid)
    return process


def start_api_foreground(config: RuntimeConfig = RuntimeConfig()) -> int:
    if not config.ace_step_dir.exists():
        raise RuntimeError(f"ACE-Step runtime is not installed: {config.ace_step_dir}")
    uv = resolve_uv_executable()
    if uv is None:
        raise RuntimeError("uv.exe was not found. Run `autotransition runtime setup` first.")
    process = subprocess.run(
        [str(uv), "run", "acestep-api", "--host", config.api_host, "--port", str(config.api_port)],
        cwd=config.ace_step_dir,
        check=False,
        env=build_runtime_env(),
    )
    return process.returncode


def stop_runtime_process_tree(pid: int, config: RuntimeConfig = RuntimeConfig()) -> bool:
    if sys.platform == "win32":
        completed = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True)
        for process in find_runtime_processes(config):
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True)
        stopped = completed.returncode == 0 or not find_runtime_processes(config)
    else:
        try:
            os.kill(pid, 15)
            time.sleep(1)
            stopped = not any(process.pid == pid for process in find_runtime_processes(config))
        except OSError:
            stopped = True
    if stopped:
        _clear_runtime_pid(pid)
    return stopped


def runtime_doctor(config: RuntimeConfig = RuntimeConfig()) -> list[RuntimeCheck]:
    checks: list[RuntimeCheck] = []
    checks.append(
        RuntimeCheck(
            name="git",
            status=CheckStatus.OK if shutil.which("git") else CheckStatus.ERROR,
            message="git is available." if shutil.which("git") else "git is not available on PATH.",
        )
    )

    uv = resolve_uv_executable()
    checks.append(
        RuntimeCheck(
            name="uv",
            status=CheckStatus.OK if uv else CheckStatus.ERROR,
            message=f"uv found at {uv}." if uv else "uv is not installed or could not be resolved.",
        )
    )

    installed = config.ace_step_dir.exists() and (config.ace_step_dir / "pyproject.toml").exists()
    checks.append(
        RuntimeCheck(
            name="runtime",
            status=CheckStatus.OK if installed else CheckStatus.ERROR,
            message=f"ACE-Step runtime installed at {config.ace_step_dir}."
            if installed
            else "ACE-Step runtime is not installed.",
        )
    )

    lock_exists = (config.ace_step_dir / "uv.lock").exists()
    checks.append(
        RuntimeCheck(
            name="sync",
            status=CheckStatus.OK if lock_exists else CheckStatus.WARN,
            message="uv lockfile exists." if lock_exists else "uv sync may not have completed yet.",
        )
    )

    running = api_health(config)
    checks.append(
        RuntimeCheck(
            name="api",
            status=CheckStatus.OK if running else CheckStatus.WARN,
            message=f"ACE-Step API reachable at {config.api_base_url}."
            if running
            else f"ACE-Step API is not reachable at {config.api_base_url}.",
        )
    )
    return checks


start_api = start_api_background


def ensure_runtime_api(config: RuntimeConfig = RuntimeConfig()) -> RuntimeStartResult:
    if api_health(config):
        return RuntimeStartResult(
            started=False,
            already_running=True,
            api_url=config.api_base_url,
            pid=None,
            message="ACE-Step API is already running.",
            managed_by_current_run=False,
        )

    status = runtime_status(config)
    if not status.installed:
        return RuntimeStartResult(
            started=False,
            already_running=False,
            api_url=config.api_base_url,
            pid=None,
            message="ACE-Step runtime is not installed. Run: autotransition setup",
            managed_by_current_run=False,
        )

    existing = find_runtime_processes(config)
    if existing:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if api_health(config):
                return RuntimeStartResult(
                    started=False,
                    already_running=True,
                    api_url=config.api_base_url,
                    pid=existing[0].pid,
                    message="ACE-Step API is already running.",
                    managed_by_current_run=False,
                )
            time.sleep(2)
        pid_list = ", ".join(str(process.pid) for process in existing)
        return RuntimeStartResult(
            started=False,
            already_running=False,
            api_url=config.api_base_url,
            pid=existing[0].pid,
            message=(
                "ACE-Step process is already running but the API is not reachable. "
                f"Existing process id(s): {pid_list}. Stop the stale runtime, then run `autotransition run` again."
            ),
            managed_by_current_run=False,
        )

    process = start_api_background(config)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        if api_health(config):
            return RuntimeStartResult(
                started=True,
                already_running=False,
                api_url=config.api_base_url,
                pid=process.pid,
                message="ACE-Step API started in the background.",
                managed_by_current_run=True,
            )
        if process.poll() is not None:
            _clear_runtime_pid(process.pid)
            return RuntimeStartResult(
                started=False,
                already_running=False,
                api_url=config.api_base_url,
                pid=process.pid,
                message=(
                    "ACE-Step API failed to start. Check data/logs/ace-step-api.err.log. "
                    f"Process exited with code {process.returncode}."
                ),
                managed_by_current_run=False,
            )
        time.sleep(2)

    return RuntimeStartResult(
        started=True,
        already_running=False,
        api_url=config.api_base_url,
        pid=process.pid,
        message="ACE-Step API is still starting in the background. Check runtime status before generating.",
        managed_by_current_run=True,
    )
