"""RVC/Applio-style voice runtime helpers."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from autotransition.config import RuntimeConfig
from autotransition.runtime.ace_step import build_runtime_env


RVC_REPO_URL = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
RVC_RUNTIME_PORT = 7897
TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"
TORCH_CUDA_VERSION = "2.4.0+cu121"


@dataclass(frozen=True)
class RvcRuntimeStatus:
    installed: bool
    api_running: bool
    api_url: str
    install_dir: str
    message: str
    install_command: str
    start_command: str
    ui_url: str
    simple_setup_command: str
    simple_start_command: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RvcRuntimeStartResult:
    started: bool
    already_running: bool
    api_url: str
    pid: int | None
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def api_base_url(config: RuntimeConfig = RuntimeConfig()) -> str:
    return f"http://{config.rvc_host}:{config.rvc_port}"


def runtime_install_dir(config: RuntimeConfig = RuntimeConfig()) -> Path:
    return config.rvc_dir.expanduser()


def _runtime_venv_dir(config: RuntimeConfig = RuntimeConfig()) -> Path:
    return runtime_install_dir(config) / ".venv"


def _runtime_python_executable(config: RuntimeConfig = RuntimeConfig()) -> Path:
    if sys.platform == "win32":
        return (_runtime_venv_dir(config) / "Scripts" / "python.exe").resolve()
    return (_runtime_venv_dir(config) / "bin" / "python").resolve()


def _python_launcher_candidates() -> list[list[str]]:
    if sys.platform == "win32":
        return [
            ["py", "-3.10"],
            ["py", "-3.9"],
            ["py", "-3"],
        ]
    return [
        ["python3.10"],
        ["python3.9"],
        ["python3"],
    ]


def _resolve_python_launcher() -> list[str]:
    for candidate in _python_launcher_candidates():
        try:
            completed = subprocess.run(
                candidate + ["-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
                capture_output=True,
                text=True,
                check=False,
                env=_runtime_env(),
            )
            if completed.returncode != 0:
                continue
            version = completed.stdout.strip()
            if version in {"3.10", "3.9"}:
                return candidate
        except Exception:
            continue
    raise RuntimeError("Python 3.10 or 3.9 is required for the Voice Work runtime. Install it and try again.")


def runtime_pid_path() -> Path:
    return Path("data/runtime/rvc-ui.pid")


def _runtime_log_path() -> Path:
    return Path("data/logs/rvc-ui.log")


def _runtime_err_log_path() -> Path:
    return Path("data/logs/rvc-ui.err.log")


def _runtime_env() -> dict[str, str]:
    env = build_runtime_env()
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    env.setdefault("LANG", "C.UTF-8")
    env.setdefault("USE_LIBUV", "0")
    env.setdefault("TORCH_DIST_INIT_USE_LIBUV", "0")
    return env


def _has_nvidia_tooling() -> bool:
    if shutil.which("nvidia-smi"):
        return True
    if sys.platform != "win32":
        return False
    candidates = [
        Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe",
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe",
    ]
    return any(candidate.exists() for candidate in candidates)


def _install_cuda_torch(python_exe: Path, install_dir: Path) -> None:
    subprocess.run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--force-reinstall",
            "--no-cache-dir",
            "--index-url",
            TORCH_CUDA_INDEX_URL,
            f"torch=={TORCH_CUDA_VERSION}",
            f"torchaudio=={TORCH_CUDA_VERSION}",
        ],
        cwd=install_dir,
        check=True,
        env=_runtime_env(),
    )


def _runtime_cuda_ready(python_exe: Path) -> bool:
    try:
        completed = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import torch; print('1' if torch.cuda.is_available() else '0')",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=_runtime_env(),
        )
        return completed.returncode == 0 and completed.stdout.strip() == "1"
    except Exception:
        return False


def _download_file(url: str, target: Path) -> None:
    if target.exists() and target.stat().st_size > 0:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(f"{target.suffix}.tmp")
    with httpx.stream("GET", url, follow_redirects=True, timeout=None) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as handle:
            for chunk in response.iter_bytes():
                if chunk:
                    handle.write(chunk)
    tmp_path.replace(target)


def _download_required_assets(config: RuntimeConfig = RuntimeConfig()) -> None:
    install_dir = runtime_install_dir(config)
    asset_targets = [
        (
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt",
            install_dir / "assets" / "hubert" / "hubert_base.pt",
        ),
        (
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt",
            install_dir / "assets" / "rmvpe" / "rmvpe.pt",
        ),
        (
            "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/onnx_dereverb_By_FoxJoy/vocals.onnx",
            install_dir / "assets" / "uvr5_weights" / "onnx_dereverb_By_FoxJoy" / "vocals.onnx",
        ),
    ]
    for url, target in asset_targets:
        _download_file(url, target)


def _required_asset_targets(config: RuntimeConfig = RuntimeConfig()) -> list[Path]:
    install_dir = runtime_install_dir(config)
    return [
        install_dir / "assets" / "hubert" / "hubert_base.pt",
        install_dir / "assets" / "rmvpe" / "rmvpe.pt",
    ]


def _required_assets_ready(config: RuntimeConfig = RuntimeConfig()) -> bool:
    return all(target.exists() and target.stat().st_size > 0 for target in _required_asset_targets(config))


def _rotate_runtime_log(path: Path) -> None:
    if not path.exists():
        return
    previous = path.with_name(f"{path.name}.previous")
    previous.unlink(missing_ok=True)
    path.replace(previous)


def _subprocess_group_kwargs() -> dict[str, object]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _write_runtime_pid(pid: int) -> None:
    path = runtime_pid_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(pid), encoding="utf-8")


def read_runtime_pid() -> int | None:
    path = runtime_pid_path()
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _clear_runtime_pid(pid: int | None = None) -> None:
    path = runtime_pid_path()
    if not path.exists():
        return
    if pid is None:
        path.unlink(missing_ok=True)
        return
    current = read_runtime_pid()
    if current == pid:
        path.unlink(missing_ok=True)


def _read_log_tail(path: Path, *, max_bytes: int = 16384) -> str:
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes), os.SEEK_SET)
        data = handle.read()
    return data.decode("utf-8", errors="ignore")


def _runtime_dependencies_ready(config: RuntimeConfig = RuntimeConfig()) -> bool:
    python_exe = _runtime_python_executable(config)
    if not python_exe.exists():
        return False
    try:
        completed = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import dotenv, torch, gradio",
            ],
            capture_output=True,
            text=True,
            check=False,
            env=_runtime_env(),
        )
        return completed.returncode == 0
    except Exception:
        return False


def startup_progress_snapshot() -> dict[str, str]:
    stdout_text = _read_log_tail(_runtime_log_path())
    stderr_text = _read_log_tail(_runtime_err_log_path())
    combined = f"{stdout_text}\n{stderr_text}".strip()
    if not combined:
        return {
            "phase": "idle",
            "message": "No RVC startup activity detected yet.",
            "detail": "",
        }
    if "Running on local URL" in combined or "http://127.0.0.1" in combined or "http://0.0.0.0" in combined:
        return {
            "phase": "ready",
            "message": "RVC runtime is ready.",
            "detail": "",
        }
    last_line = [line.strip() for line in combined.replace("\r", "\n").splitlines() if line.strip()]
    message = last_line[-1] if last_line else "RVC runtime is starting."
    return {
        "phase": "starting",
        "message": message,
        "detail": message,
    }


def _clone_command(config: RuntimeConfig = RuntimeConfig()) -> list[str]:
    return ["git", "clone", RVC_REPO_URL, str(runtime_install_dir(config))]


def _setup_command(config: RuntimeConfig = RuntimeConfig()) -> list[str]:
    python_launcher = _resolve_python_launcher()
    return python_launcher + ["-m", "venv", str(_runtime_venv_dir(config))]


def _start_command(config: RuntimeConfig = RuntimeConfig()) -> list[str]:
    python_exe = _runtime_python_executable(config)
    return [str(python_exe), "infer-web.py", "--pycmd", str(python_exe), "--port", str(config.rvc_port), "--noautoopen"]


def build_install_command(config: RuntimeConfig = RuntimeConfig()) -> str:
    cuda_step = ""
    if _has_nvidia_tooling():
        cuda_step = f' && "{_runtime_python_executable(config)}" -m pip install --upgrade --force-reinstall --no-cache-dir --index-url {TORCH_CUDA_INDEX_URL} "torch=={TORCH_CUDA_VERSION}" "torchaudio=={TORCH_CUDA_VERSION}"'
    if sys.platform == "win32":
        return f'py -3.10 -m venv "{runtime_install_dir(config) / ".venv"}" && "{_runtime_python_executable(config)}" -m pip install --upgrade "pip<24.1" "setuptools<81" wheel && "{_runtime_python_executable(config)}" -m pip install -r requirements.txt{cuda_step} && "{_runtime_python_executable(config)}" tools\\download_models.py'
    return f'python3.10 -m venv "{runtime_install_dir(config) / ".venv"}" && "{_runtime_python_executable(config)}" -m pip install --upgrade "pip<24.1" "setuptools<81" wheel && "{_runtime_python_executable(config)}" -m pip install -r requirements.txt{cuda_step} && "{_runtime_python_executable(config)}" tools/download_models.py'


def build_start_command(config: RuntimeConfig = RuntimeConfig()) -> str:
    python_exe = _runtime_python_executable(config)
    return f'"{python_exe}" infer-web.py --pycmd "{python_exe}" --port {int(config.rvc_port)} --noautoopen'


def runtime_status(config: RuntimeConfig = RuntimeConfig()) -> RvcRuntimeStatus:
    install_dir = runtime_install_dir(config)
    repo_present = install_dir.exists() and (install_dir / "pyproject.toml").exists()
    python_exe = _runtime_python_executable(config)
    venv_present = python_exe.exists()
    deps_ready = repo_present and venv_present and _runtime_dependencies_ready(config)
    assets_ready = repo_present and _required_assets_ready(config)
    installed = repo_present and venv_present
    running = api_health(config)
    managed_alive = managed_runtime_alive(config) if install_dir.exists() else False
    progress = startup_progress_snapshot()
    if running:
        message = "RVC runtime is reachable."
    elif managed_alive and progress["phase"] != "idle":
        message = progress["message"]
    elif not repo_present:
        message = "RVC runtime is not installed."
    elif not venv_present:
        message = "RVC runtime dependencies are not installed. Run Install Runtime."
    elif not deps_ready:
        message = "RVC runtime dependencies are incomplete. Run Install Runtime again."
    elif not assets_ready:
        message = "RVC runtime assets are missing. Run Install Runtime again."
    else:
        message = "RVC runtime is installed but not running."
    simple_start_command = "Restart Runtime" if running or managed_alive else "Start Runtime"
    return RvcRuntimeStatus(
        installed=installed,
        api_running=running,
        api_url=api_base_url(config),
        install_dir=str(install_dir),
        message=message,
        install_command=build_install_command(config),
        start_command=build_start_command(config),
        ui_url=api_base_url(config),
        simple_setup_command="Install Runtime",
        simple_start_command=simple_start_command,
    )


def api_health(config: RuntimeConfig = RuntimeConfig()) -> bool:
    try:
        response = httpx.get(f"{api_base_url(config)}/", timeout=config.rvc_timeout_seconds)
        return response.status_code < 500
    except Exception:
        return False


def managed_runtime_alive(config: RuntimeConfig = RuntimeConfig()) -> bool:
    pid = read_runtime_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        _clear_runtime_pid(pid)
        return False


def _any_runtime_process_alive(config: RuntimeConfig = RuntimeConfig()) -> bool:
    pid = read_runtime_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def run_install(config: RuntimeConfig = RuntimeConfig()) -> None:
    stop_runtime(config, timeout_seconds=10.0)
    install_dir = runtime_install_dir(config)
    install_dir.parent.mkdir(parents=True, exist_ok=True)
    if not install_dir.exists():
        subprocess.run(_clone_command(config), check=True, env=_runtime_env())
    if not (install_dir / "pyproject.toml").exists():
        raise RuntimeError(f"RVC runtime directory exists but does not look complete: {install_dir}")
    venv_dir = _runtime_venv_dir(config)
    if venv_dir.exists():
        python_exe = _runtime_python_executable(config)
        if python_exe.exists():
            version = subprocess.run(
                [str(python_exe), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
                capture_output=True,
                text=True,
                check=False,
                env=_runtime_env(),
            )
            if version.returncode == 0 and version.stdout.strip() in {"3.10", "3.9"}:
                pass
            else:
                shutil.rmtree(venv_dir, ignore_errors=True)
        else:
            shutil.rmtree(venv_dir, ignore_errors=True)
    if not venv_dir.exists():
        subprocess.run(_setup_command(config), cwd=install_dir, check=True, env=_runtime_env())
    python_exe = _runtime_python_executable(config)
    if not python_exe.exists():
        raise RuntimeError("RVC runtime setup finished, but the virtual environment Python was not created.")
    subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip<24.1", "setuptools<81", "wheel"], cwd=install_dir, check=True, env=_runtime_env())
    subprocess.run([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"], cwd=install_dir, check=True, env=_runtime_env())
    if _has_nvidia_tooling():
        _install_cuda_torch(python_exe, install_dir)
    subprocess.run([str(python_exe), "-m", "pip", "install", "requests"], cwd=install_dir, check=True, env=_runtime_env())
    _download_required_assets(config)
    if _has_nvidia_tooling() and not _runtime_cuda_ready(python_exe):
        raise RuntimeError("RVC runtime bootstrap completed, but CUDA torch is still unavailable.")


def start_runtime_background(config: RuntimeConfig = RuntimeConfig()) -> subprocess.Popen[bytes]:
    if not _runtime_dependencies_ready(config) or not _required_assets_ready(config):
        run_install(config)
    status = runtime_status(config)
    if not status.installed:
        raise RuntimeError(f"RVC runtime is not installed: {config.rvc_dir}")
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = _runtime_log_path()
    stderr_path = _runtime_err_log_path()
    _rotate_runtime_log(stdout_path)
    _rotate_runtime_log(stderr_path)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.touch(exist_ok=True)
    stderr_path.touch(exist_ok=True)
    runtime_pid = runtime_pid_path()
    runtime_pid.parent.mkdir(parents=True, exist_ok=True)
    runtime_pid.unlink(missing_ok=True)
    process = subprocess.Popen(
        _start_command(config),
        cwd=runtime_install_dir(config),
        stdout=stdout_path.open("wb"),
        stderr=stderr_path.open("wb"),
        env=_runtime_env(),
        **_subprocess_group_kwargs(),
    )
    _write_runtime_pid(process.pid)
    return process


def stop_runtime(config: RuntimeConfig = RuntimeConfig(), *, timeout_seconds: float = 30.0) -> bool:
    pid = read_runtime_pid()
    if pid is None and not api_health(config) and not _any_runtime_process_alive(config):
        return False
    try:
        if sys.platform == "win32" and pid is not None:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(pid)], capture_output=True, text=True, check=False)
        elif pid is not None:
            os.kill(pid, signal.SIGTERM)
    finally:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            if not managed_runtime_alive(config) and not api_health(config) and not _any_runtime_process_alive(config):
                _clear_runtime_pid(pid)
                return True
            time.sleep(1)
        _clear_runtime_pid(pid)
    return not managed_runtime_alive(config) and not api_health(config) and not _any_runtime_process_alive(config)


def ensure_runtime_api(config: RuntimeConfig = RuntimeConfig()) -> RvcRuntimeStartResult:
    if api_health(config):
        return RvcRuntimeStartResult(
            started=False,
            already_running=True,
            api_url=api_base_url(config),
            pid=read_runtime_pid(),
            message="RVC runtime is already running.",
        )
    status = runtime_status(config)
    if not status.installed:
        return RvcRuntimeStartResult(
            started=False,
            already_running=False,
            api_url=api_base_url(config),
            pid=None,
            message="RVC runtime is not installed.",
        )
    if managed_runtime_alive(config):
        deadline = time.monotonic() + max(60.0, config.api_startup_timeout_seconds)
        while time.monotonic() < deadline:
            if api_health(config):
                return RvcRuntimeStartResult(
                    started=False,
                    already_running=True,
                    api_url=api_base_url(config),
                    pid=read_runtime_pid(),
                    message="RVC runtime is already running.",
                )
            if not managed_runtime_alive(config):
                progress = startup_progress_snapshot()
                return RvcRuntimeStartResult(
                    started=False,
                    already_running=False,
                    api_url=api_base_url(config),
                    pid=read_runtime_pid(),
                    message=progress["message"] if progress["phase"] != "idle" else "RVC runtime stopped during startup.",
                )
            time.sleep(2)
        return RvcRuntimeStartResult(
            started=False,
            already_running=True,
            api_url=api_base_url(config),
            pid=read_runtime_pid(),
            message="RVC runtime is still loading.",
        )
    process = start_runtime_background(config)
    deadline = time.monotonic() + max(60.0, config.api_startup_timeout_seconds)
    while time.monotonic() < deadline:
        if api_health(config):
            return RvcRuntimeStartResult(
                started=True,
                already_running=False,
                api_url=api_base_url(config),
                pid=process.pid,
                message="RVC runtime started in the background.",
            )
        if process.poll() is not None:
            _clear_runtime_pid(process.pid)
            progress = startup_progress_snapshot()
            return RvcRuntimeStartResult(
                started=False,
                already_running=False,
                api_url=api_base_url(config),
                pid=process.pid,
                message=progress["message"] if progress["phase"] != "idle" else "RVC runtime failed to start.",
            )
        time.sleep(2)
    return RvcRuntimeStartResult(
        started=False,
        already_running=managed_runtime_alive(config),
        api_url=api_base_url(config),
        pid=read_runtime_pid(),
        message="RVC runtime is still starting.",
    )
