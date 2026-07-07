"""Managed TangoFlux runtime helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from autotransition.config import RuntimeConfig
from autotransition.runtime.ace_step import build_runtime_env


@dataclass(frozen=True)
class TangoFluxRuntimeStatus:
    install_dir: Path
    installed: bool
    ready: bool
    venv_present: bool
    python_executable: str
    message: str
    install_command: str
    simple_setup_command: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["install_dir"] = str(self.install_dir)
        return data


def runtime_install_dir(config: RuntimeConfig = RuntimeConfig()) -> Path:
    return config.tango_flux_dir.expanduser().resolve()


def _runtime_venv_dir(config: RuntimeConfig = RuntimeConfig()) -> Path:
    return runtime_install_dir(config) / ".venv"


def _runtime_python_executable(config: RuntimeConfig = RuntimeConfig()) -> Path:
    if sys.platform == "win32":
        return (_runtime_venv_dir(config) / "Scripts" / "python.exe").resolve()
    return (_runtime_venv_dir(config) / "bin" / "python").resolve()


def _runtime_env() -> dict[str, str]:
    env = build_runtime_env()
    env["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LC_ALL", "C.UTF-8")
    env.setdefault("LANG", "C.UTF-8")
    return env


def _python_launcher() -> list[str]:
    if sys.platform == "win32":
        return [sys.executable, "-m", "venv"]
    return [sys.executable, "-m", "venv"]


def _install_command(config: RuntimeConfig = RuntimeConfig()) -> str:
    return (
        f'"{_runtime_python_executable(config)}" -m pip install --upgrade pip setuptools wheel '
        f'&& "{_runtime_python_executable(config)}" -m pip install '
        f'"tangoflux @ git+https://github.com/declare-lab/TangoFlux"'
    )


def _runtime_torch_lib_dir(config: RuntimeConfig = RuntimeConfig()) -> Path:
    return _runtime_python_executable(config).parent.parent / "Lib" / "site-packages" / "torch" / "lib"


def _copy_windows_torch_dependency(config: RuntimeConfig = RuntimeConfig()) -> None:
    if sys.platform != "win32":
        return
    target_dir = _runtime_torch_lib_dir(config)
    target_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib" / "libomp140.x86_64.dll",
        Path(sys.prefix) / "Library" / "bin" / "libomp140.x86_64.dll",
        Path(sys.executable).resolve().parent.parent / "Lib" / "site-packages" / "torch" / "lib" / "libomp140.x86_64.dll",
    ]
    for source in candidates:
        if source.exists():
            shutil.copy2(source, target_dir / source.name)
            return


def runtime_status(config: RuntimeConfig = RuntimeConfig()) -> TangoFluxRuntimeStatus:
    install_dir = runtime_install_dir(config)
    python_exe = _runtime_python_executable(config)
    venv_present = python_exe.exists()
    ready = False
    if venv_present:
        try:
            completed = subprocess.run(
                [
                    str(python_exe),
                    "-c",
                    "import tangoflux, torch; print('ready')",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                env=_runtime_env(),
                timeout=30,
            )
            ready = completed.returncode == 0
        except subprocess.TimeoutExpired:
            ready = False
    installed = install_dir.exists() and venv_present
    if ready:
        message = "TangoFlux runtime is ready."
    elif installed:
        message = "TangoFlux runtime is installed but not ready."
    else:
        message = f"TangoFlux runtime is not installed at {install_dir}."
    return TangoFluxRuntimeStatus(
        install_dir=install_dir,
        installed=installed,
        ready=ready,
        venv_present=venv_present,
        python_executable=str(python_exe),
        message=message,
        install_command=_install_command(config),
        simple_setup_command="autotransition runtime setup-tangoflux",
    )


def run_install(config: RuntimeConfig = RuntimeConfig()) -> None:
    install_dir = runtime_install_dir(config)
    install_dir.mkdir(parents=True, exist_ok=True)
    legacy_nested_dir = install_dir / "runtimes"
    if legacy_nested_dir.exists():
        shutil.rmtree(legacy_nested_dir)
    venv_dir = _runtime_venv_dir(config)
    if not venv_dir.exists():
        subprocess.run(
            _python_launcher() + [str(venv_dir)],
            cwd=install_dir,
            check=True,
            env=_runtime_env(),
        )
    python_exe = _runtime_python_executable(config)
    if not python_exe.exists():
        raise RuntimeError("TangoFlux runtime setup finished, but the virtual environment Python was not created.")
    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=install_dir,
        check=True,
        env=_runtime_env(),
    )
    subprocess.run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "tangoflux @ git+https://github.com/declare-lab/TangoFlux",
        ],
        cwd=install_dir,
        check=True,
        env=_runtime_env(),
    )
    _copy_windows_torch_dependency(config)


def generate_wav(prompt: str, output_path: Path, *, steps: int, duration_seconds: int, config: RuntimeConfig = RuntimeConfig()) -> Path:
    python_exe = _runtime_python_executable(config)
    if not python_exe.exists():
        raise RuntimeError("TangoFlux runtime is not installed. Run Install Runtime.")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": prompt,
        "steps": int(steps),
        "duration_seconds": int(duration_seconds),
        "output_path": str(output_path),
    }
    script = r"""
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("LC_ALL", "C.UTF-8")
os.environ.setdefault("LANG", "C.UTF-8")

import torch
import torchaudio
from tangoflux import TangoFluxInference

payload = json.loads(sys.argv[1])
output_path = Path(sys.argv[2])
model = TangoFluxInference(name="declare-lab/TangoFlux")
audio = model.generate(
    payload["prompt"],
    steps=int(payload["steps"]),
    duration=int(payload["duration_seconds"]),
)
tensor = audio.detach().cpu() if isinstance(audio, torch.Tensor) else torch.tensor(audio)
if tensor.ndim == 1:
    tensor = tensor.unsqueeze(0)
if tensor.ndim >= 3:
    tensor = tensor.squeeze(0)
if tensor.ndim != 2:
    raise RuntimeError(f"Unexpected TangoFlux output shape: {tuple(tensor.shape)}")
if tensor.shape[0] not in {1, 2} and tensor.shape[1] in {1, 2}:
    tensor = tensor.transpose(0, 1)
if tensor.shape[0] not in {1, 2}:
    raise RuntimeError(f"Unexpected TangoFlux channel layout: {tuple(tensor.shape)}")
output_path.parent.mkdir(parents=True, exist_ok=True)
torchaudio.save(str(output_path), tensor.float().cpu(), 44100)
print(str(output_path))
"""
    try:
        completed = subprocess.run(
            [str(python_exe), "-c", script, json.dumps(payload), str(output_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=runtime_install_dir(config),
            env=_runtime_env(),
            check=False,
            timeout=config.generation_timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"TangoFlux generation timed out after {config.generation_timeout_seconds} seconds.") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "TangoFlux generation failed."
        raise RuntimeError(stderr)
    if not output_path.exists():
        raise RuntimeError("TangoFlux runtime returned no audio file.")
    return output_path
