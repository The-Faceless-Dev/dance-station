"""Managed UVR-style source separation runtime helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autotransition.config import RuntimeConfig
from autotransition.runtime.ace_step import build_runtime_env


SOURCE_SEPARATION_REPO = "https://github.com/nomadkaraoke/python-audio-separator"
DEFAULT_MODEL_FILENAME = "UVR-MDX-NET-Inst_HQ_3.onnx"
FALLBACK_MODEL_CHOICES = [
    {
        "model_filename": "UVR-MDX-NET-Inst_HQ_3.onnx",
        "friendly_name": "UVR MDX Instrumental HQ 3",
        "arch": "MDX",
        "output_stems": "vocals, instrumental",
    },
    {
        "model_filename": "UVR-MDX-NET-Inst_Main.onnx",
        "friendly_name": "UVR MDX Instrumental Main",
        "arch": "MDX",
        "output_stems": "vocals, instrumental",
    },
    {
        "model_filename": "UVR_MDXNET_KARA_2.onnx",
        "friendly_name": "UVR MDX KARA 2",
        "arch": "MDX",
        "output_stems": "vocals, instrumental",
    },
]


@dataclass(frozen=True)
class SourceSeparationRuntimeStatus:
    install_dir: Path
    installed: bool
    ready: bool
    venv_present: bool
    python_executable: str
    cli_executable: str
    message: str
    install_command: str
    simple_setup_command: str
    default_model_filename: str

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["install_dir"] = str(self.install_dir)
        return data


def runtime_install_dir(config: RuntimeConfig = RuntimeConfig()) -> Path:
    return config.source_separation_dir.expanduser().resolve()


def _runtime_venv_dir(config: RuntimeConfig = RuntimeConfig()) -> Path:
    return runtime_install_dir(config) / ".venv"


def _runtime_python_executable(config: RuntimeConfig = RuntimeConfig()) -> Path:
    if sys.platform == "win32":
        return (_runtime_venv_dir(config) / "Scripts" / "python.exe").resolve()
    return (_runtime_venv_dir(config) / "bin" / "python").resolve()


def _runtime_cli_executable(config: RuntimeConfig = RuntimeConfig()) -> Path:
    if sys.platform == "win32":
        return (_runtime_venv_dir(config) / "Scripts" / "audio-separator.exe").resolve()
    return (_runtime_venv_dir(config) / "bin" / "audio-separator").resolve()


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


def _python_launcher_candidates() -> list[list[str]]:
    if sys.platform == "win32":
        return [["py", "-3.11"], ["py", "-3.10"], ["py", "-3"]]
    return [["python3.11"], ["python3.10"], ["python3"]]


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
            if completed.returncode == 0 and completed.stdout.strip() in {"3.11", "3.10"}:
                return candidate
        except Exception:
            continue
    raise RuntimeError("Python 3.11 or 3.10 is required for the source separation runtime.")


def _install_command(config: RuntimeConfig = RuntimeConfig()) -> str:
    python_exe = _runtime_python_executable(config)
    return (
        f'"{python_exe}" -m pip install --upgrade pip setuptools wheel '
        f'&& "{python_exe}" -m pip install "audio-separator[cpu]"'
    )


def runtime_status(config: RuntimeConfig = RuntimeConfig()) -> SourceSeparationRuntimeStatus:
    install_dir = runtime_install_dir(config)
    python_exe = _runtime_python_executable(config)
    cli_exe = _runtime_cli_executable(config)
    venv_present = python_exe.exists() and cli_exe.exists()
    ready = False
    if venv_present:
        try:
            completed = subprocess.run(
                [str(python_exe), "-m", "audio_separator.utils.cli", "--env_info"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                env=_runtime_env(),
                timeout=60,
            )
            ready = completed.returncode == 0
        except subprocess.TimeoutExpired:
            ready = False
    installed = install_dir.exists() and venv_present
    if ready:
        message = "Source separation runtime is ready."
    elif installed:
        message = "Source separation runtime is installed but not ready."
    else:
        message = f"Source separation runtime is not installed at {install_dir}."
    return SourceSeparationRuntimeStatus(
        install_dir=install_dir,
        installed=installed,
        ready=ready,
        venv_present=venv_present,
        python_executable=str(python_exe),
        cli_executable=str(cli_exe),
        message=message,
        install_command=_install_command(config),
        simple_setup_command="autotransition runtime setup-source-separation",
        default_model_filename=DEFAULT_MODEL_FILENAME,
    )


def run_install(config: RuntimeConfig = RuntimeConfig()) -> None:
    install_dir = runtime_install_dir(config)
    install_dir.mkdir(parents=True, exist_ok=True)
    venv_dir = _runtime_venv_dir(config)
    if not venv_dir.exists():
        subprocess.run(
            _resolve_python_launcher() + ["-m", "venv", str(venv_dir)],
            cwd=install_dir,
            check=True,
            env=_runtime_env(),
        )
    python_exe = _runtime_python_executable(config)
    if not python_exe.exists():
        raise RuntimeError("Source separation runtime setup finished, but the virtual environment Python was not created.")
    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=install_dir,
        check=True,
        env=_runtime_env(),
    )
    subprocess.run(
        [str(python_exe), "-m", "pip", "install", "audio-separator[cpu]"],
        cwd=install_dir,
        check=True,
        env=_runtime_env(),
    )


def list_models(config: RuntimeConfig = RuntimeConfig()) -> list[dict[str, Any]]:
    cli_exe = _runtime_cli_executable(config)
    if not cli_exe.exists():
        return [dict(item) for item in FALLBACK_MODEL_CHOICES]
    try:
        completed = subprocess.run(
            [
                str(_runtime_python_executable(config)),
                "-m",
                "audio_separator.utils.cli",
                "--list_models",
                "--list_filter=vocals",
                "--list_limit=12",
                "--list_format=json",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=_runtime_env(),
            timeout=120,
        )
    except Exception:
        return [dict(item) for item in FALLBACK_MODEL_CHOICES]
    if completed.returncode != 0:
        return [dict(item) for item in FALLBACK_MODEL_CHOICES]
    try:
        payload = json.loads(completed.stdout.strip() or "[]")
    except Exception:
        return [dict(item) for item in FALLBACK_MODEL_CHOICES]
    items: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            model_filename = str(
                entry.get("model_filename")
                or entry.get("filename")
                or entry.get("model")
                or ""
            ).strip()
            if not model_filename:
                continue
            items.append(
                {
                    "model_filename": model_filename,
                    "friendly_name": str(entry.get("friendly_name") or entry.get("name") or model_filename),
                    "arch": str(entry.get("arch") or entry.get("architecture") or ""),
                    "output_stems": str(entry.get("output_stems") or entry.get("stems") or ""),
                }
            )
    return items or [dict(item) for item in FALLBACK_MODEL_CHOICES]


def separate_audio(
    *,
    source_path: Path,
    output_dir: Path,
    model_filename: str = DEFAULT_MODEL_FILENAME,
    output_format: str = "wav",
    chunk_duration: int | None = None,
    mdx_segment_size: int | None = None,
    mdx_overlap: float | None = None,
    mdx_enable_denoise: bool = False,
    config: RuntimeConfig = RuntimeConfig(),
) -> dict[str, Any]:
    python_exe = _runtime_python_executable(config)
    cli_exe = _runtime_cli_executable(config)
    if not python_exe.exists() or not cli_exe.exists():
        raise RuntimeError("Source separation runtime is not installed. Run Install Runtime.")
    source_path = source_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_cache_dir = runtime_install_dir(config) / "models"
    model_cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "model_file_dir": str(model_cache_dir),
        "model_filename": model_filename,
        "output_format": (output_format or "wav").strip().upper() or "WAV",
        "chunk_duration": int(chunk_duration) if chunk_duration and int(chunk_duration) > 0 else None,
        "mdx_segment_size": int(mdx_segment_size) if mdx_segment_size and int(mdx_segment_size) > 0 else 256,
        "mdx_overlap": float(mdx_overlap) if mdx_overlap is not None else 0.25,
        "mdx_enable_denoise": bool(mdx_enable_denoise),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
        temp_path = Path(handle.name)
    completed = subprocess.run(
        [
            str(python_exe),
            "-c",
            """
import json
import sys
from pathlib import Path

from audio_separator.separator import Separator

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
separator = Separator(
    model_file_dir=payload["model_file_dir"],
    output_dir=payload["output_dir"],
    output_format=payload["output_format"],
    use_soundfile=True,
    chunk_duration=payload["chunk_duration"],
    mdx_params={
        "hop_length": 1024,
        "segment_size": payload["mdx_segment_size"],
        "overlap": payload["mdx_overlap"],
        "batch_size": 1,
        "enable_denoise": payload["mdx_enable_denoise"],
    },
)
separator.load_model(payload["model_filename"])
result = separator.separate(payload["source_path"])
print(json.dumps({"result": result}, default=str))
""",
            str(temp_path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=runtime_install_dir(config),
        env=_runtime_env(),
        check=False,
        timeout=config.generation_timeout_seconds,
    )
    temp_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "Source separation failed."
        raise RuntimeError(stderr)
    try:
        payload = json.loads(completed.stdout.strip() or "{}")
    except Exception as exc:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "Source separation returned invalid output.") from exc
    result = payload.get("result")
    audio_files: list[Path] = []
    if isinstance(result, list):
        for item in result:
            if isinstance(item, str):
                candidate = Path(item)
                audio_files.append(candidate if candidate.is_absolute() else output_dir / candidate)
    elif isinstance(result, dict):
        for item in result.values():
            if isinstance(item, str):
                candidate = Path(item)
                audio_files.append(candidate if candidate.is_absolute() else output_dir / candidate)
    if not audio_files:
        raise RuntimeError("Source separation runtime returned no audio files.")
    audio_files = [path for path in audio_files if path.exists()]
    return {
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "audio_files": [str(path) for path in sorted(audio_files, key=lambda item: item.stat().st_mtime, reverse=True)],
        "returncode": completed.returncode,
    }
