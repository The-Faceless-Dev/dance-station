"""Seed-VC runtime helpers."""

from __future__ import annotations

import os
import json
import shutil
import signal
import subprocess
import sys
import time
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from autotransition.config import RuntimeConfig
from autotransition.runtime.ace_step import build_runtime_env


SEED_VC_REPO_URL = "https://github.com/Plachtaa/seed-vc.git"
TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu121"
TORCH_CUDA_VERSION = "2.4.0+cu121"
TORCHVISION_CUDA_VERSION = "0.19.0+cu121"
NUMPY_VERSION = "1.26.4"
PILLOW_VERSION = "11.3.0"


@dataclass(frozen=True)
class SeedVcRuntimeStatus:
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
class SeedVcRuntimeStartResult:
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
        return [["py", "-3.10"], ["py", "-3.9"], ["py", "-3"]]
    return [["python3.10"], ["python3.9"], ["python3"]]


def _resolve_python_launcher() -> list[str]:
    for candidate in _python_launcher_candidates():
        try:
            completed = subprocess.run(
                candidate + ["-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                env=_runtime_env(),
            )
            if completed.returncode == 0 and completed.stdout.strip() in {"3.10", "3.9"}:
                return candidate
        except Exception:
            continue
    raise RuntimeError("Python 3.10 or 3.9 is required for Seed-VC. Install it and try again.")


def runtime_pid_path() -> Path:
    return Path("data/runtime/seed-vc-ui.pid")


def _runtime_log_path() -> Path:
    return Path("data/logs/seed-vc.log")


def _runtime_err_log_path() -> Path:
    return Path("data/logs/seed-vc.err.log")


def _runtime_env() -> dict[str, str]:
    env = build_runtime_env()
    env.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
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
            f"torchvision=={TORCHVISION_CUDA_VERSION}",
            f"torchaudio=={TORCH_CUDA_VERSION}",
        ],
        cwd=install_dir,
        check=True,
        env=_runtime_env(),
    )


def _restore_runtime_compat_packages(python_exe: Path, install_dir: Path) -> None:
    subprocess.run(
        [
            str(python_exe),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            f"numpy=={NUMPY_VERSION}",
            f"pillow=={PILLOW_VERSION}",
        ],
        cwd=install_dir,
        check=True,
        env=_runtime_env(),
    )


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
                "import dotenv, gradio, gradio_client, torch; "
                "print('cuda' if torch.cuda.is_available() else 'cpu')",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            env=_runtime_env(),
        )
        if completed.returncode != 0:
            return False
        if _has_nvidia_tooling():
            return completed.stdout.strip() == "cuda"
        return True
    except Exception:
        return False


def startup_progress_snapshot() -> dict[str, str]:
    stdout_text = _read_log_tail(_runtime_log_path())
    stderr_text = _read_log_tail(_runtime_err_log_path())
    combined = f"{stdout_text}\n{stderr_text}".strip()
    if not combined:
        return {"phase": "idle", "message": "No Seed-VC startup activity detected yet.", "detail": ""}
    if "Running on local URL" in combined or "http://127.0.0.1" in combined or "http://0.0.0.0" in combined:
        return {"phase": "ready", "message": "Seed-VC runtime is ready.", "detail": ""}
    last_line = [line.strip() for line in combined.replace("\r", "\n").splitlines() if line.strip()]
    message = last_line[-1] if last_line else "Seed-VC runtime is starting."
    return {"phase": "starting", "message": message, "detail": message}


def _clone_command(config: RuntimeConfig = RuntimeConfig()) -> list[str]:
    return ["git", "clone", SEED_VC_REPO_URL, str(runtime_install_dir(config))]


def _setup_command(config: RuntimeConfig = RuntimeConfig()) -> list[str]:
    python_launcher = _resolve_python_launcher()
    return python_launcher + ["-m", "venv", str(_runtime_venv_dir(config))]


def _patch_runtime_entrypoint(config: RuntimeConfig = RuntimeConfig()) -> None:
    def _patch_file(app_path: Path) -> None:
        if not app_path.exists():
            return
        text = app_path.read_text(encoding="utf-8")
        updated = text
        if "import tempfile" not in updated:
            updated = updated.replace(
                "import argparse\n",
                "import argparse\nimport tempfile\n",
            )
        if "from pathlib import Path" not in updated:
            updated = updated.replace(
                "import argparse\nimport tempfile\n",
                "import argparse\nimport tempfile\nfrom pathlib import Path\n",
            )
        if "--port" not in updated:
            updated = updated.replace(
                '    parser.add_argument("--config", type=str, help="Path to the config file", default=None)\n',
                '    parser.add_argument("--config", type=str, help="Path to the config file", default=None)\n'
                '    parser.add_argument("--port", type=int, help="Port to bind the Gradio app to", default=7860)\n',
            )
        launch_before = ").launch(share=args.share,)"
        if app_path.name == "app_vc.py":
            launch_before = ").launch(server_port=args.port, share=args.share,)"
        updated = updated.replace(
            ".launch(share=args.share,)",
            ".launch(server_port=args.port, share=args.share,)",
        )
        updated = updated.replace(
            ").launch(share=args.share,)",
            ").launch(server_port=args.port, share=args.share,)",
        )
        if app_path.name == "app_vc.py":
            old_block = """    # split source condition (cond) into chunks\n    processed_frames = 0\n    generated_wave_chunks = []\n    # generate chunk by chunk and stream the output\n    while processed_frames < cond.size(1):\n        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]\n        is_last_chunk = processed_frames + max_source_window >= cond.size(1)\n        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)\n        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):\n            # Voice Conversion\n            vc_target = inference_module.cfm.inference(cat_condition,\n                                                       torch.LongTensor([cat_condition.size(1)]).to(mel2.device),\n                                                       mel2, style2, None, diffusion_steps,\n                                                       inference_cfg_rate=inference_cfg_rate)\n            vc_target = vc_target[:, :, mel2.size(-1):]\n        vc_wave = vocoder_fn(vc_target.float())[0]\n        if vc_wave.ndim == 1:\n            vc_wave = vc_wave.unsqueeze(0)\n        if processed_frames == 0:\n            if is_last_chunk:\n                output_wave = vc_wave[0].cpu().numpy()\n                generated_wave_chunks.append(output_wave)\n                output_wave = (output_wave * 32768.0).astype(np.int16)\n                mp3_bytes = AudioSegment(\n                    output_wave.tobytes(), frame_rate=sr,\n                    sample_width=output_wave.dtype.itemsize, channels=1\n                ).export(format=\"mp3\", bitrate=bitrate).read()\n                yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))\n                break\n            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()\n            generated_wave_chunks.append(output_wave)\n            previous_chunk = vc_wave[0, -overlap_wave_len:]\n            processed_frames += vc_target.size(2) - overlap_frame_len\n            output_wave = (output_wave * 32768.0).astype(np.int16)\n            mp3_bytes = AudioSegment(\n                output_wave.tobytes(), frame_rate=sr,\n                sample_width=output_wave.dtype.itemsize, channels=1\n            ).export(format=\"mp3\", bitrate=bitrate).read()\n            yield mp3_bytes, None\n        elif is_last_chunk:\n            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)\n            generated_wave_chunks.append(output_wave)\n            processed_frames += vc_target.size(2) - overlap_frame_len\n            output_wave = (output_wave * 32768.0).astype(np.int16)\n            mp3_bytes = AudioSegment(\n                output_wave.tobytes(), frame_rate=sr,\n                sample_width=output_wave.dtype.itemsize, channels=1\n            ).export(format=\"mp3\", bitrate=bitrate).read()\n            yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))\n            break\n        else:\n            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)\n            generated_wave_chunks.append(output_wave)\n            previous_chunk = vc_wave[0, -overlap_wave_len:]\n            processed_frames += vc_target.size(2) - overlap_frame_len\n            output_wave = (output_wave * 32768.0).astype(np.int16)\n            mp3_bytes = AudioSegment(\n                output_wave.tobytes(), frame_rate=sr,\n                sample_width=output_wave.dtype.itemsize, channels=1\n            ).export(format=\"mp3\", bitrate=bitrate).read()\n            yield mp3_bytes, None\n"""
            new_block = """    # split source condition (cond) into chunks and assemble one final output file\n    processed_frames = 0\n    generated_wave_chunks = []\n    while processed_frames < cond.size(1):\n        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]\n        is_last_chunk = processed_frames + max_source_window >= cond.size(1)\n        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)\n        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):\n            vc_target = inference_module.cfm.inference(\n                cat_condition,\n                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),\n                mel2,\n                style2,\n                None,\n                diffusion_steps,\n                inference_cfg_rate=inference_cfg_rate,\n            )\n            vc_target = vc_target[:, :, mel2.size(-1):]\n        vc_wave = vocoder_fn(vc_target.float())[0]\n        if vc_wave.ndim == 1:\n            vc_wave = vc_wave.unsqueeze(0)\n        if processed_frames == 0:\n            if is_last_chunk:\n                generated_wave_chunks.append(vc_wave[0].cpu().numpy())\n                break\n            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())\n            previous_chunk = vc_wave[0, -overlap_wave_len:]\n            processed_frames += vc_target.size(2) - overlap_frame_len\n        elif is_last_chunk:\n            generated_wave_chunks.append(crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))\n            processed_frames += vc_target.size(2) - overlap_frame_len\n            break\n        else:\n            generated_wave_chunks.append(crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len))\n            previous_chunk = vc_wave[0, -overlap_wave_len:]\n            processed_frames += vc_target.size(2) - overlap_frame_len\n\n    output_wave = np.concatenate(generated_wave_chunks) if generated_wave_chunks else np.zeros(0, dtype=np.float32)\n    output_wave = (output_wave * 32768.0).astype(np.int16)\n    output_dir = Path(tempfile.mkdtemp(prefix=\"seed-vc-output-\"))\n    output_path = output_dir / \"voice_conversion.wav\"\n    AudioSegment(\n        output_wave.tobytes(),\n        frame_rate=sr,\n        sample_width=output_wave.dtype.itemsize,\n        channels=1,\n    ).export(output_path, format=\"wav\")\n    return str(output_path)\n"""
        else:
            old_block = """    # split source condition (cond) into chunks\n    processed_frames = 0\n    generated_wave_chunks = []\n    # generate chunk by chunk and stream the output\n    while processed_frames < cond.size(1):\n        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]\n        chunk_f0 = interpolated_shifted_f0_alt[:, processed_frames:processed_frames + max_source_window]\n        is_last_chunk = processed_frames + max_source_window >= cond.size(1)\n        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)\n        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):\n            # Voice Conversion\n            vc_target = inference_module.cfm.inference(cat_condition,\n                                                       torch.LongTensor([cat_condition.size(1)]).to(mel2.device),\n                                                       mel2, style2, None, diffusion_steps,\n                                                       inference_cfg_rate=inference_cfg_rate)\n            vc_target = vc_target[:, :, mel2.size(-1):]\n        vc_wave = vocoder_fn(vc_target.float()).squeeze().cpu()\n        if vc_wave.ndim == 1:\n            vc_wave = vc_wave.unsqueeze(0)\n        if processed_frames == 0:\n            if is_last_chunk:\n                output_wave = vc_wave[0].cpu().numpy()\n                generated_wave_chunks.append(output_wave)\n                output_wave = (output_wave * 32768.0).astype(np.int16)\n                mp3_bytes = AudioSegment(\n                    output_wave.tobytes(), frame_rate=sr,\n                    sample_width=output_wave.dtype.itemsize, channels=1\n                ).export(format=\"mp3\", bitrate=bitrate).read()\n                yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))\n                break\n            output_wave = vc_wave[0, :-overlap_wave_len].cpu().numpy()\n            generated_wave_chunks.append(output_wave)\n            previous_chunk = vc_wave[0, -overlap_wave_len:]\n            processed_frames += vc_target.size(2) - overlap_frame_len\n            output_wave = (output_wave * 32768.0).astype(np.int16)\n            mp3_bytes = AudioSegment(\n                output_wave.tobytes(), frame_rate=sr,\n                sample_width=output_wave.dtype.itemsize, channels=1\n            ).export(format=\"mp3\", bitrate=bitrate).read()\n            yield mp3_bytes, None\n        elif is_last_chunk:\n            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len)\n            generated_wave_chunks.append(output_wave)\n            processed_frames += vc_target.size(2) - overlap_frame_len\n            output_wave = (output_wave * 32768.0).astype(np.int16)\n            mp3_bytes = AudioSegment(\n                output_wave.tobytes(), frame_rate=sr,\n                sample_width=output_wave.dtype.itemsize, channels=1\n            ).export(format=\"mp3\", bitrate=bitrate).read()\n            yield mp3_bytes, (sr, np.concatenate(generated_wave_chunks))\n            break\n        else:\n            output_wave = crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len)\n            generated_wave_chunks.append(output_wave)\n            previous_chunk = vc_wave[0, -overlap_wave_len:]\n            processed_frames += vc_target.size(2) - overlap_frame_len\n            output_wave = (output_wave * 32768.0).astype(np.int16)\n            mp3_bytes = AudioSegment(\n                output_wave.tobytes(), frame_rate=sr,\n                sample_width=output_wave.dtype.itemsize, channels=1\n            ).export(format=\"mp3\", bitrate=bitrate).read()\n            yield mp3_bytes, None\n"""
            new_block = """    # split source condition (cond) into chunks and assemble one final output file\n    processed_frames = 0\n    generated_wave_chunks = []\n    while processed_frames < cond.size(1):\n        chunk_cond = cond[:, processed_frames:processed_frames + max_source_window]\n        is_last_chunk = processed_frames + max_source_window >= cond.size(1)\n        cat_condition = torch.cat([prompt_condition, chunk_cond], dim=1)\n        with torch.autocast(device_type=device.type, dtype=torch.float16 if fp16 else torch.float32):\n            vc_target = inference_module.cfm.inference(\n                cat_condition,\n                torch.LongTensor([cat_condition.size(1)]).to(mel2.device),\n                mel2,\n                style2,\n                None,\n                diffusion_steps,\n                inference_cfg_rate=inference_cfg_rate,\n            )\n            vc_target = vc_target[:, :, mel2.size(-1):]\n        vc_wave = vocoder_fn(vc_target.float())[0]\n        if vc_wave.ndim == 1:\n            vc_wave = vc_wave.unsqueeze(0)\n        if processed_frames == 0:\n            if is_last_chunk:\n                generated_wave_chunks.append(vc_wave[0].cpu().numpy())\n                break\n            generated_wave_chunks.append(vc_wave[0, :-overlap_wave_len].cpu().numpy())\n            previous_chunk = vc_wave[0, -overlap_wave_len:]\n            processed_frames += vc_target.size(2) - overlap_frame_len\n        elif is_last_chunk:\n            generated_wave_chunks.append(crossfade(previous_chunk.cpu().numpy(), vc_wave[0].cpu().numpy(), overlap_wave_len))\n            processed_frames += vc_target.size(2) - overlap_frame_len\n            break\n        else:\n            generated_wave_chunks.append(crossfade(previous_chunk.cpu().numpy(), vc_wave[0, :-overlap_wave_len].cpu().numpy(), overlap_wave_len))\n            previous_chunk = vc_wave[0, -overlap_wave_len:]\n            processed_frames += vc_target.size(2) - overlap_frame_len\n\n    output_wave = np.concatenate(generated_wave_chunks) if generated_wave_chunks else np.zeros(0, dtype=np.float32)\n    output_wave = (output_wave * 32768.0).astype(np.int16)\n    output_dir = Path(tempfile.mkdtemp(prefix=\"seed-vc-output-\"))\n    output_path = output_dir / \"voice_conversion.wav\"\n    AudioSegment(\n        output_wave.tobytes(),\n        frame_rate=sr,\n        sample_width=output_wave.dtype.itemsize,\n        channels=1,\n    ).export(output_path, format=\"wav\")\n    return str(output_path)\n"""
        updated = updated.replace(old_block, new_block)
        if updated != text:
            app_path.write_text(updated, encoding="utf-8")

    _patch_file(runtime_install_dir(config) / "app_vc.py")
    _patch_file(runtime_install_dir(config) / "app_svc.py")


def _start_command(config: RuntimeConfig = RuntimeConfig()) -> list[str]:
    python_exe = _runtime_python_executable(config)
    return [
        str(python_exe),
        "app_svc.py",
        "--port",
        str(config.rvc_port),
        "--share",
        "false",
        "--fp16",
        "true",
        "--gpu",
        "0",
    ]


def build_install_command(config: RuntimeConfig = RuntimeConfig()) -> str:
    cuda_step = ""
    if _has_nvidia_tooling():
        cuda_step = (
            f' && "{_runtime_python_executable(config)}" -m pip install --upgrade --force-reinstall --no-cache-dir '
            f'--index-url {TORCH_CUDA_INDEX_URL} '
            f'"torch=={TORCH_CUDA_VERSION}" "torchvision=={TORCHVISION_CUDA_VERSION}" "torchaudio=={TORCH_CUDA_VERSION}"'
            f' && "{_runtime_python_executable(config)}" -m pip install --force-reinstall '
            f'"numpy=={NUMPY_VERSION}" "pillow=={PILLOW_VERSION}"'
        )
    if sys.platform == "win32":
        return (
            f'py -3.10 -m venv "{runtime_install_dir(config) / ".venv"}" && '
            f'"{_runtime_python_executable(config)}" -m pip install --upgrade "pip<24.1" "setuptools<81" wheel && '
            f'"{_runtime_python_executable(config)}" -m pip install -r requirements.txt && '
            f'"{_runtime_python_executable(config)}" -m pip install intel-openmp'
            f"{cuda_step}"
        )
    return (
        f'python3.10 -m venv "{runtime_install_dir(config) / ".venv"}" && '
        f'"{_runtime_python_executable(config)}" -m pip install --upgrade "pip<24.1" "setuptools<81" wheel && '
        f'"{_runtime_python_executable(config)}" -m pip install -r requirements.txt'
        f"{cuda_step}"
    )


def build_start_command(config: RuntimeConfig = RuntimeConfig()) -> str:
    python_exe = _runtime_python_executable(config)
    return f'"{python_exe}" app_svc.py --port {int(config.rvc_port)} --share false --fp16 true --gpu 0'


def runtime_status(config: RuntimeConfig = RuntimeConfig()) -> SeedVcRuntimeStatus:
    install_dir = runtime_install_dir(config)
    repo_present = install_dir.exists() and any((install_dir / name).exists() for name in ("app_vc.py", "app_svc.py"))
    python_exe = _runtime_python_executable(config)
    venv_present = python_exe.exists()
    deps_ready = repo_present and venv_present and _runtime_dependencies_ready(config)
    installed = repo_present and venv_present
    running = api_health(config)
    managed_alive = managed_runtime_alive(config) if install_dir.exists() else False
    progress = startup_progress_snapshot()
    contract_ok, contract_message = (True, "Seed-VC runtime contract looks correct.")
    if installed and (running or managed_alive):
        contract_ok, contract_message = runtime_contract_ready(config, mode="singing")
    if running:
        message = "Seed-VC runtime is reachable."
    elif managed_alive and progress["phase"] != "idle":
        message = progress["message"]
    elif not repo_present:
        message = "Seed-VC runtime is not installed."
    elif not venv_present:
        message = "Seed-VC runtime dependencies are not installed. Run Install Runtime."
    elif not deps_ready:
        message = "Seed-VC runtime dependencies are incomplete. Run Install Runtime again."
    else:
        message = "Seed-VC runtime is installed but not running."
    if installed and not contract_ok:
        message = f"{message} Contract mismatch: {contract_message}"
    if _has_nvidia_tooling():
        if deps_ready:
            message = f"{message} CUDA ready."
        else:
            message = f"{message} CUDA repair needed."
    simple_start_command = "Restart Runtime" if running or managed_alive else "Start Runtime"
    return SeedVcRuntimeStatus(
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


def runtime_contract_ready(
    config: RuntimeConfig = RuntimeConfig(),
    *,
    mode: str = "singing",
) -> tuple[bool, str]:
    python_exe = _runtime_python_executable(config)
    if not python_exe.exists():
        return False, "Seed-VC runtime Python is missing."
    probe_script = """
import inspect
import json
import sys

from seed_vc_wrapper import SeedVCWrapper

signature = inspect.signature(SeedVCWrapper.convert_voice)
print(json.dumps({"parameters": list(signature.parameters.keys())}))
"""
    completed = subprocess.run(
        [str(python_exe), "-c", probe_script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        cwd=runtime_install_dir(config),
        env=_runtime_env(),
        timeout=120,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "Seed-VC contract probe failed."
        return False, stderr
    try:
        payload = json.loads((completed.stdout.strip().splitlines() or ["{}"])[-1])
    except Exception:
        return False, completed.stdout.strip() or "Seed-VC contract probe returned invalid output."
    parameters = [str(item) for item in payload.get("parameters") or []]
    if mode == "singing":
        required = ["source", "target", "diffusion_steps", "length_adjust", "inference_cfg_rate", "f0_condition", "auto_f0_adjust", "pitch_shift"]
    else:
        required = ["source", "target", "diffusion_steps", "length_adjust", "inference_cfg_rate"]
    missing = [name for name in required if name not in parameters]
    if missing:
        return False, f"Seed-VC voice_conversion signature is missing expected parameters: {', '.join(missing)}"
    return True, "Seed-VC runtime contract looks correct."


def run_voice_conversion(
    *,
    source_path: Path,
    reference_path: Path,
    mode: str,
    diffusion_steps: int,
    length_adjust: float,
    inference_cfg_rate: float,
    f0_condition: bool = True,
    auto_f0_adjust: bool = True,
    pitch_shift: int = 0,
    config: RuntimeConfig = RuntimeConfig(),
) -> dict[str, Any]:
    if mode not in {"speaking", "singing"}:
        mode = "singing"
    python_exe = _runtime_python_executable(config)
    if not python_exe.exists():
        raise RuntimeError("Seed-VC runtime dependencies are not installed.")
    runtime_dir = runtime_install_dir(config)
    payload = {
        "runtime_dir": str(runtime_dir),
        "source_path": str(source_path),
        "reference_path": str(reference_path),
        "mode": mode,
        "diffusion_steps": int(diffusion_steps),
        "length_adjust": float(length_adjust),
        "inference_cfg_rate": float(inference_cfg_rate),
        "f0_condition": bool(f0_condition),
        "auto_f0_adjust": bool(auto_f0_adjust),
        "pitch_shift": int(pitch_shift),
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(payload, handle)
        temp_path = Path(handle.name)
    try:
        script = """
import json
import sys
import tempfile
from pathlib import Path

import torch
from pydub import AudioSegment

from seed_vc_wrapper import SeedVCWrapper

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
wrapper = SeedVCWrapper()
gen = wrapper.convert_voice(
    source=payload["source_path"],
    target=payload["reference_path"],
    diffusion_steps=payload["diffusion_steps"],
    length_adjust=payload["length_adjust"],
    inference_cfg_rate=payload["inference_cfg_rate"],
    f0_condition=payload["f0_condition"],
    auto_f0_adjust=payload["auto_f0_adjust"],
    pitch_shift=payload["pitch_shift"],
    stream_output=False,
)
try:
    while True:
        next(gen)
except StopIteration as stop:
    full_audio = stop.value

if full_audio is None:
    raise RuntimeError("Seed-VC runtime returned no audio.")

output_dir = Path(tempfile.mkdtemp(prefix="seed-vc-output-"))
output_path = output_dir / "voice_conversion.wav"
audio = (full_audio * 32768.0).astype("int16")
sr = wrapper.sr_f0 if payload["f0_condition"] else wrapper.sr
AudioSegment(
    audio.tobytes(),
    frame_rate=sr,
    sample_width=audio.dtype.itemsize,
    channels=1,
).export(output_path, format="wav")
print(json.dumps({"result": str(output_path)}, default=str))
"""
        completed = subprocess.run(
            [str(python_exe), "-c", script, str(temp_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            cwd=runtime_dir,
            env={**_runtime_env(), "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"},
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or "Seed-VC runtime conversion failed."
            raise RuntimeError(stderr)
        output_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        output = output_lines[-1] if output_lines else ""
        if not output:
            raise RuntimeError("Seed-VC runtime returned no output.")
        payload_result = json.loads(output)
        return {"result": payload_result.get("result")}
    finally:
        temp_path.unlink(missing_ok=True)


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
    if not any((install_dir / name).exists() for name in ("app_vc.py", "app_svc.py")):
        raise RuntimeError(f"Seed-VC runtime directory exists but does not look complete: {install_dir}")
    _patch_runtime_entrypoint(config)
    venv_dir = _runtime_venv_dir(config)
    if venv_dir.exists():
        python_exe = _runtime_python_executable(config)
        if python_exe.exists():
            version = subprocess.run(
                [str(python_exe), "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
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
        raise RuntimeError("Seed-VC runtime setup finished, but the virtual environment Python was not created.")
    subprocess.run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip<24.1", "setuptools<81", "wheel"], cwd=install_dir, check=True, env=_runtime_env())
    subprocess.run([str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"], cwd=install_dir, check=True, env=_runtime_env())
    if sys.platform == "win32":
        subprocess.run([str(python_exe), "-m", "pip", "install", "intel-openmp"], cwd=install_dir, check=True, env=_runtime_env())
    if _has_nvidia_tooling():
        _install_cuda_torch(python_exe, install_dir)
        _restore_runtime_compat_packages(python_exe, install_dir)
    _patch_runtime_entrypoint(config)


def start_runtime_background(config: RuntimeConfig = RuntimeConfig()) -> subprocess.Popen[bytes]:
    if not _runtime_dependencies_ready(config):
        run_install(config)
    status = runtime_status(config)
    if not status.installed:
        raise RuntimeError(f"Seed-VC runtime is not installed: {config.rvc_dir}")
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
            subprocess.run(
                ["taskkill", "/T", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
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


def ensure_runtime_api(config: RuntimeConfig = RuntimeConfig()) -> SeedVcRuntimeStartResult:
    if api_health(config):
        contract_ok, contract_message = runtime_contract_ready(config, mode="singing")
        return SeedVcRuntimeStartResult(
            started=False,
            already_running=True,
            api_url=api_base_url(config),
            pid=read_runtime_pid(),
            message="Seed-VC runtime is already running." if contract_ok else f"Seed-VC runtime contract mismatch: {contract_message}",
        )
    status = runtime_status(config)
    if not status.installed:
        return SeedVcRuntimeStartResult(
            started=False,
            already_running=False,
            api_url=api_base_url(config),
            pid=None,
            message="Seed-VC runtime is not installed.",
        )
    if managed_runtime_alive(config):
        deadline = time.monotonic() + max(60.0, config.api_startup_timeout_seconds)
        while time.monotonic() < deadline:
            if api_health(config):
                return SeedVcRuntimeStartResult(
                    started=False,
                    already_running=True,
                    api_url=api_base_url(config),
                    pid=read_runtime_pid(),
                    message="Seed-VC runtime is already running.",
                )
            if not managed_runtime_alive(config):
                progress = startup_progress_snapshot()
                return SeedVcRuntimeStartResult(
                    started=False,
                    already_running=False,
                    api_url=api_base_url(config),
                    pid=read_runtime_pid(),
                    message=progress["message"] if progress["phase"] != "idle" else "Seed-VC runtime stopped during startup.",
                )
            time.sleep(2)
        return SeedVcRuntimeStartResult(
            started=False,
            already_running=True,
            api_url=api_base_url(config),
            pid=read_runtime_pid(),
            message="Seed-VC runtime is still loading.",
        )
    process = start_runtime_background(config)
    deadline = time.monotonic() + max(60.0, config.api_startup_timeout_seconds)
    while time.monotonic() < deadline:
        if api_health(config):
            return SeedVcRuntimeStartResult(
                started=True,
                already_running=False,
                api_url=api_base_url(config),
                pid=process.pid,
                message="Seed-VC runtime started in the background.",
            )
        if process.poll() is not None:
            _clear_runtime_pid(process.pid)
            progress = startup_progress_snapshot()
            return SeedVcRuntimeStartResult(
                started=False,
                already_running=False,
                api_url=api_base_url(config),
                pid=process.pid,
                message=progress["message"] if progress["phase"] != "idle" else "Seed-VC runtime failed to start.",
            )
        time.sleep(2)
    return SeedVcRuntimeStartResult(
        started=False,
        already_running=managed_runtime_alive(config),
        api_url=api_base_url(config),
        pid=read_runtime_pid(),
        message="Seed-VC runtime is still starting.",
    )
