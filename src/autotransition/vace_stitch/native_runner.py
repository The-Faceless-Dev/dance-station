"""Run the upstream VACE script with a reliable tensor-to-video writer."""

from __future__ import annotations

import math
import json
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _as_video_frames(tensor: Any, *, normalize: bool, value_range: tuple[float, float]) -> Any:
    """Convert Wan's [batch, channels, frames, height, width] tensor to uint8 frames."""

    import torch

    value = tensor.detach().to(device="cpu", dtype=torch.float32)
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if value.ndim != 5:
        raise ValueError(f"expected a 4D or 5D video tensor, received shape {tuple(value.shape)}")

    low, high = value_range
    value = value.clamp(min(low, high), max(low, high))
    if normalize:
        span = high - low
        if span <= 0:
            raise ValueError("video value range must have a positive span")
        value = (value - low) / span

    batch, channels, frames, height, width = value.shape
    if channels == 1:
        value = value.repeat(1, 3, 1, 1, 1)
    elif channels < 3:
        value = torch.cat([value, value[:, -1:].repeat(1, 3 - channels, 1, 1, 1)], dim=1)
    else:
        value = value[:, :3]

    # The upstream script normally supplies a batch of one. Keep a deterministic
    # horizontal tile for completeness if a caller provides more than one item.
    columns = min(8, batch)
    rows = math.ceil(batch / columns)
    if batch > 1:
        canvas = value.new_zeros((frames, 3, rows * height, columns * width))
        for index in range(batch):
            row, column = divmod(index, columns)
            canvas[
                :, :, row * height : (row + 1) * height, column * width : (column + 1) * width
            ] = value[index].permute(1, 0, 2, 3)
        value = canvas.permute(1, 0, 2, 3).unsqueeze(0)

    return (
        value[0]
        .permute(1, 2, 3, 0)
        .mul(255)
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
        .contiguous()
        .numpy()
    )


def _cache_video(
    tensor: Any,
    save_file: str | None = None,
    fps: int = 30,
    suffix: str = ".mp4",
    nrow: int = 8,
    normalize: bool = True,
    value_range: tuple[float, float] = (-1, 1),
    retry: int = 5,
) -> str | None:
    """Write video tensors without torchvision's in-place dtype conversion."""

    del suffix, nrow, retry
    if not save_file:
        raise ValueError("VACE save_file is required")
    frames = _as_video_frames(tensor, normalize=normalize, value_range=value_range)
    height, width, channels = frames.shape[1:]
    if channels != 3:
        raise ValueError(f"video writer requires RGB frames, received {channels} channels")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(save_file),
    ]
    result = subprocess.run(command, input=frames.tobytes(), capture_output=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg video writer failed with exit code {result.returncode}: {detail[-2000:]}")
    return save_file


def _save_dir(argv: list[str]) -> Path | None:
    try:
        value = argv[argv.index("--save_dir") + 1]
    except (ValueError, IndexError):
        return None
    return Path(value)


def _write_diagnostics(save_dir: Path | None, report: dict[str, Any]) -> None:
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "runtime-diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _configure_torch_runtime() -> dict[str, Any]:
    """Configure and verify the CUDA path before upstream VACE imports its model."""

    import importlib.util

    import torch

    report: dict[str, Any] = {
        "cudaAvailable": bool(torch.cuda.is_available()),
        "attentionRequested": os.getenv("VACE_STITCH_ATTENTION_BACKEND", "auto").lower(),
        "tf32Requested": os.getenv("VACE_STITCH_TF32", "false").lower() in {"1", "true", "yes", "on"},
    }
    if not torch.cuda.is_available():
        raise RuntimeError("VACE requires CUDA, but torch.cuda.is_available() is false")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    report.update(
        {
            "device": str(device),
            "deviceName": properties.name,
            "computeCapability": f"{properties.major}.{properties.minor}",
            "totalMemoryBytes": int(properties.total_memory),
            "torchVersion": torch.__version__,
            "torchCudaVersion": torch.version.cuda,
            "flashAttention2Importable": bool(importlib.util.find_spec("flash_attn")),
            "flashAttention3Importable": bool(importlib.util.find_spec("flash_attn_interface")),
        }
    )

    if report["tf32Requested"]:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    report["tf32Enabled"] = bool(torch.backends.cuda.matmul.allow_tf32)

    # Wan's official attention module selects FlashAttention automatically when
    # its CUDA extension imports. A small real-kernel probe prevents a package
    # that imports successfully but does not support this GPU from failing deep
    # inside a paid job.
    from wan.modules import attention as wan_attention

    flash2_available = bool(getattr(wan_attention, "FLASH_ATTN_2_AVAILABLE", False))
    flash3_available = bool(getattr(wan_attention, "FLASH_ATTN_3_AVAILABLE", False))
    flash_available = flash2_available or flash3_available
    report["flashAttentionAvailable"] = flash_available
    report["flashAttention2Available"] = flash2_available
    report["flashAttention3Available"] = flash3_available
    requested = str(report["attentionRequested"])
    if requested not in {"auto", "flash_attention_2"}:
        raise RuntimeError(f"unsupported VACE attention backend: {requested}")

    selected = "sdpa"
    flash_error: str | None = None
    if flash_available and (requested == "auto" or flash2_available):
        try:
            q = torch.randn((1, 64, 4, 64), device=device, dtype=torch.float16)
            # Explicitly exercise FA2 for the strict mode. Auto mode follows
            # Wan's normal preference, which is FA3 when that extension exists.
            version = 2 if requested == "flash_attention_2" else None
            wan_attention.flash_attention(q, q, q, dtype=torch.float16, version=version)
            torch.cuda.synchronize(device)
            selected = "flash_attention_3" if version is None and flash3_available else "flash_attention_2"
        except Exception as exc:
            flash_error = f"{type(exc).__name__}: {exc}"
            # auto is allowed to use PyTorch SDPA if the extension is present
            # but the current GPU/ABI cannot execute it. Explicit flash mode is
            # strict so the Salad benchmark cannot silently change kernels.
            wan_attention.FLASH_ATTN_2_AVAILABLE = False
            wan_attention.FLASH_ATTN_3_AVAILABLE = False
    if requested == "flash_attention_2" and selected != "flash_attention_2":
        detail = f" ({flash_error})" if flash_error else ""
        raise RuntimeError(f"FlashAttention self-test failed on {properties.name}{detail}")

    report["attentionSelected"] = selected
    report["flashAttentionSelfTest"] = "passed" if selected == "flash_attention_2" else "fallback"
    if flash_error:
        report["flashAttentionError"] = flash_error
    print("[vace-runtime] " + json.dumps(report, sort_keys=True), flush=True)
    return report


def _patch_vace_meta_frequency_buffer() -> dict[str, Any]:
    """Materialize Wan's unregistered RoPE tensor after meta checkpoint loading.

    ``VaceWanModel.from_pretrained`` can construct the model under Accelerate's
    meta-device loader.  The positional-frequency tensor is a plain tensor,
    not a parameter or registered buffer, so it is absent from the checkpoint
    state dict and remains meta.  The upstream forward path then tries to copy
    that tensor to CUDA and fails before the first denoising step.
    """

    import torch
    from models.wan.modules.model import VaceWanModel
    from wan.modules.model import rope_params

    original = VaceWanModel.from_pretrained
    if getattr(original, "_autotransition_meta_frequency_patch", False):
        return {"patched": False, "reason": "already-installed"}

    def patched_from_pretrained(cls: Any, *args: Any, **kwargs: Any) -> Any:
        model = original(*args, **kwargs)
        frequencies = getattr(model, "freqs", None)
        if frequencies is None or bool(getattr(frequencies, "is_meta", False)):
            dim = int(model.dim)
            heads = int(model.num_heads)
            head_dim = dim // heads
            with torch.device("cpu"):
                model.freqs = torch.cat(
                    [
                        rope_params(1024, head_dim - 4 * (head_dim // 6)),
                        rope_params(1024, 2 * (head_dim // 6)),
                        rope_params(1024, 2 * (head_dim // 6)),
                    ],
                    dim=1,
                )
            return model
        return model

    patched_from_pretrained._autotransition_meta_frequency_patch = True
    VaceWanModel.from_pretrained = classmethod(patched_from_pretrained)
    return {"patched": True, "source": "models.wan.modules.model.VaceWanModel"}


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] != "--script":
        raise SystemExit("usage: python -m autotransition.vace_stitch.native_runner --script SCRIPT [VACE ARGS]")

    script = Path(sys.argv[2]).resolve()
    if not script.is_file():
        raise FileNotFoundError(script)

    from wan.utils import utils as wan_utils

    wan_utils.cache_video = _cache_video
    save_dir = _save_dir(sys.argv[3:])
    started = time.perf_counter()
    diagnostics = _configure_torch_runtime()
    diagnostics["startedAtEpochSeconds"] = time.time()
    _write_diagnostics(save_dir, diagnostics)
    # The upstream VACE entrypoint imports sibling packages such as
    # ``models.wan`` relative to its own ``vace`` directory. ``runpy`` does
    # not add the script directory to sys.path like a direct script launch.
    script_root = str(script.parent)
    if script_root not in sys.path:
        sys.path.insert(0, script_root)
    frequency_patch = _patch_vace_meta_frequency_buffer()
    diagnostics["metaFrequencyPatch"] = frequency_patch
    _write_diagnostics(save_dir, diagnostics)
    from .config import VaceStitchConfig
    from .fp8_loader import install_scaled_fp8_loader

    vace_config = VaceStitchConfig.from_env()
    install_scaled_fp8_loader(
        vace_config.checkpoint_dir,
        vace_config.checkpoint_file,
        save_dir / "vace-model-loader.json" if save_dir is not None else None,
    )
    sys.argv = [str(script), *sys.argv[3:]]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except Exception as exc:
        diagnostics["status"] = "failed"
        diagnostics["errorType"] = type(exc).__name__
        diagnostics["error"] = str(exc)
        raise
    finally:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            diagnostics["peakAllocatedBytes"] = int(torch.cuda.max_memory_allocated())
            diagnostics["peakReservedBytes"] = int(torch.cuda.max_memory_reserved())
        diagnostics["elapsedSeconds"] = round(time.perf_counter() - started, 3)
        diagnostics.setdefault("status", "succeeded")
        _write_diagnostics(save_dir, diagnostics)
        print("[vace-runtime] completed " + json.dumps(diagnostics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
