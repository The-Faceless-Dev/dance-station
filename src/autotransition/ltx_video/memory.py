from __future__ import annotations

import gc
import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GpuMemorySnapshot:
    available: bool
    device: str
    name: str | None = None
    total_bytes: int = 0
    free_bytes: int = 0
    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_allocated_bytes: int = 0
    capability: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("total_bytes", "free_bytes", "allocated_bytes", "reserved_bytes", "peak_allocated_bytes"):
            payload[f"{key[:-6]}Gb" if key.endswith("_bytes") else key] = round(payload.pop(key) / (1024**3), 3)
        return payload


def gpu_memory_snapshot(device: str = "cuda") -> GpuMemorySnapshot:
    try:
        import torch

        if not torch.cuda.is_available():
            return GpuMemorySnapshot(False, device, error="CUDA is not available")
        target = torch.device(device)
        index = target.index if target.index is not None else torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        props = torch.cuda.get_device_properties(index)
        return GpuMemorySnapshot(
            available=True,
            device=str(target),
            name=str(props.name),
            total_bytes=int(total_bytes),
            free_bytes=int(free_bytes),
            allocated_bytes=int(torch.cuda.memory_allocated(index)),
            reserved_bytes=int(torch.cuda.memory_reserved(index)),
            peak_allocated_bytes=int(torch.cuda.max_memory_allocated(index)),
            capability=f"{props.major}.{props.minor}",
        )
    except Exception as exc:
        return GpuMemorySnapshot(False, device, error=f"{type(exc).__name__}: {exc}")


def clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def estimate_peak_vram_gb(
    *,
    width: int,
    height: int,
    frames: int,
    audio: bool,
    offload_mode: str,
    quantization: str = "none",
    conditioning_count: int = 0,
) -> dict[str, float]:
    """Conservative planning estimate, not a substitute for measured peak memory.

    The estimate is intentionally exposed in job metadata so capacity decisions can be
    audited. The official block-streaming modes change the transformer residency, while
    activations still scale with latent tokens and frame count.
    """
    latent_frames = (frames - 1) // 8 + 1
    spatial_tokens = max(1, width // 32) * max(1, height // 32)
    video_tokens = latent_frames * spatial_tokens
    activation_gb = 2.1 + video_tokens * 0.00022
    if conditioning_count < 0:
        raise ValueError("conditioning_count cannot be negative")
    if offload_mode == "none":
        # The official 22B NVFP4 file is about 18.7 GiB. Keep a small amount of
        # headroom in the estimate for loader metadata and quantized buffers.
        transformer_gb = {"nvfp4-prequant": 19.5, "nvfp4-cast": 20.5, "fp8-cast": 25.0, "none": 45.0}.get(quantization, 45.0)
    else:
        transformer_gb = {"nvfp4-prequant": 5.0, "nvfp4-cast": 5.0, "fp8-cast": 6.0, "none": 6.0}.get(quantization, 6.0)
    # These components are deliberately staged by ltx-core's lifecycle helpers:
    # prompt encoding, transformer denoising, latent upscale, and VAE decode do
    # not all occupy the GPU at the same time.
    text_encoder_gb = 24.0
    embeddings_gb = 0.75
    upsampler_gb = 1.5
    vae_gb = 2.0 + (0.75 if width >= 1024 or height >= 1024 else 0.0)
    audio_gb = 2.0 if audio else 0.0
    conditioning_gb = conditioning_count * 0.15
    prompt_phase_gb = text_encoder_gb + embeddings_gb
    denoise_phase_gb = transformer_gb + activation_gb + audio_gb + conditioning_gb
    upscale_phase_gb = upsampler_gb + activation_gb * 0.25
    decode_phase_gb = vae_gb + activation_gb * 0.5
    phase_peaks = {
        "promptEncodingGb": round(prompt_phase_gb, 3),
        "denoisingGb": round(denoise_phase_gb, 3),
        "latentUpscaleGb": round(upscale_phase_gb, 3),
        "videoDecodeGb": round(decode_phase_gb, 3),
    }
    peak_phase, peak_gb = max(phase_peaks.items(), key=lambda item: item[1])
    return {
        "latentFrames": float(latent_frames),
        "videoTokens": float(video_tokens),
        "transformerResidencyGb": transformer_gb,
        "textEncoderResidencyGb": text_encoder_gb,
        "embeddingsProcessorGb": embeddings_gb,
        "upsamplerResidencyGb": upsampler_gb,
        "activationGb": round(activation_gb, 3),
        "vaeAndDecodeGb": vae_gb,
        "audioGb": audio_gb,
        "conditioningGb": round(conditioning_gb, 3),
        "phasePeaksGb": phase_peaks,
        "peakPhase": peak_phase,
        "estimatedPeakGb": round(peak_gb, 3),
    }


def build_memory_plan(
    *,
    width: int,
    height: int,
    frames: int,
    audio: bool,
    offload_mode: str,
    quantization: str,
    reserve_vram_gb: float,
    device: str = "cuda",
    conditioning_count: int = 0,
) -> dict[str, Any]:
    estimate = estimate_peak_vram_gb(
        width=width,
        height=height,
        frames=frames,
        audio=audio,
        offload_mode=offload_mode,
        quantization=quantization,
        conditioning_count=conditioning_count,
    )
    snapshot = gpu_memory_snapshot(device)
    available_gb = snapshot.free_bytes / (1024**3) if snapshot.available else 0.0
    budget_gb = max(0.0, available_gb - reserve_vram_gb)
    return {
        **estimate,
        "gpu": snapshot.to_dict(),
        "reserveGb": reserve_vram_gb,
        "availableBudgetGb": round(budget_gb, 3),
        "fitsCurrentFreeMemory": bool(snapshot.available and estimate["estimatedPeakGb"] <= budget_gb),
        "requiresPreflightGpu": True,
    }


def assert_blackwell_for_nvfp4(snapshot: GpuMemorySnapshot) -> None:
    if not snapshot.available:
        raise RuntimeError("CUDA is required for the NVFP4 LTX production runtime")
    if not snapshot.capability:
        raise RuntimeError("could not determine CUDA compute capability for NVFP4")
    major = int(snapshot.capability.split(".", 1)[0])
    if major < 10:
        raise RuntimeError(
            f"NVFP4 LTX production requires a Blackwell GPU (compute capability >= 10.0); got {snapshot.name} ({snapshot.capability})"
        )
