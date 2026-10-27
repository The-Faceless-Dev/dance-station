"""Small RMVPE adapter for extracting a clean singing F0 contour."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


class RmvpeRuntimeError(RuntimeError):
    """Raised when RMVPE cannot be loaded or cannot extract F0."""


@dataclass(frozen=True)
class RmvpeConfig:
    checkpoint: Path
    device: str = "cuda:0"
    threshold: float = 0.03


class RmvpeF0Extractor:
    """Load RMVPE and return 100 Hz F0 values in Hz, with zero for unvoiced."""

    def __init__(self, config: RmvpeConfig):
        checkpoint = Path(config.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise RmvpeRuntimeError(f"RMVPE checkpoint is missing: {checkpoint}")
        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RmvpeRuntimeError("RMVPE requested CUDA but CUDA is unavailable")
        try:
            from autotransition.vendor.rmvpe import RMVPE
        except Exception as exc:  # pragma: no cover - environment validation
            raise RmvpeRuntimeError(f"could not import vendored RMVPE: {exc}") from exc

        self.config = RmvpeConfig(
            checkpoint=checkpoint,
            device=config.device,
            threshold=float(config.threshold),
        )
        try:
            # Keep RMVPE in fp32. It is small, and this avoids pitch logits
            # changing when the singer itself runs in fp16.
            self.model = RMVPE(
                str(checkpoint),
                is_half=False,
                device=config.device,
            )
        except Exception as exc:  # pragma: no cover - depends on checkpoint/device
            raise RmvpeRuntimeError(f"could not load RMVPE: {exc}") from exc

    def extract(self, audio: torch.Tensor, sample_rate: int) -> np.ndarray:
        """Return a mono 100 Hz F0 array aligned to the supplied audio."""

        import torchaudio

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2 or audio.shape[0] < 1 or audio.shape[1] == 0:
            raise RmvpeRuntimeError("RMVPE received empty or malformed audio")
        audio = audio.detach().to(dtype=torch.float32)
        audio = audio.mean(dim=0, keepdim=True)
        if int(sample_rate) != 16_000:
            audio = torchaudio.functional.resample(audio, int(sample_rate), 16_000)
        samples = audio.squeeze(0).cpu().numpy().astype(np.float32, copy=False)
        try:
            f0 = np.asarray(
                self.model.infer_from_audio(samples, thred=self.config.threshold),
                dtype=np.float32,
            )
        except Exception as exc:  # pragma: no cover - depends on CUDA execution
            raise RmvpeRuntimeError(f"RMVPE inference failed: {exc}") from exc
        if f0.ndim != 1 or f0.size == 0 or not np.isfinite(f0).all():
            raise RmvpeRuntimeError("RMVPE returned empty or non-finite F0")
        return np.maximum(f0, 0.0)


def f0_to_midi(f0_hz: np.ndarray) -> np.ndarray:
    """Convert an F0 array in Hz to MIDI numbers, preserving rests as zero."""

    f0 = np.asarray(f0_hz, dtype=np.float32)
    midi = np.zeros_like(f0, dtype=np.float32)
    voiced = np.isfinite(f0) & (f0 > 0)
    midi[voiced] = 69.0 + 12.0 * np.log2(f0[voiced] / 440.0)
    return midi


def resample_f0_to_frames(
    f0_hz: np.ndarray,
    target_frames: int,
    *,
    source_duration_seconds: float,
) -> np.ndarray:
    """Resample RMVPE's 100 Hz contour to a target frame count.

    Pitch is interpolated in MIDI space only inside voiced regions. Voicing is
    interpolated separately so an unvoiced gap cannot become a false sustained
    note merely because its neighboring notes have different pitches.
    """

    if target_frames < 1:
        raise ValueError("target_frames must be positive")
    f0 = np.asarray(f0_hz, dtype=np.float32)
    source_duration_seconds = max(float(source_duration_seconds), 1e-6)
    source_times = np.linspace(0.0, source_duration_seconds, num=f0.size, endpoint=False)
    target_times = np.linspace(
        0.0, source_duration_seconds, num=target_frames, endpoint=False
    )
    voiced = np.isfinite(f0) & (f0 > 0)
    voiced_values = voiced.astype(np.float32)
    voiced_resampled = np.interp(target_times, source_times, voiced_values)
    midi = f0_to_midi(f0)
    valid_indices = np.flatnonzero(voiced)
    if valid_indices.size == 0:
        return np.zeros(target_frames, dtype=np.float32)
    midi_resampled = np.interp(
        target_times,
        source_times[valid_indices],
        midi[valid_indices],
    ).astype(np.float32)
    midi_resampled[voiced_resampled < 0.5] = 0.0
    return midi_resampled


def midi_contour_to_logits(
    midi: np.ndarray,
    *,
    sigma: float = 0.35,
    voiced_peak: float = 0.995,
    floor: float = 1e-4,
) -> torch.Tensor:
    """Encode continuous MIDI as the 128-bin logits YingMusic expects."""

    if sigma <= 0:
        raise ValueError("sigma must be positive")
    midi_values = np.asarray(midi, dtype=np.float32)
    bins = np.arange(128, dtype=np.float32)[None, :]
    voiced = midi_values > 0
    probabilities = np.full((midi_values.size, 128), floor, dtype=np.float32)
    if np.any(voiced):
        distances = bins - midi_values[voiced, None]
        peak = float(np.clip(voiced_peak, floor, 1.0 - floor))
        probabilities[voiced] = np.maximum(
            floor,
            peak * np.exp(-0.5 * (distances / float(sigma)) ** 2),
        )
    probabilities = np.clip(probabilities, floor, 1.0 - floor)
    logits = np.log(probabilities / (1.0 - probabilities))
    return torch.from_numpy(logits.astype(np.float32, copy=False))


def build_yingmusic_melody_condition(
    f0_hz: np.ndarray,
    *,
    target_frames: int,
    source_duration_seconds: float,
    sigma: float = 0.35,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Build `[T, 128]` logits plus a compatible boundary tensor."""

    midi = resample_f0_to_frames(
        f0_hz,
        target_frames,
        source_duration_seconds=source_duration_seconds,
    )
    logits = midi_contour_to_logits(midi, sigma=sigma)
    # The current YingMusic `some_pretrain` branches interpolate this value but
    # do not feed it into the transformer. Keep a correctly shaped neutral
    # tensor for the upstream call contract.
    bounds = torch.zeros(target_frames, 1, dtype=torch.float32)
    return logits, bounds, midi


__all__ = [
    "RmvpeConfig",
    "RmvpeF0Extractor",
    "RmvpeRuntimeError",
    "build_yingmusic_melody_condition",
    "f0_to_midi",
    "midi_contour_to_logits",
    "resample_f0_to_frames",
]
