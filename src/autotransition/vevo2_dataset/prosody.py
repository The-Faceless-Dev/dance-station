"""Vevo2-compatible prosody extraction and audio normalization.

The chromagram is the authoritative pre-quantization prosody representation.
F0 and spectral measurements are deliberately diagnostics only: Vevo2's AR
conditioning path consumes its learned prosody code, not an RMVPE contour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Vevo2ProsodyConfig:
    sample_rate: int = 24_000
    n_fft: int = 1_920
    win_length: int = 1_920
    hop_length: int = 480
    n_chroma: int = 24
    prosody_stride: int = 8

    @property
    def frame_rate_hz(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def prosody_frame_rate_hz(self) -> float:
        return self.frame_rate_hz / self.prosody_stride


def _require_librosa():
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover - depends on installation
        raise RuntimeError(
            "Vevo2 dataset enrichment requires librosa; install the "
            "vevo2-dataset dependencies before running it."
        ) from exc
    return librosa


def normalize_audio(source: Path, destination: Path, config: Vevo2ProsodyConfig) -> dict[str, Any]:
    """Write the exact mono/24 kHz audio representation used for extraction."""

    librosa = _require_librosa()
    import soundfile as sf

    audio, _ = librosa.load(str(source), sr=config.sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(destination), audio, config.sample_rate, subtype="PCM_16")
    return {
        "sample_rate": config.sample_rate,
        "channels": 1,
        "sample_count": int(audio.size),
        "duration_seconds": float(audio.size / config.sample_rate),
    }


def _frame_count(sample_count: int, config: Vevo2ProsodyConfig) -> int:
    # Vevo2 computes frame_len with integer division before calling its
    # chromagram helper. Keep this floor behavior so adapter metadata lines up
    # with the existing AR/codec path rather than adding a synthetic tail.
    return max(1, int(sample_count // config.hop_length))


def _fit_frames(values: np.ndarray, frame_count: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[0] == frame_count:
        return values
    if values.shape[0] > frame_count:
        return values[:frame_count]
    if values.shape[0] == 0:
        return np.zeros((frame_count,) + values.shape[1:], dtype=np.float32)
    padding = np.repeat(values[-1:], frame_count - values.shape[0], axis=0)
    return np.concatenate([values, padding], axis=0)


def _group_frames(values: np.ndarray, stride: int) -> np.ndarray:
    """Create a deterministic low-rate summary for timeline bookkeeping.

    This is not a substitute for Vevo2's learned prosody tokenizer. It exists
    so training metadata can reference the 6.25 Hz timeline without pretending
    that a mean chroma vector is a codec/prosody token.
    """

    count = int(math.ceil(values.shape[0] / stride))
    padded_count = count * stride
    if padded_count > values.shape[0]:
        values = np.concatenate([values, np.repeat(values[-1:], padded_count - values.shape[0], axis=0)])
    return values.reshape(count, stride, *values.shape[1:]).mean(axis=1).astype(np.float32)


def _diagnostic_f0(audio: np.ndarray, config: Vevo2ProsodyConfig, frame_count: int) -> np.ndarray | None:
    try:
        import parselmouth
    except ImportError:
        return None

    sound = parselmouth.Sound(audio, sampling_frequency=config.sample_rate)
    pitch = sound.to_pitch(
        time_step=config.hop_length / config.sample_rate,
        pitch_floor=50,
        pitch_ceiling=1100,
    )
    values = np.asarray(pitch.selected_array["frequency"], dtype=np.float32)
    return _fit_frames(values, frame_count)


def extract_prosody(
    audio_path: Path,
    output_dir: Path,
    config: Vevo2ProsodyConfig | None = None,
    include_f0_diagnostic: bool = True,
) -> dict[str, Any]:
    """Extract the Vevo2-compatible chroma intermediate and diagnostics."""

    config = config or Vevo2ProsodyConfig()
    librosa = _require_librosa()
    audio, _ = librosa.load(str(audio_path), sr=config.sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    frame_count = _frame_count(audio.size, config)

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        n_chroma=config.n_chroma,
    ).T.astype(np.float32)
    chroma = _fit_frames(chroma, frame_count)

    rms = librosa.feature.rms(
        y=audio,
        frame_length=config.win_length,
        hop_length=config.hop_length,
        center=True,
    )[0]
    rms = _fit_frames(rms[:, None], frame_count)[:, 0]
    centroid = librosa.feature.spectral_centroid(
        y=audio,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
    )[0]
    centroid = _fit_frames(centroid[:, None], frame_count)[:, 0]

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "chroma_50hz.npy", chroma)
    np.save(output_dir / "chroma_6_25hz_summary.npy", _group_frames(chroma, config.prosody_stride))
    np.save(output_dir / "energy_rms_50hz.npy", rms.astype(np.float32))
    np.save(output_dir / "spectral_centroid_50hz.npy", centroid.astype(np.float32))

    f0 = _diagnostic_f0(audio, config, frame_count) if include_f0_diagnostic else None
    if f0 is not None:
        np.save(output_dir / "f0_hz_50hz_diagnostic.npy", f0)

    artifacts = {
        "representation": "vevo2_chroma_prequantization",
        "sample_rate": config.sample_rate,
        "n_fft": config.n_fft,
        "win_length": config.win_length,
        "hop_length": config.hop_length,
        "n_chroma": config.n_chroma,
        "frame_rate_hz": config.frame_rate_hz,
        "prosody_frame_rate_hz": config.prosody_frame_rate_hz,
        "frame_count": frame_count,
        "duration_seconds": float(audio.size / config.sample_rate),
        "chroma_50hz": "chroma_50hz.npy",
        "chroma_6_25hz_summary": "chroma_6_25hz_summary.npy",
        "energy_rms_50hz": "energy_rms_50hz.npy",
        "spectral_centroid_50hz": "spectral_centroid_50hz.npy",
        "f0_hz_50hz_diagnostic": "f0_hz_50hz_diagnostic.npy" if f0 is not None else None,
        "f0_is_authoritative": False,
        "codec_tokens_extracted": False,
    }
    return artifacts
