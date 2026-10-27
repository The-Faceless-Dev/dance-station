"""F0, voicing, energy, and waveform artifacts for the dataset manager."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from autotransition.vevo2_dataset.prosody import Vevo2ProsodyConfig, extract_prosody


def extract_features(
    audio_path: Path,
    output_dir: Path,
    *,
    sample_rate: int,
    rmvpe_checkpoint: Path | None = None,
    rmvpe_device: str = "cuda:0",
) -> dict[str, Any]:
    """Prefer the project's RMVPE adapter and retain a transparent fallback.

    The manager does not claim that a Parselmouth fallback is RMVPE. Its
    provenance is recorded in the returned metadata and in the artifact names.
    """

    import librosa
    import soundfile as sf

    audio, actual_rate = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_hop = 240
    energy = librosa.feature.rms(y=audio, frame_length=960, hop_length=frame_hop, center=True)[0].astype(np.float32)
    np.save(output_dir / "energy_rms_100hz.npy", energy)

    provider = "parselmouth-equivalent"
    f0 = None
    if rmvpe_checkpoint:
        from autotransition.runtime.rmvpe import RmvpeConfig, RmvpeF0Extractor
        import torch

        waveform = torch.from_numpy(audio)
        f0 = RmvpeF0Extractor(RmvpeConfig(rmvpe_checkpoint, device=rmvpe_device)).extract(waveform, actual_rate)
        provider = "rmvpe"
    else:
        diagnostic = extract_prosody(
            audio_path,
            output_dir / "vevo2_diagnostics",
            config=Vevo2ProsodyConfig(sample_rate=sample_rate),
            include_f0_diagnostic=True,
        )
        f0_path = output_dir / "vevo2_diagnostics" / "f0_hz_50hz_diagnostic.npy"
        if f0_path.exists():
            f0 = np.load(f0_path).astype(np.float32)
    if f0 is None:
        f0 = np.zeros(max(1, int(audio.size / frame_hop)), dtype=np.float32)
        provider = "unavailable"
    f0 = np.asarray(f0, dtype=np.float32)
    voiced = (f0 > 0).astype(np.uint8)
    np.save(output_dir / "f0_hz.npy", f0)
    np.save(output_dir / "voiced.npy", voiced)
    waveform_path = output_dir / "analysis_audio.wav"
    sf.write(waveform_path, audio, sample_rate, subtype="PCM_16")
    return {
        "provider": provider,
        "device": rmvpe_device if provider == "rmvpe" else "cpu",
        "sample_rate": int(sample_rate),
        "f0_hz": "f0_hz.npy",
        "voiced": "voiced.npy",
        "energy_rms": "energy_rms_100hz.npy",
        "waveform_audio": "analysis_audio.wav",
        "f0_rate_hz": 100.0 if provider == "rmvpe" else float(sample_rate / frame_hop),
        "duration_seconds": float(audio.size / sample_rate),
        "warning": None if provider in {"rmvpe", "parselmouth-equivalent"} else "No F0 extractor was available.",
    }


def waveform_peaks(audio_path: Path, output_path: Path, *, bins: int = 2400) -> dict[str, Any]:
    import librosa

    audio, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    bins = max(64, min(int(bins), max(64, audio.size // 8)))
    chunks = np.array_split(np.abs(audio), bins)
    peaks = np.asarray([float(chunk.max()) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
    rms = np.asarray([float(np.sqrt(np.mean(chunk * chunk))) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, peaks=peaks, rms=rms, sample_rate=int(sample_rate), duration_seconds=float(audio.size / sample_rate))
    return {"path": output_path.name, "bins": int(bins), "sample_rate": int(sample_rate), "duration_seconds": float(audio.size / sample_rate)}
