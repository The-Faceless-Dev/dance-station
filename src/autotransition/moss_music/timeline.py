from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from .audio import AudioBuffer


def _finite(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def build_dense_timeline(
    audio: AudioBuffer,
    *,
    resolution_ms: int = 80,
    max_cells: int = 100000,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, absolute-time acoustic grid for visual sync."""

    hop = max(1, round(audio.sample_rate * resolution_ms / 1000))
    total_samples = int(audio.samples.size)
    cell_count = max(1, math.ceil(total_samples / hop))
    if cell_count > max_cells:
        raise ValueError(f"audio timeline requires {cell_count} cells, above the {max_cells} cell limit")

    window_size = min(4096, max(512, hop * 4))
    nfft = 1
    while nfft < window_size:
        nfft *= 2
    frequencies = np.fft.rfftfreq(nfft, 1.0 / audio.sample_rate)
    band_limits = ((0, 80), (80, 250), (250, 2000), (2000, 8000))
    band_masks = [(frequencies >= low) & (frequencies < high) for low, high in band_limits]
    frames: list[np.ndarray] = []
    rms_values: list[float] = []
    peak_values: list[float] = []
    centroids: list[float] = []
    flux_values: list[float] = []
    band_values: list[list[float]] = []
    chroma_values: list[list[float]] = []
    previous_spectrum: np.ndarray | None = None
    window = np.hanning(window_size).astype(np.float32)

    for index in range(cell_count):
        start = index * hop
        end = min(total_samples, start + hop)
        frame = audio.samples[start:end]
        frames.append(frame)
        padded = np.zeros(window_size, dtype=np.float32)
        if frame.size:
            copy_count = min(frame.size, window_size)
            padded[:copy_count] = frame[:copy_count]
        rms_values.append(_finite(np.sqrt(np.mean(padded * padded))))
        peak_values.append(_finite(np.max(np.abs(padded))))
        spectrum = np.abs(np.fft.rfft(padded * window, n=nfft)).astype(np.float32)
        total_energy = float(np.sum(spectrum) + 1e-8)
        centroids.append(_finite(float(np.sum(frequencies * spectrum) / total_energy)))
        normalized = spectrum / (float(np.linalg.norm(spectrum)) + 1e-8)
        flux_values.append(_finite(float(np.linalg.norm(normalized - previous_spectrum))) if previous_spectrum is not None else 0.0)
        previous_spectrum = normalized
        band_values.append([_finite(float(np.sum(spectrum[mask]) / total_energy)) for mask in band_masks])
        chroma = np.zeros(12, dtype=np.float32)
        valid = (frequencies >= 55) & (frequencies <= 5000)
        valid_frequencies = frequencies[valid]
        midi = np.rint(69 + 12 * np.log2(np.maximum(valid_frequencies, 1e-6) / 440.0)).astype(int)
        for pitch_class in range(12):
            chroma[pitch_class] = float(np.sum(spectrum[valid][midi % 12 == pitch_class]))
        chroma_total = float(np.sum(chroma) + 1e-8)
        chroma_values.append([_finite(float(value / chroma_total)) for value in chroma])
        if progress:
            progress(index + 1, cell_count)

    max_rms = max(rms_values) if rms_values else 0.0
    silence_threshold = max(0.003, max_rms * 0.03)
    cells: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        start_seconds = index * resolution_ms / 1000.0
        end_seconds = min(audio.duration_seconds, (index + 1) * resolution_ms / 1000.0)
        previous_flux = flux_values[index - 1] if index else 0.0
        onset_strength = max(0.0, flux_values[index] - previous_flux)
        cells.append(
            {
                "index": index,
                "start_seconds": round(start_seconds, 6),
                "end_seconds": round(max(start_seconds, end_seconds), 6),
                "center_seconds": round((start_seconds + end_seconds) / 2.0, 6),
                "rms": round(rms_values[index], 7),
                "peak": round(peak_values[index], 7),
                "onset_strength": round(_finite(onset_strength), 7),
                "spectral_centroid_hz": round(centroids[index], 4),
                "spectral_flux": round(flux_values[index], 7),
                "band_energy": {
                    "sub_bass": round(band_values[index][0], 7),
                    "bass": round(band_values[index][1], 7),
                    "mid": round(band_values[index][2], 7),
                    "high": round(band_values[index][3], 7),
                },
                "chroma": [round(value, 7) for value in chroma_values[index]],
                "is_silent": rms_values[index] <= silence_threshold,
            }
        )
    return {
        "resolution_ms": resolution_ms,
        "sample_rate": audio.sample_rate,
        "duration_seconds": round(audio.duration_seconds, 6),
        "cell_count": len(cells),
        "silence_threshold": round(silence_threshold, 7),
        "cells": cells,
    }
