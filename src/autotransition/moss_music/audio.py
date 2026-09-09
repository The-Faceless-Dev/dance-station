from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from autotransition.audio.ffmpeg import resolve_ffmpeg
from .contracts import MossAudioInput


@dataclass(frozen=True)
class AudioBuffer:
    path: Path
    samples: np.ndarray
    sample_rate: int

    @property
    def duration_seconds(self) -> float:
        return float(self.samples.size / self.sample_rate)


def acquire_audio(
    audio: MossAudioInput,
    destination: Path,
    *,
    max_bytes: int,
    timeout_seconds: float = 180.0,
) -> Path:
    """Copy a local source or download a bounded remote source into the job."""

    destination.mkdir(parents=True, exist_ok=True)
    filename = Path(audio.filename).name or "audio"
    target = destination / filename
    if audio.path is not None:
        source = audio.path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        if source.stat().st_size > max_bytes:
            raise ValueError("audio source exceeds the worker size limit")
        if source != target.resolve():
            shutil.copy2(source, target)
        return target

    request = Request(audio.source_url, headers={"Accept": "audio/*,application/octet-stream"})
    total = 0
    try:
        with urlopen(request, timeout=timeout_seconds) as response, target.open("wb") as handle:
            declared = response.headers.get("Content-Length")
            if declared and int(declared) > max_bytes:
                raise ValueError("audio source exceeds the worker size limit")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("audio source exceeds the worker size limit")
                handle.write(chunk)
    except HTTPError as exc:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"audio source HTTP {exc.code}") from exc
    except (URLError, OSError) as exc:
        target.unlink(missing_ok=True)
        raise RuntimeError(f"audio source download failed: {exc}") from exc
    if total == 0:
        target.unlink(missing_ok=True)
        raise ValueError("audio source was empty")
    return target


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return samples.astype(np.float32, copy=False)
    target_length = max(1, round(samples.size * target_rate / source_rate))
    source_positions = np.linspace(0, samples.size - 1, num=target_length, dtype=np.float64)
    return np.interp(source_positions, np.arange(samples.size), samples).astype(np.float32)


def _decode_wave(source: Path, sample_rate: int) -> np.ndarray | None:
    try:
        with wave.open(str(source), "rb") as handle:
            channels = handle.getnchannels()
            source_rate = handle.getframerate()
            width = handle.getsampwidth()
            frame_count = handle.getnframes()
            raw = handle.readframes(frame_count)
    except (wave.Error, OSError):
        return None
    if width != 2 or channels < 1:
        return None
    values = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        values = values.reshape(-1, channels).mean(axis=1)
    return _resample(values, source_rate, sample_rate)


def _write_wave(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)


def normalize_audio(
    source: Path,
    destination: Path,
    *,
    sample_rate: int = 16000,
    max_duration_seconds: float = 1800.0,
    progress: Callable[[str], None] | None = None,
) -> AudioBuffer:
    """Decode any ffmpeg-supported source to the model's canonical mono WAV."""

    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "normalized.wav"
    samples = _decode_wave(source, sample_rate) if source.suffix.lower() == ".wav" else None
    if samples is None:
        ffmpeg = resolve_ffmpeg()
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to decode MOSS-Music audio")
        if progress:
            progress("Decoding source audio with ffmpeg")
        command = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "-f",
            "s16le",
            "pipe:1",
        ]
        try:
            result = subprocess.run(command, check=False, capture_output=True, timeout=max_duration_seconds + 120)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("audio decoding exceeded the configured timeout") from exc
        if result.returncode != 0:
            detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
            raise RuntimeError(f"ffmpeg audio decode failed: {detail}")
        if not result.stdout:
            raise RuntimeError("ffmpeg returned no audio samples")
        samples = np.frombuffer(result.stdout, dtype="<i2").astype(np.float32) / 32768.0
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        raise ValueError("audio contains no samples")
    duration = samples.size / sample_rate
    if duration > max_duration_seconds:
        raise ValueError(f"audio duration {duration:.2f}s exceeds the {max_duration_seconds:.2f}s worker limit")
    _write_wave(output, samples, sample_rate)
    return AudioBuffer(path=output, samples=samples, sample_rate=sample_rate)
