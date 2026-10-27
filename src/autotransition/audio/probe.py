"""Audio probing helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import wave

from autotransition.audio.ffmpeg import require_pydub
from autotransition.audio.formats import source_extension, source_format_label, validate_supported_source


@dataclass(frozen=True)
class AudioProbe:
    path: Path
    source_extension: str
    source_format: str
    duration_seconds: float
    frame_rate: int
    channels: int
    sample_width: int

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


def probe_audio(source_path: Path) -> AudioProbe:
    if not source_path.exists():
        raise FileNotFoundError(f"Source audio not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Source audio is not a file: {source_path}")
    validate_supported_source(source_path)

    # WAV probing is part of the plan/editor boundary and does not need the
    # heavier pydub import. Keep this fallback so the alignment editor and its
    # tests can work in a minimal environment; compressed formats still use
    # the normal ffmpeg-backed pydub path below.
    if source_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(source_path), "rb") as source:
                frame_rate = source.getframerate()
                frame_count = source.getnframes()
                if frame_rate <= 0:
                    raise ValueError("WAV source has no valid frame rate.")
                return AudioProbe(
                    path=source_path,
                    source_extension=source_extension(source_path),
                    source_format=source_format_label(source_path),
                    duration_seconds=frame_count / frame_rate,
                    frame_rate=frame_rate,
                    channels=source.getnchannels(),
                    sample_width=source.getsampwidth(),
                )
        except (OSError, wave.Error):
            # Fall through to pydub for WAV containers that are not readable by
            # the stdlib wave module.
            pass

    AudioSegment = require_pydub("probe audio")

    source = AudioSegment.from_file(source_path)
    return AudioProbe(
        path=source_path,
        source_extension=source_extension(source_path),
        source_format=source_format_label(source_path),
        duration_seconds=len(source) / 1000,
        frame_rate=source.frame_rate,
        channels=source.channels,
        sample_width=source.sample_width,
    )
