"""Audio probing helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from autotransition.audio.ffmpeg import configure_pydub_ffmpeg
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
    try:
        from pydub import AudioSegment
    except ImportError as exc:
        raise RuntimeError("pydub is required to probe audio. Install the project dependencies.") from exc
    configure_pydub_ffmpeg()

    if not source_path.exists():
        raise FileNotFoundError(f"Source audio not found: {source_path}")
    if not source_path.is_file():
        raise ValueError(f"Source audio is not a file: {source_path}")
    validate_supported_source(source_path)

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
