"""ffmpeg discovery and pydub configuration."""

from __future__ import annotations

import shutil
from pathlib import Path


def resolve_ffmpeg() -> str | None:
    """Return a usable ffmpeg executable path from PATH or bundled package."""

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        bundled_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    return bundled_ffmpeg if bundled_ffmpeg and Path(bundled_ffmpeg).exists() else None


def configure_pydub_ffmpeg() -> str | None:
    """Point pydub at the resolved ffmpeg executable."""

    ffmpeg_path = resolve_ffmpeg()
    if not ffmpeg_path:
        return None
    try:
        from pydub import AudioSegment
    except ImportError:
        return ffmpeg_path
    AudioSegment.converter = ffmpeg_path
    AudioSegment.ffmpeg = ffmpeg_path
    return ffmpeg_path
