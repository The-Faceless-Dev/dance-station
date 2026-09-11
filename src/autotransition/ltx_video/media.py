from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .config import LtxVideoConfig


class LtxMediaError(RuntimeError):
    pass


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        executable = shutil.which("ffmpeg")
        if executable:
            return executable
        raise LtxMediaError("ffmpeg is required for LTX video input normalization and output encoding")


def _ffprobe() -> str:
    executable = shutil.which("ffprobe")
    if executable:
        return executable
    try:
        ffmpeg = Path(_ffmpeg())
        candidate = ffmpeg.with_name("ffprobe" + ffmpeg.suffix)
        if candidate.is_file():
            return str(candidate)
    except LtxMediaError:
        pass
    raise LtxMediaError("ffprobe is required for LTX media inspection")


def acquire_input(source_url: str, local_path: Path | None, destination: Path, config: LtxVideoConfig) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if local_path is not None:
        if not config.allow_local_inputs:
            raise LtxMediaError("local input paths are disabled for this worker")
        if not local_path.is_file():
            raise LtxMediaError(f"input file was not found: {local_path}")
        shutil.copy2(local_path, destination)
        return destination
    if not source_url:
        raise LtxMediaError("input source URL is missing")
    request = Request(source_url, headers={"User-Agent": "faceless-dancer-ltx-worker/1"})
    try:
        with urlopen(request, timeout=config.request_timeout_seconds) as response, destination.open("wb") as handle:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > config.max_download_bytes:
                    raise LtxMediaError(f"input exceeds max_download_bytes ({config.max_download_bytes})")
                handle.write(chunk)
    except HTTPError as exc:
        raise LtxMediaError(f"input download returned HTTP {exc.code}") from exc
    except (URLError, OSError) as exc:
        raise LtxMediaError(f"input download failed: {exc}") from exc
    return destination


def normalize_image(source: Path, destination: Path, *, width: int, height: int) -> Path:
    """Fit a conditioning image without distortion and pad to the target canvas."""
    command = [
        _ffmpeg(), "-y", "-loglevel", "error", "-i", str(source),
        "-vf", f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1",
        "-frames:v", "1", "-f", "image2", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise LtxMediaError(f"conditioning image normalization failed: {result.stderr[-2000:]}")
    return destination


def probe(path: Path) -> dict[str, object]:
    command = [_ffprobe(), "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise LtxMediaError(f"media probe failed: {result.stderr[-2000:]}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise LtxMediaError("ffmpeg returned invalid probe JSON") from exc


def mux_audio(video_path: Path, audio_path: Path, destination: Path, *, output_format: str) -> Path:
    if output_format == "webm":
        video_codec = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32"]
        audio_codec = ["-c:a", "libopus", "-b:a", "128k"]
    else:
        video_codec = ["-c:v", "copy"]
        audio_codec = ["-c:a", "aac", "-b:a", "192k"]
    command = [
        _ffmpeg(), "-y", "-loglevel", "error", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", *video_codec, *audio_codec, "-shortest", str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise LtxMediaError(f"audio mux failed: {result.stderr[-2000:]}")
    return destination


def transcode_video(source: Path, destination: Path, *, output_format: str) -> Path:
    if source.suffix.lower() == f".{output_format}":
        shutil.copy2(source, destination)
        return destination
    if output_format == "webm":
        codec = ["-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "32", "-c:a", "libopus"]
    else:
        codec = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac"]
    command = [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(source), *codec, str(destination)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        raise LtxMediaError(f"video transcode failed: {result.stderr[-2000:]}")
    return destination
