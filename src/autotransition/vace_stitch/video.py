"""FFmpeg helpers for VACE context construction and bridge extraction."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from autotransition.generative_dance.video import VideoProbe, probe_video, resolve_ffmpeg


class VaceVideoError(RuntimeError):
    """Raised when a video cannot satisfy the stitcher's technical contract."""


def _number(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".") or "0"


def _run(command: list[str], *, purpose: str) -> None:
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode:
        detail = result.stderr.strip()[-5000:]
        raise VaceVideoError(f"FFmpeg {purpose} failed: {detail}")


def frame_count(probe: VideoProbe, fps: int) -> int:
    if probe.frame_count is not None and probe.frame_count > 0:
        return probe.frame_count
    if probe.duration_seconds <= 0 or fps < 1:
        raise VaceVideoError(f"video has no usable frame count: {probe.to_dict()}")
    return max(1, int(round(probe.duration_seconds * fps)))


def normalize_canvas(
    source: Path,
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    background: str = "0x7f7f7f",
    preserve_alpha: bool = False,
) -> VideoProbe:
    """Resize/pad for technical compatibility without subject fitting.

    The full source frame is retained. Padding is centered only because the
    target canvas requires a fixed raster; no detected subject bounds,
    orientation, or placement transform is used here.
    """

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VaceVideoError("ffmpeg is required for VACE video preparation")
    if width < 64 or height < 64 or fps < 1:
        raise ValueError("invalid VACE canvas dimensions or FPS")
    if not source.is_file():
        raise VaceVideoError(f"video was not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=bilinear"
    if preserve_alpha:
        graph = f"[0:v]format=rgba,{scale},pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black@0,fps={fps},format=rgba[out]"
        encoder = [
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            "-vendor",
            "apl0",
        ]
    else:
        graph = (
            f"[0:v]format=rgba,{scale},pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black@0[foreground];"
            f"color=c={background}:s={width}x{height}:r={fps}[background];"
            f"[background][foreground]overlay=shortest=1:format=auto,fps={fps},format=yuv420p[out]"
        )
        encoder = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-an",
        "-filter_complex",
        graph,
        "-map",
        "[out]",
        *encoder,
        str(output),
    ]
    _run(command, purpose="canvas normalization")
    return probe_video(output)


def extract_time_range(
    source: Path,
    output: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    fps: int,
    preserve_alpha: bool = False,
) -> VideoProbe:
    """Extract a time range while retaining alpha when the source has it."""

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VaceVideoError("ffmpeg is required to extract VACE source ranges")
    if start_seconds < 0 or end_seconds <= start_seconds or fps < 1:
        raise ValueError("invalid VACE source time range")
    if not source.is_file():
        raise VaceVideoError(f"video was not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if preserve_alpha:
        graph = "fps={0},format=rgba".format(fps)
        encoder = [
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            "-vendor",
            "apl0",
        ]
    else:
        graph = f"fps={fps},format=yuv420p"
        encoder = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    command = [
        ffmpeg,
        "-y",
        "-ss",
        _number(start_seconds),
        "-i",
        str(source),
        "-t",
        _number(end_seconds - start_seconds),
        "-an",
        "-vf",
        graph,
        *encoder,
        str(output),
    ]
    _run(command, purpose="VACE source range extraction")
    return probe_video(output)


def solid_video(
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    frames: int,
    color: str,
    grayscale: bool = False,
) -> VideoProbe:
    """Create an exact-frame-count context or mask segment."""

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VaceVideoError("ffmpeg is required to create VACE masks")
    if frames < 1:
        raise ValueError("solid video requires at least one frame")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:r={fps}",
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if grayscale:
        command.extend(["-vf", "format=gray"])
    command.extend(["-movflags", "+faststart", str(output)])
    _run(command, purpose="solid VACE segment creation")
    return probe_video(output)


def concat_videos(inputs: Iterable[Path], output: Path, *, preserve_alpha: bool = False) -> VideoProbe:
    """Concatenate already-normalized segments and retain exact ordering."""

    paths = list(inputs)
    if not paths:
        raise ValueError("at least one video is required for concatenation")
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VaceVideoError("ffmpeg is required to concatenate VACE inputs")
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_file = output.with_suffix(".concat.txt")
    lines = []
    for path in paths:
        escaped = path.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    concat_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        if preserve_alpha:
            encoder = [
                "-c:v",
                "prores_ks",
                "-profile:v",
                "4",
                "-pix_fmt",
                "yuva444p10le",
                "-vendor",
                "apl0",
            ]
        else:
            encoder = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        command = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-an",
            *encoder,
            str(output),
        ]
        _run(command, purpose="VACE segment concatenation")
    finally:
        concat_file.unlink(missing_ok=True)
    return probe_video(output)


def extract_frame_span(
    source: Path,
    output: Path,
    *,
    start_frame: int,
    frame_count_value: int,
    fps: int,
    preserve_alpha: bool = False,
) -> VideoProbe:
    """Extract an exact frame span using frame indexing rather than timestamps."""

    if start_frame < 0 or frame_count_value < 1:
        raise ValueError("invalid VACE frame span")
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VaceVideoError("ffmpeg is required to extract VACE frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    end_frame = start_frame + frame_count_value - 1
    graph = f"select=between(n\\,{start_frame}\\,{end_frame}),setpts=N/FRAME_RATE/TB,fps={fps}"
    if preserve_alpha:
        graph += ",format=rgba"
        encoder = [
            "-c:v",
            "prores_ks",
            "-profile:v",
            "4",
            "-pix_fmt",
            "yuva444p10le",
            "-vendor",
            "apl0",
        ]
    else:
        encoder = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        graph,
        "-frames:v",
        str(frame_count_value),
        *encoder,
        str(output),
    ]
    _run(command, purpose="VACE generated-gap extraction")
    return probe_video(output)


@dataclass(frozen=True)
class PreparedVaceInput:
    source_video: Path
    source_mask: Path
    tail_video: Path
    head_video: Path
    tail_frames: int
    gap_frames: int
    head_frames: int
    total_frames: int


def _window_frames(seconds: float, fps: int) -> int:
    return max(1, int(round(seconds * fps)))


def _four_n_plus_one(value: int) -> int:
    """Return the smallest frame count >= value that is valid for Wan."""

    return value + ((1 - value) % 4)


def prepare_firstlastclip(
    tail_source: Path,
    head_source: Path,
    output_dir: Path,
    *,
    width: int,
    height: int,
    model_fps: int,
    gap_frames: int,
    context_before_seconds: float,
    context_after_seconds: float,
    max_window_frames: int,
    background: str,
) -> PreparedVaceInput:
    """Build VACE's preserved-tail / generated-gap / preserved-head input."""

    if gap_frames < 1:
        raise ValueError("VACE generated gap must contain at least one frame")
    output_dir.mkdir(parents=True, exist_ok=True)
    tail_probe = probe_video(tail_source)
    head_probe = probe_video(head_source)
    tail_duration = min(tail_probe.duration_seconds, context_before_seconds)
    head_duration = min(head_probe.duration_seconds, context_after_seconds)
    if tail_duration <= 0 or head_duration <= 0:
        raise VaceVideoError("VACE context clips must contain a positive duration")
    tail_count = _window_frames(tail_duration, model_fps)
    head_count = _window_frames(head_duration, model_fps)
    tail_raw = output_dir / "tail-context-raw.mp4"
    head_raw = output_dir / "head-context-raw.mp4"
    extract_time_range(
        tail_source,
        tail_raw,
        start_seconds=max(0.0, tail_probe.duration_seconds - tail_duration),
        end_seconds=tail_probe.duration_seconds,
        fps=model_fps,
    )
    extract_time_range(
        head_source,
        head_raw,
        start_seconds=0.0,
        end_seconds=head_duration,
        fps=model_fps,
    )
    tail_video = output_dir / "tail-context.mp4"
    head_video = output_dir / "head-context.mp4"
    normalize_canvas(tail_raw, tail_video, width=width, height=height, fps=model_fps, background=background)
    normalize_canvas(head_raw, head_video, width=width, height=height, fps=model_fps, background=background)
    tail_count = frame_count(probe_video(tail_video), model_fps)
    head_count = frame_count(probe_video(head_video), model_fps)
    total = _four_n_plus_one(tail_count + gap_frames + head_count)
    gap_frames += total - (tail_count + gap_frames + head_count)
    if total > max_window_frames:
        raise VaceVideoError(
            "VACE firstlastclip window exceeds the configured model frame budget: "
            f"tail={tail_count} gap={gap_frames} head={head_count} total={total} max={max_window_frames}"
        )
    gap_video = output_dir / "generated-gap-placeholder.mp4"
    gap_mask = output_dir / "generated-gap-mask.mp4"
    tail_mask = output_dir / "tail-context-mask.mp4"
    head_mask = output_dir / "head-context-mask.mp4"
    solid_video(gap_video, width=width, height=height, fps=model_fps, frames=gap_frames, color="0x7f7f7f")
    solid_video(gap_mask, width=width, height=height, fps=model_fps, frames=gap_frames, color="white", grayscale=True)
    solid_video(tail_mask, width=width, height=height, fps=model_fps, frames=tail_count, color="black", grayscale=True)
    solid_video(head_mask, width=width, height=height, fps=model_fps, frames=head_count, color="black", grayscale=True)
    source_video = output_dir / "firstlastclip-source.mp4"
    source_mask = output_dir / "firstlastclip-mask.mp4"
    concat_videos([tail_video, gap_video, head_video], source_video)
    concat_videos([tail_mask, gap_mask, head_mask], source_mask)
    source_count = frame_count(probe_video(source_video), model_fps)
    mask_count = frame_count(probe_video(source_mask), model_fps)
    if source_count != mask_count or source_count != total:
        raise VaceVideoError(
            "VACE source and mask frame counts do not match: "
            f"source={source_count} mask={mask_count} expected={total}"
        )
    return PreparedVaceInput(
        source_video=source_video,
        source_mask=source_mask,
        tail_video=tail_video,
        head_video=head_video,
        tail_frames=tail_count,
        gap_frames=gap_frames,
        head_frames=head_count,
        total_frames=total,
    )


def extract_generated_gap(
    model_output: Path,
    output: Path,
    *,
    prepared: PreparedVaceInput,
    output_width: int,
    output_height: int,
    output_fps: int,
    model_fps: int,
    background: str,
) -> VideoProbe:
    """Extract only the generated middle from a VACE full-window result."""

    result_probe = probe_video(model_output)
    # The runtime output is still on the VACE model cadence. Convert to the
    # requested output cadence only after the generated middle is owned.
    actual = frame_count(result_probe, model_fps)
    if actual == prepared.gap_frames:
        return normalize_canvas(
            model_output,
            output,
            width=output_width,
            height=output_height,
            fps=output_fps,
            background=background,
        )
    if actual != prepared.total_frames:
        raise VaceVideoError(
            "VACE runtime returned an unexpected frame count; refusing to guess at the generated gap: "
            f"actual={actual} expectedGap={prepared.gap_frames} expectedWindow={prepared.total_frames}"
        )
    middle = output.with_name(f"{output.stem}-full-window{output.suffix}")
    extract_frame_span(
        model_output,
        middle,
        start_frame=prepared.tail_frames,
        frame_count_value=prepared.gap_frames,
        fps=model_fps,
    )
    return normalize_canvas(
        middle,
        output,
        width=output_width,
        height=output_height,
        fps=output_fps,
        background=background,
    )
