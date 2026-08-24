"""Small, deterministic FFmpeg boundary for motion-driver video work."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from autotransition.audio.ffmpeg import resolve_ffmpeg
from autotransition.generative_dance.contracts import BoundaryState, CanvasContract, PlacementTransform


class VideoToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoProbe:
    path: Path
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int | None
    pixel_format: str | None
    has_alpha: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "durationSeconds": self.duration_seconds,
            "frameCount": self.frame_count,
            "pixelFormat": self.pixel_format,
            "hasAlpha": self.has_alpha,
        }


def resolve_ffprobe() -> str | None:
    direct = shutil.which("ffprobe")
    if direct:
        return direct
    ffmpeg = resolve_ffmpeg()
    if ffmpeg:
        sibling = Path(ffmpeg).with_name("ffprobe" + (".exe" if Path(ffmpeg).suffix.lower() == ".exe" else ""))
        if sibling.is_file():
            return str(sibling)
    return None


def probe_video(path: Path) -> VideoProbe:
    if not path.is_file():
        raise VideoToolError(f"video was not found: {path}")
    ffprobe = resolve_ffprobe()
    if not ffprobe:
        raise VideoToolError("ffprobe is required to inspect dance videos")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,pix_fmt,duration:stream_tags=alpha_mode",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise VideoToolError(result.stderr.strip() or f"ffprobe failed for {path}")
    try:
        payload = json.loads(result.stdout)
        stream = (payload.get("streams") or [])[0]
        frame_rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        fps = float(Fraction(str(frame_rate))) if frame_rate not in {None, "0/0"} else 0.0
        duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0.0)
        frames_value = stream.get("nb_frames")
        frame_count = int(frames_value) if frames_value and str(frames_value).isdigit() else None
        pixel_format = stream.get("pix_fmt")
        tags = stream.get("tags") or {}
        alpha_mode = str(tags.get("alpha_mode") or tags.get("ALPHA_MODE") or "")
        has_alpha = (
            str(pixel_format or "").lower().startswith(("yuva", "rgba", "argb", "abgr", "bgra", "gbrap"))
            or alpha_mode == "1"
        )
        return VideoProbe(
            path=path,
            width=int(stream.get("width") or 0),
            height=int(stream.get("height") or 0),
            fps=fps,
            duration_seconds=duration,
            frame_count=frame_count,
            pixel_format=pixel_format,
            has_alpha=has_alpha,
        )
    except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise VideoToolError(f"ffprobe returned invalid video metadata for {path}: {exc}") from exc


def _run_ffmpeg(command: list[str], *, purpose: str) -> None:
    result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        detail = result.stderr.strip()[-4000:]
        raise VideoToolError(f"FFmpeg {purpose} failed: {detail}")


def calculate_normalization_transform(
    probe: VideoProbe,
    *,
    width: int,
    height: int,
    anchor: tuple[float, float],
    subject_bounds: tuple[float, float, float, float],
    subject_margin: float,
) -> dict[str, object]:
    """Calculate a no-crop scale/pad transform for a normalized driver."""

    x0, y0, x1, y1 = subject_bounds
    if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
        raise ValueError("subject bounds must be normalized and ordered")
    if not 0 <= anchor[0] <= 1 or not 0 <= anchor[1] <= 1:
        raise ValueError("subject anchor must be normalized between 0 and 1")
    subject_width = max(0.001, x1 - x0)
    subject_height = max(0.001, y1 - y0)
    desired_scale = min(
        (width * (1 - 2 * subject_margin)) / (subject_width * probe.width),
        (height * (1 - 2 * subject_margin)) / (subject_height * probe.height),
    )
    full_frame_scale = min(width / probe.width, height / probe.height)
    scale = min(desired_scale, full_frame_scale)
    scaled_width = max(1, int(round(probe.width * scale)))
    scaled_height = max(1, int(round(probe.height * scale)))
    subject_center_x = ((x0 + x1) / 2) * scaled_width
    subject_center_y = ((y0 + y1) / 2) * scaled_height
    desired_center_x = anchor[0] * width
    desired_center_y = anchor[1] * height
    pad_x = int(round(desired_center_x - subject_center_x))
    pad_y = int(round(desired_center_y - subject_center_y))
    pad_x = max(0, min(width - scaled_width, pad_x))
    pad_y = max(0, min(height - scaled_height, pad_y))
    return {
        "sourceWidth": probe.width,
        "sourceHeight": probe.height,
        "scaledWidth": scaled_width,
        "scaledHeight": scaled_height,
        "scale": scale,
        "padX": pad_x,
        "padY": pad_y,
        "anchor": list(anchor),
        "subjectBounds": list(subject_bounds),
        "subjectMargin": subject_margin,
        "anchorErrorPixels": [
            round((pad_x + subject_center_x) - desired_center_x, 3),
            round((pad_y + subject_center_y) - desired_center_y, 3),
        ],
    }


def normalize_video(
    source: Path,
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    pixel_aspect_ratio: float = 1.0,
    anchor: tuple[float, float] = (0.5, 0.5),
    subject_bounds: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
    subject_margin: float = 0.12,
) -> VideoProbe:
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VideoToolError("ffmpeg is required to normalize dance videos")
    if not source.is_file():
        raise VideoToolError(f"video was not found: {source}")
    if width < 64 or height < 64 or fps < 1 or pixel_aspect_ratio <= 0:
        raise ValueError("invalid normalized video dimensions or frame rate")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_probe = probe_video(source)
    transform = calculate_normalization_transform(
        source_probe,
        width=width,
        height=height,
        anchor=anchor,
        subject_bounds=subject_bounds,
        subject_margin=subject_margin,
    )
    filter_graph = (
        f"scale={transform['scaledWidth']}:{transform['scaledHeight']}:flags=lanczos,"
        f"pad={width}:{height}:{transform['padX']}:{transform['padY']}:color=black,setsar={_format_filter_number(pixel_aspect_ratio)},fps={fps}"
    )
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        filter_graph,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    _run_ffmpeg(command, purpose="video normalization")
    return probe_video(output)


def extract_video_range(
    source: Path,
    output: Path,
    *,
    start_seconds: float,
    end_seconds: float,
    output_fps: int | None = None,
) -> VideoProbe:
    """Extract one source range for a sequence segment without carrying audio."""

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VideoToolError("ffmpeg is required to extract dance sequence ranges")
    if not source.is_file():
        raise VideoToolError(f"source video was not found: {source}")
    if start_seconds < 0 or end_seconds <= start_seconds:
        raise ValueError("video range must have a positive duration")
    output.parent.mkdir(parents=True, exist_ok=True)
    filters = []
    if output_fps is not None:
        if output_fps < 1 or output_fps > 120:
            raise ValueError("video range output FPS must be between 1 and 120")
        filters.append(f"fps={output_fps}")
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-ss",
        _format_filter_number(start_seconds),
        "-t",
        _format_filter_number(end_seconds - start_seconds),
        "-map",
        "0:v:0",
        "-an",
    ]
    if filters:
        command.extend(["-vf", ",".join(filters)])
    command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)])
    _run_ffmpeg(command, purpose="dance sequence range extraction")
    return probe_video(output)


def normalize_image(source: Path, output: Path, *, width: int, height: int) -> None:
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VideoToolError("ffmpeg is required to normalize reference images")
    if not source.is_file():
        raise VideoToolError(f"reference image was not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black@0,format=rgba"
    )
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-frames:v",
        "1",
        "-vf",
        filter_graph,
        "-c:v",
        "png",
        str(output),
    ]
    _run_ffmpeg(command, purpose="image normalization")
    if not output.is_file() or output.stat().st_size == 0:
        raise VideoToolError("image normalization completed without an output image")


def _format_filter_number(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".") or "0"


def _alpha_encoder_args(codec: str, crf: int) -> list[str]:
    if codec == "prores_ks":
        return ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le", "-vendor", "apl0"]
    return [
        "-c:v",
        codec,
        "-pix_fmt",
        "yuva420p",
        "-auto-alt-ref",
        "0",
        "-crf",
        str(crf),
        "-row-mt",
        "1",
        "-metadata:s:v:0",
        "alpha_mode=1",
    ]


def make_blank_video(
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    duration_seconds: float,
    transparent: bool = False,
    codec: str = "prores_ks",
    crf: int = 30,
) -> VideoProbe:
    """Create a timeline filler without inventing motion or alpha content."""

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VideoToolError("ffmpeg is required to create timeline fillers")
    if width < 1 or height < 1 or fps < 1 or duration_seconds <= 0:
        raise ValueError("invalid blank video dimensions, frame rate, or duration")
    output.parent.mkdir(parents=True, exist_ok=True)
    color = "black@0.0" if transparent else "black"
    command = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={width}x{height}:r={fps}",
        "-t",
        _format_filter_number(duration_seconds),
        "-an",
    ]
    if transparent:
        command.extend(["-vf", "format=rgba", *_alpha_encoder_args(codec, crf)])
    else:
        command.extend(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    command.append(str(output))
    _run_ffmpeg(command, purpose="timeline filler creation")
    return probe_video(output)


def extract_last_frame(source: Path, output: Path) -> Path:
    """Persist the final RGB frame used to continue the next Wan segment."""

    import cv2

    if not source.is_file():
        raise VideoToolError(f"rendered video was not found: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise VideoToolError(f"could not open rendered video: {source}")
    last_frame = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            last_frame = frame
    finally:
        capture.release()
    if last_frame is None:
        raise VideoToolError(f"rendered video contained no frames: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), last_frame):
        raise VideoToolError(f"could not write continuation frame: {output}")
    return output


def transform_boundary(boundary: BoundaryState, transform: PlacementTransform, *, canvas: CanvasContract) -> BoundaryState:
    """Apply a clip-level authored transform to boundary metadata."""

    transform.validate()
    x = 0.5 + (boundary.anchor[0] - 0.5) * transform.scale + transform.translate_x
    y = 0.5 + (boundary.anchor[1] - 0.5) * transform.scale + transform.translate_y
    foot_floor = boundary.foot_floor
    if foot_floor is not None:
        foot_floor = 0.5 + (foot_floor - 0.5) * transform.scale + transform.translate_y
    return BoundaryState(
        time_seconds=boundary.time_seconds,
        anchor=(x, y),
        subject_bounds=boundary.subject_bounds,
        foot_floor=foot_floor,
        pose_signature=boundary.pose_signature,
        confidence=boundary.confidence,
        source=boundary.source,
    )


def calculate_boundary_sync_transform(
    source: BoundaryState,
    target: BoundaryState,
) -> tuple[PlacementTransform, dict[str, object]]:
    """Calculate one transform that aligns a clip boundary without changing its motion."""

    source_foot = source.foot_floor if source.foot_floor is not None else source.anchor[1]
    target_foot = target.foot_floor if target.foot_floor is not None else target.anchor[1]
    hip_delta_y = target.anchor[1] - source.anchor[1]
    foot_delta_y = target_foot - source_foot
    # The source capture is normalized before rendering. If the two vertical
    # anchors disagree slightly, split the correction rather than introducing
    # a scale or continuously warping the clip.
    translate_x = target.anchor[0] - source.anchor[0]
    translate_y = (hip_delta_y + foot_delta_y) / 2.0
    transform = PlacementTransform(translate_x=translate_x, translate_y=translate_y)
    transform.validate()
    report = {
        "mode": "clip-boundary",
        "sourceAnchor": list(source.anchor),
        "targetAnchor": list(target.anchor),
        "sourceFootFloor": source_foot,
        "targetFootFloor": target_foot,
        "hipDeltaY": hip_delta_y,
        "footDeltaY": foot_delta_y,
        "transform": transform.to_dict(),
        "confidence": min(source.confidence, target.confidence),
    }
    return transform, report


def apply_placement(
    source: Path,
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    pixel_aspect_ratio: float = 1.0,
    placement: PlacementTransform,
    codec: str = "libvpx-vp9",
    crf: int = 30,
) -> VideoProbe:
    """Apply a saved transform to an alpha video on the canonical canvas.

    Translation is normalized to the canvas dimensions and is applied around
    the canvas center. The source is never re-centered from a newly detected
    subject box, which keeps motion authored in the driver intact.
    """

    placement.validate()
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VideoToolError("ffmpeg is required to apply dance placement")
    if width < 64 or height < 64 or fps < 1 or pixel_aspect_ratio <= 0:
        raise ValueError("invalid placement canvas dimensions or frame rate")
    if not source.is_file():
        raise VideoToolError(f"placement source was not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_probe = probe_video(source)
    if source_probe.width < 1 or source_probe.height < 1:
        raise VideoToolError(f"placement source has invalid dimensions: {source_probe.to_dict()}")
    # Wan may render at a model-native resolution (for example 384x480) while
    # the dance contract uses a larger canonical canvas (for example 640x800).
    # Fit that frame to the canvas before applying the authored relative scale.
    canvas_fit_scale = min(width / source_probe.width, height / source_probe.height)
    scale = _format_filter_number(canvas_fit_scale * placement.scale)
    translate_x = _format_filter_number(placement.translate_x * width)
    translate_y = _format_filter_number(placement.translate_y * height)
    rotation = _format_filter_number(placement.rotation_degrees * 3.141592653589793 / 180)
    pixel_aspect = _format_filter_number(pixel_aspect_ratio)
    filters = [
        # Lanczos rings across transparent gaps when upscaling RGBA video,
        # pulling foreground color/alpha into pixels that should remain clear.
        # Bilinear keeps the edge smooth without producing those halos.
        f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2:flags=bilinear",
        f"fps={fps}",
    ]
    if abs(placement.rotation_degrees) > 0.00001:
        filters.append(f"rotate={rotation}:fillcolor=black@0")
    foreground_filter = ",".join(filters + ["format=rgba"])
    composite_filter = (
        f"[0:v]{foreground_filter}[foreground];"
        f"color=c=black@0.0:s={width}x{height}:r={fps},format=rgba[background];"
        f"[background][foreground]overlay=x=(W-w)/2+{translate_x}:y=(H-h)/2+{translate_y}:shortest=1:format=auto,"
        f"setsar={pixel_aspect},format=rgba[output]"
    )
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-an",
        "-filter_complex",
        composite_filter,
        "-map",
        "[output]",
        *_alpha_encoder_args(codec, crf),
        str(output),
    ]
    _run_ffmpeg(command, purpose="transparent placement")
    result = probe_video(output)
    if result.width != width or result.height != height or result.fps <= 0:
        raise VideoToolError(f"transparent placement produced unexpected output metadata: {result.to_dict()}")
    return result


def encode_transparent_video(
    source: Path,
    output: Path,
    *,
    codec: str = "libvpx-vp9",
    crf: int = 30,
) -> VideoProbe:
    """Encode a browser-deliverable alpha video from an internal alpha source."""

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VideoToolError("ffmpeg is required to encode transparent dance videos")
    if not source.is_file():
        raise VideoToolError(f"transparent source was not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y", "-i", str(source), "-an", *_alpha_encoder_args(codec, crf), str(output)]
    _run_ffmpeg(command, purpose="transparent video encoding")
    result = probe_video(output)
    if not result.has_alpha:
        raise VideoToolError(f"transparent video encoding lost alpha: {result.to_dict()}")
    return result


def make_transparent_preview(
    source: Path,
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    pixel_aspect_ratio: float = 1.0,
    background_color: str = "0x20242c",
) -> VideoProbe:
    """Render alpha video over a normal MP4 background for reliable inspection.

    Browsers and desktop players do not consistently decode VP9's alpha plane.
    The transparent WebM remains the deliverable; this MP4 is only an inspection
    artifact so a player cannot make a correctly keyed result look opaque.
    """

    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VideoToolError("ffmpeg is required to create a transparent preview")
    if not source.is_file():
        raise VideoToolError(f"transparent preview source was not found: {source}")
    if width < 64 or height < 64 or fps < 1 or pixel_aspect_ratio <= 0:
        raise ValueError("invalid transparent preview dimensions or frame rate")
    output.parent.mkdir(parents=True, exist_ok=True)
    pixel_aspect = _format_filter_number(pixel_aspect_ratio)
    filter_graph = (
        # Decode the color and alpha planes separately, then rebuild the
        # foreground before compositing. Relying on an implicit RGBA overlay
        # can expose stale RGB values from transparent VP9 pixels as flashes
        # in the inspection MP4.
        f"[0:v]scale={width}:{height}:flags=bilinear,format=rgba,split=2[foreground_rgba][alpha_source];"
        f"[alpha_source]alphaextract,format=gray[foreground_alpha];"
        f"[foreground_rgba]format=rgb24[foreground_rgb];"
        f"[foreground_rgb][foreground_alpha]alphamerge,format=rgba[foreground];"
        # Composite in RGB so the transparent-hole colors are resolved
        # against the inspection background before final YUV subsampling.
        f"[1:v]format=gbrp[background];"
        f"[background][foreground]overlay=shortest=1:format=gbrp,"
        f"setsar={pixel_aspect},format=yuv420p[preview]"
    )
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-f",
        "lavfi",
        "-i",
        f"color=c={background_color}:s={width}x{height}:r={fps}",
        "-filter_complex",
        filter_graph,
        "-map",
        "[preview]",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    _run_ffmpeg(command, purpose="transparent preview")
    result = probe_video(output)
    if result.width != width or result.height != height or result.fps <= 0:
        raise VideoToolError(f"transparent preview produced unexpected output metadata: {result.to_dict()}")
    return result


def stabilize_alpha_video(
    source: Path,
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    python_executable: str | None = None,
    threshold_px: float = 12.0,
    window_size: int = 5,
    strength: float = 0.85,
    crf: int = 30,
    timeout_seconds: float = 1800.0,
) -> VideoProbe:
    """Correct isolated alpha-subject jumps without re-centering choreography."""

    if not source.is_file():
        raise VideoToolError(f"alpha stabilization source was not found: {source}")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("alpha stabilization window must be an odd number of at least 3")
    if threshold_px < 0 or not 0 <= strength <= 1:
        raise ValueError("alpha stabilization threshold must be non-negative and strength must be between 0 and 1")
    output.parent.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[3]
    runner = repo_root / "tools" / "generative_dance" / "stabilize_alpha_video.py"
    command = [
        python_executable or sys.executable,
        str(runner),
        "--input",
        str(source),
        "--output",
        str(output),
        "--width",
        str(width),
        "--height",
        str(height),
        "--fps",
        str(fps),
        "--threshold-px",
        str(threshold_px),
        "--window-size",
        str(window_size),
        "--strength",
        str(strength),
        "--crf",
        str(crf),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout_seconds)
    if result.returncode != 0:
        raise VideoToolError(f"alpha position stabilization failed: {result.stderr.strip()[-4000:]}")
    if not output.is_file() or output.stat().st_size == 0:
        raise VideoToolError("alpha position stabilization completed without an output")
    return probe_video(output)


def stitch_transparent_videos(
    inputs: list[Path],
    output: Path,
    *,
    width: int,
    height: int,
    fps: int,
    pixel_aspect_ratio: float = 1.0,
    codec: str = "libvpx-vp9",
    crf: int = 30,
) -> VideoProbe:
    """Concatenate same-canvas alpha clips while retaining their alpha plane."""

    if not inputs:
        raise ValueError("at least one transparent video is required")
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VideoToolError("ffmpeg is required to compose transparent dance videos")
    if pixel_aspect_ratio <= 0:
        raise ValueError("pixel aspect ratio must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_file = output.with_suffix(".concat.txt")
    concat_file.write_text(
        "".join(f"file '{path.resolve().as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for path in inputs),
        encoding="utf-8",
    )
    try:
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
            "-vf",
            f"scale={width}:{height}:flags=lanczos,fps={fps},setsar={_format_filter_number(pixel_aspect_ratio)},format=rgba",
            *_alpha_encoder_args(codec, crf),
            str(output),
        ]
        _run_ffmpeg(command, purpose="transparent dance composition")
    finally:
        concat_file.unlink(missing_ok=True)
    result = probe_video(output)
    if result.width != width or result.height != height:
        raise VideoToolError(f"transparent composition produced unexpected dimensions: {result.to_dict()}")
    if not result.has_alpha:
        raise VideoToolError(f"transparent composition lost its alpha plane: {result.to_dict()}")
    return result


def stitch_videos(inputs: list[Path], output: Path, *, width: int, height: int, fps: int) -> VideoProbe:
    """Re-encode normalized clips into one deterministic composition."""

    if not inputs:
        raise ValueError("at least one rendered segment is required")
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise VideoToolError("ffmpeg is required to compose dance videos")
    output.parent.mkdir(parents=True, exist_ok=True)
    concat_file = output.with_suffix(".concat.txt")
    concat_lines = []
    for path in inputs:
        escaped_path = path.resolve().as_posix().replace("'", "'\\''")
        concat_lines.append(f"file '{escaped_path}'")
    concat_file.write_text("\n".join(concat_lines) + "\n", encoding="utf-8")
    try:
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
            "-vf",
            f"scale={width}:{height}:flags=lanczos,fps={fps},setsar=1",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        _run_ffmpeg(command, purpose="dance composition")
    finally:
        concat_file.unlink(missing_ok=True)
    return probe_video(output)
