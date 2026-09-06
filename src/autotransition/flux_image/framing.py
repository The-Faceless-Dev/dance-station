"""Canonical avatar framing for Flux image-worker outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class AvatarFrameReport:
    source_width: int
    source_height: int
    output_width: int
    output_height: int
    subject_scale: float
    background_rgb: tuple[int, int, int]
    detected_subject_bounds: tuple[int, int, int, int] | None
    fallback_used: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["backgroundRgb"] = list(self.background_rgb)
        payload["detectedSubjectBounds"] = list(self.detected_subject_bounds) if self.detected_subject_bounds else None
        payload.pop("background_rgb")
        payload.pop("detected_subject_bounds")
        return payload


def _border_pixels(rgb: np.ndarray) -> np.ndarray:
    height, width, _ = rgb.shape
    edge = max(1, min(height, width) // 32)
    return np.concatenate(
        (
            rgb[:edge, :, :].reshape(-1, 3),
            rgb[-edge:, :, :].reshape(-1, 3),
            rgb[:, :edge, :].reshape(-1, 3),
            rgb[:, -edge:, :].reshape(-1, 3),
        ),
        axis=0,
    )


def _detect_subject(rgb: np.ndarray) -> tuple[tuple[int, int, int, int] | None, tuple[int, int, int]]:
    border = _border_pixels(rgb).astype(np.float32)
    background = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - background, axis=1)
    # The avatar prompt requests a solid background. Keep the threshold
    # conservative so anti-aliased subject edges remain inside the crop.
    threshold = max(24.0, float(np.percentile(border_distance, 99.0)) + 12.0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    mask = distance > threshold
    ys, xs = np.where(mask)
    if len(xs) < 32:
        return None, tuple(int(round(value)) for value in background)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    area = (x1 - x0) * (y1 - y0)
    if area < 256 or area > rgb.shape[0] * rgb.shape[1] * 0.92:
        return None, tuple(int(round(value)) for value in background)
    return (x0, y0, x1, y1), tuple(int(round(value)) for value in background)


def frame_avatar_image(
    source: Path,
    output: Path,
    *,
    width: int,
    height: int,
    subject_scale: float,
) -> AvatarFrameReport:
    """Fit a generated character into the canonical portrait avatar canvas.

    The subject is detected against the requested solid background, padded
    slightly, reduced to the configured canvas fraction, and composited over
    the sampled background color. The fallback still guarantees a full-canvas
    result when a generated background cannot be segmented reliably.
    """

    if width < 1 or height < 1 or not 0.1 <= subject_scale <= 1.0:
        raise ValueError("invalid avatar framing dimensions or subject scale")
    if not source.is_file():
        raise FileNotFoundError(source)
    image = Image.open(source).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    bounds, background = _detect_subject(rgb)
    fallback = bounds is None
    if bounds is None:
        crop = image
    else:
        x0, y0, x1, y1 = bounds
        padding = max(2, int(round(max(x1 - x0, y1 - y0) * 0.02)))
        crop = image.crop((max(0, x0 - padding), max(0, y0 - padding), min(image.width, x1 + padding), min(image.height, y1 + padding)))

    target_width = max(1, int(round(width * subject_scale)))
    target_height = max(1, int(round(height * subject_scale)))
    scale = min(target_width / crop.width, target_height / crop.height)
    resized = crop.resize(
        (max(1, int(round(crop.width * scale))), max(1, int(round(crop.height * scale)))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (width, height), background)
    left = (width - resized.width) // 2
    top = (height - resized.height) // 2
    canvas.paste(resized, (left, top))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG")
    return AvatarFrameReport(
        source_width=image.width,
        source_height=image.height,
        output_width=width,
        output_height=height,
        subject_scale=subject_scale,
        background_rgb=background,
        detected_subject_bounds=bounds,
        fallback_used=fallback,
    )
