from __future__ import annotations

from pathlib import Path

import pytest

from autotransition.generative_dance.contracts import BoundaryState
from autotransition.generative_dance.video import (
    VideoProbe,
    calculate_boundary_sync_transform,
    calculate_normalization_transform,
    make_transparent_preview,
)


def test_normalization_transform_preserves_full_frame_and_targets_anchor() -> None:
    probe = VideoProbe(
        path=Path("driver.mp4"),
        width=1920,
        height=1080,
        fps=30,
        duration_seconds=4,
        frame_count=120,
        pixel_format="yuv420p",
    )

    result = calculate_normalization_transform(
        probe,
        width=640,
        height=800,
        anchor=(0.5, 0.58),
        subject_bounds=(0.2, 0.1, 0.8, 0.95),
        subject_margin=0.12,
    )

    assert result["scaledWidth"] <= 640
    assert result["scaledHeight"] <= 800
    assert result["padX"] >= 0
    assert result["padY"] >= 0
    assert result["subjectBounds"] == [0.2, 0.1, 0.8, 0.95]


def test_normalization_transform_rejects_invalid_subject_bounds() -> None:
    probe = VideoProbe(Path("driver.mp4"), 640, 480, 24, 1, 24, "yuv420p")
    with pytest.raises(ValueError, match="subject bounds"):
        calculate_normalization_transform(
            probe,
            width=640,
            height=800,
            anchor=(0.5, 0.5),
            subject_bounds=(0.9, 0.1, 0.2, 0.8),
            subject_margin=0.12,
        )


def test_transparent_preview_is_exposed_as_a_video_boundary() -> None:
    assert callable(make_transparent_preview)


def test_boundary_sync_uses_clip_edges_without_rewriting_internal_motion() -> None:
    source = BoundaryState(
        time_seconds=0.0,
        anchor=(0.42, 0.60),
        subject_bounds=(0.2, 0.1, 0.8, 0.95),
        foot_floor=0.95,
        confidence=1.0,
    )
    target = BoundaryState(
        time_seconds=0.0,
        anchor=(0.50, 0.58),
        subject_bounds=(0.2, 0.1, 0.8, 0.95),
        foot_floor=0.94,
        confidence=1.0,
    )

    transform, report = calculate_boundary_sync_transform(source, target)

    assert transform.translate_x == pytest.approx(0.08)
    assert transform.translate_y == pytest.approx(-0.015)
    assert transform.scale == 1.0
    assert report["mode"] == "clip-boundary"
    assert report["sourceFootFloor"] == pytest.approx(0.95)
    assert report["targetFootFloor"] == pytest.approx(0.94)
