from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from autotransition.ltx_video.config import LtxVideoConfig
from autotransition.ltx_video.contracts import LtxVideoRequest
from autotransition.ltx_video.memory import build_memory_plan


def test_aspect_presets_and_frame_grid() -> None:
    config = LtxVideoConfig()
    request = LtxVideoRequest(prompt="a dancer", aspect_ratio="9:16", duration_seconds=3.0, frame_rate=24)

    assert request.resolve_dimensions(config) == (576, 1024)
    assert request.resolve_frames(config) == 73


def test_custom_dimensions_must_match_ltx_lattice() -> None:
    config = LtxVideoConfig()
    request = LtxVideoRequest(prompt="test", width=832, height=512, num_frames=81)
    assert request.resolve_dimensions(config) == (832, 512)
    assert request.resolved_aspect_ratio(config) == "custom"
    assert request.resolve_frames(config) == 81

    with pytest.raises(ValueError, match="divisible by 64"):
        LtxVideoRequest(prompt="test", width=833, height=512, num_frames=81).resolve_dimensions(config)


def test_duration_and_frame_count_are_mutually_exclusive() -> None:
    config = LtxVideoConfig()
    with pytest.raises(ValueError, match="either num_frames or duration_seconds"):
        LtxVideoRequest(prompt="test", num_frames=25, duration_seconds=1).validate(config)


def test_generated_audio_requires_audio_vae() -> None:
    config = replace(LtxVideoConfig(), audio_vae_path=None)
    with pytest.raises(ValueError, match="generated audio requires"):
        LtxVideoRequest(prompt="test", num_frames=25, audio_mode="generated").validate(config)


def test_memory_plan_exposes_vram_budget() -> None:
    plan = build_memory_plan(
        width=576,
        height=1024,
        frames=73,
        audio=False,
        offload_mode="none",
        quantization="nvfp4-prequant",
        reserve_vram_gb=1.5,
    )
    assert plan["videoTokens"] > 0
    assert plan["estimatedPeakGb"] > plan["transformerResidencyGb"]
    assert "fitsCurrentFreeMemory" in plan
