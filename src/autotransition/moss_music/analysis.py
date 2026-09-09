from __future__ import annotations

from typing import Any

from .audio import AudioBuffer
from .contracts import MossMusicRequest


def build_analysis(
    request: MossMusicRequest,
    audio: AudioBuffer,
    timeline: dict[str, Any] | None,
    semantic: dict[str, Any] | None,
    runtime_metadata: dict[str, Any],
) -> dict[str, Any]:
    semantic = semantic or {}
    return {
        "schema_version": 1,
        "runtime": "moss-music",
        "analysis_profile": request.analysis_profile,
        "source": {
            "filename": request.audio.filename,
            "duration_seconds": round(audio.duration_seconds, 6),
            "sample_rate": audio.sample_rate,
            "channels": 1,
        },
        "timebase": {
            "unit": "seconds",
            "resolution_ms": request.event_resolution_ms,
            "dense_feature_cadence_hz": round(1000 / request.event_resolution_ms, 6),
        },
        "dense_features": (timeline or {}).get("cells", []),
        "dense_feature_metadata": {
            key: value for key, value in (timeline or {}).items() if key != "cells"
        },
        "semantic": semantic,
        "tempo_bpm": semantic.get("tempo_bpm"),
        "key": semantic.get("key"),
        "sections": semantic.get("sections", []),
        "beats": semantic.get("beats", []),
        "chords": semantic.get("chords", []),
        "lyrics": semantic.get("lyrics", []),
        "instruments": semantic.get("instruments", []),
        "voices": semantic.get("voices", []),
        "visual_cues": semantic.get("visual_cues", []),
        "events": semantic.get("events", []),
        "warnings": semantic.get("warnings", []),
        "provenance": {
            "runtime": "moss-music",
            "runtime_metadata": runtime_metadata,
            "measured_timeline": bool(timeline),
            "semantic_model": bool(semantic),
        },
    }
