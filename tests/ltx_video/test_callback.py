from autotransition.ltx_video.callback import _artifact_role, _uploaded_artifact, request_from_payload
from autotransition.ltx_video.config import LtxVideoConfig


def test_ltx_callback_normalizes_worker_artifact_roles() -> None:
    assert _artifact_role("output.mp4") == "preview"
    assert _artifact_role("audio.wav") == "audio"
    assert _artifact_role("events.jsonl") == "metadata"
    assert _uploaded_artifact({"name": "output.mp4", "role": "primary"}, "artifact-1")["role"] == "preview"


def test_launcher_payload_preserves_aspect_conditioning_and_audio_mode() -> None:
    request = request_from_payload(
        {
            "job_id": "vast-job-1",
            "parameters": {
                "prompt": "A dancer performs a sharp turn",
                "aspect_ratio": "9:16",
                "duration_seconds": 3,
                "frame_rate": 24,
                "conditioning_images": [
                    {
                        "sourceUrl": "https://cdn.example/character.png?signature=redacted",
                        "frameIndex": 0,
                        "strength": 0.9,
                        "mode": "replace",
                    },
                    {
                        "source_url": "https://cdn.example/character-side.png",
                        "frame_index": 48,
                        "strength": 0.6,
                        "mode": "guide",
                    },
                ],
                "audio_mode": "off",
                "output_format": "webm",
            },
        }
    )

    assert request.aspect_ratio == "9:16"
    assert request.resolve_frames(LtxVideoConfig()) == 73
    assert [(item.frame_index, item.mode) for item in request.conditioning_images] == [(0, "replace"), (48, "guide")]
    assert request.output_format == "webm"


def test_launcher_payload_parses_temporal_prefix() -> None:
    request = request_from_payload(
        {
            "job_id": "vast-job-prefix",
            "parameters": {
                "prompt": "continue the camera movement",
                "num_frames": 73,
                "temporal_prefix": {
                    "source_url": "https://cdn.example/parent.mp4",
                    "start_frame": 40,
                    "frame_count": 25,
                    "source_frame_rate": 24,
                    "strength": 0.95,
                    "output_includes_prefix": True,
                },
            },
        }
    )

    assert request.temporal_prefix is not None
    assert request.temporal_prefix.start_frame == 40
    assert request.temporal_prefix.resolved_frame_count(LtxVideoConfig()) == 25
