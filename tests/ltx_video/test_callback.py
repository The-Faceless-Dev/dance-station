from autotransition.ltx_video.callback import request_from_payload
from autotransition.ltx_video.config import LtxVideoConfig


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
