from pathlib import Path

from autotransition.qwen_image_edit.config import QwenImageEditConfig
from autotransition.qwen_image_edit.contracts import request_from_payload
from autotransition.qwen_image_edit.server import _artifact_role


def test_callback_uses_metadata_for_non_image_artifacts() -> None:
    assert _artifact_role("image.png") == "preview"
    assert _artifact_role("reference-metadata.json") == "metadata"


def test_request_accepts_multiple_reference_shapes_and_ordered_loras() -> None:
    config = QwenImageEditConfig()
    request = request_from_payload(
        {
            "prompt": "combine the subjects",
            "reference_images": [
                {"sourceUrl": "https://cdn.example/a.png", "fileName": "a.png"},
                {"sourceUrl": "https://cdn.example/b.png", "fileName": "b.png"},
            ],
            "parameters": {
                "loras": [
                    {"sourceUrl": "https://cdn.example/one.safetensors", "scale": 0.7},
                    {"sourceUrl": "https://cdn.example/two.safetensors", "scale": 1.2},
                ]
            },
        },
        config,
    )

    request.validate(config)
    assert [item.source_url for item in request.references] == [
        "https://cdn.example/a.png",
        "https://cdn.example/b.png",
    ]
    assert [item.scale for item in request.loras] == [0.7, 1.2]


def test_request_requires_reference_image() -> None:
    config = QwenImageEditConfig()
    request = request_from_payload({"prompt": "edit this"}, config)

    try:
        request.validate(config)
    except ValueError as exc:
        assert str(exc) == "at least one reference image is required"
    else:
        raise AssertionError("request without a reference should fail")


def test_local_reference_is_only_allowed_when_explicitly_enabled(tmp_path: Path) -> None:
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"placeholder")
    config = QwenImageEditConfig(allow_local_references=False)
    request = request_from_payload(
        {"prompt": "edit this", "reference_images": [{"path": str(reference), "fileName": "reference.png"}]},
        config,
    )

    try:
        request.validate(config)
    except ValueError as exc:
        assert str(exc) == "local reference paths are disabled for this worker"
    else:
        raise AssertionError("local reference should be disabled")
