from pathlib import Path

import pytest

from autotransition.qwen_image.config import QwenImageConfig
from autotransition.qwen_image.contracts import QwenImageRequest, QwenLoRARequest, request_from_payload


def test_qwen_defaults_and_portrait_request() -> None:
    request = QwenImageRequest(prompt="a full-body character")
    request.validate(QwenImageConfig())
    assert (request.width, request.height, request.steps, request.cfg_scale) == (1328, 1328, 20, 2.5)
    assert request.loras == ()


def test_qwen_accepts_explicit_aspect_dimensions() -> None:
    request = QwenImageRequest(prompt="character", width=960, height=1664)
    request.validate(QwenImageConfig())


def test_qwen_rejects_bad_alignment_and_area() -> None:
    with pytest.raises(ValueError, match="aligned"):
        QwenImageRequest(prompt="test", width=961).validate(QwenImageConfig())
    with pytest.raises(ValueError, match="area"):
        QwenImageRequest(prompt="test", width=2000, height=2016).validate(QwenImageConfig())


def test_request_parser_merges_envelope_aliases_and_deduplicates_loras() -> None:
    payload = {
        "runtime": "qwen-image",
        "job_id": "job-1",
        "parameters": {
            "prompt": "a character",
            "width": 960,
            "height": 1664,
            "num_inference_steps": 12,
            "cfg": 3.0,
            "loras": [{"sourceUrl": "https://cdn.test/a.safetensors", "fileName": "a.safetensors", "scale": 0.8}],
        },
        "inputs": [
            {"role": "lora", "sourceUrl": "https://cdn.test/a.safetensors", "fileName": "a.safetensors", "scale": 0.8},
        ],
    }
    request = request_from_payload(payload)
    assert request.external_job_id == "job-1"
    assert request.steps == 12
    assert request.cfg_scale == 3.0
    assert len(request.loras) == 1


def test_local_lora_requires_explicit_development_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "style.safetensors"
    path.write_bytes(b"adapter")
    request = QwenImageRequest(prompt="test", loras=(
        QwenLoRARequest(path=path),
    ))
    with pytest.raises(ValueError, match="local LoRA"):
        request.validate(QwenImageConfig())
