from pathlib import Path

from autotransition.qwen_image.config import QwenImageConfig


def test_diffusers_preflight_uses_gguf_transformer_and_cuda_contract(tmp_path: Path) -> None:
    transformer = tmp_path / "qwen-image-2512-Q8_0.gguf"
    transformer.write_bytes(b"transformer")
    config = QwenImageConfig(
        runtime_backend="diffusers",
        diffusers_transformer_file=transformer,
        gpu_required=False,
    )

    report = config.preflight()

    assert report["ready"] is True
    assert report["runtime"] == "diffusers"
    assert report["transformerFormat"] == "GGUF"
    assert report["transformerFile"] == str(transformer)
    assert report["modelRevision"] == "Qwen/Qwen-Image-2512"


def test_diffusers_preflight_rejects_missing_transformer(tmp_path: Path) -> None:
    config = QwenImageConfig(
        runtime_backend="diffusers",
        diffusers_transformer_file=tmp_path / "missing.gguf",
        gpu_required=False,
    )

    report = config.preflight()

    assert report["ready"] is False
    assert "diffusersTransformer" in report["missing"]
