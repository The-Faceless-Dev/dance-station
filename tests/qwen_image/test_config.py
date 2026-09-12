from pathlib import Path

from autotransition.qwen_image.config import QwenImageConfig


def test_preflight_identifies_required_components(tmp_path: Path) -> None:
    binary = tmp_path / "sd-server"
    diffusion = tmp_path / "qwen-image-2512-Q8_0.gguf"
    encoder = tmp_path / "Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf"
    vae = tmp_path / "qwen_image_vae.safetensors"
    for path in (binary, diffusion, encoder, vae):
        path.write_bytes(b"component")
    report = QwenImageConfig(
        runtime_binary=binary,
        diffusion_model=diffusion,
        text_encoder=encoder,
        vae=vae,
        gpu_required=False,
    ).preflight()
    assert report["ready"] is True
    assert report["transformerQuantization"] == "Q8_0"
    assert report["textEncoderProfile"].startswith("Qwen2.5-VL")


def test_preflight_rejects_qwen3_encoder(tmp_path: Path) -> None:
    binary = tmp_path / "sd-server"
    diffusion = tmp_path / "qwen-image-2512-Q8_0.gguf"
    encoder = tmp_path / "qwen_3_4b.safetensors"
    vae = tmp_path / "qwen_image_vae.safetensors"
    for path in (binary, diffusion, encoder, vae):
        path.write_bytes(b"component")
    report = QwenImageConfig(
        runtime_binary=binary,
        diffusion_model=diffusion,
        text_encoder=encoder,
        vae=vae,
        gpu_required=False,
    ).preflight()
    assert report["ready"] is False
    assert any("abliterated" in item for item in report["diagnostics"])
