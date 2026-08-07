from argparse import Namespace
from pathlib import Path

import pytest

from tools.avatar.trellis2_generate import require_local_model, resolve_setting


def test_quality_modes_map_to_rigging_friendly_defaults():
    preview = resolve_setting(Namespace(quality="preview", pipeline_type=None, decimation_target=0, texture_size=0, max_num_tokens=1))
    runtime = resolve_setting(Namespace(quality="runtime", pipeline_type=None, decimation_target=0, texture_size=0, max_num_tokens=1))
    quality = resolve_setting(Namespace(quality="quality", pipeline_type=None, decimation_target=0, texture_size=0, max_num_tokens=1))

    assert preview == ("512", 50_000, 2048)
    assert runtime == ("1024_cascade", 150_000, 4096)
    assert quality == ("1536_cascade", 250_000, 4096)


def test_explicit_trellis_settings_override_quality_defaults():
    settings = resolve_setting(
        Namespace(
            quality="runtime",
            pipeline_type="512",
            decimation_target=75_000,
            texture_size=1024,
            max_num_tokens=12_345,
        )
    )

    assert settings == ("512", 75_000, 1024)


def test_missing_local_checkpoint_is_rejected_without_download(tmp_path: Path):
    with pytest.raises(RuntimeError, match="not mounted locally"):
        require_local_model("microsoft/TRELLIS.2-4B", allow_download=False)


def test_local_checkpoint_requires_pipeline_config(tmp_path: Path):
    with pytest.raises(RuntimeError, match="pipeline.json"):
        require_local_model(str(tmp_path), allow_download=False)

    (tmp_path / "pipeline.json").write_text("{}", encoding="utf-8")
    require_local_model(str(tmp_path), allow_download=False)


def test_local_checkpoint_requires_conditioning_models(tmp_path: Path):
    model_path = tmp_path / "trellis2"
    model_path.mkdir()
    (model_path / "pipeline.json").write_text(
        '{"args":{"image_cond_model":{"args":{"model_name":"/models/dinov3"}},'
        '"rembg_model":{"args":{"model_name":"/models/birefnet"}},'
        '"models":{"decoder":"ckpts/decoder"}}}',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="pipeline dependencies"):
        require_local_model(str(model_path), allow_download=False)

    (model_path / "ckpts").mkdir()
    (tmp_path / "dinov3").mkdir()
    (tmp_path / "dinov3" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "dinov3" / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "birefnet").mkdir()
    (tmp_path / "birefnet" / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "birefnet" / "model.safetensors").write_bytes(b"weights")
    (model_path / "ckpts" / "decoder.json").write_text("{}", encoding="utf-8")
    (model_path / "ckpts" / "decoder.safetensors").write_bytes(b"weights")
    require_local_model(str(model_path), allow_download=False)
