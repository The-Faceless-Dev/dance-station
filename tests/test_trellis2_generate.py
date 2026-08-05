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
