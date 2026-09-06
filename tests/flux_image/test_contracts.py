from pathlib import Path

import pytest

from autotransition.flux_image.config import FluxImageConfig
from autotransition.flux_image.contracts import FluxImageRequest, FluxLoRARequest


def test_flux_defaults_to_no_loras() -> None:
    request = FluxImageRequest(prompt="a studio portrait")
    request.validate(FluxImageConfig())
    assert request.loras == ()
    assert request.to_dict()["loras"] == []


def test_flux_rejects_unsupported_resolution() -> None:
    with pytest.raises(ValueError, match="resolution"):
        FluxImageRequest(prompt="test", width=512, height=512).validate(FluxImageConfig())


def test_flux_requires_safetensors_loras(tmp_path: Path) -> None:
    adapter = tmp_path / "style.bin"
    adapter.write_bytes(b"test")
    request = FluxImageRequest(
        prompt="test",
        loras=(FluxLoRARequest(source_url="https://example.test/style.bin", file_name="style.bin", path=adapter),),
    )
    with pytest.raises(ValueError, match="safetensors"):
        request.validate(FluxImageConfig())


def test_flux_defaults_to_distilled_four_steps() -> None:
    config = FluxImageConfig.from_env()
    assert FluxImageRequest(prompt="test").steps == 4
    assert config.max_steps == 4
    assert config.cpu_offload is True
