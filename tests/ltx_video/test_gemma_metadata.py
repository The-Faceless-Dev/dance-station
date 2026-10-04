from __future__ import annotations

import json

import torch
from safetensors.torch import safe_open, save_file

from tools.ltx_video.patch_gemma_metadata import patch_safetensors_metadata


def test_patch_safetensors_metadata_preserves_weights_and_adds_gemma_config(tmp_path) -> None:
    path = tmp_path / "encoder.safetensors"
    save_file(
        {"weight": torch.arange(16, dtype=torch.float32)},
        str(path),
        metadata={"model_type": "legacy"},
    )

    result = patch_safetensors_metadata(
        path,
        {
            "format": "pt",
            "gemma_config": json.dumps({"model_type": "gemma4_unified"}),
        },
    )

    assert result["dataShiftBytes"] > 0
    with safe_open(str(path), framework="pt") as handle:
        metadata = handle.metadata()
        assert json.loads(metadata["gemma_config"])["model_type"] == "gemma4_unified"
        assert torch.equal(handle.get_tensor("weight"), torch.arange(16, dtype=torch.float32))
