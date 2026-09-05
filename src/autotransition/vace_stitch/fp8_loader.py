"""Scaled-FP8 loader for the official Wan2.1 VACE model.

The public ``*_fp8_scaled.safetensors`` conversion stores FP8 linear weights
and one ``scale_weight`` tensor beside each converted weight.  Diffusers'
default loader does not understand that pair, so this module installs a small
loader for the upstream VACE class without bringing a UI runtime into the
worker.
"""

from __future__ import annotations

import json
import logging
import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


LOGGER = logging.getLogger("wan-vace-fp8")


class ScaledFP8Linear(nn.Module):
    """Linear layer for the scaled-FP8 safetensors conversion."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(
                (out_features, in_features),
                dtype=torch.float8_e4m3fn,
                device=device,
            ),
            requires_grad=False,
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=torch.bfloat16, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)
        self.register_buffer(
            "scale_weight",
            torch.ones((), dtype=torch.float32, device=device),
        )
        self._fp8_fallback_logged = False

    def _fallback(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(device=input.device, dtype=input.dtype)
        weight = weight * self.scale_weight.to(device=input.device, dtype=input.dtype)
        bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
        return torch.nn.functional.linear(input, weight, bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.weight.dtype != torch.float8_e4m3fn or not input.is_cuda:
            return self._fallback(input)

        scaled_mm = getattr(torch, "_scaled_mm", None)
        if scaled_mm is None:
            return self._fallback(input)

        original_shape = input.shape
        flattened = input.reshape(-1, original_shape[-1])
        try:
            # This follows the scaled-FP8 conversion contract: activations are
            # clamped before conversion and the checkpoint supplies the weight
            # scale.  The 5090 path uses cuBLAS FP8 matmul here.
            quantized_input = flattened.clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            scale_input = torch.ones((), dtype=torch.float32, device=input.device)
            scale_weight = self.scale_weight.to(device=input.device, dtype=torch.float32).squeeze()
            bias = self.bias.to(device=input.device, dtype=input.dtype) if self.bias is not None else None
            output = scaled_mm(
                quantized_input,
                self.weight.t(),
                out_dtype=input.dtype,
                bias=bias,
                scale_a=scale_input,
                scale_b=scale_weight,
            )
            return output.reshape(*original_shape[:-1], self.out_features)
        except (RuntimeError, TypeError) as exc:
            if not self._fp8_fallback_logged:
                LOGGER.warning(
                    "scaled FP8 matmul unavailable for layer %s->%s; falling back to dequantized linear: %s",
                    self.in_features,
                    self.out_features,
                    exc,
                )
                self._fp8_fallback_logged = True
            return self._fallback(input)


def _module_for_key(root: nn.Module, key: str) -> tuple[nn.Module, str]:
    parent_name, _, leaf = key.rpartition(".")
    module = root
    if parent_name:
        for part in parent_name.split("."):
            module = getattr(module, part)
    return module, leaf


def _assign_tensor(root: nn.Module, key: str, tensor: torch.Tensor) -> None:
    module, leaf = _module_for_key(root, key)
    if leaf == "scale_weight" and isinstance(module, ScaledFP8Linear):
        module.scale_weight = tensor.float()
        return
    if leaf in module._parameters:
        module._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
        return
    if leaf in module._buffers:
        module._buffers[leaf] = tensor
        return
    raise KeyError(f"checkpoint tensor does not map to VACE module: {key}")


def _replace_linears(root: nn.Module) -> int:
    replacements = 0
    for parent_name, parent in list(root.named_modules()):
        for leaf, child in list(parent.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            replacement = ScaledFP8Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device=child.weight.device,
            )
            setattr(parent, leaf, replacement)
            replacements += 1
    return replacements


def _materialize_frequency_buffer(model: nn.Module) -> bool:
    """Materialize VACE's unregistered RoPE tensor after meta construction."""

    frequencies = getattr(model, "freqs", None)
    if frequencies is None or not bool(getattr(frequencies, "is_meta", False)):
        return False
    from wan.modules.model import rope_params

    dim = int(model.dim)
    heads = int(model.num_heads)
    head_dim = dim // heads
    with torch.device("cpu"):
        model.freqs = torch.cat(
            [
                rope_params(1024, head_dim - 4 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
                rope_params(1024, 2 * (head_dim // 6)),
            ],
            dim=1,
        )
    return True


def load_scaled_fp8_vace_model(model_class: type[nn.Module], checkpoint_dir: Path, report_path: Path | None = None) -> nn.Module:
    """Construct the official VACE model and stream its scaled-FP8 tensors in."""

    from safetensors import safe_open

    checkpoint = checkpoint_dir / "wan2.1_vace_14B_fp8_scaled.safetensors"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"scaled VACE checkpoint was not found: {checkpoint}")
    config_path = checkpoint_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"native VACE config was not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    constructor_parameters = set(inspect.signature(model_class.__init__).parameters)
    constructor_parameters.discard("self")
    config = {
        key: value
        for key, value in config.items()
        if key in constructor_parameters
    }
    with torch.device("meta"):
        model = model_class(**config)
    linear_count = _replace_linears(model)
    keys: list[str] = []
    fp8_count = 0
    scale_count = 0
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        for key in keys:
            if key == "scaled_fp8":
                continue
            tensor = handle.get_tensor(key)
            if key.endswith(".scale_weight"):
                scale_count += 1
                _assign_tensor(model, key, tensor)
            else:
                fp8_count += int(tensor.dtype in {torch.float8_e4m3fn, torch.float8_e5m2})
                _assign_tensor(model, key, tensor)

    frequency_materialized = _materialize_frequency_buffer(model)

    meta_parameters = [name for name, value in model.named_parameters() if value.is_meta]
    if meta_parameters:
        raise RuntimeError(f"scaled VACE checkpoint left parameters unloaded: {meta_parameters[:8]}")
    report: dict[str, Any] = {
        "format": "scaled-fp8",
        "checkpoint": str(checkpoint),
        "checkpointBytes": checkpoint.stat().st_size,
        "tensorCount": len(keys),
        "fp8TensorCount": fp8_count,
        "scaleWeightCount": scale_count,
        "linearLayerCount": linear_count,
        "device": str(next(model.parameters()).device),
        "torchVersion": torch.__version__,
        "cudaAvailable": bool(torch.cuda.is_available()),
        "scaledMatmulAvailable": hasattr(torch, "_scaled_mm"),
        "metaFrequencyMaterialized": frequency_materialized,
    }
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    LOGGER.info("loaded VACE scaled-FP8 checkpoint: %s", json.dumps(report, sort_keys=True))
    return model


def install_scaled_fp8_loader(
    checkpoint_dir: Path | None,
    checkpoint_file: Path | None,
    report_path: Path | None = None,
) -> None:
    """Patch only the official VACE class used by the native child process."""

    if checkpoint_dir is None or checkpoint_file is None or not checkpoint_file.is_file():
        return
    from models.wan.modules.model import VaceWanModel

    original = VaceWanModel.from_pretrained

    @classmethod
    def from_pretrained(cls: type[nn.Module], pretrained_model_name_or_path: str | Path, *args: Any, **kwargs: Any) -> nn.Module:
        path = Path(pretrained_model_name_or_path)
        if path.resolve() == checkpoint_dir.resolve() and checkpoint_file.name.endswith("_fp8_scaled.safetensors"):
            return load_scaled_fp8_vace_model(cls, path, report_path)
        return original(pretrained_model_name_or_path, *args, **kwargs)

    VaceWanModel.from_pretrained = from_pretrained
    LOGGER.info("installed scaled-FP8 VACE loader for %s", checkpoint_file)
