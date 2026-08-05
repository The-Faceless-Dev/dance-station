"""Convert the public Diffusers FLUX.2 Klein VAE to official runtime format.

The official FLUX.2 repository loads ``ae.safetensors`` using its compact
AutoEncoder implementation, while the public Klein model repository exposes
the same weights with Diffusers names. This is a one-time model-preparation
tool, not part of a paid worker request.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Diffusers VAE safetensors file")
    parser.add_argument("output", type=Path, help="Official FLUX2 ae.safetensors path")
    return parser.parse_args()


def _mid_key(prefix: str, key: str) -> str:
    if key.startswith(f"{prefix}.mid_block.resnets.0."):
        return key.replace(f"{prefix}.mid_block.resnets.0.", "mid.block_1.", 1)
    if key.startswith(f"{prefix}.mid_block.resnets.1."):
        return key.replace(f"{prefix}.mid_block.resnets.1.", "mid.block_2.", 1)
    if key.startswith(f"{prefix}.mid_block.attentions.0.group_norm."):
        return key.replace(
            f"{prefix}.mid_block.attentions.0.group_norm.",
            "mid.attn_1.norm.",
            1,
        )
    for source, target in (
        ("to_q", "q"),
        ("to_k", "k"),
        ("to_v", "v"),
        ("to_out.0", "proj_out"),
    ):
        marker = f"{prefix}.mid_block.attentions.0.{source}."
        if key.startswith(marker):
            return key.replace(marker, f"mid.attn_1.{target}.", 1)
    return ""


def _simple_key(prefix: str, key: str) -> str:
    for source, target in (
        ("quant_conv.", "quant_conv."),
        ("conv_in.", "conv_in."),
        ("conv_out.", "conv_out."),
        ("conv_norm_out.", "norm_out."),
    ):
        marker = f"{prefix}.{source}"
        if key.startswith(marker):
            return key.replace(marker, target, 1)
    return ""


def convert(input_path: Path, output_path: Path) -> None:
    source = load_file(str(input_path), device="cpu")
    converted = {}

    for key, value in source.items():
        target = ""
        if key.startswith("encoder.down_blocks."):
            parts = key.split(".")
            level = int(parts[2])
            if parts[3] == "resnets":
                block = int(parts[4])
                rest = ".".join(parts[5:]).replace("conv_shortcut.", "nin_shortcut.", 1)
                target = f"encoder.down.{level}.block.{block}.{rest}"
            elif parts[3] == "downsamplers":
                target = f"encoder.down.{level}.downsample." + ".".join(parts[5:])
        elif key.startswith("encoder.mid_block."):
            mapped = _mid_key("encoder", key)
            target = f"encoder.{mapped}" if mapped else ""
        elif key.startswith("encoder."):
            mapped = _simple_key("encoder", key)
            target = f"encoder.{mapped}" if mapped else ""
        elif key.startswith("decoder.up_blocks."):
            parts = key.split(".")
            diff_level = int(parts[2])
            level = 3 - diff_level
            if parts[3] == "resnets":
                block = int(parts[4])
                rest = ".".join(parts[5:]).replace("conv_shortcut.", "nin_shortcut.", 1)
                target = f"decoder.up.{level}.block.{block}.{rest}"
            elif parts[3] == "upsamplers":
                target = f"decoder.up.{level}.upsample." + ".".join(parts[5:])
        elif key.startswith("decoder.mid_block."):
            mapped = _mid_key("decoder", key)
            target = f"decoder.{mapped}" if mapped else ""
        elif key.startswith("decoder."):
            mapped = _simple_key("decoder", key)
            target = f"decoder.{mapped}" if mapped else ""
        elif key.startswith("post_quant_conv."):
            target = f"decoder.{key}"
        elif key.startswith("quant_conv."):
            target = f"encoder.{key}"
        elif key.startswith("bn."):
            target = key

        if not target:
            raise ValueError(f"No conversion rule for Diffusers key: {key}")
        if target in converted:
            raise ValueError(f"Duplicate converted key: {target}")
        if target.endswith(("attn_1.q.weight", "attn_1.k.weight", "attn_1.v.weight", "attn_1.proj_out.weight")) and value.ndim == 2:
            value = value[:, :, None, None]
        converted[target] = value

    if len(converted) != 251:
        raise ValueError(f"Expected 251 converted tensors, got {len(converted)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, str(output_path), metadata={"format": "pt"})
    print(f"converted {len(converted)} tensors to {output_path}")


if __name__ == "__main__":
    args = parse_args()
    convert(args.input, args.output)
