"""Compatibility shims for the scaled-FP8 VACE checkpoint and LightX2V."""

from __future__ import annotations

import logging

import torch


LOGGER = logging.getLogger("wan-vace-lightx2v")


def install() -> None:
    from lightx2v.common.ops.mm.mm_weight import MMWeightQuantTemplate
    from lightx2v.models.networks.wan.model import WanModel
    from lightx2v.models.networks.wan.vace_model import WanVaceModel
    from lightx2v.models.networks.wan.weights.vace.transformer_weights import (
        WanVaceTransformerAttentionBlock,
    )
    from lightx2v.utils.registry_factory import MM_WEIGHT_REGISTER
    from lightx2v_platform.base.global_var import AI_DEVICE

    class ScaledVaceFP8Weight(MMWeightQuantTemplate):
        """GPU-resident linear for VACE's scalar ``scale_weight`` format."""

        def __init__(self, weight_name, bias_name, *args, **kwargs):
            super().__init__(weight_name, bias_name, *args, **kwargs)
            self.weight_scale_name = self.weight_name.removesuffix(".weight") + ".scale_weight"
            self.base_attrs = [(self.weight_name, "weight", False), (self.weight_scale_name, "weight_scale", False)]
            if self.bias_name is not None:
                self.base_attrs.append((self.bias_name, "bias", False))
            self.weight_need_transpose = True
            self.scale_force_fp32 = True

        def load_quantized(self, weight_dict):
            self.weight = weight_dict[self.weight_name]
            self.weight_scale = weight_dict[self.weight_scale_name].float()
            if self.bias_name is not None:
                self.bias = weight_dict.get(self.bias_name)

        def post_process(self):
            super().post_process()

        def apply(self, input_tensor):
            if not input_tensor.is_cuda or not hasattr(torch, "_scaled_mm"):
                raise RuntimeError("scaled VACE FP8 requires CUDA torch._scaled_mm; CPU fallback is disabled")
            original_shape = input_tensor.shape
            flat = input_tensor.reshape(-1, original_shape[-1])
            scale_a = flat.detach().float().abs().amax().clamp_min(1e-8) / 448.0
            quantized = (flat / scale_a).clamp(-448, 448).to(torch.float8_e4m3fn).contiguous()
            bias = self._get_actual_bias()
            if bias is not None:
                bias = bias.to(device=input_tensor.device, dtype=input_tensor.dtype)
            output = torch._scaled_mm(
                quantized,
                self.weight,
                out_dtype=input_tensor.dtype,
                bias=bias,
                scale_a=scale_a.to(device=input_tensor.device, dtype=torch.float32),
                scale_b=self.weight_scale.to(device=input_tensor.device, dtype=torch.float32).squeeze(),
            )
            if self.has_lora_branch:
                output = output + self.apply_lora(input_tensor)
            return output.reshape(*original_shape[:-1], output.shape[-1])

    # Keep the official config name while changing only the loader selected by
    # the scaled VACE checkpoint. The VACE-specific projection layers are
    # patched below because upstream constructs those with ``Default`` even
    # when the surrounding transformer uses the configured quantized loader.
    MM_WEIGHT_REGISTER["fp8-pertensor"] = ScaledVaceFP8Weight

    if not getattr(WanVaceTransformerAttentionBlock, "_faceless_scaled_fp8_patch", False):
        original_vace_block_init = WanVaceTransformerAttentionBlock.__init__

        def vace_block_init(
            self,
            base_block_idx,
            block_index,
            task,
            mm_type,
            config,
            create_cuda_buffer,
            create_cpu_buffer,
            block_prefix,
        ):
            original_vace_block_init(
                self,
                base_block_idx,
                block_index,
                task,
                mm_type,
                config,
                create_cuda_buffer,
                create_cpu_buffer,
                block_prefix,
            )
            projection_cls = MM_WEIGHT_REGISTER["fp8-pertensor"]
            if base_block_idx == 0:
                self.compute_phases[0].add_module(
                    "before_proj",
                    projection_cls(
                        f"{block_prefix}.{block_index}.before_proj.weight",
                        f"{block_prefix}.{block_index}.before_proj.bias",
                        create_cuda_buffer,
                        create_cpu_buffer,
                        self.lazy_load,
                        self.lazy_load_file,
                    ),
                )
            self.compute_phases[-1].add_module(
                "after_proj",
                projection_cls(
                    f"{block_prefix}.{block_index}.after_proj.weight",
                    f"{block_prefix}.{block_index}.after_proj.bias",
                    create_cuda_buffer,
                    create_cpu_buffer,
                    self.lazy_load,
                    self.lazy_load_file,
                ),
            )

        WanVaceTransformerAttentionBlock.__init__ = vace_block_init
        WanVaceTransformerAttentionBlock._faceless_scaled_fp8_patch = True
        LOGGER.info(
            "patched VACE before_proj/after_proj to use the scaled-FP8 GPU matmul loader"
        )
        print(
            "[VACE][LightX2V] scaled-FP8 VACE projection loader installed "
            "for before_proj/after_proj; CPU fallback disabled",
            flush=True,
        )

    # The upstream VACE subclass omits WanModel's dynamic-LoRA constructor
    # arguments. Preserve its normal initialization path and add those args.
    def vace_init(self, model_path, config, device, model_type="wan2.1", lora_path=None, lora_strength=1.0):
        WanModel.__init__(self, model_path, config, device, model_type, lora_path, lora_strength)

    WanVaceModel.__init__ = vace_init

    if AI_DEVICE != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LightX2V VACE requires CUDA; CPU execution is intentionally disabled")
