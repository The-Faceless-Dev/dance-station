"""Single-GPU Wan-Animate-2 runner owned by AutoTransition.

This runner deliberately bypasses workflow hosts and the official MPI process
launcher. It uses the official Wan-Animate-2 model implementation as the
architecture reference, the project-owned GGUF loader for the transformer, and
the official .pth companion models directly.

The default profile keeps T5 and CLIP on CPU, where they are used for their
short conditioning passes, and keeps the quantized transformer plus VAE on the
selected CUDA device. That is the memory policy needed for the local RTX 3080
profile and can be changed for a larger production GPU without changing the
request or artifact contract.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
import gc
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from wan_animate_2_runtime import load_transformer, prepare_runtime_cache_dirs


LOGGER = logging.getLogger("wan-animate-2")
_REFERENCE_ATTENTION_ACTIVE = False


def _sdpa_backend_context(torch: Any, backend_name: str | None = None) -> Any:
    """Select a stable SDPA backend for the FlashAttention compatibility path."""

    from contextlib import nullcontext

    backend_name = (backend_name or os.getenv("WAN_SDPA_BACKEND", "math")).strip().lower()
    if backend_name in {"", "auto", "default"}:
        return nullcontext()

    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "math": [SDPBackend.MATH],
        # ``manual`` selects the explicit path for reference attention while
        # retaining math SDPA for the larger masked denoising attention.
        "manual": [SDPBackend.MATH],
        # ``cpu`` keeps reference attention off the provider CUDA path. The
        # denoising path still uses CUDA SDPA; this is only for GPUs whose
        # driver rejects the short reference softmax.
        "cpu": [SDPBackend.MATH],
        # ``chunked`` uses this fused path only for flex attention; regular
        # attention is handled by _chunked_attention below.
        "chunked": [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
        "efficient": [SDPBackend.EFFICIENT_ATTENTION],
        "flash": [SDPBackend.FLASH_ATTENTION],
        "efficient+math": [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH],
    }
    try:
        return sdpa_kernel(backends[backend_name])
    except KeyError as exc:
        raise ValueError(
            f"unsupported WAN_SDPA_BACKEND={backend_name!r}; "
            "expected auto, cpu, manual, chunked, math, efficient, flash, or efficient+math"
        ) from exc


def _manual_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    mask: Any = None,
    causal: bool = False,
    softmax_scale: float | None = None,
) -> Any:
    """Compute the short reference attention without a fused CUDA kernel.

    Salad's RTX 3090 runtime can report a CUDA ``device not ready`` error from
    PyTorch SDPA even after the VAE and transformer have successfully run on
    CUDA. The reference pass is short enough that an explicit attention matrix
    is a practical fallback and avoids that driver-sensitive kernel path.
    """

    import torch

    scale = softmax_scale or (q.shape[-1] ** -0.5)
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if causal:
        causal_mask = torch.ones(
            (q.shape[-2], k.shape[-2]), device=q.device, dtype=torch.bool
        ).tril()
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
    return torch.matmul(probabilities, v)


def _sdpa_backend_name() -> str:
    return os.getenv("WAN_SDPA_BACKEND", "math").strip().lower() or "auto"


def _reference_sdpa_backend_name() -> str:
    return os.getenv(
        "WAN_REFERENCE_SDPA_BACKEND",
        os.getenv("WAN_SDPA_BACKEND", "math"),
    ).strip().lower() or "auto"


def _sampling_sigmas(sampling_steps: int, shift: float) -> np.ndarray:
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    return shift * sigma / (1 + (shift - 1) * sigma)


def _retrieve_sigmas(scheduler: Any, *, sigmas: np.ndarray, device: Any) -> Any:
    scheduler.set_timesteps(sigmas=sigmas, device=device)
    return scheduler.timesteps


def _sdpa_flash_attention(
    q: Any,
    k: Any,
    v: Any,
    q_lens: Any = None,
    k_lens: Any = None,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    q_scale: Any = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    dtype: Any = None,
) -> Any:
    """FlashAttention-compatible fallback using PyTorch SDPA."""

    import torch
    import torch.nn.functional as functional

    output_dtype = q.dtype
    backend_name = _reference_sdpa_backend_name()
    if backend_name == "cpu":
        return _cpu_chunked_attention(
            q,
            k,
            v,
            q_lens=q_lens,
            k_lens=k_lens,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )
    if backend_name == "chunked":
        return _chunked_attention(
            q,
            k,
            v,
            q_lens=q_lens,
            k_lens=k_lens,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            dtype=dtype,
        )
    q = q.to(dtype=dtype or torch.bfloat16).transpose(1, 2)
    k = k.to(dtype=q.dtype).transpose(1, 2)
    v = v.to(dtype=q.dtype).transpose(1, 2)
    mask = None
    if k_lens is not None:
        key_positions = torch.arange(k.size(-2), device=k.device).view(1, 1, 1, -1)
        mask = key_positions < k_lens.to(device=k.device).view(-1, 1, 1, 1)
    if q_scale is not None:
        q = q * q_scale
    if backend_name == "manual":
        output = _manual_attention(
            q,
            k,
            v,
            mask=mask,
            causal=causal if mask is None else False,
            softmax_scale=softmax_scale,
        )
    else:
        with _sdpa_backend_context(torch, backend_name):
            output = functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=dropout_p,
                is_causal=causal if mask is None else False,
                scale=softmax_scale,
            )
    return output.transpose(1, 2).to(dtype=output_dtype)


def _cpu_chunked_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    q_lens: Any = None,
    k_lens: Any = None,
    softmax_scale: float | None = None,
    q_scale: Any = None,
    causal: bool = False,
    dtype: Any = None,
) -> Any:
    """Run reference attention on CPU and return the result to the model GPU.

    Some Salad RTX 3090 nodes reject the first CUDA softmax in Wan's reference
    pass even though CUDA preprocessing and model loading succeed. Reference
    attention is short and does not participate in the denoising loop, so a
    float32 CPU implementation is a reliable compatibility fallback while the
    expensive denoising attention remains on CUDA. Use PyTorch's CPU SDPA
    implementation instead of a Python loop around matmul and softmax: the
    latter was functionally correct but too slow for a real Salad job.
    """

    import torch

    output_device = q.device
    output_dtype = q.dtype
    q_cpu = q.detach().to(device="cpu", dtype=torch.float32).transpose(1, 2).contiguous()
    k_cpu = k.detach().to(device="cpu", dtype=torch.float32).transpose(1, 2).contiguous()
    v_cpu = v.detach().to(device="cpu", dtype=torch.float32).transpose(1, 2).contiguous()
    q_lengths = (
        q_lens.detach().cpu().tolist()
        if q_lens is not None
        else [q_cpu.shape[-2]] * q_cpu.shape[0]
    )
    k_lengths = (
        k_lens.detach().cpu().tolist()
        if k_lens is not None
        else [k_cpu.shape[-2]] * k_cpu.shape[0]
    )
    scale = softmax_scale or (q_cpu.shape[-1] ** -0.5)
    if q_scale is not None:
        if hasattr(q_scale, "detach"):
            q_scale = q_scale.detach().cpu()
        q_cpu = q_cpu * q_scale
    import torch.nn.functional as functional

    chunk_size = max(256, int(os.getenv("WAN_CPU_SDPA_CHUNK_SIZE", "2048")))
    output = torch.zeros_like(q_cpu)
    key_positions = torch.arange(k_cpu.shape[-2], device="cpu").view(1, 1, 1, -1)
    k_float = k_cpu
    v_float = v_cpu
    for batch_index, (q_length, k_length) in enumerate(zip(q_lengths, k_lengths)):
        q_length = min(int(q_length), q_cpu.shape[-2])
        k_length = min(int(k_length), k_cpu.shape[-2])
        valid_keys = key_positions < k_length
        for start in range(0, q_length, chunk_size):
            end = min(start + chunk_size, q_length)
            q_chunk = q_cpu[batch_index : batch_index + 1, :, start:end, :]
            key_end = k_length
            k_batch = k_float[batch_index : batch_index + 1, :, :key_end, :]
            v_batch = v_float[batch_index : batch_index + 1, :, :key_end, :]
            valid = valid_keys[..., :key_end]
            if causal:
                query_positions = torch.arange(start, end, device="cpu").view(1, 1, -1, 1)
                valid = valid & (key_positions[..., :key_end] <= query_positions)
            attention = functional.scaled_dot_product_attention(
                q_chunk,
                k_batch,
                v_batch,
                attn_mask=None if q_lens is None and k_lens is None and not causal else valid,
                dropout_p=0.0,
                is_causal=causal if q_lens is None and k_lens is None else False,
                scale=scale,
            )
            output[batch_index : batch_index + 1, :, start:end, :] = attention
    return output.transpose(1, 2).to(device=output_device, dtype=output_dtype)


def _chunked_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    q_lens: Any = None,
    k_lens: Any = None,
    softmax_scale: float | None = None,
    q_scale: Any = None,
    causal: bool = False,
    dtype: Any = None,
) -> Any:
    """Run reference attention in bounded query chunks.

    Wan's reference self-attention is the first large CUDA operation. A
    single SDPA/matmul call can leave an Ampere device in ``device not ready``
    state even when smaller local runs succeed. Chunking bounds the temporary
    score matrix and keeps the runtime independent of the full frame area.
    """

    import torch

    output_dtype = q.dtype
    compute_dtype = dtype or torch.bfloat16
    q = q.to(dtype=compute_dtype).transpose(1, 2).contiguous()
    k = k.to(dtype=compute_dtype).transpose(1, 2).contiguous()
    v = v.to(dtype=compute_dtype).transpose(1, 2).contiguous()
    q_lengths = (
        q_lens.detach().cpu().tolist()
        if q_lens is not None
        else [q.shape[-2]] * q.shape[0]
    )
    k_lengths = (
        k_lens.detach().cpu().tolist()
        if k_lens is not None
        else [k.shape[-2]] * k.shape[0]
    )
    scale = softmax_scale or (q.shape[-1] ** -0.5)
    if q_scale is not None:
        q = q * q_scale
    chunk_size = max(64, int(os.getenv("WAN_SDPA_CHUNK_SIZE", "512")))
    output = torch.zeros_like(q)
    key_positions = torch.arange(k.shape[-2], device=k.device).view(1, 1, 1, -1)
    k_float = k.float()
    v_float = v.float()
    for batch_index, (q_length, k_length) in enumerate(zip(q_lengths, k_lengths)):
        q_length = min(int(q_length), q.shape[-2])
        k_length = min(int(k_length), k.shape[-2])
        valid_keys = key_positions < k_length
        for start in range(0, q_length, chunk_size):
            end = min(start + chunk_size, q_length)
            q_chunk = q[batch_index : batch_index + 1, :, start:end, :].float()
            scores = torch.matmul(q_chunk, k_float[batch_index : batch_index + 1].transpose(-2, -1)) * scale
            valid = valid_keys
            if causal:
                query_positions = torch.arange(start, end, device=q.device).view(1, 1, -1, 1)
                valid = valid & (key_positions <= query_positions)
            scores = scores.masked_fill(~valid, float("-inf"))
            probabilities = torch.softmax(scores, dim=-1)
            output[batch_index : batch_index + 1, :, start:end, :] = torch.matmul(
                probabilities,
                v_float[batch_index : batch_index + 1],
            ).to(dtype=output_dtype)
    return output.transpose(1, 2)


_EAGER_MASK_CACHE: dict[tuple[int, int, int, int, str], Any] = {}


def _sdpa_flex_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    q_lens: Any = None,
    k_lens: Any = None,
    block_mask: Any = None,
    kernel_options: Any = None,
    dtype: Any = None,
    score_mod: Any = None,
    q_scale: Any = None,
) -> Any:
    """Eager single-GPU replacement for the official flex-attention call.

    The Wan model's denoising mask is regular enough to materialize once per
    shape. This keeps the project's runtime independent of Triton while
    preserving the official base-vs-reference attention rule and score bias.
    """

    import torch
    import torch.nn.functional as functional

    if not isinstance(block_mask, tuple) or len(block_mask) != 3:
        raise RuntimeError("Wan eager attention received an invalid runtime mask")
    q_limit, hw, q_total = block_mask
    batch, q_len, _, _ = q.shape
    kv_len = k.shape[1]
    device = q.device
    cache_key = (q_len, kv_len, q_limit, hw, str(device))
    additive_mask = _EAGER_MASK_CACHE.get(cache_key)
    if additive_mask is None:
        q_index = torch.arange(q_len, device=device).view(q_len, 1)
        kv_index = torch.arange(kv_len, device=device).view(1, kv_len)
        q_valid = q_index < q_limit
        is_base = kv_index < q_limit
        is_first_part = kv_index < q_total
        kv_frame_1 = kv_index // hw
        rel_kv_index = kv_index - q_total
        kv_frame_2 = (rel_kv_index // hw) + 1
        kv_valid_1 = kv_index < q_limit
        kv_valid_2 = rel_kv_index < (q_limit - hw)
        kv_frame = torch.where(is_first_part, kv_frame_1, kv_frame_2)
        kv_valid = torch.where(is_first_part, kv_valid_1, kv_valid_2)
        is_conditional = (q_index // hw == kv_frame) & kv_valid
        valid = q_valid & (is_base | is_conditional)
        # A boolean mask allows PyTorch SDPA to select its fused CUDA kernel.
        # The official score modifier is a small reference-key bias; eager
        # fallback prioritizes the fused path and the Triton/official path
        # retains that modifier when available.
        _EAGER_MASK_CACHE[cache_key] = valid.view(1, 1, q_len, kv_len)
        additive_mask = _EAGER_MASK_CACHE[cache_key]

    if _sdpa_backend_name() in {"cpu", "chunked", "manual"}:
        return _masked_chunked_attention(
            q,
            k,
            v,
            mask=additive_mask,
            q_limit=q_limit,
            softmax_scale=None,
            q_scale=q_scale,
            dtype=dtype,
        )

    output_dtype = q.dtype
    q = q.to(dtype=dtype or torch.bfloat16).transpose(1, 2)
    k = k.to(dtype=q.dtype).transpose(1, 2)
    v = v.to(dtype=q.dtype).transpose(1, 2)
    if q_lens is not None or k_lens is not None:
        raise RuntimeError("Wan eager attention does not support variable sequence lengths")
    if q_scale is not None:
        q = q * q_scale
    with _sdpa_backend_context(torch):
        output = functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=additive_mask,
            dropout_p=0.0,
            is_causal=False,
        )
    return output.transpose(1, 2).to(dtype=output_dtype)


def _masked_chunked_attention(
    q: Any,
    k: Any,
    v: Any,
    *,
    mask: Any,
    q_limit: int,
    softmax_scale: float | None = None,
    q_scale: Any = None,
    dtype: Any = None,
) -> Any:
    """Apply Wan's denoising mask without one unbounded CUDA SDPA call.

    The fallback is intentionally explicit rather than relying on PyTorch's
    backend chooser.  Some Salad Ampere nodes reject the full masked SDPA
    operation with ``device not ready`` even though smaller CUDA operations
    work.  Keeping the score matrix bounded also makes the activation peak
    predictable for the Q6 model.
    """

    import torch

    output_dtype = q.dtype
    compute_dtype = dtype or q.dtype
    q = q.to(dtype=compute_dtype).transpose(1, 2).contiguous()
    k = k.to(dtype=compute_dtype).transpose(1, 2).contiguous()
    v = v.to(dtype=compute_dtype).transpose(1, 2).contiguous()
    if q_scale is not None:
        q = q * q_scale

    q_length = q.shape[-2]
    q_limit = min(int(q_limit), q_length)
    chunk_size = max(
        64,
        int(
            os.getenv(
                "WAN_FLEX_ATTENTION_CHUNK_SIZE",
                os.getenv("WAN_SDPA_CHUNK_SIZE", "128"),
            )
        ),
    )
    scale = softmax_scale or (q.shape[-1] ** -0.5)
    output = torch.zeros_like(q)
    key = k.transpose(-2, -1)
    for start in range(0, q_limit, chunk_size):
        end = min(start + chunk_size, q_limit)
        scores = torch.matmul(q[..., start:end, :], key) * scale
        valid = mask[..., start:end, :].to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~valid, float("-inf"))
        probabilities = torch.softmax(scores.float(), dim=-1).to(dtype=compute_dtype)
        output[..., start:end, :] = torch.matmul(probabilities, v)
    return output.transpose(1, 2).to(dtype=output_dtype)


def _sage_flash_attention(
    q: Any,
    k: Any,
    v: Any,
    q_lens: Any = None,
    k_lens: Any = None,
    dropout_p: float = 0.0,
    softmax_scale: Any = None,
    q_scale: Any = None,
    causal: bool = False,
    window_size: Any = (-1, -1),
    deterministic: bool = False,
    dtype: Any = None,
) -> Any:
    """Use the installed SageAttention CUDA kernel for non-varlen calls.

    SageAttention exposes a fixed-length HND interface, while Wan's official
    helper accepts optional per-batch lengths. The small batch sizes here make
    trimming each sample explicit and keep padding from changing attention.
    """

    import torch
    from sageattention import sageattn

    output_dtype = q.dtype
    compute_dtype = dtype or torch.bfloat16
    q = q if q.dtype in (torch.float16, torch.bfloat16) else q.to(compute_dtype)
    k = k if k.dtype in (torch.float16, torch.bfloat16) else k.to(compute_dtype)
    v = v if v.dtype in (torch.float16, torch.bfloat16) else v.to(compute_dtype)
    q = q.to(dtype=v.dtype)
    k = k.to(dtype=v.dtype)
    if q_scale is not None:
        q = q * q_scale

    batch, q_width, q_heads, _ = q.shape
    value_width = v.shape[-1]
    q_lengths = (
        q_lens.detach().cpu().tolist()
        if q_lens is not None
        else [q_width] * batch
    )
    k_lengths = (
        k_lens.detach().cpu().tolist()
        if k_lens is not None
        else [k.shape[1]] * batch
    )
    output = torch.zeros(
        (batch, q_width, q_heads, value_width),
        device=q.device,
        dtype=output_dtype,
    )
    for batch_index, (q_length, k_length) in enumerate(zip(q_lengths, k_lengths)):
        q_item = q[batch_index, : int(q_length)].transpose(0, 1).unsqueeze(0).contiguous()
        k_item = k[batch_index, : int(k_length)].transpose(0, 1).unsqueeze(0).contiguous()
        v_item = v[batch_index, : int(k_length)].transpose(0, 1).unsqueeze(0).contiguous()
        result = sageattn(
            q_item,
            k_item,
            v_item,
            tensor_layout="HND",
            is_causal=causal,
            sm_scale=softmax_scale,
        )
        output[batch_index, : int(q_length)] = result.squeeze(0).transpose(0, 1).to(output_dtype)
    return output


def _install_flex_attention_fallback(model: Any) -> None:
    """Install eager attention when Triton is unavailable or explicitly disabled.

    The official flex-attention compiler can select a Triton kernel that exceeds
    the register/shared-memory limit on some Ampere GPUs. The eager SDPA path
    keeps the same mask semantics while letting PyTorch choose its supported
    memory-efficient CUDA kernel.
    """

    import importlib
    import types

    backend_name = os.getenv("WAN_FLEX_ATTENTION_BACKEND", "auto").strip().lower()
    force_eager = backend_name in {"eager", "sdpa", "torch"}
    if not force_eager and importlib.util.find_spec("triton") is not None:
        return
    module = importlib.import_module("wanxiang.models.wan_animate_2_model")
    module.flex_attention = _sdpa_flex_attention

    def create_eager_mask(self: Any, origin_len: int, origin_area: Any, device: Any) -> tuple[int, int, int]:
        origin_latent_f = origin_len // 4 + 1
        hw = int(np.prod(origin_area).item() // 256)
        q_len = (origin_latent_f + 1) * hw
        q_total = int(np.ceil(q_len / 128) * 128)
        return q_len, hw, q_total

    model.create_mask = types.MethodType(create_eager_mask, model)
    LOGGER.info(
        "attention_backend=sdpa_eager fallback_reason=%s",
        "forced" if force_eager else "triton_unavailable",
    )


def _install_attention_fallback() -> None:
    """Use an available CUDA attention kernel when FlashAttention is absent."""

    import importlib

    attention = importlib.import_module("wanxiang.models.attention")
    if getattr(attention, "FLASH_VER", None) is not None:
        LOGGER.info(
            "attention_backend=flash_attention version=%s required=%s",
            getattr(attention, "FLASH_VER", None),
            os.getenv("WAN_REQUIRE_FLASH_ATTENTION", "0"),
        )
        return
    if os.getenv("WAN_REQUIRE_FLASH_ATTENTION", "0").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError(
            "FlashAttention is required for this worker profile but is not installed; "
            "refusing to silently use the slow SDPA fallback"
        )
    if importlib.util.find_spec("sageattention") is not None:
        fallback = _sage_flash_attention
        backend = "sageattention_cuda"
    else:
        fallback = _sdpa_flash_attention
        backend = "scaled_dot_product_attention"
    attention.flash_attention = fallback
    for module_name in ("wanxiang.models.clip", "wanxiang.models.wan_animate_2_model"):
        module = sys.modules.get(module_name)
        if module is not None:
            module.flash_attention = fallback
    LOGGER.info(
        "attention_backend=%s fallback=flash_attention_unavailable sdpa_backend=%s",
        backend,
        _sdpa_backend_name(),
    )


def _install_reference_attention_fallback(model: Any) -> None:
    """Use SDPA for Wan's reference pass while retaining FlashAttention for denoising.

    The reference pass has a different variable-length shape from the main
    denoising path. On some RTX 3090 nodes the FlashAttention-2 kernel passes a
    small self-test but fails on that reference shape with ``device not ready``.
    The reference pass is short; PyTorch SDPA is the safer compatibility path,
    while the expensive iterative denoising attention continues to use the
    official FlashAttention implementation.
    """

    import importlib

    backend_name = os.getenv("WAN_REFERENCE_ATTENTION_BACKEND", "sdpa").strip().lower()
    if backend_name in {"", "auto", "flash", "flash_attention"}:
        return

    attention = importlib.import_module("wanxiang.models.attention")
    model_module = importlib.import_module("wanxiang.models.wan_animate_2_model")
    original_flash = getattr(attention, "flash_attention", None)
    if original_flash is None or getattr(original_flash, "_autotransition_reference_dispatch", False):
        return

    def dispatch(*args: Any, **kwargs: Any) -> Any:
        if _REFERENCE_ATTENTION_ACTIVE:
            return _sdpa_flash_attention(*args, **kwargs)
        return original_flash(*args, **kwargs)

    dispatch._autotransition_reference_dispatch = True
    attention.flash_attention = dispatch
    model_module.flash_attention = dispatch

    transformer_type = type(model)
    original_forward_ref = transformer_type.forward_ref
    if getattr(original_forward_ref, "_autotransition_reference_wrapper", False):
        return

    def forward_ref_with_safe_attention(self: Any, *args: Any, **kwargs: Any) -> Any:
        global _REFERENCE_ATTENTION_ACTIVE

        previous = _REFERENCE_ATTENTION_ACTIVE
        _REFERENCE_ATTENTION_ACTIVE = True
        try:
            return original_forward_ref(self, *args, **kwargs)
        finally:
            _REFERENCE_ATTENTION_ACTIVE = previous

    forward_ref_with_safe_attention._autotransition_reference_wrapper = True
    transformer_type.forward_ref = forward_ref_with_safe_attention
    LOGGER.info(
        "attention_backend=flash_attention reference_backend=sdpa sdpa_backend=%s",
        _reference_sdpa_backend_name(),
    )


def _validate_flash_attention(device: Any) -> None:
    """Execute a small real CUDA kernel before loading the 14B transformer."""

    required = os.getenv("WAN_REQUIRE_FLASH_ATTENTION", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not required:
        return
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("FlashAttention validation requires a CUDA device")

    import torch
    from wanxiang.models.attention import flash_attention

    started = time.perf_counter()
    q = torch.randn(1, 128, 4, 64, device=device, dtype=torch.float16)
    k = torch.randn(1, 128, 4, 64, device=device, dtype=torch.float16)
    v = torch.randn(1, 128, 4, 64, device=device, dtype=torch.float16)
    with torch.inference_mode():
        result = flash_attention(q, k, v, dtype=torch.float16)
    torch.cuda.synchronize(device)
    if result.shape != q.shape or not torch.isfinite(result).all().item():
        raise RuntimeError("FlashAttention self-test returned an invalid CUDA result")
    LOGGER.info(
        "attention_self_test=passed backend=flash_attention shape=%s seconds=%.3f device=%s",
        tuple(result.shape),
        time.perf_counter() - started,
        torch.cuda.get_device_name(device),
    )


@dataclass(frozen=True)
class RunnerPaths:
    transformer: Path
    source: Path
    t5_checkpoint: Path
    t5_tokenizer: Path
    clip_checkpoint: Path
    clip_tokenizer: Path
    vae_checkpoint: Path

    def validate(self) -> None:
        fields = (
            ("transformer", self.transformer),
            ("official source", self.source),
            ("T5 checkpoint", self.t5_checkpoint),
            ("T5 tokenizer", self.t5_tokenizer),
            ("CLIP checkpoint", self.clip_checkpoint),
            ("CLIP tokenizer", self.clip_tokenizer),
            ("VAE checkpoint", self.vae_checkpoint),
        )
        missing = [f"{label}: {path}" for label, path in fields if not path.exists()]
        if missing:
            raise FileNotFoundError("Wan-Animate-2 runtime inputs are missing:\n" + "\n".join(missing))


@dataclass
class LoadedRuntime:
    model: Any
    vae: Any
    device: torch.device
    compute_dtype: Any
    t5_checkpoint: Path
    t5_tokenizer: Path
    clip_checkpoint: Path
    clip_tokenizer: Path
    t5_text_length: int


def _add_source(source: Path) -> None:
    source_value = str(source.resolve())
    if source_value not in sys.path:
        sys.path.insert(0, source_value)


def _dtype(name: str) -> Any:
    import torch

    values = {
        "float16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported compute dtype: {name}") from exc


def _log_memory(stage: str, device: Any) -> None:
    import torch

    if not torch.cuda.is_available() or str(device) == "cpu":
        LOGGER.info("stage=%s cuda=unavailable", stage)
        return
    LOGGER.info(
        "stage=%s cuda_allocated_mb=%.1f cuda_reserved_mb=%.1f",
        stage,
        torch.cuda.memory_allocated(device) / 1024**2,
        torch.cuda.memory_reserved(device) / 1024**2,
    )


def _require_finite_tensor(label: str, tensor: Any) -> None:
    """Reject invalid model output before it can become a black encoded video."""

    import torch

    finite = torch.isfinite(tensor)
    finite_count = int(finite.sum().item())
    total_count = int(tensor.numel())
    nan_count = int(torch.isnan(tensor).sum().item())
    positive_inf_count = int(torch.isposinf(tensor).sum().item())
    negative_inf_count = int(torch.isneginf(tensor).sum().item())
    if finite_count:
        finite_values = tensor[finite].float()
        minimum = float(finite_values.min().item())
        maximum = float(finite_values.max().item())
        mean = float(finite_values.mean().item())
    else:
        minimum = maximum = mean = float("nan")
    LOGGER.info(
        "stage=tensor_check tensor=%s finite=%s/%s nan=%s pos_inf=%s neg_inf=%s min=%.6g max=%.6g mean=%.6g",
        label,
        finite_count,
        total_count,
        nan_count,
        positive_inf_count,
        negative_inf_count,
        minimum,
        maximum,
        mean,
    )
    if finite_count != total_count:
        raise RuntimeError(
            f"Wan-Animate-2 produced non-finite {label}: "
            f"finite={finite_count}/{total_count} nan={nan_count} "
            f"pos_inf={positive_inf_count} neg_inf={negative_inf_count}"
        )


class _ScaledReferenceValues:
    """Temporarily scale cached reference values without duplicating the cache."""

    def __init__(self, values: dict[int, Any], strength: float):
        self.values = values
        self.strength = strength
        self._originals: dict[int, Any] = {}

    def __enter__(self) -> None:
        if self.strength == 1.0:
            return None
        for index, value in self.values.items():
            self._originals[index] = value
            value.mul_(self.strength)
        return None

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if self.strength == 1.0:
            return
        inverse = 1.0 / self.strength
        for index, value in self._originals.items():
            value.mul_(inverse)
        self._originals.clear()


def load_runtime(
    paths: RunnerPaths,
    *,
    device_name: str,
    compute_dtype: str,
    text_length: int,
) -> LoadedRuntime:
    """Load every component with explicit stage boundaries and memory policy."""

    import torch

    paths.validate()
    prepare_runtime_cache_dirs()
    _add_source(paths.source)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    _install_attention_fallback()
    _validate_flash_attention(device)
    dtype = _dtype(compute_dtype)

    LOGGER.info("stage=transformer_load model=%s device=%s", paths.transformer, device)
    started = time.perf_counter()
    model, store, validation = load_transformer(
        paths.transformer,
        paths.source,
        device=str(device),
    )
    _install_reference_attention_fallback(model)
    _install_flex_attention_fallback(model)
    LOGGER.info(
        "stage=transformer_ready seconds=%.2f tensors=%s raw_model_type=%s",
        time.perf_counter() - started,
        validation["tensorCount"],
        store.metadata.get("general.architecture"),
    )
    _log_memory("transformer_ready", device)

    from wanxiang.models.vae import VideoVAE

    LOGGER.info("stage=vae_load device=%s checkpoint=%s", device, paths.vae_checkpoint)
    started = time.perf_counter()
    vae = VideoVAE(
        vae_pth=str(paths.vae_checkpoint),
        device=str(device),
        dtype=dtype,
    )
    LOGGER.info("stage=vae_ready seconds=%.2f", time.perf_counter() - started)
    _log_memory("runtime_ready", device)
    LOGGER.info(
        "stage=conditioning_deferred t5=%s clip=%s policy=load-encode-release",
        paths.t5_checkpoint,
        paths.clip_checkpoint,
    )
    return LoadedRuntime(
        model=model,
        vae=vae,
        device=device,
        compute_dtype=dtype,
        t5_checkpoint=paths.t5_checkpoint,
        t5_tokenizer=paths.t5_tokenizer,
        clip_checkpoint=paths.clip_checkpoint,
        clip_tokenizer=paths.clip_tokenizer,
        t5_text_length=text_length,
    )


def _is_cuda_oom(error: RuntimeError) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _load_t5(runtime: LoadedRuntime, *, forced_device: str | None = None) -> Any:
    import torch
    from wanxiang import models
    from wanxiang.eval_i2v import T5Encoder
    from wanxiang.utils import HuggingfaceTokenizer

    # The production 3090 profile reserves CUDA for the quantized Wan
    # transformer and VAE. T5 is a short conditioning pass and belongs on
    # CPU by default; moving UMT5-XXL to CUDA can fail after the transformer
    # has reserved most of the device, even when the worker is otherwise
    # healthy. Keep an explicit CUDA override for diagnostics/large hosts.
    requested_device = (forced_device or os.getenv("WAN_T5_DEVICE", "cpu")).strip().lower()
    if requested_device not in {"auto", "cpu", "cuda"}:
        raise ValueError("WAN_T5_DEVICE must be auto, cpu, or cuda")
    use_cuda = requested_device == "cuda" or (
        requested_device == "auto" and runtime.device.type == "cuda"
    )
    target_device = runtime.device if use_cuda else torch.device("cpu")
    LOGGER.info(
        "stage=t5_load device=%s requested=%s checkpoint=%s",
        target_device,
        requested_device,
        runtime.t5_checkpoint,
    )
    started = time.perf_counter()
    model = getattr(models, "umt5_xxl")(
        encoder_only=True,
        return_tokenizer=False,
        dtype=runtime.compute_dtype,
        device="meta",
    ).eval().requires_grad_(False)
    _load_mmap_state(model, runtime.t5_checkpoint)
    if target_device.type == "cuda":
        try:
            model.to(target_device)
        except RuntimeError as error:
            if not _is_cuda_oom(error):
                raise
            LOGGER.warning(
                "stage=t5_cuda_fallback reason=oom requested=%s",
                requested_device,
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
            return _load_t5(runtime, forced_device="cpu")
    encoder = T5Encoder.__new__(T5Encoder)
    encoder.name = "umt5_xxl"
    encoder.text_len = runtime.t5_text_length
    encoder.dtype = runtime.compute_dtype
    encoder.device = target_device
    encoder.checkpoint_path = str(runtime.t5_checkpoint)
    encoder.tokenizer_path = str(runtime.t5_tokenizer)
    encoder.model = model
    encoder.tokenizer = HuggingfaceTokenizer(
        name=str(runtime.t5_tokenizer), seq_len=runtime.t5_text_length, clean="whitespace"
    )
    LOGGER.info("stage=t5_ready seconds=%.2f device=%s", time.perf_counter() - started, target_device)
    return encoder


def _load_clip(runtime: LoadedRuntime) -> Any:
    import torch
    from wanxiang.eval_i2v import CLIP
    from wanxiang.models.clip import clip_xlm_roberta_vit_h_14
    from wanxiang.utils import HuggingfaceTokenizer

    LOGGER.info("stage=clip_load device=cpu checkpoint=%s", runtime.clip_checkpoint)
    started = time.perf_counter()
    model, transforms = clip_xlm_roberta_vit_h_14(
        pretrained=False,
        return_transforms=True,
        return_tokenizer=False,
        dtype=torch.float16,
        device="meta",
    )
    _load_mmap_state(model, runtime.clip_checkpoint)
    model = model.eval().requires_grad_(False)
    encoder = CLIP.__new__(CLIP)
    encoder.name = "clip_xlm_roberta_vit_h_14"
    encoder.dtype = torch.float16
    encoder.device = "cpu"
    encoder.checkpoint_path = str(runtime.clip_checkpoint)
    encoder.tokenizer_path = str(runtime.clip_tokenizer)
    encoder.model = model
    encoder.transforms = transforms
    encoder.tokenizer = HuggingfaceTokenizer(
        name=str(runtime.clip_tokenizer), seq_len=model.max_text_len - 2, clean="whitespace"
    )
    LOGGER.info("stage=clip_ready seconds=%.2f device=cpu", time.perf_counter() - started)
    return encoder


def _load_mmap_state(model: Any, checkpoint: Path) -> None:
    """Assign checkpoint tensors without constructing a second full state copy."""

    import torch

    state = torch.load(
        str(checkpoint),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    result = model.load_state_dict(state, assign=True)
    del state
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch for {checkpoint}: "
            f"missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )


def _release_conditioner(conditioner: Any, *, label: str, device: Any) -> None:
    import torch

    del conditioner
    gc.collect()
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.empty_cache()
    LOGGER.info("stage=%s_released", label)


def _encode_t5(runtime: LoadedRuntime, prompt: str) -> Any:
    """Encode on CUDA when it fits, retrying on CPU only after a CUDA OOM."""

    t5 = _load_t5(runtime)
    try:
        try:
            return t5([prompt])[0].to(device=runtime.device, dtype=runtime.compute_dtype)
        except RuntimeError as error:
            if getattr(t5.device, "type", str(t5.device)) != "cuda" or not _is_cuda_oom(error):
                raise
            LOGGER.warning("stage=t5_cuda_fallback reason=oom during_encode")
            _release_conditioner(t5, label="t5", device=runtime.device)
            t5 = _load_t5(runtime, forced_device="cpu")
            return t5([prompt])[0].to(device=runtime.device, dtype=runtime.compute_dtype)
    finally:
        _release_conditioner(t5, label="t5", device=runtime.device)


def _read_driver(path: Path, *, width: int, height: int, frame_count: int | None = None) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open driver video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    try:
        while frame_count is None or len(frames) < frame_count:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"driver video contained no readable frames: {path}")
    while frame_count is not None and len(frames) < frame_count:
        frames.append(frames[len(frames) % len(frames)].copy())
    return (frames[:frame_count] if frame_count is not None else frames), fps


def _write_video(frames: list[np.ndarray], output: Path, fps: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"could not open output video writer: {output}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _render_window(
    runtime: LoadedRuntime,
    *,
    reference_image: Path,
    driver_video: Path,
    output: Path,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    fps: int,
    clip_len: int,
    steps: int,
    seed: int,
    reference_strength: float = 1.0,
    continuation_frames: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Render one short segment using Wan's temporal continuation contract."""

    import torch
    from einops import rearrange
    from wanxiang.eval_i2v import get_i2v_mask
    from wanxiang.models.wan_animate_2_model import sinusoidal_embedding_1d
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

    if seed < 0:
        seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(seed)

    LOGGER.info(
        "stage=render_start reference=%s driver=%s width=%s height=%s fps=%s clip_len=%s steps=%s seed=%s reference_strength=%.4f",
        reference_image,
        driver_video,
        width,
        height,
        fps,
        clip_len,
        steps,
        seed,
        reference_strength,
    )
    first_image = cv2.imread(str(reference_image), cv2.IMREAD_COLOR)
    if first_image is None:
        raise RuntimeError(f"could not read reference image: {reference_image}")
    first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2RGB)
    first_image = cv2.resize(first_image, (width, height), interpolation=cv2.INTER_AREA)
    if continuation_frames:
        continuation_frames = [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            for frame in continuation_frames
        ]
    driver_frames, source_fps = _read_driver(driver_video, width=width, height=height, frame_count=clip_len)
    LOGGER.info(
        "stage=inputs_ready driver_fps=%.3f frames=%s continuation_frames=%s",
        source_fps,
        len(driver_frames),
        len(continuation_frames or []),
    )

    # The reference image remains the identity anchor; continuation frames
    # are supplied separately as temporal context for the next window.
    target_frames = clip_len + 1
    lat_h = height // 8
    lat_w = width // 8
    lat_t = target_frames // 4 + 1 + 1
    target_shape = [lat_t, lat_h, lat_w]
    grid_sizes = torch.stack(
        [torch.tensor([lat_t, lat_h // 2, lat_w // 2], dtype=torch.long)]
    )

    device = runtime.device
    model = runtime.model
    vae = runtime.vae
    compute_dtype = runtime.compute_dtype

    conditioning = torch.tensor(np.stack(driver_frames) / 127.5 - 1.0)
    conditioning = rearrange(conditioning, "t h w c -> 1 c t h w").to(device=device, dtype=compute_dtype)
    reference = torch.tensor(first_image / 127.5 - 1.0)
    reference = rearrange(reference, "h w c -> 1 c h w").to(device=device, dtype=compute_dtype)
    ref_pixel_values = reference.unsqueeze(2)
    conditioning_pixel_values = conditioning

    noise = [
        torch.randn(
            16,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
    ]
    # The released distilled profile explicitly uses Euler flow sampling.
    # Keep the shifted sigma schedule from the official Wan implementation,
    # but use the matching Euler update instead of the base-model multistep
    # solver.
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    sigmas = _sampling_sigmas(steps, 5.0)
    timesteps = _retrieve_sigmas(scheduler, sigmas=sigmas, device=device)

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=compute_dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast_context:
        LOGGER.info("stage=vae_reference_encode")
        ref_latents = torch.stack(vae.encode(ref_pixel_values))
        _require_finite_tensor("reference_latents", ref_latents)
        mask_ref = get_i2v_mask(1, lat_h, lat_w, 1, device=device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(dtype=compute_dtype, device=device)

        # The target conditioning tensor has one reference frame followed by
        # the driver frames. Keep the target-side conditioning tensor separate
        # from the driver tensor used by forward_ref; the official pipeline
        # uses both shapes and the transformer requires them to match exactly.
        continuation_count = len(continuation_frames or [])
        mask_reft = get_i2v_mask(lat_t - 1, lat_h, lat_w, continuation_count, device=device)
        reft_frame_count = max(1, target_frames - continuation_count - 1)
        reft_parts: list[torch.Tensor] = []
        if continuation_frames:
            continuation_tensor = torch.tensor(np.stack(continuation_frames) / 127.5 - 1.0)
            continuation_tensor = rearrange(continuation_tensor, "t h w c -> c t h w")
            reft_parts.append(continuation_tensor)
        reft_parts.append(torch.zeros(3, reft_frame_count, height, width))
        reft_pixels = torch.cat(reft_parts, dim=1)
        expected_reft_t = mask_reft.shape[1]
        for _ in range(8):
            reft_latents = torch.stack(vae.encode([reft_pixels.to(device=device, dtype=compute_dtype)]))
            _require_finite_tensor("driver_condition_latents", reft_latents)
            if reft_latents.shape[2] >= expected_reft_t:
                break
            reft_pixels = torch.cat([reft_pixels, torch.zeros(3, 1, height, width)], dim=1)
        if reft_latents.shape[2] != expected_reft_t:
            raise RuntimeError(
                f"VAE temporal conditioning mismatch: expected {expected_reft_t}, "
                f"got {reft_latents.shape[2]}"
            )
        y_reft = torch.concat([mask_reft, reft_latents[0]]).to(
            dtype=compute_dtype, device=device
        )
        y = torch.concat([y_ref, y_reft], dim=1)

        clip = _load_clip(runtime)
        try:
            LOGGER.info("stage=clip_reference_encode device=cpu")
            with torch.autocast(device_type="cuda", enabled=False) if device.type == "cuda" else nullcontext():
                clip.model.visual.to(device)
                clip_context = clip.visual(
                    [ref_pixel_values[0, :, 0].to(device=device, dtype=torch.float16).unsqueeze(1)]
                ).to(dtype=compute_dtype, device=device)
                clip.model.visual.to("cpu")
                torch.cuda.empty_cache()
                LOGGER.info("stage=clip_condition_encode device=cpu")
                clip.model.visual.to(device)
                condition_clip_context = clip.visual(
                    [conditioning_pixel_values[0, :, 0].to(device=device, dtype=torch.float16).unsqueeze(1)]
                ).to(dtype=compute_dtype, device=device)
                clip.model.visual.to("cpu")
                torch.cuda.empty_cache()
        finally:
            _release_conditioner(clip, label="clip", device=device)

        t5_device = (os.getenv("WAN_T5_DEVICE", "auto").strip().lower())
        LOGGER.info("stage=t5_prompt_encode requested_device=%s", t5_device)
        context = _encode_t5(runtime, prompt)
        context_ref = context

        LOGGER.info("stage=condition_encode")
        condition_latents = torch.stack(vae.encode(conditioning_pixel_values))
        _require_finite_tensor("condition_latents", condition_latents)
        condition_lat_t = condition_latents.shape[2]
        condition_lat_h = condition_latents.shape[3]
        condition_lat_w = condition_latents.shape[4]
        condition_y = torch.concat(
            [
                get_i2v_mask(
                    condition_lat_t,
                    condition_lat_h,
                    condition_lat_w,
                    clip_len,
                    device=device,
                ),
                condition_latents[0],
            ],
            dim=0,
        ).to(dtype=compute_dtype, device=device)
        grid_sizes_ref = torch.stack(
            [torch.tensor([condition_lat_t, condition_lat_h // 2, condition_lat_w // 2], dtype=torch.long)]
        ).to(device=device)

        max_seq_len = int(np.ceil(np.prod(target_shape) / 4))
        max_seq_len_ref = int(np.ceil(np.prod(condition_latents.shape[2:]) / 4))
        ref_args = {
            "context_ref": [context_ref],
            "seq_len_ref": max_seq_len_ref,
            "clip_fea_ref": condition_clip_context,
            "y_ref": [condition_y],
        }
        cond_args = {
            "context": [context],
            "seq_len": max_seq_len,
            "clip_fea": clip_context,
            "y": [y],
            # The model's origin grid is derived from the full target frame
            # contract, including the conditioning frame. Using clip_len here
            # undercounts the temporal grid for longer clips and breaks the
            # in-context reshape during generation.
            "origin_len": target_frames,
            "origin_area": [width, height],
        }
        LOGGER.info("stage=reference_transformer_pass")
        cache_k: dict[int, Any] = {}
        cache_v: dict[int, Any] = {}
        model(
            condition_latents,
            grid_sizes=grid_sizes,
            k_cache=cache_k,
            v_cache=cache_v,
            t=torch.ones(1, device=device),
            method="forward_ref",
            **ref_args,
        )
        latents = noise
        with _ScaledReferenceValues(cache_v, reference_strength):
            LOGGER.info(
                "stage=reference_strength_applied value=%.4f cache_layers=%s",
                reference_strength,
                len(cache_v),
            )
            for index, timestep in enumerate(timesteps):
                LOGGER.info("stage=denoise step=%s/%s", index + 1, len(timesteps))
                t = torch.stack([timestep])
                prediction = model(
                    latents,
                    k_cache=cache_k,
                    v_cache=cache_v,
                    t=t,
                    grid_sizes_ref=grid_sizes_ref,
                    method="forward_gen",
                    **cond_args,
                )
                # The Wan adapter returns a one-item list here; the scheduler consumes
                # the tensor at index zero, so validate that exact value.
                _require_finite_tensor("denoise_prediction", prediction[0])
                latents[0] = scheduler.step(
                    prediction[0].unsqueeze(0),
                    timestep,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=generator,
                )[0].squeeze(0)
                _require_finite_tensor("sample_latents", latents[0])
                del prediction
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Wan allocates one leading latent for the conditioning frame. It is
        # not a generated video frame and must be removed before VAE decode.
        # Decoding it and trimming pixels afterward shifts every temporal
        # window and exposes the conditioning transition as visible repeats
        # and color changes at the start of each window.
        generated_latents = latents[0][:, 1:]
        LOGGER.info(
            "stage=vae_decode conditioning_latent_dropped=true latent_frames=%s generated_latent_frames=%s",
            int(latents[0].shape[1]),
            int(generated_latents.shape[1]),
        )
        decoded = torch.stack(vae.decode([generated_latents.to(dtype=torch.float32)]))
        _require_finite_tensor("decoded_frames", decoded)
        frame_values = rearrange(((decoded + 1.0) * 127.5), "1 c t h w -> t h w c")
        _require_finite_tensor("decoded_pixel_values", frame_values)
        frame_values = frame_values.clamp(0, 255).cpu().numpy()
        if not np.isfinite(frame_values).all():
            raise RuntimeError("Wan-Animate-2 produced non-finite pixel values after VAE decode")
        frames = frame_values.astype(np.uint8)
        nonzero_fraction = float(np.count_nonzero(frames) / max(1, frames.size))
        LOGGER.info(
            "stage=decoded_frames_ready frame_min=%s frame_max=%s frame_mean=%.6g nonzero_fraction=%.6g",
            int(frames.min()),
            int(frames.max()),
            float(frames.mean()),
            nonzero_fraction,
        )
        if int(frames.max()) == 0 or nonzero_fraction < 0.0001:
            raise RuntimeError(
                "Wan-Animate-2 produced a blank render after VAE decode; "
                "the job was not encoded as a successful video"
            )

    _write_video([frame for frame in frames], output, fps)
    metadata = {
        "referenceImage": str(reference_image),
        "driverVideo": str(driver_video),
        "output": str(output),
        "seed": seed,
        "width": width,
        "height": height,
        "fps": fps,
        "frameCount": len(frames),
        "steps": steps,
        "referenceStrength": reference_strength,
        "continuationUsed": bool(continuation_frames),
        "continuationFrames": len(continuation_frames or []),
        "backend": "autotransition-wan-animate-2-gguf",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("stage=render_complete output=%s frames=%s", output, len(frames))
    return metadata


def _read_rendered_frames(path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open rendered video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"rendered video contained no readable frames: {path}")
    return frames


def _release_window_memory(device: Any, *, stage: str) -> None:
    """Release temporaries after a window has returned all frames to host RAM."""

    import torch

    gc.collect()
    if getattr(device, "type", str(device)) == "cuda":
        torch.cuda.empty_cache()
    _log_memory(stage, device)


def _read_continuation_frames(path: Path, count: int) -> list[np.ndarray]:
    """Read the newest continuation frames in chronological order."""

    if count < 1:
        return []
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    if path.is_dir():
        image_paths = sorted(
            candidate
            for candidate in path.iterdir()
            if candidate.is_file() and candidate.suffix.lower() in image_suffixes
        )[-count:]
        frames: list[np.ndarray] = []
        for image_path in image_paths:
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"could not read continuation frame: {image_path}")
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if frames:
            return frames
        raise RuntimeError(f"continuation frame directory contained no images: {path}")
    if path.suffix.lower() in image_suffixes:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"could not read continuation frame: {path}")
        return [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"could not open continuation video: {path}")
    frames = deque(maxlen=count)
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"continuation video contained no readable frames: {path}")
    return list(frames)


def _read_last_rendered_frames(path: Path, count: int) -> list[np.ndarray]:
    if count < 1:
        return []
    return _read_rendered_frames(path)[-count:]


def _read_last_rendered_frame(path: Path) -> np.ndarray:
    return _read_last_rendered_frames(path, 1)[0]


def _read_continuation_frame(path: Path) -> np.ndarray:
    """Backward-compatible one-frame continuation reader."""

    return _read_continuation_frames(path, 1)[0]


def render_segment(
    runtime: LoadedRuntime,
    *,
    reference_image: Path,
    driver_video: Path,
    output: Path,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    fps: int,
    clip_len: int | None,
    steps: int,
    seed: int,
    max_clip_len: int = 81,
    temporal_context_frames: int = 5,
    reference_strength: float = 1.0,
    continuation_frames: Path | None = None,
    continuation_frame: Path | None = None,
) -> dict[str, Any]:
    """Render the complete driver using bounded overlapping temporal windows.

    Wan's transformer has a bounded temporal attention window. The official
    pipeline handles longer drivers by overlapping windows and carrying the
    previous window's boundary forward. The project-owned GGUF path keeps the
    same full-length contract here: no source frames are discarded merely to
    fit a short CLI default, and the output is trimmed only to the exact input
    frame count after window stitching.
    """

    if max_clip_len < 2:
        raise ValueError("max_clip_len must be at least 2")
    if temporal_context_frames not in {0, 1, 5}:
        raise ValueError("temporal_context_frames must be 0, 1, or 5")
    if temporal_context_frames >= max_clip_len:
        raise ValueError("temporal_context_frames must be smaller than max_clip_len")
    all_frames, source_fps = _read_driver(
        driver_video,
        width=width,
        height=height,
        frame_count=None,
    )
    if clip_len is not None:
        if clip_len < 1:
            raise ValueError("clip_len must be positive when supplied")
        all_frames = all_frames[:clip_len]
    if not all_frames:
        raise RuntimeError("driver video contained no readable frames")
    output_fps = int(round(fps if fps > 0 else source_fps))
    if output_fps < 1:
        raise RuntimeError(f"driver video has no usable frame rate: {source_fps}")

    window_dir = output.parent / ".windows"
    window_dir.mkdir(parents=True, exist_ok=True)
    overlap = temporal_context_frames
    step = max_clip_len - overlap
    stitched: list[np.ndarray] = []
    window_reports: list[dict[str, Any]] = []
    start = 0
    window_index = 0
    continuity_frames: list[np.ndarray] | None = None
    effective_continuation = continuation_frames or continuation_frame
    if effective_continuation is not None and temporal_context_frames > 0:
        continuity_frames = _read_continuation_frames(
            effective_continuation,
            temporal_context_frames,
        )
        LOGGER.info(
            "stage=segment_continuation_loaded path=%s frames=%s",
            effective_continuation,
            len(continuity_frames),
        )
    try:
        while start < len(all_frames):
            end = min(len(all_frames), start + max_clip_len)
            window_frames = all_frames[start:end]
            window_index += 1
            window_driver = window_dir / f"driver-{window_index:04d}.mp4"
            window_output = window_dir / f"render-{window_index:04d}.mp4"
            _write_video(window_frames, window_driver, output_fps)
            LOGGER.info(
                "stage=window_start index=%s start_frame=%s end_frame=%s input_frames=%s fps=%s",
                window_index,
                start,
                end,
                len(window_frames),
                output_fps,
            )
            report = _render_window(
                runtime,
                reference_image=reference_image,
                driver_video=window_driver,
                output=window_output,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                fps=output_fps,
                clip_len=len(window_frames),
                steps=steps,
                seed=(seed + window_index - 1) if seed >= 0 else seed,
                reference_strength=reference_strength,
                continuation_frames=continuity_frames,
            )
            # Keep one immutable stdout/stderr pair per temporal window. The
            # adapter writes logs beside the window output; retaining them is
            # what makes a Salad failure diagnosable without rerunning it.
            for log_path in window_dir.glob("*.log"):
                retained = output.parent / f"window-{window_index:04d}-{log_path.name}"
                retained.write_bytes(log_path.read_bytes())
            rendered = _read_rendered_frames(window_output)
            raw_rendered_count = len(rendered)
            if continuity_frames is not None and len(rendered) > overlap:
                # These are the carried frames used to condition this window,
                # not new timeline frames. The leading conditioning latent
                # was already removed before VAE decode above.
                rendered = rendered[overlap:]
            stitched.extend(rendered)
            continuity_frames = _read_last_rendered_frames(window_output, overlap)
            report.update(
                {
                    "windowIndex": window_index,
                    "startFrame": start,
                    "endFrame": end,
                    "inputFrameCount": len(window_frames),
                    "rawOutputFrameCount": raw_rendered_count,
                    "outputFrameCount": len(rendered),
                    "continuationUsed": report.get("continuationUsed", False),
                    "continuationFrameIndices": (
                        list(range(max(0, start - overlap), start))
                        if continuity_frames is not None
                        else []
                    ),
                    "continuationFrameTimestampsSeconds": (
                        [
                            round(frame_index / output_fps, 6)
                            for frame_index in range(max(0, start - overlap), start)
                        ]
                        if continuity_frames is not None
                        else []
                    ),
                }
            )
            window_reports.append(report)
            _release_window_memory(runtime.device, stage=f"window_{window_index}_released")
            LOGGER.info(
                "stage=window_complete index=%s stitched_frames=%s/%s",
                window_index,
                len(stitched),
                len(all_frames),
            )
            if end >= len(all_frames):
                break
            start += step
    finally:
        for path in window_dir.glob("*"):
            path.unlink(missing_ok=True)
        window_dir.rmdir()

    if len(stitched) < len(all_frames):
        if not stitched:
            raise RuntimeError("Wan Animate produced no frames after window stitching")
        stitched.extend(stitched[-1].copy() for _ in range(len(all_frames) - len(stitched)))
    stitched = stitched[: len(all_frames)]
    _write_video(stitched, output, output_fps)
    metadata = {
        "referenceImage": str(reference_image),
        "driverVideo": str(driver_video),
        "output": str(output),
        "seed": seed,
        "referenceStrength": reference_strength,
        "width": width,
        "height": height,
        "fps": output_fps,
        "sourceFps": source_fps,
        "frameCount": len(stitched),
        "sourceFrameCount": len(all_frames),
        "steps": steps,
        "temporalWindow": max_clip_len,
        "temporalOverlap": overlap,
        "windowCount": len(window_reports),
        "continuityMode": (
            "multi-frame-carry"
            if temporal_context_frames > 0
            else "disabled"
        ),
        "temporalContextFrames": temporal_context_frames,
        "initialContinuation": effective_continuation is not None,
        "windows": window_reports,
        "backend": "autotransition-wan-animate-2-gguf",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info(
        "stage=render_complete output=%s frames=%s fps=%s source_frames=%s windows=%s",
        output,
        len(stitched),
        output_fps,
        len(all_frames),
        len(window_reports),
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Wan-Animate-2 through the project-owned GGUF runtime")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--t5", type=Path, required=True)
    parser.add_argument("--t5-tokenizer", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--clip-tokenizer", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--preflight", action="store_true", help="validate files and import the official architecture only")
    parser.add_argument("--load", action="store_true", help="load the transformer and companions without rendering")
    parser.add_argument("--reference-image", type=Path)
    parser.add_argument("--driver-video", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prompt", default="a person performing a full-body dance")
    parser.add_argument("--negative-prompt", default="static, distorted body, extra limbs, cropped subject")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--fps", type=int, default=0, help="output FPS; 0 preserves the driver FPS")
    parser.add_argument("--clip-len", type=int, default=None, help="optional test window; production uses the full driver")
    parser.add_argument("--full-driver", action="store_true", help="process every driver frame with overlapping windows")
    parser.add_argument("--max-clip-len", type=int, default=81, help="maximum frames per inference window")
    parser.add_argument(
        "--temporal-context-frames",
        type=int,
        default=5,
        help="previous rendered frames carried across windows; Wan supports 0, 1, or 5",
    )
    parser.add_argument(
        "--continuation-frames",
        type=Path,
        help="directory or video containing the previous render's boundary frames",
    )
    parser.add_argument(
        "--continuation-frame",
        type=Path,
        help="deprecated alias for a single preceding render frame",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--text-length", type=int, default=256)
    parser.add_argument("--reference-strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = _parser().parse_args()
    paths = RunnerPaths(
        transformer=args.model,
        source=args.official_source,
        t5_checkpoint=args.t5,
        t5_tokenizer=args.t5_tokenizer,
        clip_checkpoint=args.clip,
        clip_tokenizer=args.clip_tokenizer,
        vae_checkpoint=args.vae,
    )
    paths.validate()
    if args.preflight:
        _add_source(paths.source)
        from wanxiang.wanxiang_animate_2_arch import WanxiangAnimate2Transformer

        LOGGER.info("preflight=ok model=%s source=%s class=%s", paths.transformer, paths.source, WanxiangAnimate2Transformer.__name__)
        return 0
    if not args.load and not (args.reference_image and args.driver_video and args.output):
        _parser().error("rendering requires --reference-image, --driver-video, and --output")
    if args.text_length < 32 or args.text_length > 512:
        _parser().error("--text-length must be between 32 and 512")
    runtime = load_runtime(
        paths,
        device_name=args.device,
        compute_dtype=args.dtype,
        text_length=args.text_length,
    )
    if args.load:
        LOGGER.info("load=ok")
        return 0
    render_segment(
        runtime,
        reference_image=args.reference_image,
        driver_video=args.driver_video,
        output=args.output,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        fps=args.fps,
        clip_len=None if args.full_driver else args.clip_len,
        steps=args.steps,
        seed=args.seed,
        max_clip_len=args.max_clip_len,
        temporal_context_frames=args.temporal_context_frames,
        reference_strength=args.reference_strength,
        continuation_frames=args.continuation_frames,
        continuation_frame=args.continuation_frame,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
