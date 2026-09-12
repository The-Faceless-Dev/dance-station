from __future__ import annotations

import gc
import json
import logging
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import LtxVideoConfig
from .contracts import LtxVideoRequest
from .media import acquire_input, mux_audio, transcode_video
from .memory import (
    assert_blackwell_for_nvfp4,
    build_memory_plan,
    clear_cuda_cache,
    gpu_memory_snapshot,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[str, float, str, dict[str, Any] | None], None]


class LtxRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str, *, stage: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.details = details or {}


@dataclass(frozen=True)
class LtxRuntimeResult:
    video_path: Path
    audio_path: Path | None
    metadata: dict[str, Any]
    intermediate_paths: tuple[Path, ...] = ()


def _emit(progress: ProgressCallback, stage: str, fraction: float, message: str, details: dict[str, Any] | None = None) -> None:
    progress(stage, max(0.0, min(1.0, float(fraction))), message, details)


def _tiling_to_dict(tiling: Any) -> dict[str, Any] | None:
    if tiling is None:
        return None
    result: dict[str, Any] = {}
    for name in ("frames", "height", "width"):
        value = getattr(tiling, name, None)
        if value is not None:
            result[name] = {
                "tileSize": int(getattr(value, "tile_size", 0)),
                "overlap": int(getattr(value, "overlap", 0)),
            }
    return result or None


def _configure_fast_attention(torch: Any, *, enforce: bool) -> dict[str, Any]:
    """Select Flash SDP and make unsupported fallback paths visible.

    LTX uses PyTorch scaled-dot-product attention in its transformer. The
    production profile is a single RTX 5090, so silently dropping to math or
    memory-efficient attention would turn a capacity/configuration problem into
    an unexplained slow generation.
    """
    if not torch.cuda.is_available():
        return {"cuda": False, "enforced": enforce}
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(not enforce)
    torch.backends.cuda.enable_math_sdp(not enforce)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(not enforce)
    result = {
        "cuda": True,
        "enforced": enforce,
        "flashSdp": bool(torch.backends.cuda.flash_sdp_enabled()),
        "memoryEfficientSdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
        "mathSdp": bool(torch.backends.cuda.math_sdp_enabled()),
    }
    if hasattr(torch.backends.cuda, "cudnn_sdp_enabled"):
        result["cudnnSdp"] = bool(torch.backends.cuda.cudnn_sdp_enabled())
    return result


@contextmanager
def _gemma_attention_context(torch: Any):
    """Use a Gemma-compatible SDP backend without changing video attention.

    Gemma 4 is invoked through Transformers and can reject the exact Flash SDP
    shape that the LTX video transformer requires. The prompt pass is short,
    so use PyTorch's universal math SDP backend only for that pass. The caller
    must restore the strict video policy after this context exits.
    """
    if not torch.cuda.is_available():
        yield {"backend": "default", "isolated": False, "device": "cpu"}
        return

    cuda = getattr(getattr(torch, "backends", None), "cuda", None)
    if cuda is None:
        raise RuntimeError("PyTorch CUDA backends are unavailable for Gemma prompt encoding")

    def _enabled(name: str) -> bool | None:
        reader = getattr(cuda, name, None)
        return bool(reader()) if callable(reader) else None

    previous = {
        "flash": _enabled("flash_sdp_enabled"),
        "memoryEfficient": _enabled("mem_efficient_sdp_enabled"),
        "math": _enabled("math_sdp_enabled"),
        "cudnn": _enabled("cudnn_sdp_enabled"),
    }

    # The production video policy intentionally disables math SDP. Explicitly
    # enable math for this short Gemma-only pass, then restore every flag.
    cuda.enable_flash_sdp(False)
    cuda.enable_mem_efficient_sdp(False)
    cuda.enable_math_sdp(True)
    if hasattr(cuda, "enable_cudnn_sdp"):
        cuda.enable_cudnn_sdp(False)

    attention = getattr(getattr(torch, "nn", None), "attention", None)
    modern_kernel = getattr(attention, "sdpa_kernel", None)
    sdp_backend = getattr(attention, "SDPBackend", None)
    try:
        if callable(modern_kernel) and sdp_backend is not None and hasattr(sdp_backend, "MATH"):
            with modern_kernel([sdp_backend.MATH]):
                yield {
                    "backend": "sdpa_math",
                    "isolated": True,
                    "api": "torch.nn.attention.sdpa_kernel",
                    "mathSdp": True,
                }
            return

        legacy_kernel = getattr(cuda, "sdp_kernel", None)
        if callable(legacy_kernel):
            with legacy_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False, enable_cudnn=False):
                yield {
                    "backend": "sdpa_math",
                    "isolated": True,
                    "api": "torch.backends.cuda.sdp_kernel",
                    "mathSdp": True,
                }
            return

        raise RuntimeError("PyTorch does not expose an isolated SDP context for Gemma prompt encoding")
    finally:
        restore = {
            "enable_flash_sdp": previous["flash"],
            "enable_mem_efficient_sdp": previous["memoryEfficient"],
            "enable_math_sdp": previous["math"],
            "enable_cudnn_sdp": previous["cudnn"],
        }
        for name, value in restore.items():
            setter = getattr(cuda, name, None)
            if callable(setter) and value is not None:
                setter(value)


class _ScopedPromptEncoder:
    """Run the shared Gemma encoder under its own attention policy."""

    def __init__(self, inner: Any, torch: Any, *, enforce_fast_attention: bool, progress: ProgressCallback):
        self.inner = inner
        self.torch = torch
        self.enforce_fast_attention = enforce_fast_attention
        self.progress = progress

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        gemma_attention: dict[str, Any]
        try:
            with _gemma_attention_context(self.torch) as gemma_attention:
                logger.info("LTX Gemma prompt encoding started attention=%s", gemma_attention)
                result = self.inner(*args, **kwargs)
        except Exception as exc:
            logger.exception("LTX Gemma prompt encoding failed attention=sdpa_math")
            raise LtxRuntimeError(
                "gemma_prompt_encoding_failed",
                f"Gemma prompt encoding failed with its isolated attention policy: {type(exc).__name__}: {exc}",
                stage="encode_prompt",
                details={"attentionPolicy": "sdpa_math_isolated"},
            ) from exc
        finally:
            video_attention = _configure_fast_attention(self.torch, enforce=self.enforce_fast_attention)
            logger.info("LTX video attention policy restored after Gemma encoding: %s", video_attention)
        _emit(self.progress, "encode_prompt", 0.5, "Gemma prompt encoded", {"attention": gemma_attention})
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _StepProgressDenoiser:
    """Report real denoiser calls without changing the upstream sampler."""

    def __init__(self, inner: Any, stage: str, total_steps: int, progress: ProgressCallback):
        self.inner = inner
        self.stage = stage
        self.total_steps = max(1, total_steps)
        self.progress = progress
        self.completed = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        result = self.inner(*args, **kwargs)
        self.completed += 1
        _emit(
            self.progress,
            self.stage,
            self.completed / self.total_steps,
            f"{self.stage.replace('_', ' ')} step {self.completed}/{self.total_steps}",
            {"completedSteps": self.completed, "totalSteps": self.total_steps},
        )
        return result


class _ProgressDiffusionStage:
    """Add per-step progress around an upstream stage without changing its sampler."""

    def __init__(self, inner: Any, progress: ProgressCallback):
        self.inner = inner
        self.progress = progress
        self.calls = 0

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        sigmas = kwargs.get("sigmas")
        denoiser = kwargs.get("denoiser")
        if sigmas is not None and denoiser is not None:
            stage = "stage_1_denoise" if self.calls == 0 else "stage_2_refine"
            kwargs["denoiser"] = _StepProgressDenoiser(
                denoiser,
                stage,
                len(sigmas) - 1,
                self.progress,
            )
        self.calls += 1
        return self.inner(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def _parse_ltx_imports() -> dict[str, Any]:
    try:
        import torch
        from ltx_core.allocator_trim_strategy import AllocatorTrimStrategy
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_core.model.transformer import LTXVideoOnlyModelConfigurator
        from ltx_core.model.transformer.compiling import CompilationConfig
        from ltx_core.model.video_vae import AUTO_TILING, DimensionSizeConfig, TileSizeConfig, get_video_chunks_number
        from ltx_core.model.video_vae.transformer import DiffVAEMode
        from ltx_pipelines.distilled import DistilledPipeline, should_use_ancestral_sampler
        from ltx_pipelines.utils.args import ImageConditioningInput
        from ltx_pipelines.utils.blocks import DiffusionStage, ImageConditioner, PromptEncoder, VideoDecoder, VideoUpsampler
        from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import (
            assert_resolution,
            ensure_tiling_config,
            image_conditionings_by_adding_guiding_latent,
            image_conditionings_by_replacing_latent,
            tiling_scale_factors_for_vae,
        )
        from ltx_pipelines.utils.media_io import encode_audio, encode_video
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.samplers import euler_ancestral_denoising_loop
        from ltx_pipelines.utils.types import ModalitySpec, OffloadMode, PipelineOutput
    except Exception as exc:
        raise LtxRuntimeError(
            "ltx_runtime_unavailable",
            f"LTX 2.5 runtime packages are unavailable: {type(exc).__name__}: {exc}",
            stage="load_models",
        ) from exc
    return locals()


def _video_only_pipeline_factory(
    imports: dict[str, Any],
    *,
    enforce_fast_attention: bool,
    vae_temporal_tile_frames: int,
    vae_temporal_overlap_frames: int,
) -> type:
    """Create a video-only variant of the official distilled two-stage pipeline.

    The upstream distilled pipeline is intentionally audio-video. This small adapter keeps its
    stage ordering and conditioning behavior, but passes ``audio=None`` and selects the official
    video-only transformer configurator. It therefore never constructs audio latent state or the
    audio VAE for silent/source-audio jobs.
    """
    torch = imports["torch"]
    AllocatorTrimStrategy = imports["AllocatorTrimStrategy"]
    GaussianNoiser = imports["GaussianNoiser"]
    DiffusionStage = imports["DiffusionStage"]
    ImageConditioner = imports["ImageConditioner"]
    PromptEncoder = imports["PromptEncoder"]
    VideoDecoder = imports["VideoDecoder"]
    VideoUpsampler = imports["VideoUpsampler"]
    LTXVideoOnlyModelConfigurator = imports["LTXVideoOnlyModelConfigurator"]
    ModelPaths = imports["ModelPaths"]
    ModalitySpec = imports["ModalitySpec"]
    SimpleDenoiser = imports["SimpleDenoiser"]
    AUTO_TILING = imports["AUTO_TILING"]
    DimensionSizeConfig = imports["DimensionSizeConfig"]
    TileSizeConfig = imports["TileSizeConfig"]
    DiffVAEMode = imports["DiffVAEMode"]
    should_use_ancestral_sampler = imports["should_use_ancestral_sampler"]
    EulerAncestralDiffusionStep = __import__(
        "ltx_core.components.diffusion_steps", fromlist=["EulerAncestralDiffusionStep"]
    ).EulerAncestralDiffusionStep
    euler_ancestral_denoising_loop = imports["euler_ancestral_denoising_loop"]
    DISTILLED_SIGMAS = imports["DISTILLED_SIGMAS"]
    STAGE_2_DISTILLED_SIGMAS = imports["STAGE_2_DISTILLED_SIGMAS"]
    assert_resolution = imports["assert_resolution"]
    ensure_tiling_config = imports["ensure_tiling_config"]
    image_conditionings_by_adding_guiding_latent = imports["image_conditionings_by_adding_guiding_latent"]
    image_conditionings_by_replacing_latent = imports["image_conditionings_by_replacing_latent"]
    tiling_scale_factors_for_vae = imports["tiling_scale_factors_for_vae"]
    PipelineOutput = imports["PipelineOutput"]
    encode_video = imports["encode_video"]
    get_video_chunks_number = imports["get_video_chunks_number"]
    VideoPixelShape = __import__("ltx_core.types", fromlist=["VideoPixelShape"]).VideoPixelShape

    class VideoOnlyDistilledPipeline:
        def __init__(self, model_paths: Any, spatial_upsampler_path: str, loras: list[Any], device: Any, quantization: Any, offload_mode: Any, compilation_config: Any, progress: ProgressCallback):
            self.device = device
            self.dtype = torch.bfloat16
            self.progress = progress
            self.enforce_fast_attention = enforce_fast_attention
            self.prompt_encoder = _ScopedPromptEncoder(
                PromptEncoder(
                    model_paths,
                    self.dtype,
                    device,
                    offload_mode=offload_mode,
                    alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
                ),
                torch,
                enforce_fast_attention=enforce_fast_attention,
                progress=progress,
            )
            self.image_conditioner = ImageConditioner(
                model_paths.video_vae(), self.dtype, device, alloc_trim_strategy=AllocatorTrimStrategy.TRIM
            )
            self.stage = DiffusionStage.from_checkpoint(
                model_paths.transformer(),
                self.dtype,
                device,
                loras=tuple(loras),
                quantization=quantization,
                offload_mode=offload_mode,
                compilation_config=compilation_config,
                alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
                model_configurator=LTXVideoOnlyModelConfigurator,
            )
            self.upsampler = VideoUpsampler(
                model_paths.video_vae(), spatial_upsampler_path, self.dtype, device, alloc_trim_strategy=AllocatorTrimStrategy.TRIM
            )
            self.video_decoder = VideoDecoder(
                model_paths.video_vae(),
                self.dtype,
                device,
                alloc_trim_strategy=AllocatorTrimStrategy.TRIM,
                diffvae_optimization=DiffVAEMode.CHUNKED_EAGER,
            )
            self.use_ancestral_sampler = should_use_ancestral_sampler(model_paths.transformer())

        def _stage_1_sampler_kwargs(self, seed: int) -> dict[str, Any]:
            if not self.use_ancestral_sampler:
                return {}
            return {
                "stepper": EulerAncestralDiffusionStep(eta=1.0, s_noise=1.0),
                "loop": __import__("functools").partial(
                    euler_ancestral_denoising_loop,
                    noise_seed=seed + 10000,
                    model_dtype=self.dtype,
                ),
            }

        @staticmethod
        def _conditionings(images: list[tuple[Any, str]], *, height: int, width: int, conditioner: Any, dtype: Any, device: Any) -> list[Any]:
            if not images:
                return []
            def build(encoder: Any) -> list[Any]:
                replace_images = [item for item, mode in images if mode == "replace"]
                guide_images = [item for item, mode in images if mode == "guide"]
                result: list[Any] = []
                if replace_images:
                    result.extend(image_conditionings_by_replacing_latent(replace_images, height, width, encoder, dtype, device))
                if guide_images:
                    result.extend(image_conditionings_by_adding_guiding_latent(guide_images, height, width, encoder, dtype, device))
                return result
            return conditioner(build)

        def _tiling_config(self, *, scale_factors: Any, height: int, width: int, frames: int) -> Any:
            auto_config = ensure_tiling_config(
                AUTO_TILING,
                scale_factors=scale_factors,
                vae_checkpoint_path=self.video_decoder.checkpoint_path,
                video_shape=VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=24.0),
                diffvae_optimization=self.video_decoder.diffvae_optimization,
                device=self.device,
            )
            if auto_config is None:
                return None
            # Keep the upstream spatial recommendation, but cap temporal VAE
            # work to a bounded frame tile. This is the decode working set, not
            # an inference window, so it does not change the requested output.
            bounded = TileSizeConfig(
                frames=DimensionSizeConfig(
                    tile_size=vae_temporal_tile_frames,
                    overlap=vae_temporal_overlap_frames,
                ),
                height=auto_config.height,
                width=auto_config.width,
            )
            return ensure_tiling_config(
                bounded,
                scale_factors=scale_factors,
                vae_checkpoint_path=self.video_decoder.checkpoint_path,
                video_shape=VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=24.0),
                diffvae_optimization=self.video_decoder.diffvae_optimization,
                device=self.device,
            )

        def __call__(self, *, prompt: str, seed: int, height: int, width: int, frame_rate: float, images: list[tuple[Any, str]], num_frames: int, stage_1_output: Path | None = None) -> Any:
            assert_resolution(height=height, width=width, is_two_stage=True)
            generator = torch.Generator(device=self.device).manual_seed(seed)
            noiser = GaussianNoiser(generator=generator)
            image_values = self.image_conditioner.resolve_crf([image for image, _mode in images])
            images = list(zip(image_values, (mode for _image, mode in images)))
            (ctx_p,) = self.prompt_encoder([prompt])
            video_context = ctx_p.video_encoding
            scale_factors = tiling_scale_factors_for_vae(self.video_decoder.checkpoint_path)
            tiling_config = self._tiling_config(
                scale_factors=scale_factors,
                height=height,
                width=width,
                frames=num_frames,
            )
            stage_1_h, stage_1_w = height // 2, width // 2
            stage_1_conditionings = self._conditionings(
                images, height=stage_1_h, width=stage_1_w, conditioner=self.image_conditioner,
                dtype=self.dtype, device=self.device,
            )
            video_state, _ = self.stage(
                denoiser=_StepProgressDenoiser(SimpleDenoiser(video_context, None), "stage_1_denoise", len(DISTILLED_SIGMAS) - 1, self.progress),
                sigmas=DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device),
                noiser=noiser,
                width=stage_1_w,
                height=stage_1_h,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(context=video_context, conditionings=stage_1_conditionings),
                audio=None,
                **self._stage_1_sampler_kwargs(seed),
            )
            if video_state is None:
                raise RuntimeError("LTX video-only stage 1 returned no video state")
            if stage_1_output is not None:
                _emit(self.progress, "decode_stage_1", 0.0, "Encoding retained stage 1 preview")
                stage_1_tiling = self._tiling_config(
                    scale_factors=scale_factors,
                    height=stage_1_h,
                    width=stage_1_w,
                    frames=num_frames,
                )
                stage_1_decoded = self.video_decoder(video_state.latent, stage_1_tiling, generator, dtype=self.dtype)
                encode_video(
                    video=stage_1_decoded,
                    fps=int(round(frame_rate)),
                    audio=None,
                    output_path=str(stage_1_output),
                    video_chunks_number=get_video_chunks_number(num_frames, stage_1_tiling),
                )
                _emit(self.progress, "decode_stage_1", 1.0, "Retained stage 1 preview")
            _emit(self.progress, "spatial_upscale", 0.0, "Upscaling stage 1 video latent")
            upscaled = self.upsampler(video_state.latent[:1])
            _emit(self.progress, "spatial_upscale", 1.0, "Latent spatial upscale complete")
            video_state, _ = self.stage(
                denoiser=_StepProgressDenoiser(SimpleDenoiser(video_context, None), "stage_2_refine", len(STAGE_2_DISTILLED_SIGMAS) - 1, self.progress),
                sigmas=STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=self.device),
                noiser=noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=video_context,
                    conditionings=self._conditionings(
                        images, height=height, width=width, conditioner=self.image_conditioner,
                        dtype=self.dtype, device=self.device,
                    ),
                    noise_scale=STAGE_2_DISTILLED_SIGMAS[0].item(),
                    initial_latent=upscaled,
                ),
                audio=None,
            )
            if video_state is None:
                raise RuntimeError("LTX video-only stage 2 returned no video state")
            _emit(self.progress, "decode_video", 0.0, "Decoding final video latent")
            decoded = self.video_decoder(video_state.latent, tiling_config, generator, dtype=self.dtype)
            return PipelineOutput(decoded, None, num_frames, tiling_config, None, video_state.latent)

    return VideoOnlyDistilledPipeline


class LtxVideoRuntime:
    def __init__(self, config: LtxVideoConfig):
        self.config = config
        self._residency_lock = threading.RLock()
        self._requires_reset = False
        self._last_cleanup_report: dict[str, Any] | None = None

    @property
    def last_cleanup_report(self) -> dict[str, Any] | None:
        return self._last_cleanup_report

    def residency_status(self) -> dict[str, Any]:
        return {
            "requiresReset": self._requires_reset,
            "memory": gpu_memory_snapshot(self.config.device).to_dict(),
            "lastCleanup": self._last_cleanup_report,
        }

    def reset_residency(self) -> dict[str, Any]:
        """Trim CUDA allocations and report whether a process restart is still needed."""
        with self._residency_lock:
            before = gpu_memory_snapshot(self.config.device)
            clear_cuda_cache()
            after = gpu_memory_snapshot(self.config.device)
            allocated_delta = before.allocated_bytes - after.allocated_bytes
            reserved_delta = before.reserved_bytes - after.reserved_bytes
            remaining_gb = after.allocated_bytes / (1024**3) if after.available else 0.0
            requires_process_restart = bool(
                after.available and remaining_gb > self.config.residency_reset_threshold_gb
            )
            report = {
                "before": before.to_dict(),
                "after": after.to_dict(),
                "releasedAllocatedGb": round(allocated_delta / (1024**3), 3),
                "releasedReservedGb": round(reserved_delta / (1024**3), 3),
                "requiresProcessRestart": requires_process_restart,
                "thresholdGb": self.config.residency_reset_threshold_gb,
            }
            self._requires_reset = requires_process_restart
            self._last_cleanup_report = report
            logger.info("LTX CUDA residency reset: %s", report)
            return report

    def preflight(self) -> dict[str, Any]:
        report = self.config.preflight()
        if self.config.enforce_nvfp4:
            snapshot = gpu_memory_snapshot(self.config.device)
            try:
                assert_blackwell_for_nvfp4(snapshot)
            except RuntimeError as exc:
                report["ready"] = False
                report["reason"] = str(exc)
                report["cuda"] = snapshot.to_dict()
        try:
            import torch

            report["attention"] = _configure_fast_attention(
                torch,
                enforce=self.config.enforce_fast_attention,
            )
            report["attentionPolicy"] = {
                "video": "flash_sdp_strict" if self.config.enforce_fast_attention else "torch_sdp_dispatch",
                "textEncoder": "sdpa_math_isolated",
            }
            cuda_report = report.get("cuda") if isinstance(report.get("cuda"), dict) else {}
            if self.config.enforce_fast_attention and cuda_report.get("available"):
                attention = report["attention"]
                if (
                    not attention.get("flashSdp")
                    or attention.get("memoryEfficientSdp")
                    or attention.get("mathSdp")
                    or attention.get("cudnnSdp")
                ):
                    report["ready"] = False
                    report["reason"] = report.get("reason") or "Flash SDP is not the only enabled CUDA attention backend"
        except Exception as exc:
            report["attention"] = {"errorType": type(exc).__name__, "error": str(exc)}
        report["residency"] = self.residency_status()
        if self._requires_reset:
            report["ready"] = False
            report["reason"] = report.get("reason") or (
                "GPU residency is above the safe reset threshold; call POST /v1/worker/reset "
                "or restart the worker process before submitting another job"
            )
        try:
            import ltx_core  # noqa: F401
            import ltx_pipelines  # noqa: F401
            report["ltxPackagesAvailable"] = True
        except Exception as exc:
            report["ltxPackagesAvailable"] = False
            report["packageError"] = f"{type(exc).__name__}: {exc}"
            report["ready"] = False
            report["reason"] = report.get("reason") or report["packageError"]
        if self.config.quantization.startswith("nvfp4"):
            try:
                from ltx_kernels import nvfp4

                kernels_ready = bool(nvfp4.is_available())
                report["ltxKernelsAvailable"] = kernels_ready
                if not kernels_ready:
                    report["ready"] = False
                    report["reason"] = report.get("reason") or f"ltx-kernels NVFP4 is unavailable: {nvfp4.unavailable_reason()}"
            except Exception as exc:
                report["ltxKernelsAvailable"] = False
                report["kernelError"] = f"{type(exc).__name__}: {exc}"
                report["ready"] = False
                report["reason"] = report.get("reason") or report["kernelError"]
        return report

    def generate(self, request: LtxVideoRequest, output_dir: Path, progress: ProgressCallback) -> LtxRuntimeResult:
        request.validate(self.config)
        if request.negative_prompt:
            raise LtxRuntimeError(
                "negative_prompt_not_supported",
                "The distilled LTX 2.5 pipeline has CFG=1 and does not support a negative prompt.",
                stage="validate_request",
            )
        if request.audio_mode == "generated" and any(item.mode == "replace" for item in request.conditioning_images):
            raise LtxRuntimeError(
                "conditioning_mode_not_supported",
                "generated-audio mode uses the upstream conditioning path; use guide conditioning or audio_mode=off/source for explicit replacement conditioning",
                stage="validate_request",
            )
        retain_intermediates = request.retain_intermediates if request.retain_intermediates is not None else self.config.keep_intermediates
        if request.audio_mode == "generated" and retain_intermediates:
            raise LtxRuntimeError(
                "audio_intermediate_not_supported",
                "stage-1 preview retention is currently supported for video-only/source-audio mode; disable retain_intermediates for generated-audio mode",
                stage="validate_request",
            )
        width, height = request.resolve_dimensions(self.config)
        resolved_aspect_ratio = request.resolved_aspect_ratio(self.config)
        frames = request.resolve_frames(self.config)
        seed = request.seed if request.seed is not None else __import__("secrets").randbelow(2**32)
        output_dir.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        current_stage = {"name": "validate_request"}
        memory_phases: dict[str, dict[str, Any]] = {}

        def report(stage: str, fraction: float, message: str, details: dict[str, Any] | None = None) -> None:
            current_stage["name"] = stage
            progress(stage, fraction, message, details)

        def capture_memory(name: str) -> dict[str, Any]:
            snapshot = gpu_memory_snapshot(self.config.device).to_dict()
            memory_phases[name] = snapshot
            return snapshot

        memory_plan = build_memory_plan(
            width=width,
            height=height,
            frames=frames,
            audio=request.audio_mode == "generated",
            offload_mode=self.config.offload_mode,
            quantization=self.config.quantization,
            reserve_vram_gb=self.config.reserve_vram_gb,
            device=self.config.device,
            conditioning_count=len(request.conditioning_images),
            vae_temporal_tile_frames=self.config.vae_temporal_tile_frames,
        )
        if not memory_plan["gpu"].get("available", False) and self.config.gpu_required:
            raise LtxRuntimeError("cuda_unavailable", "CUDA is required for LTX video generation", stage="preflight_memory", details=memory_plan)
        if memory_plan["gpu"].get("available") and not memory_plan["fitsCurrentFreeMemory"]:
            raise LtxRuntimeError("vram_budget_exceeded", "LTX job exceeds the current free VRAM budget", stage="preflight_memory", details=memory_plan)
        report("preflight_memory", 1.0, "GPU memory plan accepted", memory_plan)
        imports = _parse_ltx_imports()
        torch = imports["torch"]
        device = torch.device(self.config.device)
        attention = _configure_fast_attention(torch, enforce=self.config.enforce_fast_attention)
        report("preflight_memory", 1.0, "CUDA attention policy selected", attention)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        memory_before_inference = capture_memory("before_inference")
        model_paths = imports["ModelPaths"].from_split(
            transformer_path=str(self.config.transformer_path),
            text_encoder_path=str(self.config.text_encoder_path),
            video_vae_path=str(self.config.video_vae_path),
            audio_vae_path=str(self.config.audio_vae_path) if request.audio_mode == "generated" and self.config.audio_vae_path else None,
        )
        report("acquire_inputs", 0.0, "Acquiring conditioning inputs")
        attempt_dir = output_dir / "inputs"
        normalized_images: list[tuple[Any, str]] = []
        for index, item in enumerate(request.conditioning_images):
            source = acquire_input(item.source_url, item.path, attempt_dir / f"conditioning-{index:02d}-source", self.config)
            normalized = attempt_dir / f"conditioning-{index:02d}.png"
            from .media import normalize_image

            normalize_image(source, normalized, width=width, height=height)
            normalized_images.append((
                imports["ImageConditioningInput"](
                    path=str(normalized), frame_idx=min(item.frame_index, frames - 1), strength=item.strength, crf=item.crf
                ),
                item.mode,
            ))
            report("normalize_inputs", (index + 1) / max(1, len(request.conditioning_images)), f"Normalized conditioning image {index + 1}/{len(request.conditioning_images)}")
        source_audio: Path | None = None
        if request.audio_mode == "source":
            source_audio = acquire_input(request.source_audio_url, request.source_audio_path, attempt_dir / "source-audio", self.config)
        report("acquire_inputs", 1.0, "Inputs acquired")
        report("normalize_inputs", 1.0, "Inputs normalized")
        loras: list[Any] = []
        for index, item in enumerate(request.loras):
            path = awaitable_acquire_lora(item, attempt_dir, self.config)
            loras.append(imports["LoraPathStrengthAndSDOps"](str(path), float(item.get("strength", 1.0)), imports["LTXV_LORA_COMFY_RENAMING_MAP"]))
            logger.info("LTX LoRA prepared index=%d path=%s strength=%s", index, path, item.get("strength", 1.0))
        report("load_models", 0.0, "Loading LTX 2.5 model components", {"audioMode": request.audio_mode, "quantization": self.config.quantization, "offloadMode": self.config.offload_mode})
        pipeline: Any | None = None
        result: Any | None = None
        video_stream: Any | None = None
        metadata: dict[str, Any] = {}
        quantization: Any | None = None
        offload_mode: Any | None = None
        compilation: Any | None = None
        try:
            quantization = build_quantization(imports, self.config)
            offload_mode = imports["OffloadMode"](self.config.offload_mode)
            compilation = imports["CompilationConfig"]() if self.config.compile_transformer else None
            if request.audio_mode == "generated":
                if not self.config.audio_vae_path:
                    raise LtxRuntimeError("audio_model_missing", "generated audio requires an audio VAE path", stage="load_models")
                pipeline = imports["DistilledPipeline"](
                    model_paths=model_paths,
                    spatial_upsampler_path=str(self.config.spatial_upsampler_path),
                    loras=loras,
                    device=device,
                    quantization=quantization,
                    offload_mode=offload_mode,
                    compilation_config=compilation,
                )
                pipeline.prompt_encoder = _ScopedPromptEncoder(
                    pipeline.prompt_encoder,
                    torch,
                    enforce_fast_attention=self.config.enforce_fast_attention,
                    progress=report,
                )
                pipeline.stage = _ProgressDiffusionStage(pipeline.stage, report)
            else:
                pipeline_cls = _video_only_pipeline_factory(
                    imports,
                    enforce_fast_attention=self.config.enforce_fast_attention,
                    vae_temporal_tile_frames=self.config.vae_temporal_tile_frames,
                    vae_temporal_overlap_frames=self.config.vae_temporal_overlap_frames,
                )
                pipeline = pipeline_cls(
                    model_paths=model_paths,
                    spatial_upsampler_path=str(self.config.spatial_upsampler_path),
                    loras=loras,
                    device=device,
                    quantization=quantization,
                    offload_mode=offload_mode,
                    compilation_config=compilation,
                    progress=report,
                )
            report("load_models", 1.0, "LTX model components loaded", {"memory": capture_memory("after_model_load")})
            report("encode_prompt", 0.0, "Encoding prompt")
            # The upstream call includes prompt encoding; this event brackets that work.
            stage_1_output = output_dir / "stage-1.mp4" if retain_intermediates and request.audio_mode != "generated" else None
            pipeline_images = (
                [image for image, _mode in normalized_images]
                if request.audio_mode == "generated"
                else normalized_images
            )
            # The upstream LTX VAE uses a custom convolution path that can
            # attempt to save an input tensor for backward even during its
            # inference-only decode. ``inference_mode`` creates immutable
            # inference tensors and makes that path fail; ``no_grad`` keeps
            # autograd disabled without changing tensor semantics.
            # The pipeline returns a lazy VAE decoder iterator. Keep the
            # no-grad scope open until that iterator has been consumed; ending
            # it after pipeline() silently enabled autograd for every decoder
            # feature map and caused the 46.6 GB allocator spike.
            with torch.no_grad():
                result = pipeline(
                    prompt=request.prompt,
                    seed=seed,
                    height=height,
                    width=width,
                    frame_rate=request.frame_rate,
                    images=pipeline_images,
                    num_frames=frames,
                    **({"stage_1_output": stage_1_output} if stage_1_output is not None else {}),
                )
                report("encode_prompt", 1.0, "Prompt encoded")
                report("stage_1_denoise", 1.0, "Stage 1 denoising complete")
                report("spatial_upscale", 1.0, "Latent spatial upscale complete")
                report("stage_2_refine", 1.0, "Stage 2 refinement complete")
                report("decode_video", 0.0, "Decoding video frames", {"memory": capture_memory("before_decode")})
                render_mp4 = output_dir / "render.mp4"
                video_stream = result.video
                try:
                    imports["encode_video"](
                        video=video_stream,
                        fps=int(round(request.frame_rate)),
                        audio=result.audio if request.audio_mode == "generated" else None,
                        output_path=str(render_mp4),
                        video_chunks_number=imports["get_video_chunks_number"](result.num_frames, result.tiling_config),
                    )
                finally:
                    close_stream = getattr(video_stream, "close", None)
                    if callable(close_stream):
                        close_stream()
                    video_stream = None
                report("decode_video", 1.0, "Video decoded and encoded", {"memory": capture_memory("after_decode")})
            audio_path: Path | None = None
            if request.audio_mode == "generated" and result.audio is not None:
                audio_path = output_dir / "audio.wav"
                imports["encode_audio"](result.audio, str(audio_path))
                report("decode_audio", 1.0, "Generated audio decoded")
            elif request.audio_mode == "source" and source_audio is not None:
                report("mux_audio", 0.0, "Muxing source audio")
                audio_path = source_audio
                muxed = output_dir / "render-with-audio.mp4"
                mux_audio(render_mp4, source_audio, muxed, output_format="mp4")
                render_mp4 = muxed
                report("mux_audio", 1.0, "Source audio muxed")
            final_path = output_dir / f"output.{request.output_format}"
            if request.output_format == "mp4":
                shutil.copy2(render_mp4, final_path)
            else:
                transcode_video(render_mp4, final_path, output_format="webm")
            report("encode_output", 1.0, f"Wrote {final_path.name}")
            conditioning_manifest = {
                "schemaVersion": 1,
                "target": {"width": width, "height": height, "aspectRatio": resolved_aspect_ratio},
                "items": [
                    {**item.to_dict(), "normalizedPath": str(output_dir / "inputs" / f"conditioning-{index:02d}.png")}
                    for index, item in enumerate(request.conditioning_images)
                ],
            }
            (output_dir / "conditioning-manifest.json").write_text(json.dumps(conditioning_manifest, indent=2, default=str) + "\n", encoding="utf-8")
            metadata = {
                "schemaVersion": 1,
                "runtime": "ltx-video",
                "model": self.config.model_name,
                "pipeline": "distilled-audio-video" if request.audio_mode == "generated" else "distilled-video-only",
                "prompt": request.prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "aspectRatio": resolved_aspect_ratio,
                "frameRate": request.frame_rate,
                "numFrames": frames,
                "durationSeconds": frames / request.frame_rate,
                "audioMode": request.audio_mode,
                "conditioning": [item.to_dict() for item in request.conditioning_images],
                "conditioningCount": len(request.conditioning_images),
                "conditioningManifest": "conditioning-manifest.json",
                "loras": list(request.loras),
                "quantization": self.config.quantization,
                "offloadMode": self.config.offload_mode,
                "attention": attention,
                "textEncoderAttention": "sdpa_math_isolated",
                "memoryPlan": memory_plan,
                "memoryBeforeInference": memory_before_inference,
                "memoryAfter": gpu_memory_snapshot(self.config.device).to_dict(),
                "memoryPhases": memory_phases,
                "vaeTemporalTileFrames": self.config.vae_temporal_tile_frames,
                "vaeTemporalOverlapFrames": self.config.vae_temporal_overlap_frames,
                "decodeTiling": _tiling_to_dict(result.tiling_config),
                "elapsedSeconds": round(time.monotonic() - started, 3),
                "videoOnlyTransformer": request.audio_mode != "generated",
                "retainIntermediates": bool(retain_intermediates),
            }
            (output_dir / "generation-metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
            report("finalize", 1.0, "LTX video generation finalized", metadata)
            intermediate_paths = (stage_1_output,) if stage_1_output is not None and stage_1_output.is_file() else ()
            return LtxRuntimeResult(final_path, audio_path, metadata, intermediate_paths)
        except LtxRuntimeError:
            raise
        except Exception as exc:
            raise LtxRuntimeError(
                "ltx_inference_failed",
                f"LTX inference failed: {type(exc).__name__}: {exc}",
                stage=current_stage["name"],
                details={"memoryAfter": gpu_memory_snapshot(self.config.device).to_dict(), "memoryPhases": memory_phases},
            ) from exc
        finally:
            # Explicitly drop model and tensor owners. Mutating locals() is not
            # guaranteed to update Python's fast-local storage.
            close_stream = getattr(video_stream, "close", None)
            if callable(close_stream):
                close_stream()
            video_stream = None
            pipeline = None
            result = None
            quantization = None
            offload_mode = None
            compilation = None
            loras.clear()
            gc.collect()
            cleanup_report = self.reset_residency()
            if metadata:
                metadata["cleanup"] = cleanup_report
                (output_dir / "generation-metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


def build_quantization(imports: dict[str, Any], config: LtxVideoConfig) -> Any:
    kind = config.quantization
    if kind == "none":
        if config.enforce_nvfp4:
            raise LtxRuntimeError("quantization_policy_invalid", "NVFP4 is required by the production runtime", stage="preflight_memory")
        return None
    try:
        from ltx_pipelines.utils.quantization_factory import QuantizationKind

        return QuantizationKind(kind).to_policy(checkpoint_path=str(config.transformer_path))
    except Exception as exc:
        raise LtxRuntimeError(
            "quantization_unavailable",
            f"could not build LTX quantization policy {kind}: {type(exc).__name__}: {exc}",
            stage="load_models",
        ) from exc


def awaitable_acquire_lora(item: dict[str, Any], attempt_dir: Path, config: LtxVideoConfig) -> Path:
    raw_path = item.get("path")
    local = Path(str(raw_path)).expanduser() if raw_path else None
    source_url = str(item.get("sourceUrl") or item.get("source_url") or item.get("url") or "")
    name = Path(str(item.get("filename") or f"lora-{len(list(attempt_dir.glob('lora-*'))):02d}.safetensors")).name
    return acquire_input(source_url, local, attempt_dir / name, config)


def create_runtime(config: LtxVideoConfig | None = None) -> LtxVideoRuntime:
    return LtxVideoRuntime(config or LtxVideoConfig.from_env())
