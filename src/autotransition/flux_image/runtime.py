from __future__ import annotations

import gc
import json
import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .config import FluxImageConfig
from .contracts import FluxImageRequest


ProgressCallback = Callable[[str, float, str], None]


def _dtype(torch: Any, name: str) -> Any:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


@dataclass
class _Components:
    torch: Any
    model: Any
    autoencoder: Any
    text_encoder: Any


class FluxImageRuntime:
    """Native FLUX.2 Klein 4B runtime with process-safe job cleanup."""

    def __init__(self, config: FluxImageConfig, *, emit: Callable[..., Any] | None = None):
        self.config = config
        self.emit = emit or (lambda *_args, **_kwargs: None)
        self._components: _Components | None = None

    @staticmethod
    def _cuda_memory(torch: Any, device: Any) -> dict[str, Any]:
        if not str(device).startswith("cuda"):
            return {"device": str(device), "available": False}
        index = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        properties = torch.cuda.get_device_properties(index)
        return {
            "device": str(device),
            "deviceIndex": index,
            "deviceName": properties.name,
            "totalBytes": int(total_bytes),
            "freeBytes": int(free_bytes),
            "usedBytes": int(total_bytes - free_bytes),
            "allocatedBytes": int(torch.cuda.memory_allocated(index)),
            "reservedBytes": int(torch.cuda.memory_reserved(index)),
        }

    @classmethod
    def _cuda_probe(cls, torch: Any, device: Any) -> dict[str, Any]:
        """Exercise the CUDA context without loading model weights."""
        if not str(device).startswith("cuda"):
            return {"ready": True, "available": False, "device": str(device)}
        try:
            if not torch.cuda.is_available():
                return {"ready": False, "available": False, "device": str(device), "error": "torch.cuda.is_available() is false"}
            torch.cuda.init()
            before = cls._cuda_memory(torch, device)
            probe = torch.empty((1024, 1024), dtype=torch.float32, device=device)
            probe.fill_(1.0)
            torch.cuda.synchronize(device)
            del probe
            torch.cuda.empty_cache()
            after = cls._cuda_memory(torch, device)
            return {"ready": True, "available": True, "torchCuda": torch.version.cuda, "before": before, "after": after}
        except Exception as exc:
            return {
                "ready": False,
                "available": bool(torch.cuda.is_available()),
                "device": str(device),
                "errorType": type(exc).__name__,
                "error": str(exc),
            }

    def preflight(self) -> dict[str, Any]:
        report = self.config.preflight()
        for name, path in self.config.resolved_paths().items():
            if path and path.is_file():
                report.setdefault("files", {})[name] = {"path": str(path), "sizeBytes": path.stat().st_size}
        if report["ready"] and self.config.device.startswith("cuda"):
            try:
                import torch

                cuda = self._cuda_probe(torch, torch.device(self.config.device))
            except Exception as exc:
                cuda = {"ready": False, "available": False, "errorType": type(exc).__name__, "error": str(exc)}
            report["cuda"] = cuda
            report["ready"] = bool(cuda.get("ready"))
        return report

    def _load_components(self, progress: ProgressCallback) -> _Components:
        if self._components is not None:
            return self._components
        report = self.preflight()
        if not report["ready"]:
            raise RuntimeError("FLUX model preflight failed: " + json.dumps(report, sort_keys=True))
        try:
            import torch
            from flux2.text_encoder import Qwen3Embedder
            from flux2.util import load_ae, load_flow_model
        except Exception as exc:
            raise RuntimeError("pinned official FLUX.2 runtime dependencies are unavailable") from exc
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("FLUX worker requires CUDA but torch.cuda.is_available() is false")

        os.environ["KLEIN_4B_MODEL_PATH"] = str(self.config.model_path)
        os.environ["FLUX2_TEXT_ENCODER_PATH"] = str(self.config.text_encoder)
        os.environ["AE_MODEL_PATH"] = str(self.config.vae)
        device = torch.device(self.config.device)
        cuda_probe = self._cuda_probe(torch, device)
        self.emit("cuda_preflight", **cuda_probe)
        if not cuda_probe.get("ready"):
            raise RuntimeError("FLUX CUDA preflight failed: " + json.dumps(cuda_probe, sort_keys=True, default=str))
        encoder_device = "cpu" if self.config.cpu_offload else device
        progress("load_model", 0.05, "Loading FLUX.2 Klein 4B")
        self.emit(
            "model_load_started",
            model=self.config.model_name,
            modelPath=str(self.config.model_path),
            textEncoderPath=str(self.config.text_encoder),
            vaePath=str(self.config.vae),
            device=str(device),
            dtype=self.config.dtype,
            cpuOffload=self.config.cpu_offload,
        )
        text_encoder = Qwen3Embedder(model_spec=str(self.config.text_encoder), device=encoder_device)
        model = load_flow_model(self.config.model_name, device="cpu" if self.config.cpu_offload else device)
        progress("load_model", 0.70, "Loading FLUX VAE")
        autoencoder = load_ae(self.config.model_name, device=device)
        model.eval()
        text_encoder.eval()
        autoencoder.eval()
        self._components = _Components(torch, model, autoencoder, text_encoder)
        self.emit("model_load_finished", model=self.config.model_name, cpuOffload=self.config.cpu_offload, cudaMemory=self._cuda_memory(torch, device))
        progress("load_model", 1.0, "FLUX.2 Klein components loaded")
        return self._components

    @staticmethod
    def _reject_loras(request: FluxImageRequest) -> None:
        if request.loras:
            raise RuntimeError(
                "FLUX.2 Klein distilled does not expose native LoRA inference; "
                "use a Klein Base worker profile for LoRA jobs"
            )

    def generate(self, request: FluxImageRequest, output_path: Path, progress: ProgressCallback) -> dict[str, Any]:
        components = self._load_components(progress)
        torch = components.torch
        device = torch.device(self.config.device)
        model = components.model
        text_encoder = components.text_encoder
        autoencoder = components.autoencoder
        moved_model = False
        try:
            self._reject_loras(request)
            if request.negative_prompt:
                self.emit("negative_prompt_ignored", reason="distilled Klein uses fixed guidance")
            if request.true_cfg_scale != 1.0:
                self.emit("true_cfg_scale_ignored", value=request.true_cfg_scale, effective=1.0)
            from einops import rearrange
            from flux2.sampling import batched_prc_img, batched_prc_txt, denoise, encode_image_refs, get_schedule, scatter_ids
            from PIL import Image

            if self.config.cpu_offload:
                progress("load_model", 0.05, "Moving FLUX transformer to CUDA")
                model = model.to(device)
                moved_model = True
            progress("encode_prompt", 0.0, "Encoding prompt with Qwen3-4B")
            ctx = text_encoder([request.prompt]).to(dtype=torch.bfloat16)
            ctx, ctx_ids = batched_prc_txt(ctx)
            if self.config.cpu_offload:
                text_encoder.cpu()
                ctx = ctx.to(device)
                ctx_ids = ctx_ids.to(device)
                torch.cuda.empty_cache()
            progress("encode_prompt", 1.0, "Prompt encoded")
            seed = request.seed if request.seed is not None else int.from_bytes(os.urandom(8), "big")
            generator = torch.Generator(device=device).manual_seed(seed)
            shape = (1, 128, request.height // 16, request.width // 16)
            noise = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=device)
            latents, image_ids = batched_prc_img(noise)
            ref_tokens, ref_ids = encode_image_refs(autoencoder, [])
            timesteps = get_schedule(request.steps, latents.shape[1])
            self.emit(
                "inference_started",
                seed=seed,
                steps=request.steps,
                width=request.width,
                height=request.height,
                cpuOffload=self.config.cpu_offload,
                cudaMemory=self._cuda_memory(torch, device),
            )
            progress("denoise", 0.05, f"Denoising {request.steps} FLUX steps")
            with torch.inference_mode():
                latents = denoise(
                    model, latents, image_ids, ctx, ctx_ids, timesteps=timesteps, guidance=1.0,
                    img_cond_seq=ref_tokens, img_cond_seq_ids=ref_ids,
                )
            progress("denoise", 1.0, "Denoising complete")
            if self.config.cpu_offload and moved_model:
                progress("decode", 0.0, "Moving FLUX transformer off CUDA before decode")
                model.to("cpu")
                moved_model = False
                torch.cuda.empty_cache()
            progress("decode", 0.0, "Decoding image")
            with torch.inference_mode():
                latents = torch.cat(scatter_ids(latents, image_ids)).squeeze(2)
                image = autoencoder.decode(latents).float().clamp(-1, 1)
            image = rearrange(image[0], "c h w -> h w c")
            output = Image.fromarray((127.5 * (image + 1.0)).cpu().byte().numpy())
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output.save(output_path, format="PNG")
            progress("validate_output", 0.90, "Model PNG written; canonical avatar framing pending")
            self.emit("inference_finished", seed=seed, output=str(output_path), width=output.width, height=output.height)
            return {
                "width": output.width,
                "height": output.height,
                "seed": seed,
                "steps": request.steps,
                "model": self.config.model_name,
                "negativePromptApplied": False,
                "trueCfgScaleApplied": False,
                "loraCount": 0,
            }
        except Exception as exc:
            self.emit("inference_failed", errorType=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            if self.config.cpu_offload and moved_model:
                model.to("cpu")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
