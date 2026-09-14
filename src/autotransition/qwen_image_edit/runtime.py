from __future__ import annotations

import gc
import importlib.metadata
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .config import QwenImageEditConfig
from .contracts import QwenImageEditRequest
from .references import PreparedReference


ProgressCallback = Callable[[str, float, str], None]


class QwenImageEditDiffusersRuntime:
    def __init__(self, config: QwenImageEditConfig, *, emit: Callable[..., Any] | None = None):
        self.config = config
        self.emit = emit or (lambda *_args, **_kwargs: None)
        self._pipe: Any | None = None
        self._lock = threading.RLock()

    def preflight(self) -> dict[str, Any]:
        report = self.config.preflight()
        if not report.get("ready"):
            return report
        try:
            import torch

            report["runtimeVersions"] = {
                "python": os.sys.version.split()[0],
                "torch": torch.__version__,
                "torchCuda": torch.version.cuda,
                "diffusers": _package_version("diffusers"),
                "transformers": _package_version("transformers"),
                "accelerate": _package_version("accelerate"),
                "bitsandbytes": _package_version("bitsandbytes"),
            }
            report["cuda"] = {
                "available": bool(torch.cuda.is_available()),
                "deviceCount": torch.cuda.device_count(),
                "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            }
            if self.config.gpu_required and not report["cuda"]["available"]:
                report["ready"] = False
                report["diagnostics"].append("PyTorch cannot access a CUDA device")
        except Exception as exc:
            report["ready"] = False
            report["diagnostics"].append(f"Diffusers/PyTorch preflight failed: {exc}")
        return report

    def generate(
        self,
        request: QwenImageEditRequest,
        references: tuple[PreparedReference, ...],
        lora_paths: tuple[Any, ...],
        output_path: Path,
        progress: ProgressCallback,
    ) -> dict[str, Any]:
        with self._lock:
            progress("load_model", 0.02, "Loading the Qwen-Image-Edit-2511 Diffusers pipeline")
            pipe = self._ensure_pipeline()
            progress("encode_prompt", 0.12, "Encoding the edit prompt and reference images")
            loaded_adapters = self._load_loras(pipe, lora_paths)
            images = []
            try:
                import torch
                from PIL import Image, ImageOps

                images = [
                    ImageOps.exif_transpose(Image.open(item.path)).convert("RGB")
                    for item in references
                ]
                device = torch.device("cuda")
                generator = torch.Generator(device=device)
                if request.seed is not None:
                    generator.manual_seed(request.seed)
                else:
                    request_seed = generator.initial_seed()
                seed = request.seed if request.seed is not None else request_seed
                memory_before = _cuda_memory(torch)
                started = time.monotonic()
                self.emit(
                    "diffusers_edit_inference_started",
                    modelId=self.config.diffusers_model_id,
                    transformer=str(self.config.transformer),
                    quantization="Q8_0_GGUF",
                    dtype=self.config.diffusers_dtype,
                    computeDtype=self.config.diffusers_compute_dtype,
                    attentionBackend=self.config.diffusers_attention_backend,
                    cpuOffload=self.config.diffusers_cpu_offload,
                    width=request.width,
                    height=request.height,
                    steps=request.steps,
                    cfgScale=request.cfg_scale,
                    referenceCount=len(images),
                    referenceSizes=[list(image.size) for image in images],
                    seed=seed,
                    loraCount=len(lora_paths),
                    memoryBefore=memory_before,
                )

                def on_step_end(
                    _pipeline: Any,
                    step_index: int,
                    timestep: Any,
                    _callback_kwargs: dict[str, Any],
                ) -> dict[str, Any]:
                    fraction = 0.15 + (0.72 * (step_index + 1) / max(request.steps, 1))
                    memory = _cuda_memory(torch)
                    self.emit(
                        "diffusers_edit_denoise_step",
                        step=step_index + 1,
                        totalSteps=request.steps,
                        timestep=str(timestep),
                        progress=fraction,
                        memory=memory,
                    )
                    progress("denoise", fraction, f"Qwen Edit denoising step {step_index + 1}/{request.steps}")
                    return {}

                call_kwargs: dict[str, Any] = {
                    "image": images,
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt or " ",
                    "width": request.width,
                    "height": request.height,
                    "num_inference_steps": request.steps,
                    "true_cfg_scale": request.cfg_scale,
                    "guidance_scale": 1.0,
                    "generator": generator,
                    "callback_on_step_end": on_step_end,
                }
                self.emit(
                    "diffusers_edit_request",
                    request={**call_kwargs, "image": {"count": len(images)}, "generator": {"seed": seed}},
                )
                result = pipe(**call_kwargs)
                progress("decode", 0.9, "Diffusers VAE decoded the edited image")
                image = result.images[0] if getattr(result, "images", None) else None
                if image is None:
                    raise RuntimeError("Qwen Image Edit pipeline returned no image")
                if tuple(image.size) != (request.width, request.height):
                    raise RuntimeError(
                        f"Qwen Image Edit returned {image.size[0]}x{image.size[1]}, "
                        f"expected {request.width}x{request.height}"
                    )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(output_path, format="PNG", optimize=False)
                elapsed = time.monotonic() - started
                memory_after = _cuda_memory(torch)
                self.emit(
                    "diffusers_edit_inference_completed",
                    elapsedSeconds=elapsed,
                    output=str(output_path),
                    outputBytes=output_path.stat().st_size,
                    memoryBefore=memory_before,
                    memoryAfter=memory_after,
                )
                return {
                    "runtime": "diffusers",
                    "modelId": self.config.diffusers_model_id,
                    "revision": self.config.diffusers_revision,
                    "transformer": str(self.config.transformer),
                    "quantization": "Q8_0_GGUF",
                    "dtype": self.config.diffusers_dtype,
                    "computeDtype": self.config.diffusers_compute_dtype,
                    "attentionBackend": self.config.diffusers_attention_backend,
                    "cpuOffload": self.config.diffusers_cpu_offload,
                    "seed": seed,
                    "steps": request.steps,
                    "cfgScale": request.cfg_scale,
                    "width": request.width,
                    "height": request.height,
                    "referenceCount": len(references),
                    "loraCount": len(lora_paths),
                    "elapsedSeconds": elapsed,
                    "memoryBefore": memory_before,
                    "memoryAfter": memory_after,
                }
            finally:
                for image in images:
                    image.close()
                if loaded_adapters:
                    self._unload_loras(pipe, loaded_adapters)
                gc.collect()

    def close(self) -> None:
        with self._lock:
            self._pipe = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception as exc:
                self.emit("diffusers_cleanup_failed", errorType=type(exc).__name__, error=str(exc))

    def _ensure_pipeline(self) -> Any:
        if self._pipe is not None:
            return self._pipe
        import torch
        from diffusers import GGUFQuantizationConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel

        self.config.diffusers_cache.mkdir(parents=True, exist_ok=True)
        dtype = _torch_dtype(torch, self.config.diffusers_dtype)
        self.emit(
            "diffusers_edit_pipeline_loading",
            modelId=self.config.diffusers_model_id,
            revision=self.config.diffusers_revision,
            transformer=str(self.config.transformer),
            dtype=self.config.diffusers_dtype,
            quantization="Q8_0_GGUF",
            cacheDir=str(self.config.diffusers_cache),
        )
        transformer = QwenImageTransformer2DModel.from_single_file(
            str(self.config.transformer),
            config=self.config.diffusers_model_id,
            subfolder="transformer",
            quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
            torch_dtype=dtype,
        )
        self.emit("diffusers_edit_transformer_loaded", path=str(self.config.transformer), format="GGUF", quantization="Q8_0")
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            self.config.diffusers_model_id,
            transformer=transformer,
            torch_dtype=dtype,
            cache_dir=str(self.config.diffusers_cache),
            revision=self.config.diffusers_revision,
        )
        if self.config.diffusers_cpu_offload:
            pipe.enable_model_cpu_offload()
            placement = "model_cpu_offload"
        else:
            pipe.to("cuda")
            placement = "cuda"
        self._set_attention_backend(pipe)
        if self.config.diffusers_compile:
            pipe.transformer = torch.compile(pipe.transformer, fullgraph=True)
            self.emit("diffusers_edit_transformer_compiled")
        self._pipe = pipe
        self.emit(
            "diffusers_edit_pipeline_ready",
            placement=placement,
            components={
                "transformer": type(pipe.transformer).__name__,
                "textEncoder": type(pipe.text_encoder).__name__,
                "vae": type(pipe.vae).__name__,
            },
        )
        return pipe

    def _set_attention_backend(self, pipe: Any) -> None:
        backend = self.config.diffusers_attention_backend
        if backend == "sdpa":
            self.emit("diffusers_edit_attention_backend_selected", backend="sdpa")
            return
        names = {"flash": "flash", "flash_2": "flash", "flash_3": "_flash_3_hub"}
        if backend not in names:
            raise RuntimeError(f"unsupported Diffusers attention backend: {backend}")
        try:
            pipe.transformer.set_attention_backend(names[backend])
        except Exception as exc:
            raise RuntimeError(f"requested Diffusers attention backend {backend!r} is unavailable: {exc}") from exc
        self.emit("diffusers_edit_attention_backend_selected", backend=backend, implementation=names[backend])

    def _load_loras(self, pipe: Any, lora_paths: tuple[Any, ...]) -> list[str]:
        loaded: list[str] = []
        for index, lora in enumerate(lora_paths):
            path = Path(str(lora.path))
            adapter_name = f"job_{index}_{path.stem}"
            self.emit("diffusers_edit_lora_loading", index=index, path=str(path), scale=lora.scale)
            pipe.load_lora_weights(str(path.parent), weight_name=path.name, adapter_name=adapter_name)
            loaded.append(adapter_name)
        if loaded:
            pipe.set_adapters(loaded, adapter_weights=[float(item.scale) for item in lora_paths])
            self.emit("diffusers_edit_loras_ready", adapters=loaded)
        return loaded

    def _unload_loras(self, pipe: Any, loaded: list[str]) -> None:
        try:
            pipe.unload_lora_weights()
            self.emit("diffusers_edit_loras_unloaded", adapters=loaded)
        except Exception as exc:
            self.emit("diffusers_edit_lora_unload_failed", adapters=loaded, errorType=type(exc).__name__, error=str(exc))


def _torch_dtype(torch: Any, name: str) -> Any:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _cuda_memory(torch: Any) -> dict[str, int | bool]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "allocatedBytes": int(torch.cuda.memory_allocated()),
        "reservedBytes": int(torch.cuda.memory_reserved()),
        "maxAllocatedBytes": int(torch.cuda.max_memory_allocated()),
        "maxReservedBytes": int(torch.cuda.max_memory_reserved()),
    }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None
