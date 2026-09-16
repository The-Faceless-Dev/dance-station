# Qwen Image Diffusers Runtime

## Objective

Replace the failing stable-diffusion.cpp CUDA denoising path with the official
Diffusers `QwenImagePipeline` implementation, while preserving the existing
worker HTTP contract and the native image worker as a rollback path.

## Approach

1. Add a Diffusers runtime selected by `QWEN_IMAGE_RUNTIME=diffusers`.
2. Load `Qwen/Qwen-Image-2512` with CUDA and `torch.bfloat16`; support a
   request-independent 4-bit BitsAndBytes mode for lower-VRAM machines.
3. Use Accelerate model CPU offload when enabled so the text encoder,
   transformer, and VAE are not resident on the GPU simultaneously. The
   denoising transformer still executes on CUDA; CUDA is a hard preflight
   requirement for this path.
4. Preserve request-level prompt, negative prompt, dimensions, steps, CFG,
   seed, and multiple Diffusers LoRA inputs. Emit detailed stage, memory,
   timing, and denoising-step events to the existing durable artifact logs.
5. Add a separate Diffusers container target and dependency set. It will not
   replace the current native model-bearing image until the CUDA output has
   been verified.
6. Run static/unit checks locally, then run an actual CUDA generation in a
   disposable GPU environment and retain the request, logs, metadata, and PNG
   under `tmp/`. Destroy any paid test instance after the artifacts are saved.

## Defaults and tradeoffs

- The new target defaults to the official model repository and 4-bit
  BitsAndBytes loading so it can run within a 24--48 GB CUDA environment.
  `QWEN_IMAGE_DIFFUSERS_QUANTIZATION=none` selects full BF16 weights when the
  host has enough memory and is the quality comparison setting.
- CUDA SDPA is the portable default. An explicit Diffusers attention backend
  can be selected by environment variable, but an unavailable backend is a
  startup error rather than a silent CPU fallback.
- LoRAs are loaded only for the current job and are unloaded before the next
  job where the installed Diffusers version supports it.
- The native runtime and Dockerfile remain untouched for rollback.

## Risks

- Diffusers cannot consume the existing native GGUF transformer directly;
  the official Diffusers safetensors repository must be available locally or
  through the configured Hugging Face cache.
- Quantization and attention backend support depends on the installed PyTorch,
  Diffusers, Transformers, and bitsandbytes versions. The worker records all
  versions and CUDA details in preflight and job metadata.
- Full BF16 loading may require more system RAM than the existing GGUF image;
  the default 4-bit path avoids loading all compute weights into GPU memory at
  once.

## Verification

- `pytest` for request parsing, config validation, and runtime selection.
- Python compilation and container import checks without model loading.
- CUDA smoke generation with a small valid resolution and finite/nonuniform
  PNG validation.
- Production-shaped request through `/process`, including artifact upload
  contract, before considering the replacement usable.

## Persistent Image Correction (2026-09-16)

The successful 2026-09-13 RunPod proof used a temporary startup bootstrap on a
RunPod PyTorch image. It proved the Diffusers inference path, but it was not a
published worker image. The persistent Qwen-Image-2512 target must therefore
install the complete Python runtime in its Docker build and start the project
worker directly; it must not depend on a `dockerStartCmd` bootstrap or on a
commit-message-selected native Dockerfile.

The model-bearing target continues to bake the verified Q8 transformer into
the image. Official Diffusers pipeline files may be populated in the image's
configured Hugging Face cache as part of image assembly or mounted cache, but
runtime installation and worker startup must be self-contained. The Docker
build also avoids recursively changing ownership of the virtualenv, since the
base image's venv contains interpreter symlinks that made the previous CI
build fail in its final setup command.
