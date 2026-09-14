# Qwen Image Edit 2511 Worker

## Objective

Publish a public, model-bearing worker image for `Qwen/Qwen-Image-Edit-2511`
using the verified Qwen Image CUDA Diffusers runtime and the
`qwen-image-edit-2511-Q8_0.gguf` transformer. The worker must support one or
more reference images, ordered LoRAs with per-adapter strength, configurable
resolution/steps/CFG/seed, and the same provider-neutral and Salad queue
delivery contract as the existing Qwen Image worker.

## Approach

- Add an isolated `autotransition.qwen_image_edit` package so the existing
  Qwen Image 2512 worker cannot regress.
- Use `QwenImageEditPlusPipeline` with `image=[...]` for multi-reference
  editing. Load the Q8 GGUF transformer with Diffusers' GGUF loader and keep
  model CPU offload enabled by default for constrained GPUs.
- Accept references through `reference_images`, `referenceImages`, `images`,
  or typed `inputs` entries. Download each reference into a job-scoped
  directory, normalize EXIF orientation/RGB mode, validate dimensions and
  pixels, and remove temporary inputs after terminal completion.
- Reuse the existing request-level LoRA shape, but keep downloads and cleanup
  job-scoped. Never embed LoRA syntax in prompts.
- Persist the request, effective settings, reference metadata, preflight,
  progress events, failure traceback, and generated PNG in the final artifact
  set. Queue mode uploads every artifact to the launch-server callback.
- Add a dedicated Dockerfile and workflow that publish an edit-specific tag
  in the already-public `faceless-qwen-image-worker` GHCR package. The
  workflow verifies the exact tag through the anonymous registry endpoint;
  no private-image or visibility mutation step is added.

## Risks and checks

- The 21.8 GB Q8 transformer is included in the model-bearing image; official
  2511 text-encoder, tokenizer, scheduler, and VAE files are cached by
  Diffusers at runtime just as in the working 2512 Diffusers worker.
- Edit inference is GPU-required and has no CPU fallback. A missing CUDA
  device, transformer, or Diffusers pipeline component fails preflight with a
  diagnostic rather than producing a misleading output.
- Local tests use fake runtimes to exercise contracts, reference parsing,
  multi-reference preparation, LoRA ordering, artifact persistence, and queue
  envelope behavior. The full Q8 inference is not attempted on the local
  10 GB GPU.
