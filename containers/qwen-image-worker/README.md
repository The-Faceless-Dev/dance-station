# Qwen-Image-2512 Worker

This container runs Qwen-Image-2512 through a project-owned Python worker and
the CUDA-enabled native `stable-diffusion.cpp` server. It does not use
ComfyUI, does not download model weights during a paid job, and exposes the
same provider-neutral `/process` envelope for Salad/Vast/RunPod callers.

## Model staging

The production image requires these exact files in a local staging directory:

```text
qwen-image-2512-Q8_0.gguf
Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf
qwen_image_vae.safetensors
```

The first file is the Qwen-Image-2512 Q8_0 transformer. The second is the
abliterated/heretic Qwen2.5-VL encoder. The Dockerfile copies each file in its
own layer:

The expected source artifacts are `unsloth/Qwen-Image-2512-GGUF`,
`mradermacher/Qwen2.5-VL-7B-Instruct-abliterated-GGUF`, and the Qwen Image VAE
from `Comfy-Org/Qwen-Image_ComfyUI`. Staging is intentionally separate from
the Docker build so the worker never downloads model weights during a paid
request.

```powershell
hf download unsloth/Qwen-Image-2512-GGUF qwen-image-2512-Q8_0.gguf `
  --local-dir D:\models\qwen-image-2512
hf download mradermacher/Qwen2.5-VL-7B-Instruct-abliterated-GGUF `
  Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf `
  --local-dir D:\models\qwen-image-2512
hf download Comfy-Org/Qwen-Image_ComfyUI `
  split_files/vae/qwen_image_vae.safetensors `
  --local-dir D:\models\qwen-image-2512
```

```powershell
docker buildx build `
  --build-context qwenmodel=D:\models\qwen-image-2512 `
  --target runtime-local `
  -f containers/qwen-image-worker/Dockerfile `
  -t qwen-image-worker:local .
```

The published `runtime` target downloads the same pinned files during the CI
build, verifies their SHA-256 values, and does not download anything at job
runtime.

The worker refuses to become ready if the transformer is not identified as
Q8_0 or the text encoder filename does not identify the required abliterated
Qwen2.5-VL profile. This prevents accidentally running the existing Qwen3
encoder or a generic LLM with this pipeline.

## Runtime behavior

The Python adapter starts `sd-server` lazily on the first accepted job. Native
server stdout is retained in `runtime-server.log` and mirrored into the job
event log. The native command uses CUDA, Flash Attention, mmap, and CPU
offload by default; these are explicit environment settings, not silent CPU
fallbacks. If CUDA or any required model component is missing, preflight
reports the exact path and diagnostic.

The native job is polled through `/sdcpp/v1/jobs/{id}`. Every job retains the
request, effective settings, preflight report, output PNG, metadata, events,
native server log, and failure traceback under:

```text
/var/lib/autotransition/qwen-image-jobs/<job-id>/
```

Temporary LoRA files are downloaded only when requested, applied through the
structured native `lora` request field in caller order, and deleted at the
job cleanup boundary. Prompt-embedded LoRA syntax is not used.

## Environment

```text
QWEN_IMAGE_DIFFUSION_MODEL=/models/qwen-image-2512/qwen-image-2512-Q8_0.gguf
QWEN_IMAGE_TEXT_ENCODER=/models/qwen-image-2512/Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf
QWEN_IMAGE_VAE=/models/qwen-image-2512/qwen_image_vae.safetensors
QWEN_IMAGE_CPU_OFFLOAD=true
QWEN_IMAGE_FLASH_ATTENTION=true
QWEN_IMAGE_MMAP=true
QWEN_IMAGE_LORA_APPLY_MODE=at_runtime
QWEN_IMAGE_ALLOW_LOCAL_LORAS=false
SALAD_QUEUE_WORKER_ENABLED=true
```

Resolution, steps, CFG, seed, prompt, and ordered LoRAs are request-level
values. Defaults are 1328x1328, 20 steps, and CFG 2.5. Explicit dimensions
must be aligned to 16 pixels and remain within the worker's configured area
limit. The default area limit allows the site's portrait and landscape sizes.

## Endpoints

```text
GET  /health
GET  /ready
GET  /v1/worker/status
POST /v1/qwen-image/jobs
GET  /v1/qwen-image/jobs/{job_id}
POST /process
```
