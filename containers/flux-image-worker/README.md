# FLUX.2 Klein 4B Image Worker

This container is the native image-generation worker for The Faceless Dancer.
It uses the official FLUX.2 Klein 4B runtime with a local Qwen3-4B text
encoder and FLUX VAE. It does not use ComfyUI and does not download model
weights while processing a paid job.

The Salad entrypoint starts both the Salad HTTP queue process and the worker
adapter. Jobs expose the same durable status, progress, artifact callback,
failure/refund metadata, and optional input-adapter contract used by the
other production workers. The image build installs and verifies the pinned
Salad queue-worker binary before the image can be published.

## Model files

The image contains these files under `/models/flux2-klein-4b/`:

```text
flux-2-klein-4b.safetensors
text_encoder/
  model.safetensors
  config.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
flux_vae.safetensors
```

The Docker build copies each file in its own layer. The source files are
staged from the local ComfyUI model directory and are never modified there.

## Environment

```text
FLUX_IMAGE_MODEL_ROOT=/models/flux2-klein-4b
FLUX_IMAGE_MODEL_PATH=/models/flux2-klein-4b/flux-2-klein-4b.safetensors
FLUX_IMAGE_TEXT_ENCODER_PATH=/models/flux2-klein-4b/text_encoder
FLUX_IMAGE_VAE_PATH=/models/flux2-klein-4b/flux_vae.safetensors
FLUX_IMAGE_MODEL_NAME=flux.2-klein-4b
FLUX_IMAGE_DEVICE=cuda
FLUX_IMAGE_DTYPE=bfloat16
FLUX_IMAGE_CPU_OFFLOAD=true
FLUX_IMAGE_MAX_STEPS=4
FLUX_IMAGE_AVATAR_WIDTH=480
FLUX_IMAGE_AVATAR_HEIGHT=832
FLUX_IMAGE_AVATAR_SUBJECT_SCALE=0.70
```

CPU offload is enabled by default. The unquantized Klein transformer and
Qwen3 text encoder must not be resident on the GPU at the same time as the
denoising working memory, even on a 24 GB RTX 3090.

## Request parameters

The launcher sends the fully resolved prompt and settings. The worker does
not add prompt wrappers. `loras` remains an accepted, validated input and is
empty by default. The distilled Klein runtime currently reports a structured
unsupported-adapter failure rather than silently ignoring a supplied LoRA;
the base Klein profile is the appropriate future runtime for active LoRA
inference.

```json
{
  "runtime": "flux-image",
  "job_id": "job-id",
  "parameters": {
    "prompt": "A purple horse wearing a baseball cap",
    "width": 1328,
    "height": 1328,
    "steps": 4,
    "true_cfg_scale": 1.0,
    "seed": 123,
    "loras": []
  }
}
```

Successful jobs return `image.png` and `image-metadata.json`. Failed jobs
preserve the failure summary and event log in the durable artifact directory.

The model inference request keeps its requested resolution and sampling path.
After inference, the worker detects the requested solid background, fits the
character to approximately 70% of the canonical `480x832` portrait canvas,
and fills the remaining canvas with the sampled background color. The original
model PNG remains in the attempt directory for diagnostics; `image.png` is the
framed production artifact. Job progress reports model loading, prompt
encoding, denoising, decode, framing, and finalization stages.
