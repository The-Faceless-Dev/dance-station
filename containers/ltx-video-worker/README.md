# LTX 2.5 Video Worker

This container provides a serialized, launcher-compatible LTX 2.5 video runtime
for a single Vast Blackwell 5090 instance. It uses the pre-quantized NVFP4
distilled transformer by default. It does not use ComfyUI.

## Runtime behavior

Silent and source-audio requests use the official LTX video-only transformer
configurator. They do not construct audio latent state or load the audio VAE.
Requests that explicitly use `audio_mode=generated` use the full audio-video
pipeline and require `LTX_VIDEO_AUDIO_VAE_PATH`.

The two-stage distilled path is used for quality and speed:

1. Stage 1 generates at half resolution with the fixed 8-step distilled schedule.
2. The official latent spatial upscaler raises the latent to the requested size.
3. Stage 2 performs the fixed 4-step refinement schedule.
4. The video VAE is loaded for decode only after transformer denoising finishes.

Components are loaded lazily and released after each job. The worker records
the preflight estimate, actual CUDA memory snapshots, selected quantization,
conditioning, and every lifecycle stage. NVFP4 refuses non-Blackwell GPUs and
the worker never silently falls back to CPU or another quantization policy.
The video path remains strict Flash SDP; Gemma prompt encoding temporarily
enables math SDP in a scoped context because its attention shapes are not
compatible with the video Flash policy, then restores the video policy before
denoising.
Whole-job progress and stage-local progress are persisted in `events.jsonl`,
sent through the launch-server callback, and printed as structured events in
the Vast container log.

## Model context

The model-bearing context must have this layout:

```text
ltx-2.5/
  diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors
  text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors
  vae/ltx-2.5-video-vae-conv-bf16.safetensors
  vae/ltx-2.5-audio-vae-bf16.safetensors
  latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors
```

These are the complete required model components for the supported profiles.
The duration head is intentionally not bundled because this API requires an
explicit frame count or duration and resolves that to the LTX temporal grid
before loading models. The temporal upscaler, diffusion video VAE, and
detailing LoRA are optional profiles that are not part of this worker's
default pipeline; request LoRAs are supplied separately when supported.

Build with a named model context:

```powershell
docker buildx build `
  --build-context ltxmodels=D:\models\ltx-2.5 `
  -f containers/ltx-video-worker/Dockerfile `
  -t ltx-video-worker:local .
```

The build intentionally fails if a required model file is absent. The audio VAE
is included for `audio_mode=generated`, but the video-only path does not load it.
No weights are included in git or downloaded during a request.

Validate the bundle before starting a large image build:

```powershell
python tools/ltx_video/verify_model_bundle.py D:\models\ltx-2.5
```

For a resumable Windows staging run that downloads the five files sequentially
and writes progress logs under `tmp/ltx-model-download/`, use:

```powershell
powershell -ExecutionPolicy Bypass -File tools\ltx_video\download_model_bundle.ps1 `
  -Root D:\models\ltx-2.5
```

## HTTP endpoints

```text
GET  /health
GET  /ready
GET  /v1/worker/status
POST /v1/worker/reset
POST /v1/ltx/jobs
GET  /v1/ltx/jobs/{job_id}
GET  /v1/ltx/jobs/{job_id}/artifacts/{name}
POST /process
```

`/process` accepts the existing launch-server callback envelope used by the
Vast adapter. The direct endpoint accepts the same fields inside `parameters`.

Example request:

```json
{
  "job_id": "ltx-example-1",
  "parameters": {
    "prompt": "A dancer walks into a sunlit studio and turns toward camera",
    "aspect_ratio": "9:16",
    "num_frames": 73,
    "frame_rate": 24,
    "seed": 1234,
    "audio_mode": "off",
    "conditioning_images": [
      {
        "sourceUrl": "https://cdn.example/character.png",
        "frameIndex": 0,
        "strength": 0.85,
        "mode": "replace"
      },
      {
        "sourceUrl": "https://cdn.example/character-side.png",
        "frameIndex": 48,
        "strength": 0.65,
        "mode": "guide"
      }
    ],
    "output_format": "mp4"
  }
}
```

Aspect presets are `16:9` (`1024x576`), `9:16` (`576x1024`), and `1:1`
(`768x768`). Custom width and height are accepted when both are divisible by
64, which is required by the two-stage pipeline. Frame counts are snapped to
the LTX causal grid `8*K+1`.

`negative_prompt` is rejected for the distilled profile because it runs with
CFG 1; this is explicit rather than silently pretending to apply negative
conditioning. The request's `seed`, dimensions, frame count, audio mode,
conditioning list, and LoRAs are persisted in `generation-metadata.json`.

## Durable artifacts

Each job is stored below `LTX_VIDEO_ARTIFACT_ROOT`:

```text
job.json
events.jsonl
attempts/attempt-1/...
final/output.mp4 or output.webm
final/generation-metadata.json
final/request.json
final/memory-plan.json
final/events.jsonl
```

Generated audio additionally produces `final/audio.wav`. Failure artifacts
contain the exception type, traceback, stage, memory report, and a stable error
code suitable for launch-server refund/error handling.

The worker exposes `POST /v1/worker/reset` when no job is active. It trims the
CUDA allocator and returns before/after residency. A failed lazy decode is
closed and trimmed automatically; if allocations remain above
`LTX_VIDEO_RESIDENCY_RESET_THRESHOLD_GB`, the status becomes not-ready and the
worker requires a reset or process restart before accepting another job. The
ConvVAE decode working set defaults to 40 output frames with 16-frame overlap
and can be changed with `LTX_VIDEO_VAE_TEMPORAL_TILE_FRAMES` and
`LTX_VIDEO_VAE_TEMPORAL_OVERLAP_FRAMES`.
