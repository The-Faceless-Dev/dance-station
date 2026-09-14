# Qwen Image Edit 2511 Worker

This worker runs `Qwen/Qwen-Image-Edit-2511` through the project-owned CUDA
Diffusers runtime with the verified `qwen-image-edit-2511-Q8_0.gguf`
transformer. It is published as an edit-specific tag in the public
`ghcr.io/the-faceless-dev/faceless-qwen-image-worker` package so the existing
public package visibility and provider pull path are preserved.

The edit pipeline accepts one or more references. Callers can use any of these
request shapes:

```json
{
  "job_id": "edit-1",
  "prompt": "Put both subjects in a studio portrait",
  "reference_images": [
    {"sourceUrl": "https://cdn.example/subject-a.png", "fileName": "a.png"},
    {"sourceUrl": "https://cdn.example/subject-b.png", "fileName": "b.png"}
  ],
  "width": 1328,
  "height": 1328,
  "steps": 20,
  "cfg_scale": 2.5,
  "loras": [
    {"sourceUrl": "https://cdn.example/style.safetensors", "fileName": "style.safetensors", "scale": 0.8}
  ]
}
```

`referenceImages`, `references`, `images`, and typed `inputs` entries with a
reference/image role are also accepted. LoRAs are downloaded in caller order,
applied with their individual scales, and deleted after the job. Local paths
are disabled in the production image.

The worker exposes `/health`, `/ready`, `/v1/worker/status`, direct
`/v1/qwen-image-edit/jobs` routes, and the provider-neutral `/process` queue
route. Final jobs retain the output PNG, effective metadata, reference
metadata, preflight report, and JSONL event log. Queue mode uploads all of
those artifacts to the launch-server callback.

The Q8 transformer is in the image. The official 2511 pipeline components are
cached under `HF_HOME` when Diffusers initializes, matching the existing Qwen
Image Diffusers worker. CUDA is required; there is no silent CPU fallback.

## Startup diagnostics

The entrypoint writes `startup.log`, `startup-report.txt`, and `worker.log` to
`QWEN_IMAGE_EDIT_STARTUP_LOG_ROOT` (default:
`/var/lib/autotransition/qwen-image-edit-startup`) while also forwarding the
same events to container stdout/stderr. The report records the exact entrypoint
phase, safe configuration values, UID and path permissions, disk and memory
capacity, process and cgroup state, CUDA visibility, Python/package import
results, health-probe responses, child exit status, and the last child log
lines. Secrets and callback tokens are intentionally excluded.

`QWEN_IMAGE_EDIT_STARTUP_PROBE_SECONDS` bounds the local `/health` probe after
the HTTP process is spawned; it does not load the generation pipeline. If the
container reaches the entrypoint, a failed import, child exit, or health
timeout now produces an explicit `STARTUP_EVENT` and durable report path. If
RunPod fails before invoking the image entrypoint, only RunPod's provider-side
events can explain that failure; the absence of `entrypoint_invoked` in the
provider log is itself the diagnostic signal.

## Local image build

```powershell
docker buildx build --load --target runtime-local `
  --build-context qweneditmodel=D:\models\qwen-image-edit-2511 `
  -f containers\qwen-image-edit-worker\Dockerfile .
```

The named build context must contain `qwen-image-edit-2511-Q8_0.gguf`.
