# MOSS-Music Worker

This container runs the OpenMOSS MOSS-Music audio-analysis runtime with a durable
job API and the existing launch-server callback contract. It returns semantic
MOSS-Music interpretation together with a measured dense timeline whose default
cell size is 80 ms.

## Runtime modes

The default image starts the OpenMOSS `moss-audio` SGLang backend in the same
container and runs one analysis job at a time. Set
`MOSS_MUSIC_SGLANG_AUTOSTART=false` when SGLang is supplied by another process.
Set `SALAD_QUEUE_WORKER_ENABLED=false` for direct HTTP dispatch, which is the
intended starting point for a future Vast adapter.
The entrypoint sets writable HOME, XDG cache, and FlashInfer workspace paths and
waits for `/ready` before starting the external queue worker. This also makes the
normal image entrypoint usable when a provider launches the container as root.

The worker does not download model weights while processing a job. Build the final
image with a model-bearing named context:

```powershell
docker buildx build `
  --build-context mossmodel=D:\models\moss-music-8b-instruct `
  -f containers/moss-music-worker/Dockerfile `
  -t moss-music-worker:local .
```

The official backend setup is based on the OpenMOSS SGLang `moss-audio` branch.
Pin the branch to a tested commit before publishing a production image.
The readiness endpoint rejects a production profile when CUDA is unavailable, the
model directory is missing, or the SGLang backend is unhealthy; it does not fall
back to CPU inference.

## Important environment values

```text
MOSS_MUSIC_MODEL_ROOT=/models/moss-music-8b-instruct
MOSS_MUSIC_MODEL_NAME=MOSS-Music-8B-Instruct
MOSS_MUSIC_BACKEND=sglang
MOSS_MUSIC_SGLANG_URL=http://127.0.0.1:30000
MOSS_MUSIC_EVENT_RESOLUTION_MS=80
MOSS_MUSIC_GPU_REQUIRED=true
MOSS_MUSIC_ALLOW_LOCAL_AUDIO_PATHS=false
SALAD_QUEUE_WORKER_ENABLED=true
```

`MOSS_MUSIC_EVENT_RESOLUTION_MS` is bounded by the worker and may be changed per
request inside the configured range. The dense grid is measured from the audio;
MOSS-generated semantic timestamps are kept separately and are never treated as
an exact substitute for the measured timeline.

## HTTP API

Health and status:

```text
GET /health
GET /ready
GET /v1/worker/status
```

Direct job submission:

```json
{
  "job_id": "local-test-1",
  "audio": {"path": "/data/song.wav", "filename": "song.wav"},
  "analysis_profile": "visual_sync_v1",
  "event_resolution_ms": 80,
  "include_dense_features": true,
  "include_semantic_events": true,
  "temperature": 0
}
```

```text
POST /v1/moss/jobs
GET  /v1/moss/jobs/{job_id}
POST /process
```

The final `analysis.json` is the canonical artifact. MOSS semantic interpretation
is collected in focused overview, rhythm, harmony, and lyrics/voices passes while
the same backend remains loaded; all raw pass responses are retained alongside
`moss-response.json`. `moss-raw.txt`, `moss-semantic.json`,
`audio-metadata.json`, `moss-runtime.json`, `request.json`, and `events.jsonl` are
retained for provenance and debugging. The `/process`
response includes the uploaded artifact IDs and remote-safe manifest. Direct
jobs can retrieve each durable file from
`/v1/moss/jobs/{job_id}/artifacts/{artifact_name}`. The worker applies only the
30-minute audio limit by default. There is no caller-facing token limit. For each
focused pass, the worker derives the available output budget from the checkpoint's
actual context length minus the measured text and audio input; this avoids
SGLang's accidental 4096-token default without inventing a smaller ceiling.
