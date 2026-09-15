# YuE2-3B Worker

This worker runs the native `audio.cpp` YuE2 implementation with the public
YuE2 GGUF package:

- `yue2-3b-q8_0.gguf`
- `yue2-vae-f16.gguf`
- the four package sidecars under `sidecars/`

The worker does not use Python YuE, ComfyUI, a CPU fallback, or an implicit
model download. `/ready` remains unavailable until the native CLI, both GGUF
files, all sidecars, and the CUDA GPU are present.

## Build

Download the model package into a directory whose contents are the repository
files, then build with that directory as the model context:

```powershell
docker build --build-context yue2model=D:/models/yue2-3b-gguf -f Dockerfile -t yue2-worker .
```

The image compiles audio.cpp from its YuE2-enabled `dev` revision and uses the
same HTTP queue worker contract as the other project workers. Configure
`YUE2_MODEL_ROOT`, `YUE2_AUDIOCPP_CLI`, and the request defaults with
environment variables when running it.

## Request

Direct local request:

```json
{
  "lyrics": "[Verse]\nA quiet street is waking up.\n[Chorus]\nCarry the rhythm home.",
  "style": "English, indie pop, warm synths, crisp drums, clear lead vocal",
  "cot": "off",
  "num_inference_steps": 8,
  "seed": 12345,
  "request_options": {}
}
```

POST it to `/v1/yue2/jobs`. The durable job directory contains `audio.wav`,
the exact command, preflight report, runtime metadata, native stdout/stderr,
request, and event log. `/process` accepts the launch-server callback contract
and uploads all of those artifacts before completing the job.

The upstream GGUF package is licensed CC BY-NC 4.0; verify that license is
appropriate before using this worker commercially.

For a duration-limited request, send `duration_seconds`. The worker converts
that value to YuE2's native semantic codec-token ceiling (25 tokens per
second) unless `semantic_max_tokens` is explicitly supplied in
`request_options`.
