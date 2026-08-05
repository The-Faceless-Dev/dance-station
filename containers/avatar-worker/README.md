# Avatar Worker Container

This image runs the image-to-mesh-to-rig worker on a CUDA GPU. It contains the
pinned inference adapters and the provisioned model weights required by the
production profile. Model files are kept out of git and added during private
image provisioning.

## Adapter commands

The worker accepts argv templates through environment variables. Available
placeholders are:

* Image generation: `{prompt_file}`, `{negative_prompt_file}`, `{output}`, `{seed}`, `{reference_image}`, `{quality}`
* Mesh generation: `{image}`, `{output_dir}`, `{output}`, `{quality}`
* Rig generation: `{input}`, `{output}`, `{manifest_output}`, `{quality}`

The repository includes `tools/avatar/flux2_klein_generate.py`, a non-
interactive wrapper around the official FLUX.2 Python API. The production
image includes the pinned FLUX.2 checkout and uses these paths:

```text
AVATAR_IMAGE_COMMAND=python /app/tools/avatar/flux2_klein_generate.py --prompt-file {prompt_file} --negative-prompt-file {negative_prompt_file} --output {output} --seed {seed} --reference-image {reference_image}
KLEIN_4B_MODEL_PATH=/models/flux2/flux-2-klein-4b.safetensors
AE_MODEL_PATH=/models/flux2/ae.safetensors
FLUX2_TEXT_ENCODER_PATH=/models/flux2/Qwen3-4B-FP8
```

The wrapper reads the worker prompt files and writes exactly the requested
image output. The negative prompt is retained in the attempt metadata even
though the distilled Klein sampler does not expose a separate negative-prompt
condition. `FLUX2_TEXT_ENCODER_PATH` is optional; when set, it points at a
local Qwen3-4B-FP8 directory so the worker does not resolve the text encoder
from the network. The worker does not download weights inside a paid job.

If using the public Diffusers-formatted VAE from the Klein model repository,
convert it once before deployment:

```text
python /app/tools/avatar/convert_flux2_vae.py /models/flux2/vae/diffusion_pytorch_model.safetensors /models/flux2/ae.safetensors
```

Stable Fast 3D can use the existing `run.py` entry point:

```text
AVATAR_MESH_COMMAND=python /models/stable-fast-3d/run.py {image} --pretrained-model /models/stable-fast-3d-checkpoint --output-dir {output_dir} --texture-resolution 2048 --remesh_option triangle --target_vertex_count 100000
```

## TRELLIS.2 mesh profile

`Dockerfile.trellis2` and `docker-compose.trellis2.yml` are a separate
production profile. The existing `Dockerfile` remains the Stable Fast 3D
rollback profile. TRELLIS.2 is pinned to the upstream commit recorded in its
Dockerfile and its CUDA extensions are compiled into that image, not during a
paid job.

The official TRELLIS.2-4B checkpoint is provisioned into the image. The worker
expects:

```text
/models/trellis2/pipeline.json
/models/flux2/...
/models/SkinTokens/...
```

`TRELLIS2_MODEL_PATH` defaults to `/models/trellis2` and
`TRELLIS2_ALLOW_MODEL_DOWNLOAD` defaults to `0`, so a missing checkpoint fails
clearly instead of downloading model weights while processing a paid request.
The wrapper maps the worker quality modes as follows:

| Worker quality | TRELLIS.2 pipeline | Default GLB target |
| --- | --- | ---: |
| `preview` | `512` | 50,000 vertices |
| `runtime` | `1024_cascade` | 150,000 vertices |
| `quality` | `1536_cascade` | 250,000 vertices |

Override `TRELLIS2_PIPELINE_TYPE`, `TRELLIS2_MAX_NUM_TOKENS`,
`TRELLIS2_DECIMATION_TARGET`, or `TRELLIS2_TEXTURE_SIZE` for a specific GPU
profile. The generated mesh then goes through the same mesh validation,
TokenRig rigging, deformation validation, retries, and refund signaling as the
Stable Fast 3D path.

The upstream project currently documents Linux and at least 24 GB of NVIDIA
VRAM as requirements. That matches the production RTX 3090 target; the local
10 GB RTX 3080 can test the surrounding worker and pipeline contracts but
cannot honestly run TRELLIS.2-4B inference.

TokenRig can use the repository runner shipped with this project:

```text
AVATAR_RIG_COMMAND=python /app/tools/tokenrig/adaptive_runner.py --skintokens-repo /models/SkinTokens --input {input} --output {output} --manifest-output {manifest_output} --profile auto --use-transfer
```

On Windows development machines, `tools/avatar/tokenrig_wsl_bridge.py` can
wrap the same Linux runtime while translating local drive paths to WSL paths.

The worker has a dependency-free glTF skinning validator enabled by default. It
applies a deterministic torso, arm, and leg diagnostic pose to the actual
inverse-bind matrices and vertex weights, then checks that the mesh moves and
that each major region has weighted vertices. Set
`AVATAR_REQUIRE_DEFORMATION_VALIDATOR=0` only for lightweight unit tests or
development fixtures that are not complete skinned GLBs.

## Production diagnostics

Every job writes an append-only `events.jsonl` file under
`AVATAR_ARTIFACT_ROOT/<job-id>/`. Each attempt also keeps complete
`<component>.stdout.log` and `<component>.stderr.log` files, even when an
attempt is discarded for retry. The worker mirrors structured events to
container stdout, including:

* request and idempotency decisions
* GPU lease wait/acquire/release and readiness information
* every stage transition and progress value
* exact adapter argv, timeout, start/end time, return code, and output paths
* live FLUX, TRELLIS.2, TokenRig, and validator output
* output file sizes and SHA-256 values
* validation reports, exception tracebacks, retry decisions, cleanup, and
  terminal `refundRequired` state

Salad may truncate a very long deployment log, so the mounted artifact volume
is the authoritative complete diagnostic record. Search the deployment log by
`jobId`, then inspect that job's `events.jsonl` and `attempts/` directory.

An optional pinned Blender/headless runtime can add renderer-specific checks.
It must write JSON to `{output}` and return a nonzero exit code or
`{"ok":false}` when a known test motion produces invalid deformation. The
built-in check still runs first when `AVATAR_REQUIRE_DEFORMATION_VALIDATOR=1`.

## Runtime policy

* One GPU job is admitted at a time.
* Every model subprocess has a stage timeout and is terminated as a process tree.
* CUDA caches are released after every attempt.
* Intermediate mesh/rig outputs are deleted after a failed attempt unless debug retention is enabled.
* `AVATAR_MAX_ATTEMPTS` is capped by the application at three total attempts.
* A terminal failed job includes `refundRequired=true`; the launch server owns the payment keys and consumes that signal through its existing idempotent refund flow.
