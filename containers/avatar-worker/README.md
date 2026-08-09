# Avatar Worker Container

This image runs the image-to-mesh-to-rig worker on a CUDA GPU. It contains the
pinned inference adapters and the model weights required by the production
profile. Model files are kept out of git and added through a named Docker build
context during the release build.

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
FLUX2_TEXT_ENCODER_PATH=/models/flux2/Qwen3-4B
```

The wrapper reads the worker prompt files and writes exactly the requested
image output. The negative prompt is retained in the attempt metadata even
though the distilled Klein sampler does not expose a separate negative-prompt
condition. `FLUX2_TEXT_ENCODER_PATH` is optional; when set, it points at a
local Qwen3-4B directory so the worker does not resolve the text encoder
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

The official TRELLIS.2-4B checkpoint and its conditioning models are copied
into separate image layers during the release build. The worker expects:

```text
/models/trellis2/pipeline.json
/models/dinov3/config.json and model.safetensors
/models/birefnet/config.json and model.safetensors
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

Build the production image in two normal layered steps. The first step builds
the runtime and compiled CUDA extensions; the second adds the five model
families as separate layers. The named context should contain `flux2`,
`trellis2`, `dinov3`, `birefnet`, and `SkinTokens` directories and should not
include unused model families:

```powershell
docker buildx build --platform linux/amd64 `
  --file containers/avatar-worker/Dockerfile.trellis2 `
  --tag faceless-avatar-worker:trellis2-runtime `
  --load .

docker buildx build --platform linux/amd64 `
  --build-context models='D:\path\to\avatar-models' `
  --build-arg TRELLIS2_RUNTIME_IMAGE=faceless-avatar-worker:trellis2-runtime `
  --file containers/avatar-worker/Dockerfile.trellis2.salad `
  --tag ghcr.io/ORG/faceless-avatar-worker:TAG `
  --push .
```

The release Dockerfile deliberately does not use a locally imported or
flattened base image. Keep the runtime image and model context on the same
build host until the registry push finishes.

If the existing model-bearing r2 image is already present on the build host,
and rebuilding the CUDA runtime would exceed available disk space, use the
additive overlay only after verifying that the base contains Flux, TRELLIS.2,
SkinTokens, and the Salad queue binary:

```powershell
docker buildx build --platform linux/amd64 `
  --build-context models='D:\path\to\avatar-models' `
  --file containers/avatar-worker/Dockerfile.trellis2.salad.overlay `
  --tag ghcr.io/ORG/faceless-avatar-worker:TAG `
  --push .
```

The overlay adds only DINOv3, BiRefNet, and the current worker source. It is
not a replacement for the normal two-stage build when sufficient disk space
is available.

For source-only changes such as canonical skeleton, TokenRig, or reskin
updates, use `Dockerfile.trellis2.salad.source-overlay` against the current
model-bearing image. It copies the current `src` and `tools` without touching
the large model layers and verifies that the canonical/reskin entry points are
present in the resulting image.

The release image includes the verified Salad HTTP job-queue worker and starts
both processes under `faceless-avatar-entrypoint.sh`. The queue transport calls
the avatar adapter at `POST /process` on port `8080`; the existing direct avatar
API remains available under `/v1/avatar/jobs` for local development. Queue jobs
download their reference image, run the durable image-to-mesh-to-rig pipeline,
send schema-valid progress callbacks, upload the generated preview/metadata
artifacts, and only then send the completion callback. Set
`SALAD_QUEUE_WORKER_ENABLED=false` only when running the image for direct local
HTTP testing without Salad transport.

The upstream project currently documents Linux and at least 24 GB of NVIDIA
VRAM as requirements. That matches the production RTX 3090 target; the local
10 GB RTX 3080 can test the surrounding worker and pipeline contracts but
cannot honestly run TRELLIS.2-4B inference.

When creating the Salad container group, provide the existing read-only GHCR
credential through the group's registry authentication settings. Salad pulls
GHCR images into its own registry and requires those credentials even when a
package is intended to be public. Do not put the credential in this image,
the repository, or a paid-job payload.

TokenRig can use the repository runner shipped with this project:

```text
AVATAR_RIG_COMMAND=python /app/tools/avatar/rig_runner.py --skintokens-repo /models/SkinTokens --input {input} --output {output} --manifest-output {manifest_output} --profile auto --use-transfer
AVATAR_RESKIN_COMMAND=python /app/tools/avatar/reskin_runner.py --skintokens-repo /models/SkinTokens --input {input} --output {output} --profile {profile} --manifest-output {manifest_output} --profile-mode auto --use-transfer
```

The reskin endpoint accepts an existing mesh and editable canonical profile,
fits the `humanoid-v1` skeleton, runs SkinTokens with that skeleton, restores
stable joint names, and keeps the source mesh, profile, skeleton, output GLB,
manifest, and validation diagnostics together in the job artifact directory.

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
For the first-pass result, read `final/failure-summary.json` or request
`GET /v1/avatar/jobs/<job-id>/failure-summary`; it contains the terminal code,
stage, missing/invalid roles, retry history, and retained artifact names.

An optional pinned Blender/headless runtime can add renderer-specific checks.
It must write JSON to `{output}` and return a nonzero exit code or
`{"ok":false}` when a known test motion produces invalid deformation. The
built-in check still runs first when `AVATAR_REQUIRE_DEFORMATION_VALIDATOR=1`.

## Runtime policy

* One GPU job is admitted at a time.
* Every model subprocess has a stage timeout and is terminated as a process tree.
* CUDA caches are released after every attempt.
* Failed attempts retain a bounded `final/debug-attempt-*` bundle containing
  the source image, mesh, rig, manifest, validation report, and logs; model
  caches and unrelated intermediate files are still deleted.
* The Salad adapter uploads that failure bundle before sending the explicit
  launch-server failure callback, so rejected rigs remain downloadable for
  inspection and refund processing.
* `AVATAR_MAX_ATTEMPTS` is capped by the application at three total attempts.
* A terminal failed job includes `refundRequired=true`; the launch server owns the payment keys and consumes that signal through its existing idempotent refund flow.
