# AGENTS.md

## Project Context

This repository is distributed by **The Faceless Dancer** and is intended to become a public, reusable pipeline for AI-generated music transitions.

The goal is to help users generate, extend, transition, and organize music clips in a way that is practical for creators, streamers, visualizers, rhythm-game experiments, and other public-facing media projects.

Build this as a professional open-source tool, not a one-off script.

## Development Workflow

Before making changes, create a plan first.

For any meaningful feature, refactor, UI change, or architectural decision:

1. Add a plan in `docs/plans/`.
2. Explain the intended approach, affected files, tradeoffs, and risks.
3. Wait for approval before implementation unless the task is clearly tiny or explicitly marked as safe to proceed.

Keep plans direct and implementation-focused.

## Registry Credentials

GHCR publishing and private-image pulls must use the repository/deployment credential configured for `D:\faceless-launch-server` (or the equivalent CI repository token). The exact CI publishing workflow is:

`D:\autotransition\.github\workflows\publish-wan-animate-overlay.yml`

That workflow authenticates with `secrets.GHCR_PUSH_TOKEN || secrets.GITHUB_TOKEN`; GitHub injects `GITHUB_TOKEN` only while the workflow runs, so it should not be expected in a local `.env` or process environment. Use that workflow for Wan code-only overlays and inspect it before concluding that GHCR credentials are missing. Never substitute the local Docker Desktop credential, a local GitHub login, or an unrelated machine credential. Keep the established public package names and publish path unchanged. If the repository workflow or its configured secret is unavailable, report that explicitly rather than silently switching credentials or changing package visibility.

### GHCR Publish Gate

The absence of `GHCR_PUSH_TOKEN`, `GITHUB_TOKEN`, or `SALAD_GHCR_*` in a local
`.env`, process environment, or Docker credential store is **never** a reason
to stop a publish, ask the user for credentials, or report that GHCR access is
missing. Those checks do not test the GitHub Actions credential path. The
workflow reference must be inspected and the appropriate CI publish path must
be run or verified first.

`GITHUB_TOKEN` is intentionally ephemeral and exists only inside the GitHub
Actions job. `GHCR_PUSH_TOKEN` is a repository/environment secret and its value
must not be copied to disk. `SALAD_GHCR_READ_TOKEN` is a pull credential, not a
publish credential, and must never be used for a push. Never use the local
Docker credential helper, local GitHub login, or any other machine credential
as a substitute.

For a locally built image, distinguish the image source from the credential
source: use the repository's CI/deployment credential and an appropriate
model-bearing or code-overlay workflow. The Wan overlay workflow is only for
Wan code-only overlays; it must not be treated as the publish path for a new
model-bearing image such as LTX. If no appropriate CI workflow exists, record
that exact workflow limitation and continue preparing the publish path; do
not misdiagnose it as a missing local token. A credential blocker may only be
reported after the correct CI workflow has been inspected and its actual run
or configured-secret lookup has failed.

## Wan Overlay Publishing

### Vast Cost Hard Stop

**Never select, allocate, restart, or launch a Vast instance costing more than
$1.00 USD per hour. Period.** This is a hard safety limit, not a preference or
a tie-breaker. Reject offers above the limit before allocation, even when their
network speed is higher. Among offers at or below $1.00/hour, prefer the
fastest eligible single-GPU machine and record both network speed and hourly
cost before starting it. If no eligible offer is at or below $1.00/hour, stop
without creating an instance and report that no safe offer is available.

Wan production images use a model-bearing base plus a small code/dependency
overlay. Preserve this process exactly; do not create a new Docker build or
stack successive overlays for a code/config-only change.

1. Identify the last known-good model-bearing base tag. For the current Vast
   Wan Q6 lineage, that base is `wan-q6-vast-direct-20260905-vace-lightx2v-r22`
   with 120 layers. Treat that as the base for code-only VACE/Animate updates
   until a newer base is explicitly verified.
2. Use `.github/workflows/publish-wan-animate-overlay.yml` and
   `tools/publish_wan_overlay_manifest.py`. Do not publish with a local Docker
   credential, a separately invented Dockerfile, or a prior overlay as the
   base just because its tag is newer.
3. The current code-only Wan overlay adds five layers: the runtime code layer
   plus SciPy, imageio-ffmpeg, prometheus-client, and pyzmq. The expected
   result from the current base is therefore 125 layers. Do not publish an
   image above the registry/runtime layer-depth limit; in particular, never
   stack those five layers on top of r23/r24 or any other image that already
   contains them.
4. Before starting any paid Vast instance, verify the published tag and layer
   count:

   ```powershell
   docker buildx imagetools inspect `
     ghcr.io/the-faceless-dev/faceless-wan-animate-worker:<tag> --raw `
     | ConvertFrom-Json | Select-Object -ExpandProperty layers `
     | Measure-Object | Select-Object -ExpandProperty Count
   ```

   The command must show the expected count, and the GitHub Actions publish
   workflow must be completed successfully. If Vast reports `failed to
   register layer: max depth exceeded`, destroy that failed instance and do
   not retry it; inspect the manifest lineage and rebuild from the known-good
   base.
5. Keep runtime settings separate from image composition. Animate must keep
   LightX2V disabled. The verified VACE LightX2V defaults from the successful
   5090 run are: 4 inference steps, sample shift 5, guide scale 5, model
   offload disabled, T5 CPU offload enabled, and FlashAttention 2 with no
   fallback. Do not let generic Animate `steps` or `shift` fields override
   VACE settings.
6. After the tag passes the manifest check, update the launcher’s Vast image
   reference, start one single-GPU 5090 instance, verify `/v1/worker/status`,
   and run the complete production-shaped test: Animate segments, VACE bridge,
   loop, matte/transparent output, and final artifact assembly. Do not report
   success from a worker start or an Animate-only result.
7. Retain the request, progress/events, VACE metadata, stdout/stderr, and final
   artifacts under `tmp/`. Only destroy the paid instance after the complete
   test succeeds or after a confirmed failure has been diagnosed.

The essential rule is: **reuse the known-good image lineage and publisher;
change only the requested runtime/code layer; prove the final manifest before
paying for an inference instance.**

## Architecture Expectations

Keep the project modular and easy to hack.

Separate concerns clearly:

* audio analysis
* audio slicing and stitching
* model inference
* transition generation
* scoring/selection
* queue/playback/export logic
* UI/API layer
* configuration
* logging/debugging

Avoid hardcoding model paths, timing values, prompts, formats, or output locations inside core logic. Put configurable behavior in a central, organized config area.

Design modules so users can replace pieces later, such as switching models, changing BPM/key detection, adding new scoring rules, or swapping the UI.

## Public Repo Standards

Assume other developers will clone this repo and try to run it.

Prioritize:

* clear file names
* readable code
* small functions
* useful comments where behavior is not obvious
* practical defaults
* helpful errors
* documented setup steps
* no secret keys or personal paths committed
* examples that work out of the box

Keep the repo organized enough that someone can understand the project structure without reading every file.

## UI Expectations

The UI should look professional, clean, and intentional.

Do not build a bare developer panel unless specifically requested. Even early UI should feel like a real tool from The Faceless Dancer.

Use a polished layout with clear spacing, readable typography, obvious controls, useful status feedback, and sane empty/error/loading states.

The UI should make the pipeline understandable:

* current source clip
* selected tail/context
* transition settings
* target prompt/style
* generation status
* candidate outputs
* chosen/exported result

Design UI components so they can grow into a public creator-facing app.

## Branding

Use **The Faceless Dancer** as the distributor/brand.

Keep branding tasteful and minimal. Avoid locking core functionality to branding-specific code. Public users should be able to fork, rebrand, or customize the project without fighting the architecture.

## Code Quality Rules

Prefer simple, reliable code over clever code.

Do not create giant files. Split code when responsibilities diverge.

Do not silently swallow errors. Surface useful messages.

Do not introduce unnecessary dependencies.

When adding dependencies, explain why they are needed in the plan.

Keep generated files, model outputs, cache folders, and large media out of git unless explicitly intended.

## Audio Pipeline Priorities

The transition pipeline should be designed around repeatable steps:

1. Take the ending of the current generated clip.
2. Use it as context for the next generated clip.
3. Generate or repaint a continuation.
4. Score/check the result.
5. Export a clean transition-ready audio segment.

Keep timing, overlap, fade, BPM, key, prompt, seed, and model settings trackable as metadata.

Generated outputs should be organized predictably so users can inspect and reuse them.

## Cross-Repo Work

Dance Station is expected to connect with the main Faceless Dancer website repo at `D:\faceless-dancer-site`.

When work touches both repos:

* Treat each repo as independent. Check `git status` in both before coordinated work.
* Use the shell/tool `workdir` to operate in the correct repo instead of copying files between folders.
* Before editing the site repo, read and follow `D:\faceless-dancer-site\AGENTS.md`.
* Keep branches, commits, and reviews separate between Dance Station and the site repo.
* Do not mix Dance Station changes and site changes into one commit.
* Prefer coordinated branch names when a feature spans both repos, such as `library-sync` here and `library-platform` in the site repo.
* For shared library contracts, define the public schema/API shape in the site repo first, especially under its `shared/` area, then mirror or consume that contract from Dance Station.
* Preserve local-first behavior in Dance Station. Public library and website integration should add sync/export/import paths without breaking offline local libraries.
* Avoid leaving both repos half-finished. If work spans both, clearly state what is complete in each repo and what remains.

## Agent Behavior

When working in this repo:

* plan before building
* keep changes modular
* preserve public usability
* update docs when behavior changes
* avoid personal-machine assumptions
* keep UI polished
* make the project easy for other developers to understand, fork, and extend
