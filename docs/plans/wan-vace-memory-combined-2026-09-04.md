# Wan VACE Memory-Compatible Overlay

## Objective

Publish a combined Wan Animate plus VACE image for the Vast two-window validation. The image must retain the already-tested r2 memory behavior while adding the VACE bridge and loop pipeline.

## Approach

- Rebase the existing VACE overlay workflow on `wan-q6-vast-direct-20260904-block-offload-r2`.
- Keep the VACE checkpoint and pinned VACE/Wan2.1 sources in the overlay layer.
- Preserve CPU reference-cache storage, transformer block offload, VAE offload, raw GGUF cache disabled, and required fast attention settings from r2.
- Run one request containing two 3-second Animate segments, one inter-segment VACE bridge, and one VACE loop bridge.
- Validate the returned MP4, job metadata, per-stage diagnostics, timings, and downloaded artifacts.
- Destroy the Vast instance after all artifacts and logs have been saved locally.

## Affected Files

- `.github/workflows/publish-wan-animate-vace-vast.yml`
- `docs/plans/wan-vace-memory-combined-2026-09-04.md`

## Risks

- The VACE checkpoint adds substantial VRAM pressure; the test must confirm that the r2 Animate memory path remains active in the combined image.
- VACE bridge and loop processing may fail independently of Animate; retain the full job-result and failure diagnostics.
- Vast startup and registry pulls are external operations, so startup events and image identity must be recorded.
