# Wan VACE Memory-Compatible Overlay

## Objective

Publish a combined Wan Animate plus VACE image for the Vast two-window validation. The image must retain the already-tested r2 memory behavior while adding the VACE bridge and loop pipeline.

## Approach

- Rebase the existing VACE overlay workflow on `wan-q6-vast-direct-20260904-block-offload-r2`.
- Keep the VACE checkpoint and pinned VACE/Wan2.1 sources in the overlay layer.
- Add the optional Real-ESRGAN and RIFE payloads to a final combined image by
  reusing only the existing public quality-stage registry layers; do not unpack
  the large Animate and VACE images on a hosted build runner.
- Preserve CPU reference-cache storage, transformer block offload, VAE offload, raw GGUF cache disabled, and required fast attention settings from r2.
- Run one request containing two 3-second Animate segments, one inter-segment VACE bridge, and one VACE loop bridge.
- Validate the returned MP4, job metadata, per-stage diagnostics, timings, and downloaded artifacts.
- Destroy the Vast instance after all artifacts and logs have been saved locally.

## Affected Files

- `.github/workflows/publish-wan-animate-vace-vast.yml`
- `tools/publish_wan_quality_overlay.py`
- `docs/plans/wan-vace-memory-combined-2026-09-04.md`

## Risks

- The VACE checkpoint adds substantial VRAM pressure; the test must confirm that the r2 Animate memory path remains active in the combined image.
- VACE bridge and loop processing may fail independently of Animate; retain the full job-result and failure diagnostics.
- Vast startup and registry pulls are external operations, so startup events and image identity must be recorded.

## Pending Follow-up: Final Timeline Assembly

The production-shaped validation generated both VACE jobs, but the delivered
timeline did not contain both results:

- `bridge-1` was generated, but its extracted gap appears as an effectively
  held final frame in the middle of the assembled video.
- `bridge-2` was generated as the enabled loop bridge, but it is absent from
  the final RGB timeline.
- The final base and enhancement outputs are `8.125` seconds. With two
  three-second segments and two approximately `2.125` second generated gaps,
  the expected assembled duration is approximately `10.25` seconds.

Before another production run, fix and verify:

1. Record the ordered RGB and alpha concat inputs, including each path, probe,
   bridge ID, loop flag, and expected duration.
2. Ensure every generated normal bridge and the loop bridge is included exactly
   once in the final timeline before enhancement or interpolation.
3. Validate final duration against the segment durations plus all bridge and
   loop durations, failing instead of silently returning a truncated timeline.
4. Inspect the extracted middle frames for each VACE output so a held context
   frame cannot be accepted as a successful bridge.
5. Give per-bridge diagnostic artifacts unique uploaded names; currently files
   such as `generated-gap.mp4` and `vace-output.mp4` collide between bridges.

## VACE GPU-Resident LightX2V Validation

### Objective

Test the VACE 14B bridge with the official Wan2.1 T2V LightX2V CFG-distill
LoRA while keeping the VACE transformer resident on the target GPU. The
isolated run must prove that the supported VACE inference path loads the
required transformer, attention backend, LoRA, and VAE without silently
falling back to CPU model offload.

### Approach

- Use the official `Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors`
  through LightX2V's `wan2.1_vace` runner and `lora_configs` contract.
- Keep the VACE DiT on CUDA for the entire bridge; allow only one-time text
  encoding on CPU if required by the memory budget, and report every module's
  device in diagnostics.
- Use the LoRA's intended four-step distillation profile as the isolated test
  default, with LoRA strength, steps, guidance, and checkpoint paths exposed
  through environment variables.
- Preserve the existing VACE input preparation, alpha/matting, artifact, and
  callback contracts.
- Make the final timeline assembly explicit and validate ordered inputs,
  per-part frame counts, bridge uniqueness, and final duration before any
  enhancement stage runs.
- Run one isolated VACE bridge on a single target Vast GPU, compare elapsed
  time and output metadata against the prior 50-step CPU-offloaded baseline,
  save all diagnostics/artifacts locally, and destroy the paid instance after
  the test.

### Affected Files

- `src/autotransition/vace_stitch/config.py`
- `src/autotransition/vace_stitch/runtime.py`
- `src/autotransition/vace_stitch/native_runner.py`
- `src/autotransition/vace_stitch/worker.py`
- `tools/publish_wan_vace_overlay.py`
- the VACE publish workflow and worker image overlay files

### Risks

- The published T2V LoRA is supported by LightX2V's VACE runner, but its
  compatibility with the scaled-FP8 VACE checkpoint must be verified by an
  actual isolated inference, not assumed from matching model family names.
- A GPU-resident 14B VACE transformer may exceed available memory at the
  current window size. The worker must fail with a clear preflight/OOM report;
  it must not silently switch to a slower CPU-DiT path.
- The four-step profile may trade quality for speed. Keep the old runtime tag
  and all sampling controls available so the result can be compared and
  reverted without changing the client contract.
