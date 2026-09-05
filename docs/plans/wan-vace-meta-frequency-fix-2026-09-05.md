# Wan VACE Meta-Frequency Fix

## Objective

Allow the production Wan VACE 14B FP8 runtime to use model offloading without
crashing on its unregistered RoPE frequency tensor.

## Approach

- Patch the native VACE wrapper immediately after the official model class is
  imported and before the upstream entrypoint constructs the model.
- Rebuild the missing frequency tensor on CPU when checkpoint loading leaves it
  on the `meta` device. The normal upstream forward path then moves it to the
  active CUDA device.
- Publish this as a code-only overlay on the already working r4 image so the
  large Animate, VACE, VAE, and postprocessing layers are reused unchanged.
- Run an isolated VACE smoke test before repeating the full two-window Animate,
  bridge, loop, Real-ESRGAN, and RIFE pipeline.

## Risks

- The patch changes only initialization of a deterministic positional tensor;
  model weights and sampling settings are unchanged.
- If the isolated VACE smoke test exposes another runtime issue, preserve its
  diagnostics and fix that issue before spending time on the full run.
