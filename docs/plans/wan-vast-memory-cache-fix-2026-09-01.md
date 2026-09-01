# Wan Animate Vast Memory and Cache Fix

## Objective

Make the existing public Wan Animate Q6 worker complete the production-shaped
Vast generation without CPU dequantization fallback or avoidable CUDA memory
pressure, while keeping the existing image/package and job contract unchanged.

## Approach

1. Give the non-root `wan` process writable, persistent cache directories for
   Triton, TorchInductor, Hugging Face, and matplotlib through the existing
   worker image overlay. This removes the `/root/.triton` permission failure.
2. Keep quantized GGUF raw bytes in host memory by default and make the cache
   policy visible in startup diagnostics. Do not add a second GPU copy of the
   Q6 weights.
3. Add explicit memory cleanup and memory telemetry around conditioning,
   denoising, and VAE decode so one job/window cannot retain temporary tensors.
4. Provide Triton's linker with a writable `libcuda.so` alias when the host
   exposes only the driver soname `libcuda.so.1`; this is required by the Q6
   CUDA dequantization path on Vast and does not modify the host filesystem.
5. Force the project-owned eager bounded-attention fallback on runtimes where
   Triton is installed, and select its chunked SDPA path explicitly. This
   avoids unsupported high-register Flex Attention kernels on RTX 5090.
6. Fix the direct worker status endpoint so production failures can be queried
   without the current coroutine/type error.
7. Include the runtime module and container environment in the same small
   overlay layer, increment only the existing worker tag, and run focused
   tests plus a live Vast generation before cleanup.
8. Keep CUDA dequantized linear projections in the configured inference dtype
   instead of inheriting the official block's temporary FP32 input dtype. This
   prevents the 40-layer reference K/V cache from doubling in size on the
   32 GiB RTX 5090, and log its exact byte size and dtype before denoising.
9. Enforce the same dtype on any cache values still promoted by an upstream
   autocast operation before denoising begins, and record how many values were
   normalized.
10. Preserve the official reference-pass FP32 input contract; normalize only
    the completed K/V cache after `forward_ref` returns so the memory fix does
    not break Wan's reference assertions.

## Affected Files

- `tools/generative_dance/wan_animate_2_runtime.py`
- `tools/generative_dance/wan_animate_2_runner.py`
- `src/autotransition/generative_dance/salad_server.py`
- `tools/publish_wan_overlay_manifest.py`
- `.github/workflows/publish-wan-animate-overlay.yml`

## Risks and Verification

- Disabling the GPU raw cache can trade a small amount of dequantization work
  for lower peak VRAM; it is the correct default for a 32 GiB RTX 5090 profile.
- Cache paths must be created and owned by `wan` without changing model layers.
- Vast may expose only `libcuda.so.1`; the runtime must make Triton's `-lcuda`
  link succeed without requiring root access or changing the host image.
- Verify Python compilation, focused worker tests, image manifest publication,
  worker readiness, a completed production-shaped generation, returned media,
  and the final destruction of the temporary Vast instance.
