# Qwen Image CLI/Server Diagnosis

## Objective

Determine whether the Qwen-Image-2512 white-output failure comes from the
worker's HTTP/server request path or from the underlying CUDA diffusion
execution path.

## Approach

1. Verify the pinned stable-diffusion.cpp commit, CUDA build image, model file
   hashes, and the request fields used by the worker.
2. Build a disposable diagnostic image containing matching `sd-cli` and
   `sd-server` binaries from the same stable-diffusion.cpp checkout.
3. Run equivalent tests through direct CLI and native HTTP server paths using
   the same Q8 diffusion model, text encoder, VAE, prompt, dimensions, seed,
   scheduler, CFG, steps, and flow shift.
4. Capture stdout/stderr, request/response payloads, tensor finiteness
   statistics, timing, and binary/runtime metadata under `tmp/`.
5. Classify the failure and document the result without changing production
   defaults until the failing layer is identified.

## Risks and Controls

- The diagnostic image is disposable and must not replace the current worker
  image.
- No paid instance may be left running after the tests.
- The existing verified model hashes must be enforced during image assembly.
- Direct CLI and server tests must use identical inference parameters.
- A CPU smoke test, if needed, is only a finiteness check at a minimal valid
  resolution; it is not a quality benchmark.
