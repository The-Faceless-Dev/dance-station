# Qwen Image Edit Startup Diagnostics

## Objective

Make Qwen Image Edit startup failures diagnosable from provider logs and
durable files. A provider can fail before the container process starts, which
cannot be fixed from inside the image, but every failure after entrypoint
execution must include enough context to identify the failing boundary.

## Approach

- Extend the entrypoint with a writable startup log directory and a durable
  startup report.
- Capture shell errors with the failing command, source line, exit status,
  process list, filesystem capacity, UID/home/path permissions, image/runtime
  metadata, CUDA device visibility, and dependency import diagnostics.
- Tee entrypoint and child-process output to both provider stdout/stderr and
  startup log files so RunPod logs remain useful while the pod is reachable.
- Start the HTTP worker with captured stdout/stderr, probe `/health` with a
  bounded startup window, and emit an explicit ready or startup-failed
  marker. Preserve the child exit code and diagnostic paths.
- Keep secrets out of diagnostics by logging only an allowlist of safe
  configuration values and never dumping the full environment.
- Add shell/static tests and document the diagnostic files and their limits.

## Affected Files

- `containers/qwen-image-edit-worker/entrypoint.sh`
- `containers/qwen-image-edit-worker/Dockerfile`
- `containers/qwen-image-edit-worker/README.md`
- `tests/qwen_image_edit/test_entrypoint.py`

## Risks and Checks

- Startup probing must not load the full generation pipeline; `/health` only
  performs the existing runtime preflight.
- The worker remains single-process for inference and keeps the existing
  direct and Salad queue contracts.
- Provider failures before entrypoint execution will still only be visible in
  provider-side events; the image will explicitly record that it never
  reached entrypoint execution only when external logs show no startup marker.
- Run `bash -n`, the focused Qwen tests, and the existing repository test
  suite relevant to the worker before publishing a new image.
