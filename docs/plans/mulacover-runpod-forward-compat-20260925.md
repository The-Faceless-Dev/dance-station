# MuLaCover RunPod Forward Compatibility

## Objective

Make the public MuLaCover worker complete a real reference-audio generation on
RunPod without passing callback arguments that the pinned MuLaCover pipeline
does not support.

## Approach

- Inspect the installed/upstream `MuLaCoverGenPipeline` call signatures.
- Filter optional cancellation and progress keyword arguments against the
  loaded pipeline method signature for `_forward` and `postprocess`.
- Keep stage-level progress reporting in the worker when the installed pipeline
  has no per-step callback support.
- Add focused unit coverage for the compatibility helper.
- Run the existing MuLaCover test suite and publish the corrected public worker
  image through the existing MuLaCover workflow.
- Verify the replacement image pulls anonymously, then run one real reference
  audio generation on the existing secure RunPod RTX 6000 Ada pod and retain
  all artifacts locally. Leave that pod running after success.

## Affected Files

- `src/autotransition/mulacover/runtime.py`
- `tests/test_mulacover_worker.py`
- `.github/workflows/publish-mulacover-worker.yml` only if the publish run
  requires no workflow change; otherwise leave the established workflow intact.

## Risks and Validation

- The compatibility filter must not hide required arguments or change model
  inputs.
- The current upstream pipeline has no callback parameters, so inference will
  report coarse stage progress; it must still persist progress, diagnostics,
  audio, symbolic MIDI, and metadata artifacts.
- Do not create a second paid pod. Reuse pod `jwod69epcrqlk6` and leave it
  running after the successful test.
