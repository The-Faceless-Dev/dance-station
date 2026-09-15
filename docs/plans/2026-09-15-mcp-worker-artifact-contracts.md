# MCP Worker Artifact Contracts

## Objective

Make every MCP-supported worker return artifacts that satisfy the launch
server's public callback contract, so completed jobs expose usable media and
metadata instead of failing during callback validation.

## Scope

- Normalize MOSS, LTX, Qwen Image, and Qwen Image Edit callback roles and
  variants to the launch server schema.
- Keep final images and videos as `preview`, generated audio as `audio`, and
  JSON, JSONL, and text diagnostics as `metadata`.
- Preserve the existing YuE2, Flux, Wan, and VACE contracts unless the audit
  finds a concrete schema violation.
- Add focused regression coverage for callback role mapping and worker
  artifact declarations.
- Publish only affected worker updates through their established public CI
  workflows.

## Launcher Coordination

Update launch-server completion classification so `moss-music` succeeds with
metadata or chart artifacts and does not enqueue an audio waveform task.
Keep media runtimes and audio-generation runtimes unchanged.

## Risks And Verification

- Existing artifact variants are package-level API data; remove invalid
  worker-specific variants rather than expanding the schema for arbitrary
  values.
- Run targeted Python tests and compile checks before pushing.
- Run launcher build and job-service tests, including a MOSS metadata-only
  completion case.
- Verify every published tag is anonymously pullable in CI.
