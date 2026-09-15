# YuE2 Callback Contract Fix

## Problem

The first live MCP YuE2 request completed its inference but the launcher rejected
the artifact upload because the worker copied the obsolete `primary` role and
sent an unregistered `yue2-music` variant. The launcher contract uses `audio`
for generated audio and `metadata` for diagnostic files.

## Approach

- Correct the YuE2 callback adapter to classify `audio.wav` as `audio`.
- Use the existing registered `merged` variant for the generated audio and omit
  the variant header for metadata artifacts.
- Add a regression test that inspects the actual upload headers for audio and
  metadata files.
- Rebuild the existing public YuE2 worker image through its established GitHub
  Actions workflow and rerun the same live MCP request.

## Scope and Risk

This changes only callback metadata; inference, model files, request parsing,
and artifact contents remain unchanged. The launcher remains backward-compatible
and does not need a schema change.

## Verification

- Run the focused YuE2 callback test and the full Python test suite.
- Verify the rebuilt public tag is anonymously pullable.
- Submit and poll a 60-second holder-free YuE2 request through the production MCP
  endpoint, then retain the returned artifacts locally.
