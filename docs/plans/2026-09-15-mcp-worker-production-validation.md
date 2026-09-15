# MCP Worker Production Validation

## Objective

Restore the LTX image to its separate public package and validate every MCP-supported worker end to end through the live launcher, including progress, artifact delivery, and failure/refund behavior.

## Approach

- Change only the LTX code-overlay publisher's image repository back to `faceless-ltx-video-worker`; preserve the existing `ltx-2.5-nvfp4-5090-r1` launcher tag.
- Use the established repository CI credential path and verify the exact LTX tag anonymously before compute is allocated.
- Run one minimal, production-shaped MCP request for each supported worker with the configured test wallet and private signer file, using the live launcher/provider policy.
- Poll every job through terminal status, events, and artifact URLs; retain request and response evidence under `tmp/` without storing private keys.
- Stop and destroy every paid provider instance after its successful test, and record any failure before retrying.

## Risks and Checks

- A worker may be standby/free but still fail readiness or artifact-contract validation.
- MCP requests must use HTTPS inputs and the exact payment/free-holder flow; never send the private key to the server.
- Do not alter WAN image references or stack a Wan overlay onto an already-overlaid image.
- Do not report success unless the returned artifact URL is usable and the job exposes meaningful progress and terminal events.
