# VACE Working Defaults

## Goal

Restore the VACE settings from the clean Piccolo/Yoda production tests and make
those settings the defaults in the worker image used by production.

## Changes

- Set the VACE configuration defaults to 4 inference steps and sample shift 5.
- Add the same explicit environment overrides to the Wan code-overlay workflow.
- Keep Animate LightX2V disabled and keep VACE LightX2V enabled separately.
- Add regression coverage for the VACE defaults.
- Publish a new immutable overlay tag containing the VACE configuration and the
  previously fixed shared adapter file.
- Base the replacement overlay on the known 120-layer r22 image so the final
  image stays below Vast's layer-registration depth limit; do not stack another
  five dependency layers on top of r23.

## Tradeoffs and risks

- Four VACE steps are the known-good LightX2V configuration and should be much
  faster, but are not a general-purpose high-step VACE setting.
- Explicit overlay environment values prevent inherited base-image settings
  from silently changing the production behavior.
- Request-level overrides remain available for intentional future experiments;
  normal requests without overrides use the verified defaults.

## Verification

- Run focused VACE configuration tests.
- Inspect the generated overlay inputs and workflow diff.
- Run a production-style two-clip Animate/VACE/loop job and verify the VACE
  metadata reports 4 steps and shift 5 before destroying the paid instance.
