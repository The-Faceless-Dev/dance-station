# Generative Dance Controls and Pricing

## Scope

Carry the generative-dance postprocessing choices from Dance Station through
the site API and launch server to the existing Wan/Vast worker, and add
duration-based pricing for the generative dance runtime. Audit the existing
Flux image Salad path and only add missing contract or configuration support.

## Implementation

1. **Client request controls**
   - Add two clearly labeled checkboxes to the generative dance submission
     controls: 2x video enhancement and motion interpolation to 48 FPS.
   - Default both to the quality settings used by the previously verified
     successful runs: 2x enhancement and 48 FPS interpolation.
   - Include the choices in `parameters.postprocess` on the submitted request
     and show the resulting choices in the price/request state without changing
     the dance sequence format.

2. **Site and launcher contract**
   - Validate the optional postprocess object at the site boundary and launch
     server boundary, including bounded enhancement scale and target FPS.
   - Preserve the object unchanged through payment quoting, job creation, and
     Salad/Vast dispatch.
   - Keep the existing Flux image request and Salad routing contract intact;
     add only any missing validation or artifact-routing coverage found during
     the audit.

3. **Worker request override**
   - Extend the existing Wan worker to apply the request-level postprocess
     choices to the VACE postprocess stages, falling back to the worker’s
     environment defaults when the request omits them.
   - Keep Animate LightX2V and all existing VACE settings unchanged.
   - Record the effective postprocess configuration in progress metadata and
     final job metadata so the launcher and artifacts are auditable.

4. **Duration pricing**
   - Add FACELESS and SOL per-second add-on fields for Wan generative dance to
     the launcher payment settings, database migration, admin form, client
     pricing contract, and quote calculation.
   - Charge the configured rate for each submitted sequence second, with the
     existing holder-free behavior preserved: holder status removes the base
     charge but not configured duration add-ons.
   - Keep defaults at zero so existing production prices do not change until
     explicitly configured in launcher admin.

5. **Verification and release hygiene**
   - Add focused tests for request validation/forwarding, dynamic worker
     postprocess behavior, pricing, and Flux image Salad routing.
   - Run site and launcher tests/builds plus worker unit tests.
   - Commit site, launcher, and worker changes separately, leaving unrelated
     dirty worktree changes untouched.
   - No image publish or paid Vast/Salad run is part of this change unless a
     later deployment request explicitly asks for it.

## Risks and tradeoffs

- Request-level postprocess settings must be bounded so callers cannot select
  arbitrary expensive or unsupported stages.
- Defaults remain compatible with existing jobs; the client defaults will match
  the verified quality preset, while jobs from older clients continue using
  worker environment defaults.
- The duration add-on is a database/admin setting rather than an environment
  variable, so pricing can change without rebuilding or redeploying a worker.
