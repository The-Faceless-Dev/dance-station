# Strict ACE-Step API Health Plan

## Problem

Autotransition can treat the configured ACE-Step API URL as running when it is actually a different HTTP service, stale proxy, or RunPod error page. The current health check accepts any `/health` response below HTTP 500, so 404/405/HTML responses can mark the runtime as ready. Generation then fails later with confusing messages such as `405 Method Not Allowed` or a non-JSON RunPod 502 page.

## Approach

- Make ACE-Step health checks strict enough to reject wrong services:
  - `/health` must return HTTP 200.
  - The response must not look like an HTML page.
- Improve ACE-Step API errors with method, URL, status code, and a short response preview.
- Add tests for strict health rejection and clearer API errors.

## Affected Files

- `src/autotransition/runtime/ace_step.py`
- `src/autotransition/models/acestep_api.py`
- `tests/test_runtime.py`
- `tests/test_acestep_api.py`

## Tradeoffs and Risks

- Some ACE-Step versions may return an empty HTTP 200 health body. That should still be accepted.
- Non-200 health responses are now considered not running, which is correct for avoiding accidental connections to unrelated services.
