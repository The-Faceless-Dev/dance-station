# Handle Non-JSON ACE-Step Model List

## Problem

Generation can fail with a server 500 when ACE-Step returns an empty or non-JSON response from `/v1/models`. The UI can still show the model as installed because that status is based on local model assets, while generation checks the running ACE-Step API before repainting.

## Approach

- Treat `/v1/models` as a best-effort readiness check because ACE-Step API behavior differs across environments.
- If `/v1/models` fails or returns non-JSON, continue to `/v1/init` so ACE-Step can initialize the requested model directly.
- Convert non-JSON responses from required generation endpoints into `AceStepApiError` so the UI reports a clean failed generation instead of crashing the FastAPI route.
- Add a regression test for a non-JSON `/v1/models` response.

## Affected Files

- `src/autotransition/models/acestep_api.py`
- `tests/test_acestep_api.py`

## Tradeoffs and Risks

- A broken `/v1/models` endpoint will no longer stop generation immediately; if the runtime is truly unhealthy, `/v1/init` or generation will return the actionable failure instead.
- This keeps model install status and runtime initialization separate, which matches how the app already behaves.
