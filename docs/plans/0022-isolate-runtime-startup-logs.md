# Isolate Runtime Startup Logs Plan

## Problem

ACE-Step runtime logs are appended across managed starts. If a previous run was interrupted with `KeyboardInterrupt` or failed with a traceback, the startup status parser can continue reading that old failure and report it during the next run.

That makes current runtime status unreliable.

## Approach

- Rotate existing managed ACE-Step stdout/stderr logs before starting a new managed runtime.
- Open the current run's logs in write mode instead of append mode.
- Keep one previous copy for debugging without poisoning current status.
- Add regression coverage that `start_api_background` truncates/replaces current logs.

## Affected Files

- `src/autotransition/runtime/ace_step.py`
- `tests/test_runtime.py`

## Tradeoffs and Risks

- Only one previous log copy is retained. This is enough for local debugging and avoids unbounded log growth.
- Advanced users who want continuous ACE-Step logs can still start ACE-Step separately outside `autotransition run`.
