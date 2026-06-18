# Runtime Startup Status Plan

## Problem

`autotransition run` currently waits for ACE-Step readiness without telling the user what ACE-Step is doing. A long startup can be normal, but a silent wait feels broken and gives no difference between dependency sync, model loading, GPU initialization, or an actual failure.

## Approach

- Reuse the existing runtime activity summarizer that already parses ACE-Step logs for the UI.
- Add an optional startup status callback to `ensure_runtime_api`.
- Emit status updates when the phase/message changes while waiting for:
  - existing ACE-Step processes to become healthy
  - newly started ACE-Step API process to become healthy
- Include the last observed runtime status in timeout/failure messages.
- Keep the CLI simple: `autotransition run` prints useful startup status before the UI launches.

## Affected Files

- `src/autotransition/runtime/ace_step.py`
- `src/autotransition/cli.py`
- `tests/test_runtime.py`

## Tradeoffs and Risks

- Log parsing is best-effort. If ACE-Step changes log formats, the fallback still reports the latest meaningful line.
- Status output is intentionally throttled by message changes so the terminal does not spam duplicate lines.
