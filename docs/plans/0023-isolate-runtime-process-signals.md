# Isolate Runtime Process Signals Plan

## Problem

ACE-Step can show `KeyboardInterrupt` during startup even when the user did not intentionally stop it. The managed ACE-Step subprocess is launched in the same terminal/process group as Autotransition, so terminal or parent-process interrupts can reach the child directly while it is importing heavy modules like Torch.

This makes startup look like an ACE-Step crash and can interrupt long imports.

## Approach

- Start the managed ACE-Step API process in its own process/session group.
- Keep Autotransition responsible for stopping the managed runtime through the existing shutdown cleanup path.
- Preserve stdout/stderr log capture.
- Add regression coverage that the subprocess launch uses isolation flags.

## Affected Files

- `src/autotransition/runtime/ace_step.py`
- `tests/test_runtime.py`

## Tradeoffs and Risks

- Ctrl+C on `autotransition run` will no longer directly interrupt ACE-Step; Autotransition will stop the managed runtime in its `finally` cleanup.
- If Autotransition is force-killed, ACE-Step may require the existing stale-runtime detection/cleanup path on the next run.
