# Wait For Runtime Before UI Plan

## Problem

`autotransition run` can start the Autotransition UI after ACE-Step is launched but before the ACE-Step API is ready. The command prints `ACE-Step API is still starting in the background`, then the UI lets users attempt generation, which fails with `ACE-Step API is not running`.

That breaks the intended public workflow: one setup command, one run command, then generate from the UI.

## Approach

- Add a configurable ACE-Step API startup timeout separate from normal request timeouts.
- Increase the managed startup wait so first-run `uv run` and ACE-Step import/startup time can complete.
- If ACE-Step does not become healthy before the timeout, stop the process Autotransition started and do not launch the UI.
- Keep the UI-only and no-runtime-autostart options for advanced/debug use.
- Pass the runtime config used by the CLI into the UI app so status/generation use the same host/port.

## Affected Files

- `src/autotransition/config.py`
- `src/autotransition/runtime/ace_step.py`
- `src/autotransition/cli.py`
- `src/autotransition/ui/app.py`
- runtime/UI tests

## Tradeoffs and Risks

- `autotransition run` may spend longer waiting before the UI appears, but that is better than opening a UI that cannot generate.
- If users intentionally want the UI without ACE-Step ready, `autotransition run --ui-only` remains available.
