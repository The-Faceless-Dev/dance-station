# 0032 Remove Margin From Primary Workflow

## Problem

The UI still exposes `ACE repaint margin seconds` as if it is part of the normal continuation process. That is misleading.

The corrected default workflow is:

1. Generate the new section with ACE-Step `text2music`.
2. Stitch it after the selected continuation point.
3. Optionally run ACE-Step `repaint` around the boundary later.

ACE-Step does not have a normal continuation parameter named margin. Autotransition introduced that term only to decide the size of an optional boundary repaint region. It should not be shown as a primary setting or preset value.

## Intended Change

Remove margin from the normal user workflow:

- Remove the `ACE repaint margin seconds` field from the primary Transition panel.
- Remove margin from preset/default language.
- Stop sending or relying on a margin value during default generation.
- Keep boundary repaint disabled by default.

If boundary repaint stays available, move it behind a clearly optional advanced boundary-repair section later, with names tied to what it actually does:

- boundary repaint enabled
- boundary repaint seconds before marker
- boundary repaint seconds after marker

Do not expose this as part of the normal continuation flow.

## Affected Files

- `src/autotransition/ui/static/index.html`
- `src/autotransition/ui/static/app.js`
- `src/autotransition/ui/app.py`
- `src/autotransition/config.py`
- `src/autotransition/presets.py`
- `tests/test_ui_api.py`
- README/docs wording

## Validation

- Default UI request should generate via text2music and stitch without any repaint/margin value.
- Existing API compatibility can continue accepting `repaint_overlap_seconds`, but it should default to `0` and not be shown as a normal control.
- Full test suite should pass.
