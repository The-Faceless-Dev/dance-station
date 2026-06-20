# Match Working ACE-Step Defaults

## Problem

The direct ACE-Step curl command produced clean music, while the app's raw ACE-Step output was still garbled. The app request still differs from the working command by sending extra explicit fields and by omitting/defaulting some metadata differently.

## Approach

- Make the app's default text-to-music payload match the working curl payload.
- Do not force `/v1/init` for text-to-music generation by default; the working direct request used the already-loaded ACE-Step primary model.
- Omit `model` from the text-to-music payload by default so ACE-Step uses its active primary model instead of Autotransition forcing XL Base.
- Keep user-adjustable knobs for prompt, duration, BPM, key, seed, inference steps, guidance scale, and shift.
- Default prompt, duration, BPM, and key to the working values so the payload remains identical unless the user changes them.
- Stop sending extra ACE-Step fields that the working command did not send by default: `vocal_language`, `instruction`, `infer_method`, `use_tiled_decode`, `constrained_decoding`, `use_cot_caption`, `use_cot_language`, and `allow_lm_batch`.
- Keep debug `ace-request.json` output so every app run can be compared to the working curl command.

## Affected Files

- `src/autotransition/models/acestep_api.py`
- `src/autotransition/ui/static/index.html`
- `src/autotransition/ui/static/app.js`
- `tests/test_acestep_api.py`
- `tests/test_ui_api.py`

## Risks

- `120 BPM` and `C minor` are not musically neutral defaults, but they match the first proven-good generation command and remain editable.
- Removing explicit defaults relies on ACE-Step REST defaults for those fields, which is exactly what the working command did.
- The UI's selected model may not match the active ACE-Step runtime primary model until runtime inventory is exposed directly in the app; for now the working path deliberately follows the runtime primary model.
