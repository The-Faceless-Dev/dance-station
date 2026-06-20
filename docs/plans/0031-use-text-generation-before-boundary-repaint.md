# 0031 Use Text Generation Before Boundary Repaint

## Problem

Autotransition still produces garbled/static audio when asking ACE-Step `repaint` to generate a new appended section from padded/empty space. This confirms the failure is not just a bad overlap/margin setting.

The current path uses one ACE-Step task for two different jobs:

1. Create new musical material from the prompt.
2. Make the boundary seamless with the source audio.

ACE-Step `repaint` is designed as an audio editing task: repaint a selected range of an existing source while preserving the rest. It is not the same as prompt-first text-to-music generation, and ACE-Step explicitly skips the normal LM planning/code-generation path for repaint/cover/extract tasks. That matches the symptom: the request runs, but the newly padded region does not adhere to the prompt and sounds like unstructured diffusion.

## Correct Direction

Split continuation into two ACE-Step-aligned stages:

1. Generate coherent new music with ACE-Step `text2music`.
2. Stitch it after the selected source point.
3. Optionally use ACE-Step `repaint` only around the join, where it has real audio on both sides.

Repaint should smooth or regenerate the transition boundary. It should not be the primary mechanism for inventing the entire future section from silence.

## Intended Workflow

### Stage 1: Prompted continuation candidate

Call ACE-Step with:

- `task_type="text2music"`
- `thinking=true` so the LM/code planning path can be used
- `prompt` from the user
- `lyrics="[Instrumental]"` by default unless the UI later exposes lyrics
- `vocal_language="unknown"`
- `audio_duration=<new section seconds>`
- selected model/inference/seed/guidance settings
- BPM/key hints when provided

This should produce a real prompted music segment instead of asking repaint to hallucinate audio into silence.

### Stage 2: Build transition composite

Create a working composite:

- source audio up to the selected continuation point
- generated new section after that point

For UI playback, store both:

- raw generated section
- transition composite

The results list should make the composite the primary playable output.

### Stage 3: Optional boundary repaint

If the user sets an ACE repaint margin/crossfade:

- create a composite audio file with source + generated section
- run ACE-Step `repaint` over only the boundary region
- `repainting_start = continuation_point - repaint_margin`
- `repainting_end = continuation_point + repaint_margin` or a configurable forward boundary window
- use `chunk_mask_mode="explicit"`
- use ACE-Step repaint mode/strength/crossfade controls

This gives repaint real audio before and after the boundary, which matches its documented purpose.

If repaint margin is `0`, skip ACE-Step repaint and use only waveform stitching/crossfade if configured.

## UI / Settings

Expose controls based on real ACE-Step behavior:

- Source context seconds: used for preview/metadata and possibly prompt construction, not as fake model overlap.
- New section seconds: maps to text2music `audio_duration`.
- ACE repaint margin seconds: only used for optional boundary repaint.
- ACE repaint controls: only apply to Stage 3.
- ACE generation controls: apply to Stage 1 text2music.

Do not present repaint as the primary generation method for creating the full added section.

## Affected Files

- `src/autotransition/models/acestep_api.py`
- `src/autotransition/models/acestep.py`
- `src/autotransition/generation/`
- `src/autotransition/audio/`
- `src/autotransition/pipeline/source_selection.py`
- `src/autotransition/ui/app.py`
- `src/autotransition/ui/static/app.js`
- `src/autotransition/ui/static/index.html`
- `tests/test_acestep_api.py`
- `tests/test_ui_api.py`
- README/docs describing the generation flow

## Tradeoffs

- This is a larger change than repaint-only generation, but it matches ACE-Step's task model.
- Text2music may not perfectly continue the exact source groove on its own, but it should produce coherent prompt-following audio.
- Boundary repaint/crossfade becomes the continuity tool instead of the content-generation tool.

## Validation

- Unit-test Stage 1 payload sends `task_type="text2music"`, `thinking=true`, `audio_duration=<new section seconds>`, and no `src_audio`.
- Unit-test Stage 3 payload sends `task_type="repaint"` only for an existing composite audio file and only around the boundary region.
- Unit-test result metadata records raw generation path, composite path, and optional boundary repaint path.
- Run the full test suite.
- Manually generate with the same test settings that produced static before and confirm the primary output is coherent prompt-following music.
