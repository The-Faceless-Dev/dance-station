# Seamless Transition Boundary

Status: superseded by `0038-acestep-repaint-transition.md`.

This plan proposed waveform trimming/crossfading after generation. That is not
the right primary solution for Autotransition: the transition should be produced
by ACE-Step's audio-conditioned repaint path, not by post-editing an
unconditioned text-to-music clip.

## Problem

Generation now works, but the final composite is not a transition. Autotransition currently writes:

```text
source audio up to marker + full generated clip from 0s
```

If the generated clip starts with silence, a fade-in, or a loose intro, that silence lands directly after the selected source marker. The result is a gap and then the generated section starts.

## Approach

1. Add post-generation boundary preparation before stitching:
   - Detect leading near-silence in the raw generated audio.
   - Trim that leading silence up to a conservative maximum.
   - Keep the raw generated file unchanged for inspection.

2. Add a short waveform crossfade at the join:
   - Use the last `N` milliseconds of source before the marker.
   - Crossfade into the trimmed generated section.
   - Default to a practical value, such as `750ms`.

3. Keep the behavior configurable:
   - Add central defaults for generated-leading-silence trim and stitch crossfade.
   - Keep the UI simple initially; advanced controls can be exposed after the default sounds right.

4. Preserve metadata:
   - Save trim duration and crossfade duration in `composite.json`.
   - Keep raw generation path and final composite path separate.

## Affected Files

- `src/autotransition/audio/compose.py`
- `src/autotransition/config.py`
- `src/autotransition/ui/app.py`
- `tests/test_audio_scaffolds.py`
- `tests/test_ui_api.py`

## Risks

- Silence detection can over-trim very quiet intentional intros if thresholds are too aggressive.
- Crossfade can smear transients if too long.
- This is waveform-level smoothing, not true musical repaint. It should remove the blank gap and make the boundary usable, but later ACE-Step boundary repaint may still improve musical continuity.
