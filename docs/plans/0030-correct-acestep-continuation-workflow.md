# 0030 Correct ACE-Step Continuation Workflow

## Problem

Autotransition is still producing garbled/static continuation audio when extending a selected point in a source song. The current implementation treats ACE-Step REST `repaint` as the primary continuation engine and feeds it a short tail scaffold. That is too narrow and likely the wrong abstraction for the user goal.

The actual product goal is:

1. Load a full song.
2. Pick where the new music should continue from.
3. Generate a seamless addition in the prompted style.
4. Return listenable candidate results in the UI.

Current controls such as context seconds, overlap seconds, and repaint-existing mode are implementation choices. They should not be treated as requirements just because they exist in the current UI. Keep, rename, or remove them based on what ACE-Step actually accepts and what produces a clean continuation.

## Research Findings

- ACE-Step REST `repaint` is a source-audio editing task. It does not use the LM planning path the same way text-to-music generation does.
- ACE-Step can repaint/continue around a region, but the REST API does not expose a clear first-class `extend` task.
- Third-party ACE-Step extension workflows use a full-source/total-duration approach rather than uploading only a short tail as the whole source.
- The current short-tail repaint path can technically run but may not provide enough musical structure, metadata, or planning to create a coherent new section.
- The local ACE-Step runtime exposes repaint timing as `repainting_start` and `repainting_end`.
- The local ACE-Step runtime exposes boundary blending as `repaint_latent_crossfade_frames` and `repaint_wav_crossfade_sec`.
- The local ACE-Step runtime exposes repaint behavior as `chunk_mask_mode`, `repaint_mode`, and `repaint_strength`.
- The local ACE-Step runtime does not expose a parameter named `overlap`. Any overlap-like UI control in Autotransition must map to a real ACE-Step concept or to our own post-processing stitch crossfade.

## Intended Approach

### 1. Replace Autotransition-specific overlap assumptions with ACE-Step concepts

Rename the internal workflow around continuation rather than repaint scaffolds:

- user-selected continuation point
- optional source context window
- desired new-section duration
- prompt/style guidance
- generated continuation candidates

Do not keep a control named or modeled as "overlap" unless the implementation proves it is needed. If a boundary control is useful, map it explicitly to one of:

- ACE-Step repaint range: `repainting_start` / `repainting_end`
- ACE-Step latent boundary blend: `repaint_latent_crossfade_frames`
- ACE-Step waveform boundary blend: `repaint_wav_crossfade_sec`
- Autotransition post-processing stitch crossfade after generation

If none of those are part of the chosen continuation strategy, remove the overlap control from the primary workflow.

### 2. Add an ACE-Step continuation strategy layer

Create a strategy module that can choose between:

- native ACE-Step extension if the installed runtime exposes a working pipeline path
- full-context REST repaint fallback if native extension is not available
- generate-and-stitch fallback if repaint cannot produce clean outpainting

This separates "what the app needs" from "which ACE-Step API shape happens to be available."

### 3. Verify the native ACE-Step path before wiring UI defaults

Add a small internal diagnostic command or helper that compares:

- current short-tail REST repaint
- full-source/full-context REST repaint
- native pipeline extension if available

Use the same selected marker, prompt, duration, seed, model, and settings. The goal is to identify which path produces coherent continuation before making it the default.

### 4. Prefer the context shape ACE-Step actually needs

For continuation generation, stop assuming the uploaded source should be only the current `context + overlap` window.

Use either:

- the full source song up to the continuation point, when ACE-Step can handle it, or
- a larger configurable context window with clear limits if full source is too expensive.

The app can still export only the needed generated addition after the model returns audio.

The context-size control should mean "how much source audio ACE-Step receives before the continuation point." It should not imply that ACE-Step supports a separate overlap parameter.

### 5. Make instrumental and prompt intent explicit

When the user does not provide lyrics, send explicit instrumental intent instead of leaving vocal behavior ambiguous:

- lyrics: `[Instrumental]`
- vocal language / vocal mode: instrumental or none, depending on the ACE-Step field supported by the runtime
- prompt remains the user style description

Expose lyrics/vocal controls later, but default to the non-vocal continuation path for this tool.

### 6. Keep the UI simple

The primary UI should stay focused on:

- source audio
- continuation point
- style prompt
- new-section seconds
- generated candidates
- status/logs

Advanced ACE-Step settings can remain available when they correspond to real runtime parameters:

- repaint start/end behavior, if the chosen strategy uses repaint
- latent crossfade frames
- waveform crossfade seconds
- repaint mode
- repaint strength
- mask mode
- inference steps/guidance/seed/model
- lyrics/vocal intent

Do not expose fake knobs. Do not expose "overlap" just because Autotransition used that term earlier.

## Affected Files

- `src/autotransition/pipeline/source_selection.py`
- `src/autotransition/audio/selection.py`
- `src/autotransition/models/acestep_api.py`
- `src/autotransition/models/acestep.py`
- `src/autotransition/ui/app.py`
- `src/autotransition/ui/static/app.js`
- `src/autotransition/ui/static/index.html`
- `tests/test_acestep_api.py`
- `tests/test_source_selection.py`
- `tests/test_ui_api.py`
- README/runtime docs if commands or recommended workflow change

## Tradeoffs

- Full-source or larger-context continuation may be slower and use more memory, but it gives ACE-Step more musical structure.
- Native pipeline integration may be less stable than REST if ACE-Step changes internals, but it may be required for true continuation quality.
- Generate-and-stitch may produce better musical coherence than repaint-only outpainting, but boundary matching needs careful stitching and metadata tracking.

## Risks

- ACE-Step 1.5 may not expose a reliable native extension path in the installed runtime.
- Long source context may exceed practical runtime limits on low-VRAM machines.
- The API may accept parameters that are documented but ignored internally for repaint tasks.

## Validation

- Add tests proving the continuation plan no longer hardcodes overlap as a user-facing requirement.
- Add tests for payload construction for the selected continuation strategy.
- Run the existing test suite.
- Manually test one short prompt with a selected source marker and verify the UI returns a playable result in the results list.
- Compare at least one generated continuation from the old short-tail repaint path against the chosen corrected path.
