# Dance Station

Dance Station is an AI-assisted audio workstation distributed by **The Faceless Dancer**. It started as an Autotransition pipeline for extending music with ACE-Step, and is growing into a full local audio editing and music creation app.

The goal is to give creators, streamers, visualizers, rhythm-game experimenters, and media builders one practical local workspace for generating, extending, separating, editing, performing, arranging, and reusing music clips.

The app currently includes eight main work areas:

- **Autotransition**: continue a source song from a selected point and create a natural transition into newly generated music.
- **Extraction**: separate useful musical parts from a song using ACE-Step extraction.
- **Sound Generation**: create new music directly from a prompt and generate short sound effects from text.
- **Voice Work**: manage reusable voice assets, train clones, and convert speech with an embedded RVC runtime.
- **LoKr Training**: build captioned music datasets, preprocess them with Side-Step, train LoKr style adapters, and use completed adapters during generation.
- **Instrument Lab**: create and edit instrument performances with tracks, piano-roll editing, computer-key input, sampled instruments, and SFZ imports.
- **Audio Editor**: edit, repair, record, export, and arrange audio with the integrated AudioMass editor.
- **Rhythm Beat Lab**: author rhythm-game beat charts from a source song and extracted layers, then save final chart assets into the local library.

All saved outputs become reusable Dance Station assets. Transitions, generations, extractions, merges, edits, instrument clips, instrument tracks, voice assets, speech generations, datasets, LoKr adapters, and rhythm beat charts can be labeled, played or inspected in the UI, loaded into compatible tabs, and reused as source material for the next step.

The current command-line package is still named `autotransition`, so setup and run commands use that executable.

## General Setup

Use a conda environment so audio and model dependencies stay isolated from your system Python.

```powershell
conda env create -f environment.yml
conda activate autotransition
python -m pip install -e ".[dev]"
```

First-time setup:

```powershell
autotransition setup
```

This installs the app dependencies plus the external ACE-Step and Side-Step runtimes used by the generation, extraction, transition, and LoKr training workflows.

Run the full local app:

```powershell
autotransition run
```

Then open the printed local URL, usually:

```text
http://127.0.0.1:7860
```

The UI includes ACE-Step runtime status, system checks, recent outputs, playable results, and local logs. Use this command if you need to check the local environment:

```powershell
autotransition doctor
```

`autotransition run` starts the ACE-Step API when needed and stops the ACE-Step process it started when the app shuts down. If ACE-Step was already running before the app started, Dance Station leaves that process alone.

## Autotransition

Autotransition is the continuation workflow for turning the end of an existing song or generated clip into a prompted next section.

The primary workflow is:

1. Load a source song.
2. Select the point where new music should continue.
3. Enter a target prompt and transition settings.
4. Generate a new prompted section with ACE-Step text-to-music.
5. Use ACE-Step repaint across the boundary so the continuation transitions naturally.
6. Listen to the result in the UI, then optionally use that result as the next source.

Autotransition is currently built around the working ACE-Step 1.5 XL Turbo runtime path. The app creates repaint scaffolds internally from the selected source and generated continuation. Generated outputs include a `Use as Source` action so you can chain another transition from the result.

## Extraction

The Track Extraction section lets you upload a song, choose a part to extract, and listen to the completed separations in the UI.

Supported extraction targets include vocals, drums, bass, guitar, synth, strings, and other ACE-Step-supported parts. Extraction uses ACE-Step Base in the runtime's secondary slot while the transition workflow stays on the active Turbo runtime path.

Completed extractions are saved as reusable results. You can label individual extracted parts, select multiple parts, merge them into a new labeled result, and play those merged outputs in the UI.

The Extraction tab also includes a vocal separation workflow powered by a managed UVR-style MDX runtime. It can separate a source song into instrumental and vocals stems, save the instrumental stem as the primary result, and add both outputs to the local library for reuse in other tabs. Advanced cleanup controls are available for model selection, segment size, overlap, and denoise tuning when you want to push bleed-through lower.

## Sound Generation

The Sound Generation section creates new music directly from a text prompt and also exposes a dedicated sound effects panel powered by TangoFlux.

Music generation supports instrumental generation by default, plus optional vocal generation with a lyrics field and vocal language hint. Voice character is controlled through the prompt, such as "female vocal", "male baritone", "choir", or "rap verse"; ACE-Step does not expose a fixed singer selector through the local runtime path used here.

The music controls expose the settings that are useful for the active ACE-Step model path, including duration, seed, steps, guidance, shift, sampler mode, tiled decoding, DCW, and velocity settings. Turbo and Base generation use different defaults based on the working settings found during testing.

The sound effects panel uses TangoFlux text-to-audio generation for short cues and effect beds. It accepts a text prompt, duration, step count, and output format. TangoFlux is installed through the normal project setup path and its outputs are saved alongside the other local assets.

Completed generations are listed in the UI with playable audio and saved metadata. If you have trained LoKr adapters, the Sound Generation tab can select one, set its strength, and apply it to a compatible ACE-Step model while generating.

## Voice Work

Voice Work adds reusable voice assets, local voice training, sample conversion, and text-to-speech generation through an embedded RVC runtime.

The first-pass workflow is:

1. add a voice asset from one or more reference audio files and optional embedding files
2. train or refresh a voice clone from those local assets
3. select that saved clone inside the Voice Work tab
4. enter text or upload a sample for conversion
5. run the embedded RVC workflow and save the result as a reusable speech asset

Voice assets are directory-backed. Each voice can include copied reference audio, embeddings, and metadata under `data/voice-work/voices/`. Generated speech and converted samples are saved under `data/voice-work/outputs/` and become part of the local library.

Dance Station manages the local RVC runtime from the app and exposes runtime status plus launch controls directly in the Voice Work tab.

## LoKr Training

The LoKr Training section builds reusable style adapters with Side-Step.

The workflow is:

1. Create or load a dataset.
2. Add audio from disk or from existing Dance Station creations.
3. Add captions and lyrics for each entry. Instrumental entries can use `[Instrumental]`.
4. Set dataset-level defaults such as genre, language, trigger tag, and whether the whole set is instrumental.
5. Run preprocess to build the Side-Step tensor dataset.
6. Start LoKr training from the preprocessed tensors.
7. Monitor epoch, step, and loss progress in the UI.
8. Use the completed LoKr from the Generation tab.

Each preprocess or training run is saved under `data/lokr-training/runs/` with its own metadata, logs, and outputs. Reusing the same dataset creates a new training run and a new adapter; it does not overwrite earlier trained LoKrs.

The UI prevents starting another Side-Step preprocess/training run while one is already active. Running jobs can be stopped from the LoKr Training panel, and the log view can be cleared without deleting saved run files or trained adapters.

Default LoKr training is configured for practical local use: batch size 1, gradient accumulation, gradient checkpointing, encoder offload, and AdamW 8-bit. These defaults are intended to make training possible on smaller GPUs while preserving useful quality and speed.

## Instrument Lab

The Instrument Lab section creates playable instrument clips and editable instrument tracks directly in the browser.

It includes:

- computer-key note entry
- clickable piano keys
- transport-synced recording with a short count-in
- a piano-roll editor with cursor positioning, zooming, scrolling, note selection, group move, delete, copy, and paste
- multiple instrument tracks
- imported Dance Station creations as audio layers
- per-track playback/mute controls for recording and rendering
- composite clip saving
- individual editable instrument track saving
- preview rendering and playable saved results

The instrument system uses a manifest-backed instrument bank. Current support includes built-in synth patches, packaged sampled instruments, and imported SFZ instruments with uploaded sample files. Imported SFZ instruments are converted into Dance Station's sample-region format and appear under the SoundFonts / User Instruments category.

Saved Instrument Lab clips and instrument tracks are listed in the UI, can be loaded back into Instrument Lab for editing, and become reusable assets for the other tabs.

## Audio Editor

The Audio Editor section embeds a local vendored copy of AudioMass inside Dance Station.

AudioMass provides browser-side waveform editing, selections, trim/cut/copy/paste workflows, effects, repair tools, recording, export, and multitrack editing with clips, fades, crossfades, mixer controls, session save/open, and mixdown.

The editor can still open files from your device through AudioMass itself. Dance Station also lists prior app outputs in the editor tab, including transitions, music generations, instrument clips, extractions, merges, and saved edits. Each item is shown with its label and category, can be opened directly in AudioMass, and can be relabeled from the asset list.

Use `Save Edited Result` to store the current embedded AudioMass edit back into Dance Station under the edits category with a custom name. Saved edits become reusable assets in the same editor list.

Dance Station serves the editor from the same local app at:

```text
http://127.0.0.1:7860/audiomass/
```

The integrated copy preserves the upstream AudioMass MIT license and third-party notices under `src/autotransition/vendor/audiomass/`.

## Rhythm Beat Lab

Rhythm Beat Lab moves rhythm chart authoring into standalone Dance Station instead of keeping it in a separate site admin workflow.

The current standalone workflow lets you:

1. create a local rhythm beat project
2. attach a full source song from disk or from an existing Dance Station asset
3. extract ACE-Step stems directly from the attached source song and add them back into the project as linked tracks
4. add other extracted or generated layers as linked tracks
5. run local hybrid-style beat analysis on the full song or any linked track
6. extract lyrics locally from the source song with Faster-Whisper, then edit them in place
7. drag-select beat ranges on the chart and save them as named beat-selection layers
8. merge saved beat-selection layers into candidate chart results, or promote a single layer directly when that is the better result
9. promote one candidate as the final chart and save it into the local library as a `rhythm_game` asset

Final rhythm beat assets are exported as chart JSON plus the linked song audio so they can publish through the same library flow as the other Dance Station asset types.

## ACE-Step Runtime

Normal users should use:

```powershell
autotransition setup
```

Then:

```powershell
autotransition run
```

First-run runtime and model downloads can take a while and require enough disk space. Dance Station manages the ACE-Step runtime through the app command so users do not need to start ACE-Step separately.

LoKr training uses Side-Step, which is installed by the same setup command. To install only the Side-Step runtime later, use:

```powershell
autotransition runtime setup-sidestep
```

## Project Layout

```text
src/autotransition/
  audio/        Audio probing, slicing, silence, merge, and composition helpers.
  models/       ACE-Step runtime/API integration.
  pipeline/     Transition planning and scaffold state.
  runtime/      External runtime setup/status helpers for ACE-Step, Side-Step, and RVC.
  scoring/      Candidate scoring interfaces.
  ui/           Local web UI and API.
  ui/static/    Browser UI assets, including the instrument bank.
  vendor/       Vendored browser tools such as AudioMass.
  cli.py        Command-line entry point.
  config.py     Central configuration defaults.
```

## Avatar Worker

The repository also contains a paid-job-safe avatar worker boundary:

```powershell
$env:AVATAR_IMAGE_COMMAND = 'python tools/avatar/flux2_klein_generate.py --prompt-file {prompt_file} --negative-prompt-file {negative_prompt_file} --output {output} --seed {seed} --reference-image {reference_image}'
$env:AVATAR_MESH_COMMAND = 'python C:\models\stable-fast-3d\run.py {image} --pretrained-model C:\models\stable-fast-3d-checkpoint --output-dir {output_dir} --texture-resolution 2048 --remesh_option triangle --target_vertex_count 100000'
$env:AVATAR_RIG_COMMAND = 'python tools/tokenrig/adaptive_runner.py --skintokens-repo C:\models\SkinTokens --input {input} --output {output} --manifest-output {manifest_output} --profile auto --use-transfer'
autotransition avatar-worker --host 0.0.0.0 --port 8090
```

For the TRELLIS.2 production worker, use the standard layered image published
from `containers/avatar-worker/Dockerfile.trellis2.salad`. The image includes
the pinned FLUX.2 Klein image stage, Qwen text encoder, TRELLIS.2 checkpoints,
and SkinTokens rigging checkpoints. The worker is configured to fail rather
than download weights at runtime. Salad still requires GHCR registry
authentication when creating the container group, even for a public GHCR
image; provide a read-only GitHub Packages credential to the group
configuration and never put it in the image or repository.

The image emits structured per-job JSONL logs for model loading, inference,
rigging, validation, retries, cleanup, and failures. The local RTX 3080 can
load the full pipeline and complete FLUX.2 plus TRELLIS sparse/shape sampling,
but full mesh export requires the production RTX 3090 memory target.

The image command is deliberately an adapter boundary because FLUX.2 Klein
inference entry points vary by pinned checkout. The worker adds the humanoid
full-body/A-pose policy, persists job state under `AVATAR_ARTIFACT_ROOT`,
validates the source image, skin, semantic `humanoid-v1` manifest, weights, and
an actual deterministic glTF skinning deformation report, and retries a broken output up to three total
attempts. A terminal failure is returned with `refundRequired=true`; payment
keys remain in the launch server, which owns the idempotent refund.

## Current Limitations

- Candidate scoring is only an interface placeholder.
- ACE-Step first-run runtime/model downloads can take a long time and require enough disk space.
- Track extraction uses ACE-Step Base and may require more startup/download time than transition generation.
- LoKr training requires the Side-Step runtime and can take substantial GPU time depending on dataset size and training settings.
- Voice Work depends on a reachable local RVC runtime.
- TangoFlux sound effects follow the upstream TangoFlux license and are intended for non-commercial research-style use unless you have the right to use and redistribute that model separately.
- Audio loading, merging, and scaffold generation depend on `pydub` and `ffmpeg`.
- SFZ import supports a practical subset of SFZ regions and sample mapping. Native binary `.sf2` import is not implemented yet.
