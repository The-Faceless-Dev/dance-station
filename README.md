# Dance Station

Dance Station is an AI music workstation distributed by **The Faceless Dancer**. It is built for creators, streamers, visualizers, rhythm-game experiments, and other media projects that need practical tools for generating, extending, separating, and organizing music clips.

The app currently includes three main work areas:

- **Autotransition**: continue a source song from a selected point and create a natural transition into newly generated music.
- **Extraction**: separate useful musical parts from a song using ACE-Step extraction.
- **Generation**: create new music directly from a prompt.

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

## Generation

The Music Generation section creates new music directly from a text prompt.

It exposes the generation controls that are useful for the active ACE-Step model path, including duration, seed, steps, guidance, shift, sampler mode, tiled decoding, DCW, and velocity settings. Turbo and Base generation use different defaults based on the working settings found during testing.

Completed generations are listed in the UI with playable audio and saved metadata.

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

## Project Layout

```text
src/autotransition/
  audio/        Audio probing, slicing, silence, merge, and composition helpers.
  models/       ACE-Step runtime/API integration.
  pipeline/     Transition planning and scaffold state.
  scoring/      Candidate scoring interfaces.
  ui/           Local web UI and API.
  cli.py        Command-line entry point.
  config.py     Central configuration defaults.
```

## Current Limitations

- Candidate scoring is only an interface placeholder.
- ACE-Step first-run runtime/model downloads can take a long time and require enough disk space.
- Track extraction uses ACE-Step Base and may require more startup/download time than transition generation.
- Audio loading, merging, and scaffold generation depend on `pydub` and `ffmpeg`.
