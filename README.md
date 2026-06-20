# Autotransition

Autotransition is an early-stage pipeline for creating AI-generated music transitions. It is distributed by **The Faceless Dancer** and is intended to become a public, reusable tool for creators, streamers, visualizers, rhythm-game experiments, and other media projects.

The first workflow focuses on ACE-Step continuation from a selected point in a source song:

1. Load a source song.
2. Select the point where new music should continue.
3. Generate a new prompted section with ACE-Step text-to-music.
4. Stitch the generated section after the selected point.
5. Optionally repaint only the boundary region when ACE repaint margin is enabled.
6. Return playable generated candidates for review and export.

ACE-Step runtime integration is managed through the local app command. The transition scaffold is created internally from the selected part of the source song, then sent to ACE-Step for repaint/continuation generation.

## Setup

Use a conda environment so audio and model dependencies stay isolated from your system Python.

```powershell
conda env create -f environment.yml
conda activate autotransition
python -m pip install -e ".[dev]"
```

You also need `ffmpeg` available on your `PATH` for broad audio format support. Conda installs it through `environment.yml`.

## Quickstart

Prepare a repaint scaffold from an existing clip:

```powershell
autotransition scaffold path\to\clip.wav --preset smooth-continuation
```

By default, scaffold files and metadata are written under `data/scaffolds/`, which is ignored by git.

## Local UI

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

The UI includes transition presets, direct settings, ACE-Step model status, system status, recent outputs, and local logs.

The main UI workflow is `Generate Transition`: load a song, select the continuation point, enter the target prompt/settings, choose an ACE-Step model, and generate. Internal repaint scaffolds are created automatically and shown only as output details/logs.

`autotransition run` starts the ACE-Step API when needed and stops the ACE-Step process it started when the app shuts down. If ACE-Step was already running before the app started, Autotransition leaves that process alone.

## ACE-Step Runtime Setup

Normal users should use:

```powershell
autotransition setup
```

Then:

```powershell
autotransition run
```

Advanced runtime/debug commands are documented in `docs/ace-step-runtime.md`.

## Presets

Initial presets are intentionally creator-friendly:

- `smooth-continuation`
- `energy-build`
- `breakdown`
- `genre-shift`
- `dj-bridge`

Presets set practical defaults for source context length, generated duration, prompt language, and optional BPM/key hints. ACE-Step repaint margin defaults to `0` so repaint is not used as the main content generator; advanced users can enable it for boundary smoothing from the CLI or UI.

## ACE-Step Models

Autotransition keeps model selection separate from transition planning. Repaint-capable ACE-Step profiles are listed through:

```powershell
autotransition models list
```

Install a selected profile from Hugging Face:

```powershell
autotransition models install acestep-v15-turbo
```

The future generation command and UI should also be able to download the selected model automatically when it is missing, after clearly showing the model name, source, local path, and hardware notes.

## Project Layout

```text
src/autotransition/
  audio/        Audio slicing, silence, and stitching helpers.
  models/       Interfaces for future ACE-Step repaint integration.
  pipeline/     Transition planning and scaffold state.
  scoring/      Candidate scoring interfaces.
  cli.py        Command-line entry point.
  config.py     Central configuration defaults.
  presets.py    Named transition presets.
```

## Current Limitations

- Candidate scoring is only an interface placeholder.
- ACE-Step first-run model downloads can take a long time and require enough disk space for the selected model.
- Audio loading and scaffold generation depend on `pydub` and `ffmpeg`.

See `docs/plans/` for implementation plans.
