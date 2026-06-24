# Autotransition

Autotransition is an early-stage pipeline for creating AI-generated music transitions. It is distributed by **The Faceless Dancer** and is intended to become a public, reusable tool for creators, streamers, visualizers, rhythm-game experiments, and other media projects.

The primary workflow focuses on ACE-Step continuation from a selected point in a source song:

1. Load a source song.
2. Select the point where new music should continue.
3. Enter a target prompt and transition settings.
4. Generate a new prompted section with ACE-Step text-to-music.
5. Use ACE-Step repaint across the boundary so the continuation transitions naturally.
6. Listen to the result in the UI, then optionally use that result as the next source.

ACE-Step runtime integration is managed through the local app command. Autotransition is currently built around the working ACE-Step 1.5 XL Turbo runtime path. The app keeps generation requests on the active ACE runtime path and creates repaint scaffolds internally from the selected source and generated continuation.

## Setup

Use a conda environment so audio and model dependencies stay isolated from your system Python.

```powershell
conda env create -f environment.yml
conda activate autotransition
python -m pip install -e ".[dev]"
```

You also need `ffmpeg` available on your `PATH` for broad audio format support. Conda installs it through `environment.yml`.

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

The UI includes direct prompt/settings controls, ACE-Step runtime status, recent outputs, playable results, and local logs.

The main UI workflow is `Generate Transition`: load a song, select the continuation point, enter the target prompt/settings, and generate. Internal repaint scaffolds are created automatically and shown only as output details/logs. Generated outputs include a `Use as Source` action so you can chain another transition from the result.

The app also includes a `Track Extraction` tab. Upload a song, choose a track such as vocals, drums, bass, guitar, synth, or strings, then run ACE-Step extraction. Extraction uses ACE-Step Base in the runtime's secondary slot while the transition workflow stays on the active Turbo runtime path. Completed extractions are listed in the UI with playable audio.

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

The current UI does not expose model selection. It presents the workflow as ACE-Step 1.5 XL Turbo and uses the active ACE-Step runtime path that the app starts.

## Project Layout

```text
src/autotransition/
  audio/        Audio slicing, silence, and stitching helpers.
  models/       ACE-Step runtime/API integration.
  pipeline/     Transition planning and scaffold state.
  scoring/      Candidate scoring interfaces.
  cli.py        Command-line entry point.
  config.py     Central configuration defaults.
```

## Current Limitations

- Candidate scoring is only an interface placeholder.
- ACE-Step first-run runtime/model downloads can take a long time and require enough disk space.
- Track extraction uses ACE-Step Base and may require more startup/download time than transition generation.
- Audio loading and scaffold generation depend on `pydub` and `ffmpeg`.
