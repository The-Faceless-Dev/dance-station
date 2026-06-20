# UI Notes

Autotransition should feel like a creator-facing tool, not a raw model control panel.

## Primary Workflow

The main screen should make the transition pipeline understandable at a glance:

1. Source clip
2. Selected tail/context
3. Transition settings
4. Target prompt or style
5. Generation status
6. Candidate outputs
7. Chosen/exported result

## Configuration Model

Use simple presets first, then expose advanced controls when needed.

Preset-level controls:

- Smooth continuation
- Energy build
- Breakdown
- Genre shift
- DJ-friendly bridge

Advanced controls:

- Context seconds
- ACE-Step repaint margin seconds
- New section seconds
- Output format
- BPM hint
- Key hint
- Seed
- Candidate count
- Model path/profile
- Scoring rules

## Product Direction

The UI should let non-technical users generate useful transitions without knowing the model internals. Technical users should still be able to inspect and modify the underlying settings, metadata, prompts, and output folders.
