# MOSS Hierarchical Visual Events

## Objective

Produce useful musical event data for visual synchronization without asking the
language model to emit an exhaustive 80 ms event grid. Preserve the raw MOSS
responses while generating a compact structured view containing song structure,
harmony, lyrics/voices, and salient within-section visual-impact events.

## Approach

1. Keep the measured 80 ms timeline as the source for dense energy and timing
   features.
2. Retain focused MOSS passes for overview, rhythm, harmony, and lyrics/voices,
   but make the rhythm pass request sparse, meaningful events rather than every
   beat/onset.
3. Add a bounded visual-impact pass that labels meaningful builds, drops,
   breaks, fills, transitions, vocal/instrument entries, and other changes with
   timestamps, intensity, emotional direction, evidence, and confidence.
4. Make parsing tolerant of pass-specific responses: useful fields remain
   usable when optional wrapper fields are absent or a response is partially
   malformed. Raw text remains authoritative and is always stored.
5. Keep the worker's 30-minute audio limit and remove no data because of
   normalization errors. Never retry solely because structured parsing fails.

## Affected Files

- `src/autotransition/moss_music/prompting.py`: pass definitions and compact
  visual-event schema.
- `src/autotransition/moss_music/parser.py`: tolerant pass parsing and visual
  event normalization.
- `src/autotransition/moss_music/worker.py`: merge the new pass and preserve
  partial structured results.
- `src/autotransition/moss_music/analysis.py`: expose visual-impact events in
  the analysis artifact.
- `tests/moss_music/test_moss_music.py`: parser, merge, and worker coverage.

## Tradeoffs and Risks

- MOSS timestamps and emotional labels remain model estimates; exact timing is
  anchored by measured audio features where possible.
- Sparse semantic events provide less raw detail than an exhaustive generated
  event list, but avoid repetition, truncation, and unusable JSON.
- The raw pass artifacts remain available for future offline recovery or parser
  improvements.

## Verification

- Run focused MOSS tests and the full available test suite.
- Run a complete `crown.mp3` analysis on the MOSS worker.
- Verify non-empty raw artifacts, structured visual-impact events or explicit
  warnings, artifact checksums, and recorded runtime metrics.
