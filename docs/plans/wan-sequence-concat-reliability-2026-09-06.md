# Wan Sequence Concat Reliability

## Objective

Make final RGB and alpha sequence composition consume every ordered segment,
bridge, and loop bridge without relying on timestamp behavior from the FFmpeg
concat demuxer.

## Approach

- Replace the demuxer-based composition in the generative-dance video helpers
  with an explicit FFmpeg concat filter graph.
- Keep the existing canvas, FPS, codec, alpha, and output contracts unchanged.
- Add focused tests proving that all inputs, including the final loop bridge,
  are represented in the composition command and output duration.
- Republish the shallow Wan overlay and rerun the same two-segment plus loop
  Vast test before treating the worker as ready.

## Risks

- The filter graph must remain bounded for long sequences; it will be built
  from the already-normalized inputs and will not hold all decoded frames in
  application memory.
- Alpha and RGB paths need separate codec/filter handling so transparency is
  preserved.
