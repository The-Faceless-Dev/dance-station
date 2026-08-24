"""Correct isolated foreground-position jumps in an alpha video.

The generated dance is allowed to move continuously. This stage only corrects
an isolated centroid outlier relative to a short neighboring-frame median, which
handles diffusion-frame jumps without recentering every frame or flattening the
choreography.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stabilize isolated alpha-video position jumps")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--threshold-px", type=float, default=12.0)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--strength", type=float, default=0.85)
    parser.add_argument("--crf", type=int, default=30)
    return parser.parse_args()


def _foreground_center(frame: Any, np: Any) -> tuple[float, float] | None:
    alpha = frame[:, :, 3]
    ys, xs = np.where(alpha >= 16)
    if len(xs) < 8:
        return None
    return (float(xs.min() + xs.max()) / 2.0, float(ys.min() + ys.max()) / 2.0)


def _median_center(centers: list[tuple[float, float] | None], np: Any) -> tuple[float, float] | None:
    values = [value for value in centers if value is not None]
    if len(values) < 3:
        return None
    return (
        float(np.median([value[0] for value in values])),
        float(np.median([value[1] for value in values])),
    )


def main() -> int:
    args = _args()
    if not args.input.is_file():
        raise SystemExit(f"alpha input was not found: {args.input}")
    if args.width < 64 or args.height < 64 or args.fps <= 0:
        raise SystemExit("invalid alpha-video dimensions or FPS")
    if args.window_size < 3 or args.window_size % 2 == 0:
        raise SystemExit("window size must be an odd number of at least 3")
    if args.threshold_px < 0 or not 0 <= args.strength <= 1:
        raise SystemExit("threshold must be non-negative and strength must be between 0 and 1")
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise SystemExit("alpha position stabilization requires opencv-python and numpy") from exc
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required for alpha position stabilization")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    decoder = subprocess.Popen(
        [
            ffmpeg,
            "-v",
            "error",
            "-c:v",
            "libvpx-vp9",
            "-i",
            str(args.input),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgba",
            "-s",
            f"{args.width}x{args.height}",
            "-r",
            f"{args.fps:.8f}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libvpx-vp9",
            "-pix_fmt",
            "yuva420p",
            "-auto-alt-ref",
            "0",
            "-crf",
            str(args.crf),
            "-row-mt",
            "1",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    frame_bytes = args.width * args.height * 4
    radius = args.window_size // 2
    frames: deque[Any] = deque()
    centers: deque[tuple[float, float] | None] = deque()
    corrected = 0
    total = 0

    def process_one(frame: Any, center_window: list[tuple[float, float] | None]) -> None:
        nonlocal corrected
        # The first queued frame is emitted now; later frames only provide the
        # local context used to identify an isolated positional outlier.
        center = center_window[0]
        target = _median_center(center_window, np)
        if center is not None and target is not None:
            dx = target[0] - center[0]
            dy = target[1] - center[1]
            distance = float((dx * dx + dy * dy) ** 0.5)
            if distance > args.threshold_px:
                matrix = np.float32([[1, 0, dx * args.strength], [0, 1, dy * args.strength]])
                frame = cv2.warpAffine(
                    frame,
                    matrix,
                    (args.width, args.height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0, 0),
                )
                corrected += 1
        encoder.stdin.write(frame.tobytes())

    try:
        while True:
            raw = decoder.stdout.read(frame_bytes)
            if not raw:
                break
            if len(raw) != frame_bytes:
                raise RuntimeError("alpha decoder ended with a partial frame")
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((args.height, args.width, 4)).copy()
            frames.append(frame)
            centers.append(_foreground_center(frame, np))
            total += 1
            if len(frames) < args.window_size:
                continue
            process_one(frames.popleft(), list(centers))
            centers.popleft()
        while frames:
            window = list(centers)
            if window:
                window = (window + [window[-1]] * radius)[: args.window_size]
            else:
                window = [None] * args.window_size
            process_one(frames.popleft(), window)
            centers.popleft()
    finally:
        if decoder.stdout:
            decoder.stdout.close()
        decoder_stderr = decoder.stderr.read().decode("utf-8", errors="replace") if decoder.stderr else ""
        decoder_return = decoder.wait()
        if encoder.stdin:
            encoder.stdin.close()
    encoder_stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    encoder_return = encoder.wait()
    if decoder_return != 0:
        raise SystemExit(f"alpha decoder failed with code {decoder_return}: {decoder_stderr[-4000:]}")
    if encoder_return != 0:
        raise SystemExit(f"alpha encoder failed with code {encoder_return}: {encoder_stderr[-4000:]}")
    if total == 0 or not args.output.is_file() or args.output.stat().st_size == 0:
        raise SystemExit("alpha position stabilization produced no frames")
    print(f"[alpha-stabilize] frames={total} corrected={corrected} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
