"""Low-cost frame diagnostics for VACE composition boundaries."""

from __future__ import annotations

from hashlib import sha256
import subprocess
from pathlib import Path
from typing import Any

from autotransition.generative_dance.video import probe_video, resolve_ffmpeg

from .video import frame_count


def _read_frame(path: Path, index: int) -> bytes:
    probe = probe_video(path)
    count = frame_count(probe, max(1, int(round(probe.fps))))
    if index < 0 or index >= count:
        raise ValueError(f"frame index {index} is outside {path.name} ({count} frames)")
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for VACE frame diagnostics")
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            f"select=eq(n\\,{index})",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0 or not result.stdout:
        detail = result.stderr.decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"could not decode frame {index} from {path}: {detail}")
    expected = probe.width * probe.height * 3
    if len(result.stdout) != expected:
        raise RuntimeError(
            f"decoded frame {index} from {path.name} has {len(result.stdout)} bytes; expected {expected}"
        )
    return result.stdout


def _frame_summary(frame: bytes) -> dict[str, Any]:
    if not frame:
        return {"sha256": None, "meanRgb": None}
    channels = [0, 0, 0]
    for index in range(0, len(frame), 3):
        channels[0] += frame[index]
        channels[1] += frame[index + 1]
        channels[2] += frame[index + 2]
    pixels = len(frame) // 3
    return {
        "sha256": sha256(frame).hexdigest(),
        "meanRgb": [round(value / pixels, 3) for value in channels],
    }


def compare_frame_bytes(first: bytes, second: bytes) -> dict[str, Any]:
    if len(first) != len(second):
        return {
            "sameShape": False,
            "meanAbsoluteByteDelta": None,
            "changedByteRatio": None,
        }
    changed = 0
    difference = 0
    for left, right in zip(first, second):
        delta = abs(left - right)
        difference += delta
        changed += delta > 0
    size = len(first)
    return {
        "sameShape": True,
        "meanAbsoluteByteDelta": round(difference / size, 4) if size else 0.0,
        "changedByteRatio": round(changed / size, 6) if size else 0.0,
    }


def compare_video_frames(path: Path, first_index: int, second_index: int) -> dict[str, Any]:
    first = _read_frame(path, first_index)
    second = _read_frame(path, second_index)
    return {
        "video": str(path),
        "firstFrame": first_index,
        "secondFrame": second_index,
        "first": _frame_summary(first),
        "second": _frame_summary(second),
        "comparison": compare_frame_bytes(first, second),
    }


def part_boundaries(frame_counts: list[int]) -> list[dict[str, int]]:
    """Return the exact final-timeline indices where adjacent parts meet."""

    boundaries: list[dict[str, int]] = []
    cursor = 0
    for part_index, count in enumerate(frame_counts):
        if count < 1:
            raise ValueError(f"part {part_index} has no frames")
        if part_index:
            boundaries.append(
                {
                    "partIndex": part_index,
                    "leftFrame": cursor - 1,
                    "rightFrame": cursor,
                }
            )
        cursor += count
    return boundaries


def analyze_video_seams(path: Path, boundaries: list[dict[str, int]]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    probe = probe_video(path)
    count = frame_count(probe, max(1, int(round(probe.fps))))
    for boundary in boundaries:
        try:
            report = compare_video_frames(path, boundary["leftFrame"], boundary["rightFrame"])
            report["partIndex"] = boundary["partIndex"]
            if boundary["leftFrame"] > 0:
                report["leftPartInternal"] = compare_video_frames(
                    path,
                    boundary["leftFrame"] - 1,
                    boundary["leftFrame"],
                )
            if boundary["rightFrame"] + 1 < count:
                report["rightPartInternal"] = compare_video_frames(
                    path,
                    boundary["rightFrame"],
                    boundary["rightFrame"] + 1,
                )
        except Exception as exc:
            report = {
                "partIndex": boundary["partIndex"],
                "leftFrame": boundary["leftFrame"],
                "rightFrame": boundary["rightFrame"],
                "errorType": type(exc).__name__,
                "error": str(exc),
            }
        reports.append(report)
    return reports
