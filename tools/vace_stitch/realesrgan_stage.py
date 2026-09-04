"""Upscale an RGB video with the project-owned Real-ESRGAN stage."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# BasicSR versions used by the official Real-ESRGAN runtime still import the
# removed torchvision.transforms.functional_tensor compatibility module.
try:
    import torchvision.transforms.functional_tensor as _functional_tensor
except ModuleNotFoundError:  # pragma: no cover - depends on torchvision version
    import torchvision.transforms.functional as _functional_tensor

    sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor

from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

from autotransition.generative_dance.video import probe_video, resolve_ffmpeg


def _build_upsampler(model_path: Path, scale: int, tile_size: int, fp16: bool) -> RealESRGANer:
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=scale,
    )
    return RealESRGANer(
        scale=scale,
        model_path=str(model_path),
        model=model,
        tile=tile_size,
        tile_pad=16,
        pre_pad=0,
        half=fp16 and torch.cuda.is_available(),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def _run(input_video: Path, output_video: Path, model_path: Path, scale: int, tile_size: int, fp16: bool) -> None:
    if scale not in {2, 4}:
        raise RuntimeError(f"Real-ESRGAN supports scale 2 or 4, got {scale}")
    if not model_path.is_file() or model_path.stat().st_size == 0:
        raise RuntimeError(f"Real-ESRGAN checkpoint is missing or empty: {model_path}")
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for the Real-ESRGAN stage")
    source = probe_video(input_video)
    if source.width < 1 or source.height < 1 or source.fps <= 0:
        raise RuntimeError(f"input video has invalid dimensions or FPS: {source.to_dict()}")

    output_video.parent.mkdir(parents=True, exist_ok=True)
    reader = cv2.VideoCapture(str(input_video))
    if not reader.isOpened():
        raise RuntimeError(f"could not open input video: {input_video}")
    writer = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{source.width * scale}x{source.height * scale}",
            "-r",
            f"{source.fps:.12g}",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    upsampler = _build_upsampler(model_path, scale, tile_size, fp16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processed = 0
    try:
        assert writer.stdin is not None
        with torch.inference_mode():
            while True:
                ok, frame_bgr = reader.read()
                if not ok:
                    break
                enhanced_bgr, _ = upsampler.enhance(frame_bgr, outscale=scale)
                enhanced_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
                enhanced_rgb = np.asarray(enhanced_rgb, dtype=np.uint8)
                if enhanced_rgb.shape[:2] != (source.height * scale, source.width * scale):
                    raise RuntimeError(
                        "Real-ESRGAN returned unexpected dimensions: "
                        f"expected={source.width * scale}x{source.height * scale} "
                        f"actual={enhanced_rgb.shape[1]}x{enhanced_rgb.shape[0]}"
                    )
                writer.stdin.write(np.ascontiguousarray(enhanced_rgb).tobytes())
                processed += 1
                if processed % 30 == 0:
                    print(f"[realesrgan] frames={processed} device={device} scale={scale} tile={tile_size}", flush=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        writer.stdin.close()
        writer_return = writer.wait(timeout=300)
        if writer_return != 0:
            detail = writer.stderr.read().decode(errors="replace") if writer.stderr else ""
            raise RuntimeError(f"Real-ESRGAN encoder failed: {detail[-4000:]}")
    finally:
        reader.release()
        if writer.poll() is None:
            writer.kill()
            writer.wait()
    if processed != source.frame_count:
        raise RuntimeError(f"Real-ESRGAN frame count mismatch: expected={source.frame_count} actual={processed}")
    output = probe_video(output_video)
    print(
        f"[realesrgan] complete inputFrames={processed} outputFrames={output.frame_count} "
        f"input={source.width}x{source.height} output={output.width}x{output.height} "
        f"fps={output.fps} durationSeconds={output.duration_seconds:.6f}",
        flush=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    _run(args.input, args.output, args.model, args.scale, args.tile_size, args.fp16)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
