"""Project-owned BiRefNet video matting runner.

This optional runtime uses the official Transformers loading boundary exposed by
BiRefNet. It decodes the complete input video, runs the model over every frame
in bounded batches, and writes an alpha-capable ProRes 4444 stream. ProRes is
used as the internal interchange format because FFmpeg cannot decode VP9 alpha
back into an alpha plane reliably. The worker adapter owns
the request contract; this file owns only model execution.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BiRefNet matting over every video frame")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--raw-output", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--model", default="ZhengPeng7/BiRefNet-matting")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16", choices=("float32", "float16", "bfloat16"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=1024)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--edge-feather", type=float, default=0.0)
    parser.add_argument("--despill-strength", type=float, default=1.0)
    parser.add_argument("--background-key-distance", type=float, default=80.0)
    parser.add_argument("--background-key-strength", type=float, default=1.0)
    parser.add_argument("--crf", type=int, default=30)
    return parser.parse_args()


def _load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation
    except ImportError as exc:
        raise RuntimeError(
            "BiRefNet runtime requires torch, torchvision, transformers, Pillow, and opencv-python"
        ) from exc
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("BiRefNet requested CUDA but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    source = args.checkpoint if args.checkpoint and Path(args.checkpoint).is_dir() else args.model
    model = AutoModelForImageSegmentation.from_pretrained(source, trust_remote_code=True)
    model.to(device)
    model.eval()
    if device.type == "cuda" and args.dtype == "float16":
        model.half()
    elif device.type == "cuda" and args.dtype == "bfloat16":
        model.bfloat16()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return cv2, np, torch, Image, (model, preprocess, device)


def _predict_batch(
    frames: list[Any],
    *,
    cv2: Any,
    np: Any,
    torch: Any,
    image_type: Any,
    runtime: tuple[Any, Any, Any],
    threshold: float,
    edge_feather: float,
    despill_strength: float,
    background_key_distance: float,
    background_key_strength: float,
) -> list[Any]:
    model, preprocess, device = runtime
    tensors = [preprocess(image_type.fromarray(frame[:, :, ::-1].copy())).unsqueeze(0) for frame in frames]
    batch = torch.cat(tensors, dim=0).to(device)
    if device.type == "cuda" and next(model.parameters()).dtype == torch.float16:
        batch = batch.half()
    elif device.type == "cuda" and next(model.parameters()).dtype == torch.bfloat16:
        batch = batch.bfloat16()
    with torch.inference_mode():
        result = model(batch)
        if hasattr(result, "logits"):
            prediction = result.logits
        elif isinstance(result, (list, tuple)):
            prediction = result[-1]
        elif isinstance(result, dict):
            prediction = result.get("logits")
            if prediction is None:
                prediction = result.get("predictions")
            if prediction is None:
                prediction = next(iter(result.values()))
        else:
            prediction = result
        if prediction.ndim == 3:
            prediction = prediction.unsqueeze(1)
        alpha = prediction.sigmoid().float().detach().cpu().numpy()
    output: list[Any] = []
    for frame, mask in zip(frames, alpha):
        mask_image = (mask[0] * 255).clip(0, 255).astype(np.uint8)
        mask_image = cv2.resize(mask_image, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        if edge_feather > 0:
            kernel_size = max(3, int(round(edge_feather * 40)) * 2 + 1)
            mask_image = cv2.GaussianBlur(mask_image, (kernel_size, kernel_size), 0)
        if threshold > 0:
            mask_image = np.where(mask_image / 255.0 >= threshold, mask_image, 0).astype(np.uint8)
        frame_float = frame.astype(np.float32)
        alpha_float = mask_image.astype(np.float32) / 255.0
        background = None
        if background_key_strength > 0 or despill_strength > 0:
            height, width = frame.shape[:2]
            top_bottom = max(2, int(round(height * 0.04)))
            left = int(round(width * 0.2))
            right = max(left + 1, int(round(width * 0.8)))
            border = np.concatenate(
                [frame[:top_bottom, left:right].reshape(-1, 3), frame[-top_bottom:, left:right].reshape(-1, 3)],
                axis=0,
            ).astype(np.float32)
            background = np.median(border, axis=0)
        if background is not None and background_key_strength > 0:
            # Suppress residual pixels that still match the known solid source
            # background, even when BiRefNet's broad mask includes them inside
            # the character silhouette.
            color_distance = np.linalg.norm(frame_float - background, axis=2)
            window = max(12.0, background_key_distance * 0.5)
            background_likelihood = np.clip(
                (background_key_distance + window - color_distance) / (2.0 * window),
                0.0,
                1.0,
            )
            alpha_float *= 1.0 - background_key_strength * background_likelihood
            mask_image = np.clip(alpha_float * 255.0, 0, 255).astype(np.uint8)
        # BiRefNet gives us a straight alpha mask, but the RGB frame still has
        # the source background mixed into partially transparent edge pixels.
        # Unmix that solid background before encoding, otherwise source color
        # remains as a halo around the character.
        if despill_strength > 0:
            if background is None:
                raise RuntimeError("despill background was not estimated")
            # Do not unmix very uncertain matte pixels. Their predicted alpha
            # is not a trustworthy coverage value, and dividing by it creates
            # bright red/green/blue spikes at the contour.
            safe_alpha = np.maximum(alpha_float, 0.5)
            unmixed = (frame_float - background * (1.0 - alpha_float)[..., None]) / safe_alpha[..., None]
            # Low-confidence pixels are deliberately darkened rather than
            # amplified. This trades a tiny amount of edge softness for no
            # colored background fringe.
            premultiplied = frame_float * np.clip(alpha_float * 2.0, 0.0, 1.0)[..., None]
            edge_weight = np.clip((alpha_float - 0.25) / 0.30, 0.0, 1.0) * despill_strength
            frame_float = premultiplied * (1.0 - edge_weight[..., None]) + unmixed * edge_weight[..., None]
            frame_float = np.where(alpha_float[..., None] < 0.08, 0.0, frame_float)

            frame = np.clip(frame_float, 0, 255).astype(np.uint8)
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba[:, :, 3] = mask_image
        output.append(rgba)
    return output


def main() -> int:
    args = _parse_args()
    if args.batch_size < 1 or not 0 <= args.threshold <= 1:
        raise SystemExit("batch size must be positive and threshold must be between 0 and 1")
    if not 0 <= args.edge_feather <= 1:
        raise SystemExit("edge feather must be between 0 and 1")
    if not args.input.is_file():
        raise SystemExit(f"input video was not found: {args.input}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg is required for BiRefNet alpha encoding")
    cv2, np, torch, image_type, runtime = _load_runtime(args)
    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise SystemExit(f"could not open input video: {args.input}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    if width < 1 or height < 1 or fps <= 0:
        capture.release()
        raise SystemExit("input video has invalid dimensions or FPS")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "prores_ks",
        "-profile:v",
        "4",
        "-pix_fmt",
        "yuva444p10le",
        "-vendor",
        "apl0",
        str(args.output),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    frames: list[Any] = []
    frame_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
            if frames and (len(frames) >= args.batch_size or not ok):
                for rgba in _predict_batch(
                    frames,
                    cv2=cv2,
                    np=np,
                    torch=torch,
                    image_type=image_type,
                    runtime=runtime,
                    threshold=args.threshold,
                    edge_feather=args.edge_feather,
                    despill_strength=args.despill_strength,
                    background_key_distance=args.background_key_distance,
                    background_key_strength=args.background_key_strength,
                ):
                    encoder.stdin.write(rgba.tobytes())
                    frame_count += 1
                frames.clear()
                if frame_count % max(args.batch_size * 10, 1) == 0:
                    print(f"[birefnet] frames={frame_count}", flush=True)
            if not ok:
                break
    finally:
        capture.release()
        if encoder.stdin:
            encoder.stdin.close()
    stderr = encoder.stderr.read().decode("utf-8", errors="replace") if encoder.stderr else ""
    return_code = encoder.wait()
    if return_code != 0:
        raise SystemExit(f"alpha encoder failed with code {return_code}: {stderr[-4000:]}")
    if frame_count == 0 or not args.output.is_file() or args.output.stat().st_size == 0:
        raise SystemExit("BiRefNet produced no output frames")
    print(f"[birefnet] complete frames={frame_count} output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
