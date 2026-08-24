"""Optional Wan-Animate-2 Diffusers subprocess for the local POC.

This file intentionally keeps heavyweight Torch/Diffusers imports outside the
Autotransition web process. Install the exact Wan/Diffusers environment in the
runtime that owns the checkpoint, then point GENERATIVE_DANCE_WAN_COMMAND at
this script with the placeholders documented in README.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Wan-Animate-2 image-driven dance segment.")
    parser.add_argument("--model", required=True, help="Pinned local path or Hugging Face model id.")
    parser.add_argument("--reference-image", required=True, type=Path)
    parser.add_argument("--driver-video", required=True, type=Path)
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    try:
        import torch
        from diffusers import WanAnimate2Pipeline
        from diffusers.utils import export_to_video, load_image
    except ImportError as exc:
        raise SystemExit(
            "Wan Diffusers runtime is missing. Install the pinned Wan-Animate-2 "
            f"environment in the worker runtime: {exc}"
        ) from exc

    if not args.reference_image.is_file():
        raise SystemExit(f"reference image not found: {args.reference_image}")
    if not args.driver_video.is_file():
        raise SystemExit(f"driving video not found: {args.driver_video}")
    if not args.prompt_file.is_file():
        raise SystemExit(f"prompt file not found: {args.prompt_file}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"[wan-animate] loading model={args.model} device={device} dtype={dtype}", flush=True)
    pipe = WanAnimate2Pipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)
    prompt = args.prompt_file.read_text(encoding="utf-8")
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    print(
        f"[wan-animate] rendering reference={args.reference_image} driver={args.driver_video} "
        f"width={args.width} height={args.height} steps={args.steps}",
        flush=True,
    )
    result = pipe(
        image=load_image(str(args.reference_image)),
        driving_video=str(args.driver_video),
        prompt=prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(result.frames[0], str(args.output), fps=args.fps)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise SystemExit("Wan pipeline returned without a video output")
    print(f"[wan-animate] output={args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
