"""Run the project-owned VACE motion-interpolation stage with RIFE."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch

from autotransition.generative_dance.video import probe_video, resolve_ffmpeg


def _load_rife(source_root: Path, checkpoint: Path, device: torch.device) -> torch.nn.Module:
    """Load only RIFE's model architecture; no ComfyUI runtime is required."""

    architecture_path = source_root / "vfi_models" / "rife" / "rife_arch.py"
    if not architecture_path.is_file():
        raise RuntimeError(f"RIFE architecture was not found: {architecture_path}")

    # The upstream architecture imports this one device helper from ComfyUI.
    # Supplying the helper here keeps the worker's runtime independent of the
    # ComfyUI application and node registry.
    comfy = types.ModuleType("comfy")
    model_management = types.ModuleType("comfy.model_management")
    model_management.get_torch_device = lambda: device
    comfy.model_management = model_management
    sys.modules.setdefault("comfy", comfy)
    sys.modules.setdefault("comfy.model_management", model_management)

    spec = importlib.util.spec_from_file_location("autotransition_vace_rife_arch", architecture_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load RIFE architecture: {architecture_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.IFNet(arch_ver="4.7")
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    if device.type == "cuda":
        model = model.half()
    return model


def _run(input_video: Path, output_video: Path, source_root: Path, checkpoint: Path, target_fps: int) -> None:
    ffmpeg = resolve_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required for the RIFE stage")
    source_probe = probe_video(input_video)
    input_fps = source_probe.fps
    if input_fps <= 0 or target_fps <= input_fps:
        raise RuntimeError(f"RIFE requires a target FPS above the input FPS: input={input_fps} target={target_fps}")
    if abs(target_fps - (input_fps * 2)) > 0.05:
        raise RuntimeError(f"RIFE stage currently supports 2x interpolation: input={input_fps} target={target_fps}")
    if source_probe.width < 1 or source_probe.height < 1:
        raise RuntimeError(f"input video has invalid dimensions: {source_probe.to_dict()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_rife(source_root, checkpoint, device)
    dtype = next(model.parameters()).dtype
    frame_bytes = source_probe.width * source_probe.height * 3
    output_video.parent.mkdir(parents=True, exist_ok=True)
    reader = subprocess.Popen(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(input_video), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
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
            f"{source_probe.width}x{source_probe.height}",
            "-r",
            str(target_fps),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_video),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def read_frame() -> np.ndarray | None:
        assert reader.stdout is not None
        data = reader.stdout.read(frame_bytes)
        if len(data) != frame_bytes:
            return None
        return np.frombuffer(data, dtype=np.uint8).reshape(source_probe.height, source_probe.width, 3).copy()

    def write_frame(frame: np.ndarray) -> None:
        assert writer.stdin is not None
        writer.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    try:
        previous = read_frame()
        if previous is None:
            raise RuntimeError("input video contained no decodable frames")
        pair_count = 0
        with torch.inference_mode():
            while True:
                current = read_frame()
                if current is None:
                    write_frame(previous)
                    break
                first = torch.from_numpy(previous).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255.0)
                second = torch.from_numpy(current).permute(2, 0, 1).unsqueeze(0).to(device).float().div_(255.0)
                if dtype == torch.float16:
                    first = first.half()
                    second = second.half()
                middle = model(
                    first,
                    second,
                    timestep=0.5,
                    scale_list=[8, 4, 2, 1],
                    training=False,
                    fastmode=True,
                    ensemble=True,
                )
                rendered = (middle[0].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
                write_frame(previous)
                write_frame(rendered)
                previous = current
                pair_count += 1
                if pair_count % 120 == 0:
                    print(f"[rife] pairs={pair_count} device={device} dtype={dtype}", flush=True)
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
        assert writer.stdin is not None
        writer.stdin.close()
        writer_return = writer.wait(timeout=300)
        reader_return = reader.wait(timeout=30)
        if writer_return != 0:
            detail = writer.stderr.read().decode(errors="replace") if writer.stderr else ""
            raise RuntimeError(f"RIFE encoder failed: {detail[-4000:]}")
        if reader_return != 0:
            detail = reader.stderr.read().decode(errors="replace") if reader.stderr else ""
            raise RuntimeError(f"RIFE decoder failed: {detail[-4000:]}")
        output_probe = probe_video(output_video)
        expected_frames = (pair_count * 2) + 1
        print(
            f"[rife] complete inputFrames={pair_count + 1} outputFrames={output_probe.frame_count} "
            f"expectedFrames={expected_frames} inputFps={input_fps} outputFps={output_probe.fps} "
            f"durationSeconds={output_probe.duration_seconds:.6f}",
            flush=True,
        )
        if output_probe.frame_count != expected_frames:
            raise RuntimeError(f"RIFE output frame count mismatch: expected={expected_frames} actual={output_probe.frame_count}")
    finally:
        for process in (reader, writer):
            if process.poll() is None:
                process.kill()
                process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--target-fps", type=int, required=True)
    args = parser.parse_args()
    _run(args.input, args.output, args.source_root, args.checkpoint, args.target_fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
