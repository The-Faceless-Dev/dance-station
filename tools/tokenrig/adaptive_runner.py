"""Adaptive TokenRig launcher for low- and high-VRAM inference machines.

This wrapper keeps model and checkpoint files outside this repository and
changes only runtime policy: profile selection, attention backend, beam width,
and inference-only execution.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    beams: int
    attention: str


def detect_profile(requested: str, total_vram_gib: float) -> RuntimeProfile:
    if requested == "low-vram":
        return RuntimeProfile("low-vram", beams=1, attention="sdpa")
    if requested == "quality":
        return RuntimeProfile("quality", beams=10, attention="flash_attention_2")
    if total_vram_gib < 14.0:
        return RuntimeProfile("low-vram", beams=1, attention="sdpa")
    return RuntimeProfile("quality", beams=10, attention="flash_attention_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SkinTokens TokenRig with adaptive VRAM settings.")
    parser.add_argument("--skintokens-repo", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", type=Path, default=None)
    parser.add_argument(
        "--front-yaw-degrees",
        type=float,
        default=180.0,
        help="Global yaw written to the generated manifest so the model faces the runtime camera.",
    )
    parser.add_argument("--profile", choices=("auto", "low-vram", "quality"), default="auto")
    parser.add_argument("--attention", choices=("auto", "sdpa", "flash_attention_2"), default="auto")
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=2.0)
    parser.add_argument("--use-skeleton", action="store_true")
    parser.add_argument("--use-transfer", action="store_true")
    parser.add_argument("--use-postprocess", action="store_true")
    parser.add_argument(
        "--bpy-timeout",
        type=int,
        default=300,
        help="Seconds to wait for the Blender asset server (WSL startup can be slow).",
    )
    parser.add_argument("--model-ckpt", default="experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt")
    parser.add_argument("--hf-path", default=None)
    return parser.parse_args()


def collect_files(input_path: Path, supported: set[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(path for path in input_path.rglob("*") if path.suffix.lower() in supported)


def output_paths(files: Sequence[Path], input_path: Path, output_path: Path) -> list[Path]:
    if len(files) == 1 and output_path.suffix.lower() == ".glb":
        return [output_path]
    return [(output_path / file.relative_to(input_path)).with_suffix(".glb") for file in files]


def configure_attention_backend(attention: str) -> None:
    if attention != "flash_attention_2":
        return

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    flash_attention = ALL_ATTENTION_FUNCTIONS["flash_attention_2"]

    def compatible_flash_attention(module, query, key, value, *args, **kwargs):
        target_dtype = query.dtype
        if key.dtype != target_dtype:
            key = key.to(target_dtype)
        if value.dtype != target_dtype:
            value = value.to(target_dtype)
        return flash_attention(module, query, key, value, *args, **kwargs)

    ALL_ATTENTION_FUNCTIONS.register("flash_attention_2", compatible_flash_attention)


def main() -> None:
    args = parse_args()
    # Set allocator policy before importing torch so it applies to the CUDA
    # context created by the first model allocation.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    repo = args.skintokens_repo.resolve()
    if not (repo / "demo.py").is_file():
        raise SystemExit(f"SkinTokens demo.py was not found under {repo}")
    if not torch_available():
        raise SystemExit("CUDA-enabled PyTorch is required for TokenRig inference.")

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. TokenRig requires an NVIDIA CUDA device.")
    try:
        import bpy
        import numpy
    except Exception as exc:
        raise SystemExit(f"TokenRig runtime import check failed: {type(exc).__name__}: {exc}") from exc
    numpy_major = int(numpy.__version__.split(".", 1)[0])
    if numpy_major >= 2:
        raise SystemExit(f"TokenRig requires NumPy 1.x for Blender/SkinTokens compatibility; found {numpy.__version__}")
    total_vram_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    profile = detect_profile(args.profile, total_vram_gib)
    beams = args.num_beams if args.num_beams is not None else profile.beams
    attention = profile.attention if args.attention == "auto" else args.attention
    if beams < 1:
        raise SystemExit("--num-beams must be at least 1")

    files = collect_files(args.input.resolve(), {".obj", ".fbx", ".glb"})
    if not files:
        raise SystemExit(f"No supported 3D files found under {args.input}")
    outputs = output_paths(files, args.input.resolve(), args.output.resolve())
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    os.environ.setdefault("SKINTOKENS_ATTENTION", attention)
    configure_attention_backend(attention)

    import demo

    print(
        f"[tokenrig] profile={profile.name} vram_gib={total_vram_gib:.2f} "
        f"beams={beams} attention={attention} max_length=2048 inference_mode=true "
        f"numpy={numpy.__version__} bpy={bpy.app.version_string}"
    )
    server_proc = demo.start_bpy_server()
    try:
        demo.wait_for_bpy_server(timeout=args.bpy_timeout)
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            demo.run_rig(
                files,
                args.top_k,
                args.top_p,
                args.temperature,
                args.repetition_penalty,
                beams,
                args.use_skeleton,
                args.use_transfer,
                args.use_postprocess,
                outputs,
                args.model_ckpt,
                args.hf_path,
            )
    finally:
        if server_proc.poll() is None:
            server_proc.terminate()

    peak_allocated = torch.cuda.max_memory_allocated() / (1024**3)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024**3)
    print(
        f"[tokenrig] peak_vram_allocated_gib={peak_allocated:.2f} "
        f"peak_vram_reserved_gib={peak_reserved:.2f}"
    )

    if args.manifest_output is not None:
        if len(outputs) != 1:
            raise SystemExit("--manifest-output requires a single input/output GLB.")
        from manifest import write_manifest

        write_manifest(
            outputs[0],
            args.manifest_output.resolve(),
            front_yaw_degrees=args.front_yaw_degrees,
        )
        print(f"[OK] Wrote humanoid-v1 manifest: {args.manifest_output.resolve()}")


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    main()
