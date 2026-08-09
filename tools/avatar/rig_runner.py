"""Run TokenRig and normalize its output to the canonical humanoid skeleton."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autotransition.avatar.canonical_skeleton import (  # noqa: E402
    canonicalize_skinned_glb,
    fit_profile,
    write_skeleton_glb,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TokenRig and emit a canonical humanoid rig.")
    parser.add_argument("--skintokens-repo", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--front-yaw-degrees", type=float, default=0.0)
    parser.add_argument("--profile", choices=("auto", "low-vram", "quality"), default="auto")
    parser.add_argument("--attention", choices=("auto", "sdpa", "flash_attention_2"), default="auto")
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--use-transfer", action="store_true")
    parser.add_argument("--use-postprocess", action="store_true")
    parser.add_argument("--bpy-timeout", type=int, default=300)
    parser.add_argument("--model-ckpt", default="experiments/articulation_xl_quantization_256_token_4/grpo_1400.ckpt")
    parser.add_argument("--hf-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="tokenrig-canonical-", dir=args.output.parent))
    canonical_skeleton = staging / "canonical-skeleton.glb"
    raw_rig = staging / "skinned-rig.glb"
    raw_manifest = staging / "skinned-manifest.json"
    adaptive_runner = PROJECT_ROOT / "tools" / "tokenrig" / "adaptive_runner.py"
    profile = fit_profile(args.input, mesh_file=args.input.name)
    write_skeleton_glb(args.input, profile, canonical_skeleton)
    command = [
        sys.executable,
        str(adaptive_runner),
        "--skintokens-repo",
        str(args.skintokens_repo),
        "--input",
        str(canonical_skeleton),
        "--output",
        str(raw_rig),
        "--manifest-output",
        str(raw_manifest),
        "--front-yaw-degrees",
        str(args.front_yaw_degrees),
        "--profile",
        args.profile,
        "--attention",
        args.attention,
        "--bpy-timeout",
        str(args.bpy_timeout),
        "--model-ckpt",
        args.model_ckpt,
        "--use-skeleton",
    ]
    if args.hf_path is not None:
        command.extend(("--hf-path", args.hf_path))
    if args.num_beams is not None:
        command.extend(("--num-beams", str(args.num_beams)))
    if args.use_transfer:
        command.append("--use-transfer")
    if args.use_postprocess:
        command.append("--use-postprocess")

    try:
        subprocess.run(command, check=True)
        if not raw_rig.is_file() or not raw_manifest.is_file():
            raise RuntimeError("TokenRig did not produce both a rig and a manifest")
        canonicalize_skinned_glb(raw_rig, args.output, profile, args.manifest_output)
        print(
            f"[rig] canonicalized output={args.output} manifest={args.manifest_output} "
            f"profileMode={profile['source']['mode']}"
        )
    finally:
        shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    main()
