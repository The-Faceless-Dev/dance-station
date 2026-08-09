"""Run SkinTokens skin-only inference and restore canonical joint names."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from autotransition.avatar.canonical_skeleton import canonicalize_skinned_glb  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Skin a mesh using an existing canonical skeleton.")
    parser.add_argument("--skintokens-repo", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path, help="canonical skeleton GLB")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--profile-mode", choices=("auto", "low-vram", "quality"), default="auto")
    parser.add_argument("--attention", choices=("auto", "sdpa", "flash_attention_2"), default="auto")
    parser.add_argument("--num-beams", type=int, default=None)
    parser.add_argument("--use-transfer", action="store_true")
    parser.add_argument("--use-postprocess", action="store_true")
    args = parser.parse_args()
    adaptive_runner = PROJECT_ROOT / "tools" / "tokenrig" / "adaptive_runner.py"
    command = [
        sys.executable,
        str(adaptive_runner),
        "--skintokens-repo",
        str(args.skintokens_repo),
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--profile",
        args.profile_mode,
        "--attention",
        args.attention,
        "--use-skeleton",
    ]
    if args.num_beams is not None:
        command.extend(("--num-beams", str(args.num_beams)))
    if args.use_transfer:
        command.append("--use-transfer")
    if args.use_postprocess:
        command.append("--use-postprocess")
    subprocess.run(command, check=True)
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    canonicalize_skinned_glb(args.output, args.output, profile, args.manifest_output)
    print(f"[reskin] canonicalized output={args.output} manifest={args.manifest_output}")


if __name__ == "__main__":
    main()
