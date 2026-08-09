"""Fit and export the canonical humanoid skeleton for a mesh."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autotransition.avatar.canonical_skeleton import fit_profile, write_manifest, write_skeleton_glb


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit humanoid-v1 and write a skeleton-only GLB.")
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="skeleton-only GLB output")
    parser.add_argument("--profile-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--seed-rig", type=Path, default=None)
    parser.add_argument("--seed-manifest", type=Path, default=None)
    args = parser.parse_args()
    if (args.seed_rig is None) != (args.seed_manifest is None):
        parser.error("--seed-rig and --seed-manifest must be provided together")
    profile = fit_profile(
        args.mesh,
        seed_rig=args.seed_rig,
        seed_manifest=args.seed_manifest,
    )
    args.profile_output.parent.mkdir(parents=True, exist_ok=True)
    args.profile_output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    write_skeleton_glb(args.mesh, profile, args.output)
    write_manifest(profile, args.manifest_output, model_file=args.output.name)
    print(json.dumps({"skeleton": str(args.output), "profile": str(args.profile_output), "manifest": str(args.manifest_output)}, indent=2))


if __name__ == "__main__":
    main()
