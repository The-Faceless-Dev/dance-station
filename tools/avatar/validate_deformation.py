"""CLI for the dependency-free avatar deformation diagnostic.

This is also useful when inspecting a generated GLB outside the worker. It
uses the same validator that runs before a paid avatar job is finalized.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autotransition.avatar.validation import validate_deformation


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate humanoid GLB skin deformation")
    parser.add_argument("glb", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = validate_deformation(args.glb, args.manifest)
    payload = {"ok": report.ok, "checks": list(report.checks), "details": report.details}
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
