"""Run the Linux TokenRig adapter from a Windows development checkout."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


def wsl_path(value: str) -> str:
    path = Path(value).resolve()
    drive = path.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"expected an absolute Windows path: {value}")
    return f"/mnt/{drive}{path.as_posix()[2:]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skintokens-repo", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--profile", default="auto")
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--use-transfer", action="store_true")
    parser.add_argument("--bpy-timeout", type=int, default=600)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = "/mnt/d/skintokens-venv311/bin/python"
    runner = "/mnt/d/autotransition/tools/tokenrig/adaptive_runner.py"
    command = [
        python,
        runner,
        "--skintokens-repo",
        wsl_path(args.skintokens_repo),
        "--input",
        wsl_path(args.input),
        "--output",
        wsl_path(args.output),
        "--manifest-output",
        wsl_path(args.manifest_output),
        "--profile",
        args.profile,
        "--attention",
        args.attention,
        "--num-beams",
        str(args.num_beams),
        "--bpy-timeout",
        str(args.bpy_timeout),
    ]
    if args.use_transfer:
        command.append("--use-transfer")
    return subprocess.run(["wsl.exe", "bash", "-lc", shlex.join(command)], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
