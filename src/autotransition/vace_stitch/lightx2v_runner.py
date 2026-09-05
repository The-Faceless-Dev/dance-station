"""Run one LightX2V VACE inference with the worker compatibility layer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--config-json", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--frame-num", required=True, type=int)
    parser.add_argument("--src-video", required=True)
    parser.add_argument("--src-mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--save-result-path", required=True)
    args = parser.parse_args()

    source_root = Path(args.source_root)
    sys.path.insert(0, str(source_root))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(source_root), os.environ.get("PYTHONPATH", "")) if part
    )

    from .lightx2v_compat import install

    install()
    import lightx2v.infer as infer

    sys.argv = [
        "lightx2v.infer",
        "--model_cls",
        "wan2.1_vace",
        "--task",
        "vace",
        "--model_path",
        args.model_path,
        "--config_json",
        args.config_json,
        "--num_frames",
        str(args.frame_num),
        "--src_video",
        args.src_video,
        "--src_mask",
        args.src_mask,
        "--prompt",
        args.prompt,
        "--seed",
        str(args.seed),
        "--save_result_path",
        args.save_result_path,
    ]
    infer.main()


if __name__ == "__main__":
    main()
