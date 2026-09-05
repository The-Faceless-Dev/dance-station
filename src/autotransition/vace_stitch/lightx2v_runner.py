"""Run one LightX2V VACE inference with the worker compatibility layer."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import types
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

    # The upstream package initializer imports every LightX2V runner. That
    # eagerly loads unrelated model integrations and their optional
    # dependencies before VACE can start. Install a namespace package so this
    # worker imports only the VACE/Wan path below.
    package_dir = source_root / "lightx2v"
    if not package_dir.is_dir():
        raise RuntimeError(f"LightX2V package directory is missing: {package_dir}")
    lightx2v_package = types.ModuleType("lightx2v")
    lightx2v_package.__path__ = [str(package_dir)]
    lightx2v_package.__package__ = "lightx2v"
    sys.modules["lightx2v"] = lightx2v_package

    import lightx2v_platform.set_ai_device  # noqa: F401,E402

    from .lightx2v_compat import install

    install()

    import torch
    import torch.distributed as dist
    from loguru import logger

    # Import registry ops and exactly one model runner. In particular, do not
    # import lightx2v.infer: its CLI registers every model family in the tree.
    importlib.import_module("lightx2v.common.ops")
    importlib.import_module("lightx2v.models.runners.wan.wan_vace_runner")
    from lightx2v.utils.input_info import init_empty_input_info, update_input_info_from_dict
    from lightx2v.utils.registry_factory import RUNNER_REGISTER
    from lightx2v.utils.set_config import print_config, set_config
    from lightx2v.utils.utils import seed_all, validate_config_paths
    from lightx2v_platform.base.global_var import AI_DEVICE

    if AI_DEVICE != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("LightX2V VACE requires a CUDA device; CPU execution is disabled")

    runtime_args = argparse.Namespace(
        model_cls="wan2.1_vace",
        task="vace",
        support_tasks=[],
        model_path=args.model_path,
        config_json=args.config_json,
        seed=args.seed,
        prompt=args.prompt,
        negative_prompt="",
        src_ref_images=None,
        src_video=args.src_video,
        src_mask=args.src_mask,
        save_result_path=args.save_result_path,
        return_result_tensor=False,
        target_video_length=args.frame_num,
        parallel=False,
    )
    seed_all(runtime_args.seed)
    config = set_config(runtime_args)
    # ``target_video_length`` is an InputInfo field in upstream LightX2V and
    # is therefore filtered from the CLI-to-config copy. Set it explicitly so
    # the worker's prepared window length is honored rather than the JSON
    # default of 81 frames.
    config["target_video_length"] = args.frame_num
    validate_config_paths(config)
    if config.get("cpu_offload"):
        raise RuntimeError("LightX2V VACE transformer CPU offload is disabled; set cpu_offload=false")

    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    logger.info(
        "[VACE][LightX2V] VACE-only runtime device={} gpu={} model_cls={} "
        "frames={} steps={} lora_configs={} cpu_offload={} t5_cpu_offload={} vae_cpu_offload={}",
        AI_DEVICE,
        device_name,
        config.get("model_cls"),
        config.get("target_video_length"),
        config.get("infer_steps"),
        config.get("lora_configs"),
        config.get("cpu_offload"),
        config.get("t5_cpu_offload"),
        config.get("vae_cpu_offload"),
    )
    print_config(config)

    input_info = init_empty_input_info("vace")
    update_input_info_from_dict(input_info, vars(runtime_args))
    runner = RUNNER_REGISTER["wan2.1_vace"](config)
    try:
        runner.init_modules()
        logger.info("[VACE][LightX2V] Required VACE runner modules initialized")
        runner.run_pipeline(input_info)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
