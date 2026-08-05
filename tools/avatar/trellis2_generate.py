"""Generate one textured mesh with the official TRELLIS.2 image-to-3D API.

The worker invokes this file through its existing command adapter. TRELLIS.2
and its CUDA extensions stay outside the application package so the Stable
Fast 3D runtime can remain available as a rollback option.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import time
import traceback


_STARTED = time.monotonic()


def emit(event: str, **fields: object) -> None:
    print(
        json.dumps(
            {
                "timestamp": time.time(),
                "elapsedSeconds": round(time.monotonic() - _STARTED, 3),
                "component": "trellis2",
                "event": event,
                **fields,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def file_info(path: Path) -> dict[str, object]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {"path": str(path), "exists": False, "error": str(exc)}
    return {"path": str(path), "exists": True, "sizeBytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


QUALITY_DEFAULTS = {
    "preview": {
        "pipeline_type": "512",
        "decimation_target": 50_000,
        "texture_size": 2048,
    },
    "runtime": {
        "pipeline_type": "1024_cascade",
        "decimation_target": 150_000,
        "texture_size": 4096,
    },
    "quality": {
        "pipeline_type": "1536_cascade",
        "decimation_target": 250_000,
        "texture_size": 4096,
    },
}


def patch_dinov3_transformers_compat() -> None:
    """Bridge the DINOv3 module layout used by current Transformers."""
    from trellis2.modules import image_feature_extractor

    extractor = image_feature_extractor.DinoV3FeatureExtractor
    if getattr(extractor.extract_features, "_faceless_compat", False):
        return

    import torch.nn.functional as functional

    def extract_features(self, image):
        image = image.to(self.model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = self.model.embeddings(image, bool_masked_pos=None)
        position_embeddings = self.model.rope_embeddings(image)
        encoder = getattr(self.model, "layer", None)
        if encoder is None:
            encoder = self.model.model.layer
        for layer_module in encoder:
            hidden_states = layer_module(
                hidden_states,
                position_embeddings=position_embeddings,
            )
        return functional.layer_norm(hidden_states, hidden_states.shape[-1:])

    extract_features._faceless_compat = True
    extractor.extract_features = extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality", choices=tuple(QUALITY_DEFAULTS), default="runtime")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model-path",
        default=os.getenv("TRELLIS2_MODEL_PATH", "/models/trellis2"),
        help="Local TRELLIS.2 checkpoint directory or an explicit Hugging Face id.",
    )
    parser.add_argument(
        "--pipeline-type",
        choices=("512", "1024", "1024_cascade", "1536_cascade"),
        default=os.getenv("TRELLIS2_PIPELINE_TYPE"),
    )
    parser.add_argument(
        "--max-num-tokens",
        type=int,
        default=int(os.getenv("TRELLIS2_MAX_NUM_TOKENS", "49152")),
    )
    parser.add_argument(
        "--decimation-target",
        type=int,
        default=int(os.getenv("TRELLIS2_DECIMATION_TARGET", "0")),
    )
    parser.add_argument(
        "--texture-size",
        type=int,
        default=int(os.getenv("TRELLIS2_TEXTURE_SIZE", "0")),
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        default=os.getenv("TRELLIS2_ALLOW_MODEL_DOWNLOAD", "0").lower() in {"1", "true", "yes"},
    )
    return parser.parse_args()


def resolve_setting(args: argparse.Namespace) -> tuple[str, int, int]:
    defaults = QUALITY_DEFAULTS[args.quality]
    pipeline_type = args.pipeline_type or defaults["pipeline_type"]
    decimation_target = args.decimation_target or defaults["decimation_target"]
    texture_size = args.texture_size or defaults["texture_size"]
    if decimation_target < 10_000:
        raise ValueError("TRELLIS2 decimation target must be at least 10000")
    if texture_size < 256:
        raise ValueError("TRELLIS2 texture size must be at least 256")
    if args.max_num_tokens < 1:
        raise ValueError("TRELLIS2 max token count must be positive")
    return pipeline_type, decimation_target, texture_size


def require_local_model(model_path: str, *, allow_download: bool) -> None:
    path = Path(model_path)
    if path.is_dir():
        if not (path / "pipeline.json").is_file():
            raise RuntimeError(f"TRELLIS.2 model directory is missing pipeline.json: {path}")
        return
    if not allow_download:
        raise RuntimeError(
            "TRELLIS.2 model is not mounted locally; provision the checkpoint before processing "
            "a job or pass --allow-model-download explicitly"
        )


def main() -> int:
    args = parse_args()
    emit(
        "started",
        input=file_info(args.image) if args.image.is_file() else {"path": str(args.image), "exists": False},
        output=str(args.output),
        outputDir=str(args.output_dir),
        quality=args.quality,
        seed=args.seed,
        model=args.model_path,
        allowModelDownload=args.allow_model_download,
        pipelineTypeOverride=args.pipeline_type,
        maxNumTokens=args.max_num_tokens,
    )
    if not args.image.is_file():
        emit("input_validation_failed", error=f"input image was not found: {args.image}")
        raise FileNotFoundError(f"input image was not found: {args.image}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_type, decimation_target, texture_size = resolve_setting(args)
    emit(
        "settings_resolved",
        pipelineType=pipeline_type,
        decimationTarget=decimation_target,
        textureSize=texture_size,
        maxNumTokens=args.max_num_tokens,
    )
    require_local_model(args.model_path, allow_download=args.allow_model_download)
    emit("model_checkpoint_validated", model=args.model_path)

    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Heavy imports are intentionally delayed so command validation and --help
    # remain usable in lightweight development environments.
    import torch
    from PIL import Image
    import o_voxel
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    patch_dinov3_transformers_compat()

    if not torch.cuda.is_available():
        raise RuntimeError("TRELLIS.2 requires a CUDA-enabled NVIDIA GPU")

    # Keep the .glb suffix on the temporary file because trimesh selects its
    # exporter from the filename extension.
    temporary_output = args.output.with_name(f"{args.output.stem}.tmp{args.output.suffix}")
    temporary_output.unlink(missing_ok=True)
    pipeline = None
    mesh = None
    try:
        emit("model_loading_started", model=args.model_path)
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
        rembg_model = getattr(getattr(pipeline, "rembg_model", None), "model", None)
        if rembg_model is not None:
            # Some BiRefNet safetensors exports advertise fp16 weights while
            # the TRELLIS preprocessing path supplies float32 tensors. Keep
            # this small segmentation model in float32 so both public and
            # permission-gated compatible checkpoints behave consistently.
            rembg_model.float()
            emit("rembg_dtype_normalized", dtype="float32")
        pipeline.cuda()
        emit("model_loading_finished", cudaDeviceCount=torch.cuda.device_count(), cudaDeviceName=torch.cuda.get_device_name(0))
        with Image.open(args.image) as image:
            image = image.convert("RGBA") if image.mode == "RGBA" else image.convert("RGB")
            with torch.inference_mode():
                emit("inference_started", imageMode=image.mode, imageSize=image.size, pipelineType=pipeline_type)
                mesh = pipeline.run(
                    image,
                    seed=args.seed,
                    pipeline_type=pipeline_type,
                    max_num_tokens=args.max_num_tokens,
                )[0]
                emit(
                    "inference_finished",
                    vertexCount=int(getattr(mesh.vertices, "shape", [0])[0]),
                    faceCount=int(getattr(mesh.faces, "shape", [0])[0]),
                )

                emit("glb_export_started", decimationTarget=decimation_target, textureSize=texture_size)
                glb = o_voxel.postprocess.to_glb(
                    vertices=mesh.vertices,
                    faces=mesh.faces,
                    attr_volume=mesh.attrs,
                    coords=mesh.coords,
                    attr_layout=mesh.layout,
                    voxel_size=mesh.voxel_size,
                    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                    decimation_target=decimation_target,
                    texture_size=texture_size,
                    remesh=True,
                    remesh_band=1,
                    remesh_project=0,
                    verbose=True,
                )
                glb.export(str(temporary_output), extension_webp=True)
                emit("glb_export_finished", output=file_info(temporary_output))
        temporary_output.replace(args.output)
        emit("finished", output=file_info(args.output), pipelineType=pipeline_type)
        return 0
    except Exception as exc:
        emit("failed", errorType=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        emit("cleanup_started")
        temporary_output.unlink(missing_ok=True)
        del mesh
        del pipeline
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
        emit("cleanup_finished")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        emit("process_failed", errorType=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        print(f"TRELLIS.2 generation failed: {exc}", flush=True)
        raise
