"""Non-interactive FLUX.2 Klein 4B wrapper for the avatar worker.

Run this with the official ``black-forest-labs/flux2`` checkout on
``PYTHONPATH``. The model and autoencoder weights are resolved from the
official environment variables, so inference never downloads weights inside a
paid job.
"""

from __future__ import annotations

import argparse
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
                "component": "flux2-klein",
                "event": event,
                **fields,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        ),
        flush=True,
    )


def file_info(path: Path) -> dict[str, object]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        return {"path": str(path), "exists": False, "error": str(exc)}
    return {"path": str(path), "exists": True, "sizeBytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-file", required=True, type=Path)
    parser.add_argument("--negative-prompt-file", required=False, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", nargs="?", type=int, default=None, const=None)
    parser.add_argument("--reference-image", nargs="?", type=Path, default=None, const=None)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--quality", default="runtime")
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt_file.read_text(encoding="utf-8")
    negative_prompt = args.negative_prompt_file.read_text(encoding="utf-8") if args.negative_prompt_file else ""
    emit(
        "started",
        model=os.getenv("FLUX2_AVATAR_MODEL", "flux.2-klein-4b"),
        promptCharacters=len(prompt),
        promptSha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        negativePromptCharacters=len(negative_prompt),
        output=str(args.output),
        outputWidth=args.width,
        outputHeight=args.height,
        quality=args.quality,
        seed=args.seed,
        referenceImage=file_info(args.reference_image) if args.reference_image else None,
        cpuOffload=args.cpu_offload,
        modelPath=os.getenv("KLEIN_4B_MODEL_PATH"),
        textEncoderPath=os.getenv("FLUX2_TEXT_ENCODER_PATH"),
        autoencoderPath=os.getenv("AE_MODEL_PATH"),
    )
    # Imports are intentionally delayed until after argument validation. The
    # worker can reject a bad request without creating a CUDA context.
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    import torch
    from PIL import Image
    from einops import rearrange
    from flux2.sampling import batched_prc_img, batched_prc_txt, denoise, encode_image_refs, get_schedule, scatter_ids
    from flux2.text_encoder import Qwen3Embedder
    from flux2.util import load_ae, load_flow_model, load_text_encoder

    model_name = os.getenv("FLUX2_AVATAR_MODEL", "flux.2-klein-4b")
    references = [Image.open(args.reference_image).convert("RGB")] if args.reference_image else []
    device = torch.device("cuda")
    model = text_encoder = ae = None
    try:
        emit("model_loading_started", modelName=model_name)
        text_encoder_path = os.getenv("FLUX2_TEXT_ENCODER_PATH")
        encoder_device = "cpu" if args.cpu_offload else device
        if text_encoder_path:
            text_encoder = Qwen3Embedder(
                model_spec=text_encoder_path,
                device=encoder_device,
            )
        else:
            text_encoder = load_text_encoder(model_name, device=encoder_device)
        model = load_flow_model(
            model_name,
            device="cpu" if args.cpu_offload else device,
        )
        ae = load_ae(model_name, device=device)
        model.eval()
        text_encoder.eval()
        ae.eval()
        emit("model_loading_finished", device=str(device), textEncoderDevice=encoder_device)
        with torch.inference_mode():
            emit("inference_started", referenceCount=len(references))
            ref_tokens, ref_ids = encode_image_refs(ae, references)
            ctx = text_encoder([prompt]).to(torch.bfloat16)
            ctx, ctx_ids = batched_prc_txt(ctx)
            if args.cpu_offload:
                text_encoder = text_encoder.cpu()
                model = model.to(device)
                ctx = ctx.to(device)
                ctx_ids = ctx_ids.to(device)
                torch.cuda.empty_cache()
            seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(8), "big")
            generator = torch.Generator(device="cuda").manual_seed(seed)
            shape = (1, 128, args.height // 16, args.width // 16)
            noise = torch.randn(shape, generator=generator, dtype=torch.bfloat16, device=device)
            x, x_ids = batched_prc_img(noise)
            timesteps = get_schedule(4, x.shape[1])
            x = denoise(
                model,
                x,
                x_ids,
                ctx,
                ctx_ids,
                timesteps=timesteps,
                guidance=1.0,
                img_cond_seq=ref_tokens,
                img_cond_seq_ids=ref_ids,
            )
            x = torch.cat(scatter_ids(x, x_ids)).squeeze(2)
            image = ae.decode(x).float().clamp(-1, 1)
            image = rearrange(image[0], "c h w -> h w c")
            output = Image.fromarray((127.5 * (image + 1.0)).cpu().byte().numpy())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            output.save(args.output, format="PNG")
            emit("inference_finished", output=file_info(args.output), seed=seed)
    except Exception as exc:
        emit("failed", errorType=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        emit("cleanup_started")
        del ae, model, text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        emit("cleanup_finished")


if __name__ == "__main__":
    main()
