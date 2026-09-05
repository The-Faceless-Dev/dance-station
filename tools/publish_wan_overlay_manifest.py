#!/usr/bin/env python3
"""Publish a small Wan Animate overlay without rebuilding the model layers."""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import io
import json
import os
import tarfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
DOCKER_LAYER = "application/vnd.docker.image.rootfs.diff.tar.gzip"
OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
OCI_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"
MANIFEST_ACCEPT = ", ".join((DOCKER_MANIFEST, OCI_MANIFEST, "application/vnd.docker.distribution.manifest.list.v2+json", "application/vnd.oci.image.index.v1+json"))


class RegistryError(RuntimeError):
    def __init__(self, method: str, url: str, status: int, body: str) -> None:
        super().__init__(f"{method} {url} returned HTTP {status}: {body[:500]}")
        self.status = status


def registry_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    data: bytes | None = None,
) -> tuple[int, dict[str, str], bytes]:
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.status, dict(response.headers.items()), response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RegistryError(method, url, exc.code, body) from exc


def get_registry_token(repo: str, username: str, password: str) -> str:
    basic = base64.b64encode(f"{username}:{password}".encode()).decode()
    query = urllib.parse.urlencode(
        {"service": "ghcr.io", "scope": f"repository:{repo}:pull,push"}
    )
    _, _, body = registry_request(
        "GET",
        f"https://ghcr.io/token?{query}",
        headers={"Authorization": f"Basic {basic}"},
    )
    token = json.loads(body)["token"]
    if not token:
        raise RuntimeError("GHCR returned an empty registry token")
    return token


def add_file_entries(tar: tarfile.TarFile, repo_root: Path) -> None:
    entries: list[tuple[str, Path]] = []
    runtime_root = repo_root / "src" / "autotransition" / "generative_dance"
    for path in runtime_root.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(runtime_root).as_posix()
        entries.append((f"app/src/autotransition/generative_dance/{relative}", path))

    vace_root = repo_root / "src" / "autotransition" / "vace_stitch"
    for path in vace_root.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(vace_root).as_posix()
        entries.append((f"app/src/autotransition/vace_stitch/{relative}", path))

    runner = repo_root / "tools" / "generative_dance" / "wan_animate_2_runner.py"
    entries.append(("app/tools/generative_dance/wan_animate_2_runner.py", runner))
    runtime = repo_root / "tools" / "generative_dance" / "wan_animate_2_runtime.py"
    entries.append(("app/tools/generative_dance/wan_animate_2_runtime.py", runtime))

    directories = {
        "app",
        "app/src",
        "app/src/autotransition",
        "app/src/autotransition/generative_dance",
        "app/src/autotransition/vace_stitch",
        "app/tools",
        "app/tools/generative_dance",
    }
    for archive_name in sorted(directories):
        info = tarfile.TarInfo(archive_name)
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        info.mode = 0o755
        info.type = tarfile.DIRTYPE
        info.size = 0
        tar.addfile(info)

    for archive_name, path in sorted(entries):
        info = tarfile.TarInfo(archive_name)
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        info.mode = 0o644
        info.size = path.stat().st_size
        with path.open("rb") as source:
            tar.addfile(info, source)


def make_overlay_layer(repo_root: Path) -> tuple[bytes, str, str]:
    raw_buffer = io.BytesIO()
    with tarfile.open(fileobj=raw_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        add_file_entries(archive, repo_root)
    raw = raw_buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    raw_digest = hashlib.sha256(raw).hexdigest()
    compressed_digest = hashlib.sha256(compressed).hexdigest()
    return compressed, raw_digest, compressed_digest


def make_lightx2v_layer(checkpoint: Path) -> tuple[bytes, str, str]:
    """Build a standalone adapter layer so later code overlays stay small."""

    if not checkpoint.is_file():
        raise FileNotFoundError(f"LightX2V checkpoint was not found: {checkpoint}")
    raw_buffer = io.BytesIO()
    archive_name = "models/wan-animate-2/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"
    with tarfile.open(fileobj=raw_buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for directory in ("models", "models/wan-animate-2"):
            info = tarfile.TarInfo(directory)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        info = tarfile.TarInfo(archive_name)
        info.uid = info.gid = 0
        info.uname = info.gname = ""
        info.mtime = 0
        info.mode = 0o644
        info.size = checkpoint.stat().st_size
        with checkpoint.open("rb") as source:
            archive.addfile(info, source)
    raw = raw_buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=6, mtime=0)
    return compressed, hashlib.sha256(raw).hexdigest(), hashlib.sha256(compressed).hexdigest()


def make_scipy_layer(wheel: Path) -> tuple[bytes, str, str]:
    """Add the Linux CPython SciPy wheel without rebuilding the model image."""

    if not wheel.is_file():
        raise FileNotFoundError(f"SciPy wheel was not found: {wheel}")
    prefix = "opt/conda/lib/python3.11/site-packages"
    raw_buffer = io.BytesIO()
    directories: set[str] = set()
    with zipfile.ZipFile(wheel) as archive, tarfile.open(fileobj=raw_buffer, mode="w", format=tarfile.USTAR_FORMAT) as output:
        for member in sorted(archive.infolist(), key=lambda item: item.filename):
            if member.is_dir():
                continue
            archive_name = f"{prefix}/{member.filename}"
            parent = archive_name.rpartition("/")[0]
            parts = parent.split("/")
            for index in range(1, len(parts) + 1):
                directory = "/".join(parts[:index])
                if directory in directories:
                    continue
                directories.add(directory)
                info = tarfile.TarInfo(directory)
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                info.mode = 0o755
                info.type = tarfile.DIRTYPE
                info.size = 0
                output.addfile(info)
            data = archive.read(member)
            info = tarfile.TarInfo(archive_name)
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = 0
            info.mode = 0o755 if member.filename.endswith((".so", ".so.1")) else 0o644
            info.size = len(data)
            output.addfile(info, io.BytesIO(data))
    raw = raw_buffer.getvalue()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    return compressed, hashlib.sha256(raw).hexdigest(), hashlib.sha256(compressed).hexdigest()


def upload_blob(repo: str, token: str, digest: str, payload: bytes) -> None:
    base = f"https://ghcr.io/v2/{repo}"
    auth = {"Authorization": f"Bearer {token}"}
    try:
        status, _, _ = registry_request("HEAD", f"{base}/blobs/{digest}", headers=auth)
        if status == 200:
            return
    except RegistryError as exc:
        if exc.status != 404:
            raise

    _, response_headers, _ = registry_request(
        "POST", f"{base}/blobs/uploads/", headers=auth
    )
    location = response_headers.get("Location")
    if not location:
        raise RuntimeError("GHCR did not return a blob upload location")
    location = urllib.parse.urljoin("https://ghcr.io", location)
    separator = "&" if "?" in location else "?"
    upload_url = f"{location}{separator}digest={urllib.parse.quote(digest)}"
    registry_request(
        "PUT",
        upload_url,
        headers={**auth, "Content-Type": "application/octet-stream"},
        data=payload,
    )


def publish(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    code_layer, code_raw_digest, code_compressed_digest = make_overlay_layer(repo_root)
    lightx_layer = None
    lightx_raw_digest = None
    lightx_compressed_digest = None
    if not args.code_only:
        lightx_layer, lightx_raw_digest, lightx_compressed_digest = make_lightx2v_layer(
            Path(args.lightx2v_checkpoint).resolve()
        )
    scipy_layer = None
    scipy_raw_digest = None
    scipy_compressed_digest = None
    if args.scipy_wheel:
        scipy_layer, scipy_raw_digest, scipy_compressed_digest = make_scipy_layer(
            Path(args.scipy_wheel).resolve()
        )
    if args.dry_run:
        result = {
            "dryRun": True,
            "codeLayerDigest": f"sha256:{code_compressed_digest}",
            "codeLayerBytes": len(code_layer),
        }
        if lightx_layer is not None:
            result.update(
                {
                    "lightx2vLayerDigest": f"sha256:{lightx_compressed_digest}",
                    "lightx2vLayerBytes": len(lightx_layer),
                }
            )
        if scipy_layer is not None:
            result.update(
                {
                    "scipyLayerDigest": f"sha256:{scipy_compressed_digest}",
                    "scipyLayerBytes": len(scipy_layer),
                }
            )
        return result

    username = os.environ.get("REGISTRY_USERNAME")
    password = os.environ.get("REGISTRY_TOKEN")
    if not username or not password:
        raise RuntimeError("REGISTRY_USERNAME and REGISTRY_TOKEN are required")

    token = get_registry_token(args.repo, username, password)
    auth = {
        "Authorization": f"Bearer {token}",
        "Accept": MANIFEST_ACCEPT,
    }
    base_url = f"https://ghcr.io/v2/{args.repo}"
    _, _, manifest_body = registry_request(
        "GET",
        f"{base_url}/manifests/{args.base_tag}",
        headers=auth,
    )
    base_manifest = json.loads(manifest_body)
    if base_manifest.get("mediaType") not in {DOCKER_MANIFEST, OCI_MANIFEST}:
        raise RuntimeError(
            f"base image is not a single Docker/OCI image manifest: {base_manifest.get('mediaType')}"
        )

    _, _, config_body = registry_request(
        "GET",
        f"{base_url}/blobs/{base_manifest['config']['digest']}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    config = json.loads(config_body)
    diff_ids = config.setdefault("rootfs", {}).setdefault("diff_ids", [])
    history = config.setdefault("history", [])
    diff_ids.append(f"sha256:{code_raw_digest}")
    history.append({"created_by": "COPY payload/ /", "comment": "Wan Animate runtime overlay"})
    if lightx_layer is not None:
        diff_ids.append(f"sha256:{lightx_raw_digest}")
        history.append({"created_by": "COPY lightx2v/ /", "comment": "Wan Animate LightX2V adapter"})
    if scipy_layer is not None:
        diff_ids.append(f"sha256:{scipy_raw_digest}")
        history.append({"created_by": "COPY scipy wheel /", "comment": "SciPy runtime dependency for Real-ESRGAN"})
    image_config = config.setdefault("config", {})
    env = list(image_config.get("Env") or [])
    env_overrides = {
        "HOME": "/home/wan",
        "XDG_CACHE_HOME": "/home/wan/.cache",
        "TRITON_CACHE_DIR": "/home/wan/.cache/triton",
        "TORCHINDUCTOR_CACHE_DIR": "/home/wan/.cache/torchinductor",
        "HF_HOME": "/home/wan/.cache/huggingface",
        "TRANSFORMERS_CACHE": "/home/wan/.cache/huggingface/transformers",
        "TORCH_HOME": "/home/wan/.cache/torch",
        "MPLCONFIGDIR": "/home/wan/.cache/matplotlib",
        # The Vast target is the RTX 5090. Keep the quantized Q6 bytes on the
        # device, but retain a reserve for activations and VAE work. A hard
        # failure is preferable to silently dropping to host-transfer latency.
        "WAN_GGUF_GPU_RAW_CACHE": "1",
        "WAN_GGUF_GPU_RAW_RESERVE_MB": "4096",
        "WAN_GGUF_DEQUANT_BACKEND": "triton",
        "WAN_GGUF_DEQUANT_DTYPE": "bfloat16",
        "WAN_T5_DEVICE": "cpu",
        # Production 5090 runs must use the fused CUDA paths. There is no
        # eager, manually chunked, or CPU attention fallback in this profile.
        "WAN_FLEX_ATTENTION_COMPILE_MODE": "max-autotune",
        "WAN_FLEX_ATTENTION_BACKEND": "official",
        "WAN_REQUIRE_FLEX_ATTENTION": "1",
        "WAN_REQUIRE_FLASH_ATTENTION": "1",
        "WAN_REFERENCE_ATTENTION_BACKEND": "flash",
        "WAN_SDPA_BACKEND": "auto",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "GENERATIVE_DANCE_WAN_REFERENCE_STRENGTH": "1.25",
        "GENERATIVE_DANCE_WAN_STEPS": "4",
        "GENERATIVE_DANCE_WAN_MIN_STEPS": "4",
        "GENERATIVE_DANCE_WAN_LIGHTX2V_ENABLED": "1",
        "GENERATIVE_DANCE_WAN_LIGHTX2V_CHECKPOINT": "/models/wan-animate-2/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "GENERATIVE_DANCE_WAN_LIGHTX2V_STRENGTH": "1.0",
        # The 32 GB RTX 5090 profile fits the Q6 + LightX2V stack with the
        # proven 17-frame window. The base image used by the overlay carried
        # a 33-frame value, which OOMs once the adapter activations are added.
        "GENERATIVE_DANCE_WAN_TEMPORAL_WINDOW": "17",
        "GENERATIVE_DANCE_WAN_TEMPORAL_CONTEXT_FRAMES": "5",
        "WAN_LIGHTX2V_ENABLED": "1",
        "WAN_LIGHTX2V_CHECKPOINT": "/models/wan-animate-2/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "WAN_LIGHTX2V_STRENGTH": "1.0",
        "GENERATIVE_DANCE_STAGE_TIMEOUT_SECONDS": "7200",
        "VACE_STITCH_STAGE_TIMEOUT_SECONDS": "7200",
    }
    if not args.code_only:
        keys = set(env_overrides) | {
            "GENERATIVE_DANCE_JOB_TIMEOUT_SECONDS",
            "VACE_STITCH_JOB_TIMEOUT_SECONDS",
            "GENERATIVE_DANCE_WAN_RENDER_TIMEOUT_SECONDS",
            "VACE_STITCH_RUNTIME_TIMEOUT_SECONDS",
        }
        env = [value for value in env if value.split("=", 1)[0] not in keys]
        env.extend(f"{key}={value}" for key, value in env_overrides.items())
    image_config["Env"] = env
    config_bytes = json.dumps(config, separators=(",", ":"), ensure_ascii=False).encode()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    upload_blob(args.repo, token, f"sha256:{code_compressed_digest}", code_layer)
    if lightx_layer is not None:
        upload_blob(args.repo, token, f"sha256:{lightx_compressed_digest}", lightx_layer)
    if scipy_layer is not None:
        upload_blob(args.repo, token, f"sha256:{scipy_compressed_digest}", scipy_layer)
    upload_blob(args.repo, token, f"sha256:{config_digest}", config_bytes)

    base_media_type = base_manifest.get("mediaType", DOCKER_MANIFEST)
    base_layer_media_type = (
        base_manifest.get("layers", [{}])[0].get("mediaType")
        or (OCI_LAYER if base_media_type == OCI_MANIFEST else DOCKER_LAYER)
    )
    new_layers = [
        {
            "mediaType": base_layer_media_type,
            "size": len(code_layer),
            "digest": f"sha256:{code_compressed_digest}",
        }
    ]
    if lightx_layer is not None:
        new_layers.append(
            {
                "mediaType": base_layer_media_type,
                "size": len(lightx_layer),
                "digest": f"sha256:{lightx_compressed_digest}",
            }
        )
    if scipy_layer is not None:
        new_layers.append(
            {
                "mediaType": base_layer_media_type,
                "size": len(scipy_layer),
                "digest": f"sha256:{scipy_compressed_digest}",
            }
        )
    manifest = {
        "schemaVersion": 2,
        "mediaType": base_media_type,
        "config": {
            "mediaType": base_manifest["config"]["mediaType"],
            "size": len(config_bytes),
            "digest": f"sha256:{config_digest}",
        },
        "layers": [
            *base_manifest["layers"],
            *new_layers,
        ],
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":")).encode()
    registry_request(
        "PUT",
        f"{base_url}/manifests/{args.tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": base_media_type,
            "Accept": MANIFEST_ACCEPT,
        },
        data=manifest_bytes,
    )
    result = {
        "tag": args.tag,
        "layers": len(manifest["layers"]),
        "compressedBytes": sum(layer["size"] for layer in manifest["layers"]),
        "codeOverlayBytes": len(code_layer),
        "configDigest": f"sha256:{config_digest}",
        "codeOverlayDigest": f"sha256:{code_compressed_digest}",
    }
    if lightx_layer is not None:
        result.update(
            {
                "lightx2vOverlayBytes": len(lightx_layer),
                "lightx2vOverlayDigest": f"sha256:{lightx_compressed_digest}",
            }
        )
    if scipy_layer is not None:
        result.update(
            {
                "scipyOverlayBytes": len(scipy_layer),
                "scipyOverlayDigest": f"sha256:{scipy_compressed_digest}",
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--repo", default="the-faceless-dev/faceless-wan-animate-worker")
    parser.add_argument("--base-tag", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--lightx2v-checkpoint")
    parser.add_argument("--scipy-wheel", help="Linux CPython 3.11 SciPy wheel to add for optional Real-ESRGAN")
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="Append only the small runtime code layer; inherit all model and adapter layers",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.code_only and not args.lightx2v_checkpoint:
        parser.error("--lightx2v-checkpoint is required unless --code-only is used")
    print(json.dumps(publish(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
