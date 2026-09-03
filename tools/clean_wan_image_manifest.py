#!/usr/bin/env python3
"""Publish a tag with duplicate content-addressed layers removed."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from publish_wan_overlay_manifest import (
    MANIFEST_ACCEPT,
    get_registry_token,
    registry_request,
    upload_blob,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--source-tag", required=True)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument(
        "--remove-digest",
        action="append",
        default=[],
        help="Remove this content-addressed layer digest in addition to duplicate layers",
    )
    args = parser.parse_args()

    username = os.environ.get("REGISTRY_USERNAME")
    password = os.environ.get("REGISTRY_TOKEN")
    if not username or not password:
        raise RuntimeError("REGISTRY_USERNAME and REGISTRY_TOKEN are required")

    token = get_registry_token(args.repo, username, password)
    base = f"https://ghcr.io/v2/{args.repo}"
    auth = {"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT}
    _, _, manifest_body = registry_request(
        "GET", f"{base}/manifests/{args.source_tag}", headers=auth
    )
    manifest = json.loads(manifest_body)
    _, _, config_body = registry_request(
        "GET",
        f"{base}/blobs/{manifest['config']['digest']}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    config = json.loads(config_body)
    layers = manifest.get("layers") or []
    diff_ids = (config.get("rootfs") or {}).get("diff_ids") or []
    history = config.get("history") or []
    if len(layers) != len(diff_ids):
        raise RuntimeError(
            f"expected aligned image metadata, layers={len(layers)} diff_ids={len(diff_ids)}"
        )

    seen: set[str] = set()
    remove_digests = {
        value if value.startswith("sha256:") else f"sha256:{value}"
        for value in args.remove_digest
    }
    kept_layers = []
    kept_diff_ids = []
    kept_history = []
    removed: list[str] = []
    layer_index = 0
    for history_item in history:
        if history_item.get("empty_layer") is True:
            kept_history.append(history_item)
            continue
        if layer_index >= len(layers):
            raise RuntimeError("image history has more non-empty entries than layers")
        layer = layers[layer_index]
        diff_id = diff_ids[layer_index]
        layer_index += 1
        digest = str(layer["digest"])
        if digest in remove_digests:
            removed.append(digest)
            continue
        if digest in seen:
            removed.append(digest)
            continue
        seen.add(digest)
        kept_layers.append(layer)
        kept_diff_ids.append(diff_id)
        kept_history.append(history_item)
    if layer_index != len(layers):
        raise RuntimeError("image history has fewer non-empty entries than layers")
    if not removed:
        raise RuntimeError(
            f"source {args.source_tag} contains none of the requested duplicate or removable layers"
        )

    manifest["layers"] = kept_layers
    config.setdefault("rootfs", {})["diff_ids"] = kept_diff_ids
    config["history"] = kept_history
    config_bytes = json.dumps(config, separators=(",", ":"), ensure_ascii=False).encode()
    config_digest = f"sha256:{hashlib.sha256(config_bytes).hexdigest()}"
    upload_blob(args.repo, token, config_digest, config_bytes)
    manifest["config"] = {
        **manifest["config"],
        "size": len(config_bytes),
        "digest": config_digest,
    }
    registry_request(
        "PUT",
        f"{base}/manifests/{args.target_tag}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": manifest.get(
                "mediaType", "application/vnd.oci.image.manifest.v1+json"
            ),
            "Accept": MANIFEST_ACCEPT,
        },
        data=json.dumps(manifest, separators=(",", ":")).encode(),
    )
    print(
        json.dumps(
            {
                "sourceTag": args.source_tag,
                "targetTag": args.target_tag,
                "layers": len(kept_layers),
                "removed": removed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"clean_manifest_error={type(exc).__name__}: {exc}", file=sys.stderr)
        raise
