from __future__ import annotations

import json
import struct
from pathlib import Path

from autotransition.avatar.canonical_skeleton import (
    CANONICAL_PARENTS,
    REQUIRED_CANONICAL_ROLES,
    canonicalize_skinned_glb,
    fit_profile,
    read_glb,
    validate_profile,
    write_glb,
    write_manifest,
    write_skeleton_glb,
)
from autotransition.avatar.validation import read_glb_json, validate_glb, validate_manifest


def write_mesh_glb(path: Path) -> None:
    points = [
        (-0.4, -0.5, -0.1),
        (0.4, -0.5, -0.1),
        (-0.3, 0.5, 0.1),
        (0.3, 0.5, 0.1),
    ]
    binary = b"".join(struct.pack("<fff", *point) for point in points)
    payload = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": "geometry_0", "mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(points), "type": "VEC3"}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(binary)}],
        "buffers": [{"byteLength": len(binary)}],
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    binary += b"\0" * ((4 - len(binary) % 4) % 4)
    total = 12 + 8 + len(encoded) + 8 + len(binary)
    path.write_bytes(
        b"glTF"
        + struct.pack("<II", 2, total)
        + struct.pack("<II", len(encoded), 0x4E4F534A)
        + encoded
        + struct.pack("<II", len(binary), 0x004E4942)
        + binary
    )


def test_canonical_profile_and_skeleton_glb_are_named_and_valid(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.glb"
    skeleton = tmp_path / "canonical-skeleton.glb"
    manifest = tmp_path / "manifest.json"
    write_mesh_glb(mesh)

    profile = fit_profile(mesh)
    assert validate_profile(profile) == []
    assert set(profile["joints"]) == set(CANONICAL_PARENTS)
    write_skeleton_glb(mesh, profile, skeleton)
    write_manifest(profile, manifest, model_file=skeleton.name)

    payload = read_glb_json(skeleton)
    names = {node.get("name") for node in payload["nodes"]}
    assert set(CANONICAL_PARENTS).issubset(names)
    assert not any(name.startswith("bone_") for name in names if isinstance(name, str))
    assert validate_glb(skeleton, require_skin=False).ok
    assert validate_manifest(manifest, payload).ok
    assert set(json.loads(manifest.read_text())["bones"]) == set(REQUIRED_CANONICAL_ROLES)
    assert read_glb(skeleton).binary


def test_existing_canonical_skeleton_can_be_updated_for_reskin(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.glb"
    original = tmp_path / "original-rig.glb"
    updated = tmp_path / "updated-rig.glb"
    write_mesh_glb(mesh)

    profile = fit_profile(mesh)
    write_skeleton_glb(mesh, profile, original)
    profile["joints"]["head"]["position"][0] += 0.01
    write_skeleton_glb(original, profile, updated)

    original_payload = read_glb_json(original)
    updated_payload = read_glb_json(updated)
    original_names = {node.get("name") for node in original_payload["nodes"]}
    updated_names = {node.get("name") for node in updated_payload["nodes"]}
    assert set(CANONICAL_PARENTS).issubset(updated_names)
    assert len(updated_payload["nodes"]) == len(original_payload["nodes"])
    assert len(updated_payload["skins"]) == len(original_payload["skins"])
    assert validate_glb(updated, require_skin=False).ok
    assert original_names == updated_names


def test_skintokens_output_repairs_symmetric_lower_limb_influences(tmp_path: Path) -> None:
    mesh = tmp_path / "mesh.glb"
    skinned = tmp_path / "skintokens-output.glb"
    normalized = tmp_path / "normalized.glb"
    manifest = tmp_path / "manifest.json"
    write_mesh_glb(mesh)

    profile = fit_profile(mesh)
    write_skeleton_glb(mesh, profile, skinned)
    document = read_glb(skinned)
    payload = document.payload
    primitive = payload["meshes"][0]["primitives"][0]
    skin_joints = payload["skins"][0]["joints"]
    joint_order = {node_index: order for order, node_index in enumerate(skin_joints)}
    left_lower = next(index for index, node in enumerate(payload["nodes"]) if node.get("name") == "leftLowerLeg")
    right_lower = next(index for index, node in enumerate(payload["nodes"]) if node.get("name") == "rightLowerLeg")

    # Simulate SkinTokens assigning the positive-x geometry to the left lower
    # leg and the negative-x geometry to the right lower leg. The normalizer
    # must use influence regions, not bone order, to repair that swap.
    joints_data = bytes(
        [
            joint_order[right_lower], 0, 0, 0,
            joint_order[left_lower], 0, 0, 0,
            joint_order[right_lower], 0, 0, 0,
            joint_order[left_lower], 0, 0, 0,
        ]
    )
    weights_data = b"".join(struct.pack("<ffff", 1.0, 0.0, 0.0, 0.0) for _ in range(4))
    binary = bytearray(document.binary)

    def append_accessor(data: bytes, *, component_type: int, shape: str, count: int) -> int:
        while len(binary) % 4:
            binary.append(0)
        offset = len(binary)
        binary.extend(data)
        view_index = len(payload["bufferViews"])
        payload["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(data)})
        accessor_index = len(payload["accessors"])
        payload["accessors"].append({"bufferView": view_index, "componentType": component_type, "count": count, "type": shape})
        return accessor_index

    primitive["attributes"]["JOINTS_0"] = append_accessor(joints_data, component_type=5121, shape="VEC4", count=4)
    primitive["attributes"]["WEIGHTS_0"] = append_accessor(weights_data, component_type=5126, shape="VEC4", count=4)
    payload["buffers"][0]["byteLength"] = len(binary)
    write_glb(skinned, type(document)(payload=payload, binary=bytes(binary)))

    canonicalize_skinned_glb(skinned, normalized, profile, manifest)
    normalized_payload = read_glb(normalized).payload
    extras = normalized_payload["extras"]["canonicalSkeleton"]
    assert any(item["leftRole"] == "leftLowerLeg" for item in extras["influenceSwaps"])
    assert normalized_payload["nodes"][right_lower]["name"] == "leftLowerLeg"
    assert normalized_payload["nodes"][left_lower]["name"] == "rightLowerLeg"
    assert validate_glb(normalized).ok
    assert validate_manifest(manifest, normalized_payload).ok
