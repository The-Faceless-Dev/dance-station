"""Canonical humanoid skeleton fitting and skeleton-only GLB construction.

The canonical skeleton is deliberately independent of TokenRig's generated
``bone_N`` names.  A profile stores world-space joint positions and stable
semantic names.  The profile can be edited, rebuilt into a skeleton-only GLB,
and passed to SkinTokens with ``--use_skeleton`` for a fresh skinning pass.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CANONICAL_SKELETON_ID = "humanoid-v1"
CANONICAL_PROFILE_VERSION = 1

CANONICAL_PARENTS: dict[str, str | None] = {
    "root": None,
    "hips": "root",
    "spine": "hips",
    "chest": "spine",
    "neck": "chest",
    "head": "neck",
    # These names intentionally match the site capture skeleton.  The
    # runtime maps the semantic roles upperArmLeft/Right onto leftArm/rightArm
    # rather than inventing a second, incompatible arm topology here.
    "leftShoulder": "chest",
    "leftArm": "leftShoulder",
    "leftForearm": "leftArm",
    "leftHand": "leftForearm",
    "rightShoulder": "chest",
    "rightArm": "rightShoulder",
    "rightForearm": "rightArm",
    "rightHand": "rightForearm",
    "leftUpperLeg": "hips",
    "leftLowerLeg": "leftUpperLeg",
    "leftFoot": "leftLowerLeg",
    "leftToe": "leftFoot",
    "rightUpperLeg": "hips",
    "rightLowerLeg": "rightUpperLeg",
    "rightFoot": "rightLowerLeg",
    "rightToe": "rightFoot",
}

# Manifest roles are stable API names.  Canonical joint names are the
# model-independent topology consumed by the site motion retargeter.
CANONICAL_ROLE_TO_JOINT: dict[str, str] = {
    "hips": "hips",
    "spine": "spine",
    "chest": "chest",
    "head": "head",
    "shoulderLeft": "leftShoulder",
    "shoulderRight": "rightShoulder",
    "upperArmLeft": "leftArm",
    "upperArmRight": "rightArm",
    "forearmLeft": "leftForearm",
    "forearmRight": "rightForearm",
    "upperLegLeft": "leftUpperLeg",
    "upperLegRight": "rightUpperLeg",
    "lowerLegLeft": "leftLowerLeg",
    "lowerLegRight": "rightLowerLeg",
    "footLeft": "leftFoot",
    "footRight": "rightFoot",
}

REQUIRED_CANONICAL_ROLES = (
    "hips",
    "spine",
    "chest",
    "head",
    "shoulderLeft",
    "shoulderRight",
    "upperArmLeft",
    "upperArmRight",
    "forearmLeft",
    "forearmRight",
    "upperLegLeft",
    "upperLegRight",
    "lowerLegLeft",
    "lowerLegRight",
    "footLeft",
    "footRight",
)


@dataclass(frozen=True)
class GlbDocument:
    payload: dict[str, Any]
    binary: bytes


def _align4(value: int) -> int:
    return (value + 3) & ~3


def read_glb(path: Path) -> GlbDocument:
    data = path.read_bytes()
    if len(data) < 20 or data[:4] != b"glTF":
        raise ValueError(f"not a GLB file: {path}")
    version, declared_length = struct.unpack_from("<II", data, 4)
    if version != 2 or declared_length != len(data):
        raise ValueError(f"invalid GLB header: {path}")
    offset = 12
    payload: dict[str, Any] | None = None
    binary = b""
    while offset + 8 <= len(data):
        length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + length]
        offset += length
        if chunk_type == 0x4E4F534A:
            value = json.loads(chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("GLB JSON root must be an object")
            payload = value
        elif chunk_type == 0x004E4942:
            binary = chunk
    if payload is None:
        raise ValueError("GLB does not contain a JSON chunk")
    return GlbDocument(payload=payload, binary=binary)


def write_glb(path: Path, document: GlbDocument) -> None:
    encoded_json = json.dumps(document.payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    encoded_json += b" " * (_align4(len(encoded_json)) - len(encoded_json))
    binary = document.binary + b"\0" * (_align4(len(document.binary)) - len(document.binary))
    total_length = 12 + 8 + len(encoded_json) + (8 + len(binary) if binary else 0)
    chunks = [struct.pack("<II", len(encoded_json), 0x4E4F534A), encoded_json]
    if binary:
        chunks.extend((struct.pack("<II", len(binary), 0x004E4942), binary))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, total_length) + b"".join(chunks))


def _identity() -> list[list[float]]:
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]


def _matmul(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][index] * right[index][column] for index in range(4)) for column in range(4)]
        for row in range(4)
    ]


def _node_matrix(node: dict[str, Any]) -> list[list[float]]:
    matrix = node.get("matrix")
    if isinstance(matrix, list) and len(matrix) == 16:
        # glTF matrices are column-major in the JSON representation.
        return [[float(matrix[column * 4 + row]) for column in range(4)] for row in range(4)]
    translation = node.get("translation", [0.0, 0.0, 0.0])
    scale = node.get("scale", [1.0, 1.0, 1.0])
    rotation = node.get("rotation", [0.0, 0.0, 0.0, 1.0])
    x, y, z, w = (float(value) for value in rotation)
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    sx, sy, sz = (float(value) for value in scale)
    rotation_matrix = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0.0],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), 0.0],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    rotation_matrix[0][0] *= sx
    rotation_matrix[1][0] *= sx
    rotation_matrix[2][0] *= sx
    rotation_matrix[0][1] *= sy
    rotation_matrix[1][1] *= sy
    rotation_matrix[2][1] *= sy
    rotation_matrix[0][2] *= sz
    rotation_matrix[1][2] *= sz
    rotation_matrix[2][2] *= sz
    rotation_matrix[0][3] = float(translation[0])
    rotation_matrix[1][3] = float(translation[1])
    rotation_matrix[2][3] = float(translation[2])
    return rotation_matrix


def node_world_matrices(document: GlbDocument) -> dict[str, list[list[float]]]:
    nodes = document.payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("GLB has no nodes")
    parents: dict[int, int] = {}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        for child in node.get("children", []):
            if isinstance(child, int):
                parents[child] = index
    cache: dict[int, list[list[float]]] = {}

    def resolve(index: int) -> list[list[float]]:
        if index in cache:
            return cache[index]
        local = _node_matrix(nodes[index])
        parent = parents.get(index)
        world = _matmul(resolve(parent), local) if parent is not None else local
        cache[index] = world
        return world

    result: dict[str, list[list[float]]] = {}
    for index, node in enumerate(nodes):
        if isinstance(node, dict) and isinstance(node.get("name"), str):
            result[node["name"]] = resolve(index)
    return result


def _position(matrix: list[list[float]]) -> tuple[float, float, float]:
    return (matrix[0][3], matrix[1][3], matrix[2][3])


def _inverse_translation_matrix(position: tuple[float, float, float]) -> list[float]:
    """Return a column-major inverse bind matrix for a translation-only joint."""

    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        -position[0], -position[1], -position[2], 1.0,
    ]


def read_positions(document: GlbDocument) -> list[tuple[float, float, float]]:
    payload = document.payload
    meshes = payload.get("meshes")
    accessors = payload.get("accessors")
    views = payload.get("bufferViews")
    if not isinstance(meshes, list) or not isinstance(accessors, list) or not isinstance(views, list):
        raise ValueError("GLB does not contain mesh accessors")
    position_accessor: int | None = None
    for mesh in meshes:
        for primitive in mesh.get("primitives", []) if isinstance(mesh, dict) else []:
            attributes = primitive.get("attributes", {}) if isinstance(primitive, dict) else {}
            if isinstance(attributes.get("POSITION"), int):
                position_accessor = attributes["POSITION"]
                break
        if position_accessor is not None:
            break
    if position_accessor is None:
        raise ValueError("GLB has no POSITION accessor")
    accessor = accessors[position_accessor]
    view = views[accessor["bufferView"]]
    if accessor.get("componentType") != 5126 or accessor.get("type") != "VEC3":
        raise ValueError("canonical fitting requires float VEC3 positions")
    count = int(accessor["count"])
    base_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    stride = int(view.get("byteStride", 12))
    if stride < 12 or base_offset + (count - 1) * stride + 12 > len(document.binary):
        raise ValueError("POSITION accessor exceeds the GLB binary buffer")
    return [
        tuple(struct.unpack_from("<fff", document.binary, base_offset + index * stride))
        for index in range(count)
    ]


def _bounds(points: Iterable[tuple[float, float, float]]) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    values = list(points)
    if not values:
        raise ValueError("mesh contains no vertices")
    return (
        tuple(min(point[axis] for point in values) for axis in range(3)),
        tuple(max(point[axis] for point in values) for axis in range(3)),
    )


def _slice_extent(points: list[tuple[float, float, float]], y: float, half_window: float, center_x: float) -> float:
    candidates = [abs(point[0] - center_x) for point in points if abs(point[1] - y) <= half_window]
    return max(candidates, default=0.0)


def _geometry_profile(points: list[tuple[float, float, float]], mesh_file: str) -> dict[str, Any]:
    minimum, maximum = _bounds(points)
    height = maximum[1] - minimum[1]
    if height <= 1e-6:
        raise ValueError("mesh has no usable vertical extent")
    center_x = sum(point[0] for point in points) / len(points)
    center_z = sum(point[2] for point in points) / len(points)
    ground = minimum[1]

    def y_at(ratio: float) -> float:
        return ground + height * ratio

    def extent_at(ratio: float) -> float:
        return _slice_extent(points, y_at(ratio), height * 0.035, center_x)

    hip_extent = max(extent_at(0.48), extent_at(0.54), height * 0.08)
    arm_band = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
    shoulder_ratio = max(arm_band, key=extent_at)
    shoulder_extent = max(extent_at(ratio) for ratio in arm_band)
    maximum_extent = max(abs(point[0] - center_x) for point in points)
    shoulder_half = max(height * 0.08, min(shoulder_extent * 0.52, height * 0.24))
    hip_half = max(height * 0.055, min(hip_extent * 0.45, height * 0.18))
    arm_reach = max(height * 0.12, min(maximum_extent - shoulder_half - height * 0.01, height * 0.42))
    shoulder_y_ratio = max(0.62, min(0.76, shoulder_ratio + 0.08))
    arm_y_ratio = max(0.45, min(0.72, shoulder_ratio))

    joints: dict[str, tuple[float, float, float]] = {
        "root": (center_x, ground, center_z),
        "hips": (center_x, y_at(0.50), center_z),
        "spine": (center_x, y_at(0.61), center_z),
        "chest": (center_x, y_at(0.72), center_z),
        "neck": (center_x, y_at(0.82), center_z),
        "head": (center_x, y_at(0.90), center_z),
        "leftUpperLeg": (center_x - hip_half, y_at(0.49), center_z),
        "leftLowerLeg": (center_x - hip_half, y_at(0.26), center_z),
        "leftFoot": (center_x - hip_half, y_at(0.06), center_z + height * 0.015),
        "leftToe": (center_x - hip_half, y_at(0.025), center_z + height * 0.10),
        "rightUpperLeg": (center_x + hip_half, y_at(0.49), center_z),
        "rightLowerLeg": (center_x + hip_half, y_at(0.26), center_z),
        "rightFoot": (center_x + hip_half, y_at(0.06), center_z + height * 0.015),
        "rightToe": (center_x + hip_half, y_at(0.025), center_z + height * 0.10),
        "leftShoulder": (center_x - shoulder_half * 0.48, y_at(shoulder_y_ratio), center_z),
        "leftArm": (center_x - shoulder_half, y_at(arm_y_ratio), center_z),
        "leftForearm": (center_x - shoulder_half - arm_reach * 0.52, y_at(arm_y_ratio - 0.045), center_z),
        "leftHand": (center_x - shoulder_half - arm_reach, y_at(arm_y_ratio - 0.09), center_z),
        "rightShoulder": (center_x + shoulder_half * 0.48, y_at(shoulder_y_ratio), center_z),
        "rightArm": (center_x + shoulder_half, y_at(arm_y_ratio), center_z),
        "rightForearm": (center_x + shoulder_half + arm_reach * 0.52, y_at(arm_y_ratio - 0.045), center_z),
        "rightHand": (center_x + shoulder_half + arm_reach, y_at(arm_y_ratio - 0.09), center_z),
    }
    return _profile_from_joints(joints, mesh_file, source_mode="geometry")


def _profile_from_joints(
    joints: dict[str, tuple[float, float, float]],
    mesh_file: str,
    *,
    source_mode: str,
    source_rig: str | None = None,
) -> dict[str, Any]:
    normalized = {
        name: {
            "parent": CANONICAL_PARENTS[name],
            "position": [float(value) for value in joints[name]],
        }
        for name in CANONICAL_PARENTS
        if name in joints
    }
    missing = [name for name in CANONICAL_PARENTS if name not in normalized]
    if missing:
        raise ValueError(f"canonical profile is missing joints: {', '.join(missing)}")
    lengths = {
        name: math.dist(normalized[name]["position"], normalized[parent]["position"])
        for name, parent in CANONICAL_PARENTS.items()
        if parent is not None
    }
    return {
        "schemaVersion": CANONICAL_PROFILE_VERSION,
        "skeletonId": CANONICAL_SKELETON_ID,
        "meshFile": mesh_file,
        "source": {"mode": source_mode, "seedRig": source_rig},
        "coordinateSystem": {"up": "Y", "front": "+Z", "units": "mesh_units"},
        "joints": normalized,
        "boneLengths": lengths,
        "requiredRoles": list(REQUIRED_CANONICAL_ROLES),
    }


def fit_profile(
    mesh: Path,
    *,
    mesh_file: str | None = None,
    seed_rig: Path | None = None,
    seed_manifest: Path | None = None,
) -> dict[str, Any]:
    """Fit a canonical profile, optionally using an old rig as an initializer."""

    mesh_document = read_glb(mesh)
    points = read_positions(mesh_document)
    profile = _geometry_profile(points, mesh_file or mesh.name)
    if seed_rig is None or seed_manifest is None:
        profile["source"]["meshSha256"] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        return profile

    manifest = json.loads(seed_manifest.read_text(encoding="utf-8"))
    mapping = manifest.get("bones") if isinstance(manifest, dict) else None
    if not isinstance(mapping, dict):
        raise ValueError("seed manifest does not contain a bones mapping")
    seed_document = read_glb(seed_rig)
    world = node_world_matrices(seed_document)
    seed_positions = {
        role: _position(world[name])
        for role, name in mapping.items()
        if isinstance(role, str) and isinstance(name, str) and name in world
    }
    seed_positions_by_joint: dict[str, tuple[float, float, float]] = {}
    for role, point in seed_positions.items():
        seed_positions_by_joint[CANONICAL_ROLE_TO_JOINT.get(role, role)] = point
    required_joints = [CANONICAL_ROLE_TO_JOINT[role] for role in REQUIRED_CANONICAL_ROLES]
    if not all(role in seed_positions_by_joint for role in required_joints):
        missing = [role for role in required_joints if role not in seed_positions_by_joint]
        profile["source"]["mode"] = "geometry-seed-rejected"
        profile["source"]["seedRig"] = seed_rig.name
        profile["source"]["seedRejectedRoles"] = missing
        profile["source"]["meshSha256"] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        return profile

    target = profile["joints"]
    target_hip = tuple(target["hips"]["position"])
    source_hip = seed_positions_by_joint["hips"]
    mesh_min, mesh_max = _bounds(points)
    mesh_height = mesh_max[1] - mesh_min[1]
    out_of_bounds = [
        role
        for role, point in seed_positions_by_joint.items()
        if point[0] < mesh_min[0] - mesh_height * 0.15
        or point[0] > mesh_max[0] + mesh_height * 0.15
        or point[1] < mesh_min[1] - mesh_height * 0.15
        or point[1] > mesh_max[1] + mesh_height * 0.15
    ]
    if out_of_bounds:
        profile["source"]["mode"] = "geometry-seed-rejected"
        profile["source"]["seedRig"] = seed_rig.name
        profile["source"]["seedRejectedRoles"] = out_of_bounds
        profile["source"]["meshSha256"] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        return profile
    source_y_values = [seed_positions_by_joint[role][1] for role in ("leftFoot", "rightFoot", "head")]
    source_height = max(source_y_values) - min(source_y_values)
    scale = (mesh_max[1] - mesh_min[1]) / source_height if source_height > 1e-6 else 1.0
    source_ground = min(source_y_values)
    aligned: dict[str, tuple[float, float, float]] = {}
    for name in CANONICAL_PARENTS:
        if name in seed_positions_by_joint:
            point = seed_positions_by_joint[name]
            aligned[name] = (
                target_hip[0] + (point[0] - source_hip[0]) * scale,
                mesh_min[1] + (point[1] - source_ground) * scale,
                target_hip[2] + (point[2] - source_hip[2]) * scale,
            )
        else:
            aligned[name] = tuple(target[name]["position"])
    aligned_out_of_bounds = [
        name
        for name, point in aligned.items()
        if point[0] < mesh_min[0] - mesh_height * 0.05
        or point[0] > mesh_max[0] + mesh_height * 0.05
        or point[1] < mesh_min[1] - mesh_height * 0.05
        or point[1] > mesh_max[1] + mesh_height * 0.05
    ]
    if aligned_out_of_bounds:
        profile["source"]["mode"] = "geometry-seed-rejected"
        profile["source"]["seedRig"] = seed_rig.name
        profile["source"]["seedRejectedRoles"] = aligned_out_of_bounds
        profile["source"]["meshSha256"] = hashlib.sha256(mesh.read_bytes()).hexdigest()
        return profile
    result = _profile_from_joints(
        aligned,
        mesh_file or mesh.name,
        source_mode="tokenrig-initialized-geometry-fit",
        source_rig=seed_rig.name,
    )
    result["source"]["meshSha256"] = hashlib.sha256(mesh.read_bytes()).hexdigest()
    return result


def validate_profile(profile: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if profile.get("schemaVersion") != CANONICAL_PROFILE_VERSION:
        errors.append("unsupported canonical profile schema")
    if profile.get("skeletonId") != CANONICAL_SKELETON_ID:
        errors.append("profile does not target humanoid-v1")
    joints = profile.get("joints")
    if not isinstance(joints, dict):
        return [*errors, "profile joints must be an object"]
    for name, parent in CANONICAL_PARENTS.items():
        joint = joints.get(name)
        if not isinstance(joint, dict):
            errors.append(f"missing joint: {name}")
            continue
        if joint.get("parent") != parent:
            errors.append(f"invalid parent for {name}")
        position = joint.get("position")
        if not isinstance(position, list) or len(position) != 3 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in position):
            errors.append(f"invalid position for {name}")
    return errors


def write_skeleton_glb(mesh: Path, profile: dict[str, Any], output: Path) -> Path:
    errors = validate_profile(profile)
    if errors:
        raise ValueError("invalid canonical skeleton profile: " + "; ".join(errors))
    document = read_glb(mesh)
    payload = json.loads(json.dumps(document.payload))
    nodes = payload.setdefault("nodes", [])
    if not isinstance(nodes, list):
        raise ValueError("GLB nodes must be an array")
    existing_names = {node.get("name") for node in nodes if isinstance(node, dict)}
    collisions = set(CANONICAL_PARENTS) & existing_names
    if collisions:
        raise ValueError(f"mesh already contains canonical skeleton names: {', '.join(sorted(collisions))}")
    joints = profile["joints"]
    indexes = {name: len(nodes) + index for index, name in enumerate(CANONICAL_PARENTS)}
    for name in CANONICAL_PARENTS:
        position = joints[name]["position"]
        parent = CANONICAL_PARENTS[name]
        if parent is None:
            local = position
        else:
            parent_position = joints[parent]["position"]
            local = [position[index] - parent_position[index] for index in range(3)]
        children = [child for child, child_parent in CANONICAL_PARENTS.items() if child_parent == name]
        node: dict[str, Any] = {"name": name, "translation": [float(value) for value in local]}
        if children:
            node["children"] = [indexes[child] for child in children]
        nodes.append(node)
    mesh_node_indexes = [index for index, node in enumerate(nodes[: len(nodes) - len(CANONICAL_PARENTS)]) if isinstance(node, dict) and isinstance(node.get("mesh"), int)]
    if not mesh_node_indexes:
        raise ValueError("mesh GLB has no mesh node to bind to the canonical skeleton")
    joint_positions = [tuple(joints[name]["position"]) for name in CANONICAL_PARENTS]
    inverse_bind_binary = b"".join(struct.pack("<16f", *_inverse_translation_matrix(position)) for position in joint_positions)
    binary_offset = _align4(len(document.binary))
    combined_binary = document.binary + b"\0" * (binary_offset - len(document.binary)) + inverse_bind_binary
    payload.setdefault("bufferViews", []).append(
        {"buffer": 0, "byteOffset": binary_offset, "byteLength": len(inverse_bind_binary)}
    )
    inverse_view_index = len(payload["bufferViews"]) - 1
    payload.setdefault("accessors", []).append(
        {
            "bufferView": inverse_view_index,
            "componentType": 5126,
            "count": len(CANONICAL_PARENTS),
            "type": "MAT4",
        }
    )
    inverse_accessor_index = len(payload["accessors"]) - 1
    skin_index = len(payload.get("skins", [])) if isinstance(payload.get("skins"), list) else 0
    payload["skins"] = list(payload.get("skins", [])) if isinstance(payload.get("skins"), list) else []
    payload["skins"].append(
        {
            "joints": [indexes[name] for name in CANONICAL_PARENTS],
            "skeleton": indexes["root"],
            "inverseBindMatrices": inverse_accessor_index,
        }
    )
    for index in mesh_node_indexes:
        nodes[index]["skin"] = skin_index
    buffers = payload.setdefault("buffers", [])
    if not buffers:
        buffers.append({"byteLength": len(combined_binary)})
    else:
        buffers[0]["byteLength"] = len(combined_binary)
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        payload["scene"] = 0
        payload["scenes"] = [{"nodes": [indexes["root"]]}]
    else:
        scene = scenes[int(payload.get("scene", 0)) if isinstance(payload.get("scene", 0), int) else 0]
        scene_nodes = scene.setdefault("nodes", [])
        if indexes["root"] not in scene_nodes:
            scene_nodes.append(indexes["root"])
    extras = payload.setdefault("extras", {})
    extras["canonicalSkeleton"] = {
        "id": CANONICAL_SKELETON_ID,
        "profileVersion": CANONICAL_PROFILE_VERSION,
        "requiredRoles": list(REQUIRED_CANONICAL_ROLES),
    }
    write_glb(output, GlbDocument(payload=payload, binary=combined_binary))
    return output


def _indexed_world_matrices(document: GlbDocument) -> dict[int, list[list[float]]]:
    nodes = document.payload.get("nodes")
    if not isinstance(nodes, list):
        raise ValueError("GLB has no nodes")
    parents: dict[int, int] = {}
    for index, node in enumerate(nodes):
        if isinstance(node, dict):
            for child in node.get("children", []):
                if isinstance(child, int):
                    parents[child] = index
    cache: dict[int, list[list[float]]] = {}

    def resolve(index: int) -> list[list[float]]:
        if index in cache:
            return cache[index]
        local = _node_matrix(nodes[index])
        parent = parents.get(index)
        result = _matmul(resolve(parent), local) if parent is not None else local
        cache[index] = result
        return result

    return {index: resolve(index) for index in range(len(nodes))}


def canonicalize_skinned_glb(
    input_path: Path,
    output_path: Path,
    profile: dict[str, Any],
    manifest_path: Path,
) -> Path:
    """Rename SkinTokens' topology-preserved joints to canonical role names.

    SkinTokens preserves the supplied skeleton topology and positions but its
    Blender export currently renames joints to ``bone_N``.  Match the output
    joints against the fitted profile within the preserved parent/child graph,
    then write stable names and a canonical manifest.
    """

    errors = validate_profile(profile)
    if errors:
        raise ValueError("invalid canonical skeleton profile: " + "; ".join(errors))
    document = read_glb(input_path)
    payload = json.loads(json.dumps(document.payload))
    nodes = payload.get("nodes")
    skins = payload.get("skins")
    if not isinstance(nodes, list) or not isinstance(skins, list) or not skins or not isinstance(skins[0], dict):
        raise ValueError("SkinTokens output does not contain a skin")
    joints = skins[0].get("joints")
    if not isinstance(joints, list) or not joints:
        raise ValueError("SkinTokens output does not contain skin joints")
    joint_set = {index for index in joints if isinstance(index, int)}
    parents: dict[int, int] = {}
    children: dict[int, list[int]] = {index: [] for index in joint_set}
    for parent_index in joint_set:
        node = nodes[parent_index]
        if not isinstance(node, dict):
            continue
        for child in node.get("children", []):
            if isinstance(child, int) and child in joint_set:
                parents[child] = parent_index
                children[parent_index].append(child)
    world = _indexed_world_matrices(GlbDocument(payload=payload, binary=document.binary))
    expected = {name: tuple(value["position"]) for name, value in profile["joints"].items()}
    root_candidates = [index for index in joint_set if index not in parents]
    if not root_candidates:
        raise ValueError("SkinTokens output has no skin root")
    mapping: dict[str, int] = {}
    used: set[int] = set()
    for role, parent_role in CANONICAL_PARENTS.items():
        candidates = root_candidates if parent_role is None else children.get(mapping.get(parent_role, -1), [])
        candidates = [candidate for candidate in candidates if candidate not in used]
        if not candidates:
            raise ValueError(f"SkinTokens output cannot map canonical role: {role}")
        selected = min(
            candidates,
            key=lambda candidate: math.dist(_position(world[candidate]), expected[role]),
        )
        mapping[role] = selected
        used.add(selected)
    for role, node_index in mapping.items():
        nodes[node_index]["name"] = role
    write_glb(output_path, GlbDocument(payload=payload, binary=document.binary))
    write_manifest(profile, manifest_path, model_file=output_path.name)
    return output_path


def write_manifest(profile: dict[str, Any], output: Path, *, model_file: str) -> Path:
    errors = validate_profile(profile)
    if errors:
        raise ValueError("invalid canonical skeleton profile: " + "; ".join(errors))
    payload = {
        # The runtime manifest contract is still schema version 1. The
        # editable canonical profile has its own version inside the manifest.
        "schemaVersion": 1,
        "modelId": Path(model_file).stem,
        "label": Path(model_file).stem,
        "skeletonId": CANONICAL_SKELETON_ID,
        "modelFile": model_file,
        "orientation": {"yawRadians": 0.0},
        "generatedBy": {"rigGenerator": "SkinTokens --use_skeleton", "skeletonFitter": "canonical-skeleton-v1"},
        "bones": {role: CANONICAL_ROLE_TO_JOINT[role] for role in REQUIRED_CANONICAL_ROLES},
        "requiredBones": list(REQUIRED_CANONICAL_ROLES),
        "canonicalProfile": profile,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output
