"""Build a conservative humanoid-v1 manifest for a TokenRig GLB.

TokenRig currently exports generated joints as bone_N names. This mapper uses
the generated hierarchy and local joint offsets to identify the torso, two
arms, and two legs. Missing roles remain missing in the manifest instead of
being guessed, so the dance client can report incomplete rig coverage.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


REQUIRED_ROLES = [
    "hips",
    "spine",
    "chest",
    "head",
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
]


def _children(scene: Any, name: str, bones: set[str]) -> list[str]:
    children = scene.graph.transforms.children.get(name, [])
    return [child for child in children if child in bones]


def _local_x(scene: Any, name: str) -> float:
    transform, _ = scene.graph[name]
    return float(transform[0, 3])


def _local_y(scene: Any, name: str) -> float:
    transform, _ = scene.graph[name]
    return float(transform[1, 3])


def _terminal_y(scene: Any, name: str, bones: set[str]) -> float:
    children = _children(scene, name, bones)
    if not children:
        return _local_y(scene, name)
    return max(_terminal_y(scene, child, bones) for child in children)


def _chain_length(scene: Any, name: str, bones: set[str]) -> int:
    children = _children(scene, name, bones)
    if not children:
        return 1
    return 1 + max(_chain_length(scene, child, bones) for child in children)


def _longest_chain(scene: Any, name: str, bones: set[str]) -> list[str]:
    children = _children(scene, name, bones)
    if not children:
        return [name]
    child = max(children, key=lambda candidate: _chain_length(scene, candidate, bones))
    return [name, *_longest_chain(scene, child, bones)]


def _next_upward_child(scene: Any, name: str, bones: set[str]) -> str | None:
    children = _children(scene, name, bones)
    if not children:
        return None
    # The torso terminates at the highest point. This avoids confusing an arm
    # branch for the torso when an intermediate arm joint sits above the pelvis.
    return max(children, key=lambda candidate: (_terminal_y(scene, candidate, bones), -_local_x(scene, candidate)))


def _root_bone(scene: Any, bones: set[str]) -> str:
    parents = scene.graph.transforms.parents
    roots = [name for name in bones if parents.get(name) not in bones]
    if not roots:
        raise ValueError("TokenRig GLB does not contain a bone root.")
    return sorted(roots, key=lambda value: int(value.removeprefix("bone_")) if value.removeprefix("bone_").isdigit() else value)[0]


def build_manifest(
    glb_path: Path,
    model_id: str | None = None,
    front_yaw_degrees: float = 180.0,
) -> dict[str, Any]:
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError("trimesh is required to build a TokenRig manifest.") from exc

    scene = trimesh.load(glb_path, force="scene")
    bones = {name for name in scene.graph.nodes if name.startswith("bone_")}
    if not bones:
        raise ValueError(f"No TokenRig bone_N nodes found in {glb_path}.")
    if not math.isfinite(front_yaw_degrees):
        raise ValueError("front_yaw_degrees must be finite.")

    root = _root_bone(scene, bones)
    torso = [root]
    while len(torso) < 6:
        child = _next_upward_child(scene, torso[-1], bones)
        if child is None:
            break
        torso.append(child)

    mapped: dict[str, str] = {}
    if len(torso) > 1:
        mapped["hips"] = torso[1]
    if len(torso) > 2:
        mapped["spine"] = torso[2]
    if len(torso) > 3:
        mapped["chest"] = torso[3]
    if len(torso) > 4:
        mapped["neck"] = torso[4]
    if len(torso) > 5:
        mapped["head"] = torso[5]

    torso_next = torso[4] if len(torso) > 4 else None
    chest = torso[3] if len(torso) > 3 else None
    arm_roots = [] if chest is None else [
        child for child in _children(scene, chest, bones)
        if child != torso_next and _chain_length(scene, child, bones) >= 3
    ]
    arm_roots.sort(key=lambda name: _local_x(scene, name))
    for side, arm_root in zip(("Left", "Right"), arm_roots[:2]):
        mapped[f"upperArm{side}"] = arm_root
        forearm = _children(scene, arm_root, bones)
        if forearm:
            mapped[f"forearm{side}"] = max(forearm, key=lambda name: _chain_length(scene, name, bones))

    leg_roots = [
        child for child in _children(scene, root, bones)
        if child != (torso[1] if len(torso) > 1 else None) and _chain_length(scene, child, bones) >= 3
    ]
    leg_roots.sort(key=lambda name: _local_x(scene, name))
    for side, leg_root in zip(("Left", "Right"), leg_roots[:2]):
        mapped[f"upperLeg{side}"] = leg_root
        lower_leg = _children(scene, leg_root, bones)
        if lower_leg:
            lower = max(lower_leg, key=lambda name: _chain_length(scene, name, bones))
            mapped[f"lowerLeg{side}"] = lower
            foot = _children(scene, lower, bones)
            if foot:
                mapped[f"foot{side}"] = max(foot, key=lambda name: _chain_length(scene, name, bones))

    return {
        "schemaVersion": 1,
        "modelId": model_id or glb_path.stem,
        "label": glb_path.stem,
        "skeletonId": "humanoid-v1",
        "modelFile": glb_path.name,
        "orientation": {
            "yawRadians": math.radians(front_yaw_degrees),
        },
        "generatedBy": {
            "rigGenerator": "SkinTokens TokenRig"
        },
        "bones": mapped,
        "requiredBones": REQUIRED_ROLES,
    }


def write_manifest(glb_path: Path, output_path: Path, front_yaw_degrees: float = 180.0) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_manifest(glb_path, front_yaw_degrees=front_yaw_degrees), indent=2) + "\n",
        encoding="utf-8",
    )
