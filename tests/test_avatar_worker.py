from __future__ import annotations

import asyncio
import json
import struct
import zlib
from pathlib import Path

from autotransition.avatar.artifacts import AvatarArtifactStore
from autotransition.avatar.contracts import AvatarRequest
from autotransition.avatar.pipeline import AvatarPipeline
from autotransition.avatar.worker import AvatarWorker
from autotransition.config import AvatarConfig


def write_png(path: Path, width: int = 2, height: int = 2) -> None:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + bytes([180, 180, 180, 255]) * width for _ in range(height))
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def write_glb(path: Path, *, valid: bool = True) -> None:
    if not valid:
        path.write_bytes(b"not-a-glb")
        return
    roles = [
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
    nodes = [{"name": role} for role in roles]
    manifest = {
        "schemaVersion": 1,
        "skeletonId": "humanoid-v1",
        "bones": {role: role for role in roles},
    }
    payload = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": nodes,
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2}}]}],
        "skins": [{"joints": list(range(len(nodes))), "skeleton": 0}],
        "accessors": [{"count": 3}, {"count": 3}, {"count": 3}],
        "extras": {"manifest": manifest},
    }
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((4 - len(encoded) % 4) % 4)
    total = 12 + 8 + len(encoded)
    path.write_bytes(b"glTF" + struct.pack("<II", 2, total) + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded)


class FakeMeshGenerator:
    def __init__(self, *, nested: bool = False):
        self.nested = nested

    def generate(self, *, image: Path, output_dir: Path, quality: str) -> Path:
        result_dir = output_dir / "0" if self.nested else output_dir
        result_dir.mkdir(parents=True, exist_ok=True)
        result = result_dir / "mesh.glb"
        write_glb(result)
        return result


class FakeRigGenerator:
    def __init__(self, failures: int):
        self.failures = failures
        self.calls = 0

    def generate(self, *, mesh: Path, output: Path, manifest: Path, quality: str) -> tuple[Path, Path]:
        self.calls += 1
        write_glb(output, valid=self.calls > self.failures)
        manifest.write_text(
            json.dumps({
                "skeletonId": "humanoid-v1",
                "bones": {
                    role: role
                    for role in (
                        "hips", "spine", "chest", "head", "upperArmLeft", "upperArmRight",
                        "forearmLeft", "forearmRight", "upperLegLeft", "upperLegRight",
                        "lowerLegLeft", "lowerLegRight", "footLeft", "footRight",
                    )
                },
            }),
            encoding="utf-8",
        )
        return output, manifest


def make_pipeline(
    tmp_path: Path,
    rig_failures: int,
    *,
    require_deformation_validator: bool = False,
    nested_mesh: bool = False,
) -> tuple[AvatarPipeline, FakeRigGenerator]:
    config = AvatarConfig(
        artifact_root=tmp_path / "jobs",
        max_attempts=3,
        min_image_width=2,
        min_image_height=2,
        max_reference_pixels=100,
        max_image_bytes=100_000,
        require_deformation_validator=require_deformation_validator,
    )
    rig = FakeRigGenerator(rig_failures)
    return (
        AvatarPipeline(
            config,
            image_generator=None,
            mesh_generator=FakeMeshGenerator(nested=nested_mesh),
            rig_generator=rig,
            store=AvatarArtifactStore(config.artifact_root),
        ),
        rig,
    )


def test_avatar_retries_broken_rig_then_returns_valid_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    write_png(source)
    pipeline, rig = make_pipeline(tmp_path, rig_failures=2, nested_mesh=True)
    request = AvatarRequest(description="a blue teddy bear", reference_image=source, max_attempts=3, payment_intent_id="payment-1")
    job = pipeline.create_job(request)

    result = pipeline.run(request, job_id=job.id)

    assert result.status == "succeeded"
    assert result.refund_required is False
    assert rig.calls == 3
    assert {artifact.name for artifact in result.artifacts} >= {"avatar.glb", "manifest.json", "diagnostics.json"}
    persisted = pipeline.store.read_job(job.id)
    assert len(persisted["attempts"]) == 2
    assert persisted["status"] == "succeeded"
    assert persisted["request"]["payment_intent_id"] == "payment-1"


def test_avatar_refunds_after_validation_retry_budget_is_exhausted(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    write_png(source)
    pipeline, rig = make_pipeline(tmp_path, rig_failures=99)
    request = AvatarRequest(description="a green robot", reference_image=source, max_attempts=3)
    job = pipeline.create_job(request)

    result = pipeline.run(request, job_id=job.id)

    assert result.status == "failed"
    assert result.refund_required is True
    assert result.refund_reason == "avatar_output_validation_failed_after_retries"
    assert result.to_dict()["refundRequired"] is True
    assert result.to_dict()["failureCode"] == "avatar_validation_exhausted"
    assert result.failure is not None
    assert result.failure.code == "avatar_validation_exhausted"
    assert rig.calls == 3
    persisted = pipeline.store.read_job(job.id)
    assert persisted["refund_required"] is True
    assert persisted["failure"]["code"] == "avatar_validation_exhausted"


def test_avatar_worker_reconciles_interrupted_paid_job(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    write_png(source)
    pipeline, _ = make_pipeline(tmp_path, rig_failures=0)
    job = pipeline.create_job(AvatarRequest(description="a yellow robot", reference_image=source))
    payload = pipeline.store.read_job(job.id)
    payload["status"] = "running"
    pipeline.store._atomic_json(pipeline.store.job_dir(job.id) / "job.json", payload)

    assert pipeline.store.reconcile_interrupted_jobs() == [job.id]
    persisted = pipeline.store.read_job(job.id)
    assert persisted["status"] == "failed"
    assert persisted["failureCode"] == "avatar_worker_interrupted"
    assert persisted["refundRequired"] is True


def test_avatar_builtin_deformation_validation_exhausts_retries(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    write_png(source)
    pipeline, rig = make_pipeline(tmp_path, rig_failures=0, require_deformation_validator=True)
    request = AvatarRequest(description="a test humanoid", reference_image=source, max_attempts=3)
    job = pipeline.create_job(request)

    result = pipeline.run(request, job_id=job.id)

    assert result.status == "failed"
    assert result.failure is not None
    assert result.failure.code == "avatar_validation_exhausted"
    assert result.refund_reason == "avatar_output_validation_failed_after_retries"
    assert rig.calls == 3


def test_avatar_worker_queues_and_persists_completion(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    write_png(source)
    pipeline, _ = make_pipeline(tmp_path, rig_failures=0)
    worker = AvatarWorker(pipeline)
    try:
        job = asyncio.run(worker.submit(AvatarRequest(description="a queued humanoid", reference_image=source)))
        future = worker._futures[job.id]
        future.result(timeout=30)
        persisted = pipeline.store.read_job(job.id)
        assert persisted["status"] == "succeeded"
        assert {item["name"] for item in persisted["artifacts"]} >= {"avatar.glb", "manifest.json", "diagnostics.json"}
    finally:
        worker.shutdown()
