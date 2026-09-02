from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autotransition.avatar.adapters.base import AvatarAdapterError
from autotransition.generative_dance.artifacts import ArtifactStore
from autotransition.generative_dance.compose import build_driver_composition_plan
from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.contracts import BoundaryState, CanvasContract, DanceDriver
from autotransition.generative_dance.contracts import AvatarReference, RenderedSegment
from autotransition.generative_dance.image_pipeline import build_reference_prompt
from autotransition.generative_dance.identity import audit_segment_identity
from autotransition.generative_dance.service import GenerativeDanceService
from autotransition.generative_dance.video import VideoProbe
from autotransition.generative_dance.wan_animate import WanAnimate2LiteAdapter
from autotransition.generative_dance.worker import (
    _resolve_identity_seed,
    _resolve_reference_strength,
    _sequence_segments_are_adjacent,
)
from autotransition.generative_dance.worker import GenerativeDanceWorker


def test_native_wan_runner_render_path_uses_selected_runtime_dtype() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runner.py").read_text(encoding="utf-8")
    render_start = source.index("def _render_window(")
    render_end = source.index("def render_segment(", render_start)
    render_source = source[render_start:render_end]

    assert "compute_dtype = runtime.compute_dtype" in render_source
    assert "dtype=torch.bfloat16" not in render_source


def test_native_wan_runner_accepts_configured_temporal_window() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runner.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--max-clip-len"' in source
    assert "max_clip_len=args.max_clip_len" in source
    assert "_masked_chunked_attention" in source
    assert 'WAN_FLEX_ATTENTION_CHUNK_SIZE' in source
    assert "WAN_T5_DEVICE" in source
    assert "_encode_t5" in source


def test_native_wan_runner_has_bounded_denoising_attention_mode() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runner.py").read_text(encoding="utf-8")

    assert 'if _sdpa_backend_name() in {"cpu", "chunked", "manual"}' in source
    assert '"WAN_FLEX_ATTENTION_CHUNK_SIZE"' in source


def test_native_wan_runtime_provides_writable_cuda_linker_alias() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runtime.py").read_text(encoding="utf-8")

    assert "libcuda.so.1" in source
    assert "LIBRARY_PATH" in source
    assert "cuda-link" in source


def test_native_wan_runtime_has_explicit_cuda_dequant_dtype_policy() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runtime.py").read_text(encoding="utf-8")
    runner_source = Path("tools/generative_dance/wan_animate_2_runner.py").read_text(encoding="utf-8")

    assert 'WAN_GGUF_DEQUANT_DTYPE' in source
    assert "torch.bfloat16" in source
    assert "result.dtype != weight.dtype" not in source
    assert "reference_cache_ready" in runner_source
    assert "class _TypedReferenceCache" in runner_source
    assert "cache_k = _TypedReferenceCache(compute_dtype)" in runner_source
    assert "_normalize_reference_cache" in runner_source


def test_wan_overlay_requires_fused_5090_attention_and_raw_cache() -> None:
    source = Path("tools/publish_wan_overlay_manifest.py").read_text(encoding="utf-8")

    assert '"WAN_FLEX_ATTENTION_BACKEND": "official"' in source
    assert '"WAN_REQUIRE_FLEX_ATTENTION": "1"' in source
    assert '"WAN_REQUIRE_FLASH_ATTENTION": "1"' in source
    assert '"WAN_REFERENCE_ATTENTION_BACKEND": "flash"' in source
    assert '"WAN_GGUF_GPU_RAW_CACHE": "1"' in source
    assert '"WAN_GGUF_DEQUANT_DTYPE": "bfloat16"' in source
    assert '"GENERATIVE_DANCE_WAN_STEPS": "4"' in source
    assert '"GENERATIVE_DANCE_WAN_MIN_STEPS": "4"' in source
    assert '"GENERATIVE_DANCE_WAN_LIGHTX2V_ENABLED": "1"' in source
    assert '"WAN_LIGHTX2V_CHECKPOINT": "/models/wan-animate-2/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"' in source


def test_lightx2v_uses_the_official_four_step_schedule() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runner.py").read_text(encoding="utf-8")
    assert "def _lightx2v_sampling_sigmas(sampling_steps: int)" in source
    assert "if sampling_steps != 4" in source
    assert "return np.asarray([1.0, 0.75, 0.5, 0.25]" in source


def test_lightx2v_maps_official_names_to_the_runtime_transformer_names() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runtime.py").read_text(encoding="utf-8")
    assert 'parts[0] == "diffusion_model"' in source
    assert 'parts.insert(2, "block")' in source
    assert 'replaced_linears.get(target)' in source


def test_native_wan_runner_keeps_one_seed_across_temporal_windows() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runner.py").read_text(encoding="utf-8")
    assert "windows_use_same_seed=true" in source
    assert "seed=effective_seed" in source


def test_native_wan_runner_drops_conditioning_latent_before_vae_decode() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runner.py").read_text(encoding="utf-8")
    render_start = source.index("def _render_window(")
    render_end = source.index("def render_segment(", render_start)
    render_source = source[render_start:render_end]

    assert "generated_latents = latents[0][:, 1:]" in render_source
    assert "vae.decode([generated_latents.to(dtype=torch.float32)])" in render_source
    assert "Wan returns the conditioning frame as the first decoded" not in render_source


def test_native_wan_runner_applies_reference_strength_to_cached_values() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runner.py").read_text(encoding="utf-8")

    assert "class _ScaledReferenceValues" in source
    assert "with _ScaledReferenceValues(cache_v, reference_strength)" in source
    assert "value.mul_(self.strength)" in source


def test_native_wan_command_passes_configured_temporal_window(tmp_path: Path) -> None:
    config = GenerativeDanceConfig(
        artifact_root=tmp_path,
        wan_backend="native",
        wan_temporal_window=33,
        wan_transformer_checkpoint=tmp_path / "model.gguf",
        wan_official_source=tmp_path / "source",
        wan_t5_checkpoint=tmp_path / "t5.pth",
        wan_t5_tokenizer=tmp_path / "t5-tokenizer",
        wan_clip_checkpoint=tmp_path / "clip.pth",
        wan_clip_tokenizer=tmp_path / "clip-tokenizer",
        wan_vae_checkpoint=tmp_path / "vae.pth",
    )
    adapter = WanAnimate2LiteAdapter(config)

    assert "--max-clip-len" in adapter.native_command
    assert "{temporal_window}" in adapter.native_command
    assert "--temporal-context-frames" in adapter.native_command
    assert "{temporal_context_frames}" in adapter.native_command
    assert "--continuation-frames" not in adapter.native_command


def test_native_wan_command_passes_reference_strength(tmp_path: Path) -> None:
    config = GenerativeDanceConfig(
        artifact_root=tmp_path,
        wan_backend="native",
        wan_reference_strength=1.25,
        wan_transformer_checkpoint=tmp_path / "model.gguf",
        wan_official_source=tmp_path / "source",
        wan_t5_checkpoint=tmp_path / "t5.pth",
        wan_t5_tokenizer=tmp_path / "t5-tokenizer",
        wan_clip_checkpoint=tmp_path / "clip.pth",
        wan_clip_tokenizer=tmp_path / "clip-tokenizer",
        wan_vae_checkpoint=tmp_path / "vae.pth",
    )
    adapter = WanAnimate2LiteAdapter(config)

    assert "--reference-strength" in adapter.native_command
    assert "{reference_strength}" in adapter.native_command


def test_native_wan_command_passes_lightx2v_checkpoint_and_strength(tmp_path: Path) -> None:
    config = GenerativeDanceConfig(
        artifact_root=tmp_path,
        wan_backend="native",
        wan_lightx2v_enabled=True,
        wan_lightx2v_checkpoint=tmp_path / "lightx2v.safetensors",
        wan_transformer_checkpoint=tmp_path / "model.gguf",
        wan_official_source=tmp_path / "source",
        wan_t5_checkpoint=tmp_path / "t5.pth",
        wan_t5_tokenizer=tmp_path / "t5-tokenizer",
        wan_clip_checkpoint=tmp_path / "clip.pth",
        wan_clip_tokenizer=tmp_path / "clip-tokenizer",
        wan_vae_checkpoint=tmp_path / "vae.pth",
        wan_inference_steps=4,
        wan_min_inference_steps=4,
    )
    adapter = WanAnimate2LiteAdapter(config)

    assert "--lightx2v" in adapter.native_command
    assert "{wan_lightx2v_checkpoint}" in adapter.native_command
    assert "--lightx2v-strength" in adapter.native_command
    assert "{wan_lightx2v_strength}" in adapter.native_command


def test_identity_seed_is_stable_for_a_job_but_random_when_unconfigured() -> None:
    assert _resolve_identity_seed({"seed": 2**32 + 17}) == 17
    first = _resolve_identity_seed({})
    second = _resolve_identity_seed({})

    assert 0 <= first < 2**32
    assert 0 <= second < 2**32


def test_reference_strength_can_be_overridden_per_request() -> None:
    config = GenerativeDanceConfig(wan_reference_strength=1.25)

    assert _resolve_reference_strength(config, {}) == pytest.approx(1.25)
    assert _resolve_reference_strength(config, {"referenceStrength": 1.5}) == pytest.approx(1.5)
    with pytest.raises(ValueError, match="reference strength"):
        _resolve_reference_strength(config, {"reference_strength": 0})


def test_temporal_context_defaults_to_wan_five_frame_guidance() -> None:
    config = GenerativeDanceConfig()

    assert config.wan_temporal_window == 81
    assert config.wan_temporal_context_frames == 5


def test_temporal_window_settings_can_be_overridden_at_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENERATIVE_DANCE_WAN_TEMPORAL_WINDOW", "97")
    monkeypatch.setenv("GENERATIVE_DANCE_WAN_TEMPORAL_CONTEXT_FRAMES", "1")

    config = GenerativeDanceConfig.from_env()

    assert config.wan_temporal_window == 97
    assert config.wan_temporal_context_frames == 1


def test_identity_audit_reports_score_and_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import autotransition.generative_dance.identity as identity_module

    config = GenerativeDanceConfig(
        artifact_root=tmp_path,
        identity_audit_enabled=True,
        identity_audit_command=("identity-scorer", "{output}"),
        identity_audit_threshold=0.8,
    )

    def fake_run(command: tuple[str, ...], **kwargs: object) -> None:
        values = kwargs["values"]
        assert isinstance(values, dict)
        output = Path(str(values["output"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"score": 0.86, "metric": "test"}), encoding="utf-8")

    monkeypatch.setattr(identity_module, "run_adapter_command", fake_run)
    report = audit_segment_identity(
        config,
        reference_image=tmp_path / "reference.png",
        render_video=tmp_path / "render.mp4",
        output_dir=tmp_path / "audit",
        segment_id="segment-2",
        seed=13,
        attempt=0,
    )

    assert report["score"] == pytest.approx(0.86)
    assert report["threshold"] == pytest.approx(0.8)
    assert report["passed"] is True
    assert (tmp_path / "audit" / "identity-audit.json").is_file()


def test_native_wan_rejects_unsupported_distilled_step_count(tmp_path: Path) -> None:
    config = GenerativeDanceConfig(
        artifact_root=tmp_path,
        wan_backend="native",
        wan_transformer_checkpoint=tmp_path / "model.gguf",
        wan_official_source=tmp_path / "source",
        wan_t5_checkpoint=tmp_path / "t5.pth",
        wan_t5_tokenizer=tmp_path / "t5-tokenizer",
        wan_clip_checkpoint=tmp_path / "clip.pth",
        wan_clip_tokenizer=tmp_path / "clip-tokenizer",
        wan_vae_checkpoint=tmp_path / "vae.pth",
    )
    adapter = WanAnimate2LiteAdapter(config)

    with pytest.raises(AvatarAdapterError, match="requires at least 10 inference steps"):
        adapter.render(
            segment_id="segment-test",
            reference=object(),  # type: ignore[arg-type]
            driver=object(),  # type: ignore[arg-type]
            output_dir=tmp_path / "render",
            inference_steps=1,
        )


def test_native_wan_continuation_argument_is_added_only_when_needed() -> None:
    source = Path("tools/generative_dance/wan_animate_2_runner.py").read_text(encoding="utf-8")

    assert 'parser.add_argument(\n        "--continuation-frames"' in source
    assert 'parser.add_argument(\n        "--temporal-context-frames"' in source
    assert "continuation_frames=args.continuation_frames" in source
    assert "continuation_frame=args.continuation_frame" in source


def test_sequence_continuity_carries_only_across_adjacent_segments() -> None:
    assert not _sequence_segments_are_adjacent(0.0, None, fps=24)
    assert _sequence_segments_are_adjacent(2.0, 2.0, fps=24)
    assert _sequence_segments_are_adjacent(2.02, 2.0, fps=24)
    assert not _sequence_segments_are_adjacent(2.2, 2.0, fps=24)


def test_worker_sequence_resets_continuation_at_source_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autotransition.generative_dance.worker as worker_module

    config = GenerativeDanceConfig(
        artifact_root=tmp_path,
        identity_audit_enabled=True,
        identity_audit_command=("identity-scorer",),
        identity_audit_max_retries=1,
    )
    calls: list[Path | None] = []
    audit_calls: list[tuple[str, int]] = []
    render_count = 0

    class FakeService:
        def get_reference(self, reference_id: str) -> AvatarReference:
            return AvatarReference(
                id=reference_id,
                description="reference",
                prompt="reference",
                source_image=tmp_path / "reference.png",
                normalized_image=tmp_path / "reference-normalized.png",
                matte_image=None,
                canvas=CanvasContract(),
                metadata_path=tmp_path / "reference.json",
            )

        def create_driver(self, *, source: Path, label: str, output_fps: int) -> DanceDriver:
            boundary = BoundaryState(0.0, (0.5, 0.58), (0.1, 0.1, 0.9, 0.95))
            return DanceDriver(
                id=f"driver-{label}",
                label=label,
                source_video=source,
                normalized_video=source,
                duration_seconds=1.0,
                canvas=CanvasContract(fps=output_fps),
                start_boundary=boundary,
                end_boundary=boundary,
                metadata_path=source.with_suffix(".json"),
            )

        def render(self, **kwargs: object) -> RenderedSegment:
            nonlocal render_count
            render_count += 1
            continuation = kwargs.get("continuation_frames")
            calls.append(continuation if isinstance(continuation, Path) else None)
            output = tmp_path / f"render-{render_count}.mp4"
            metadata = output.with_suffix(".json")
            output.write_bytes(b"render")
            metadata.write_text("{}", encoding="utf-8")
            driver = kwargs["driver_id"]
            reference = kwargs["reference_id"]
            return RenderedSegment(
                id=f"segment-{render_count}",
                driver_id=str(driver),
                reference_id=str(reference),
                output_video=output,
                duration_seconds=1.0,
                canvas=CanvasContract(),
                metadata_path=metadata,
                model_revision="test-model",
                prompt=str(kwargs.get("prompt") or ""),
            )

    service = FakeService()
    worker = GenerativeDanceWorker(config, service=service)  # type: ignore[arg-type]
    worker._write(
        "sequence-test",
        {"id": "sequence-test", "status": "queued", "stage": "queued", "progress": 0.0},
    )

    def fake_download(job_id: str, url: str, name: str, *, prefix: str, suffixes: set[str]) -> Path:
        target = worker.job_dir(job_id) / f"{prefix}.mp4"
        target.write_bytes(b"source")
        return target

    def fake_extract(source: Path, output: Path, *, start_seconds: float, end_seconds: float, output_fps: int) -> object:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"segment")
        return SimpleNamespace(path=output)

    def fake_make_blank(output: Path, **_: object) -> SimpleNamespace:
        output.write_bytes(b"gap")
        return SimpleNamespace(path=output)

    def fake_stitch(inputs: list[Path], output: Path, **_: object) -> None:
        output.write_bytes(b"stitched")
        return VideoProbe(output, 640, 800, 24.0, len(inputs), len(inputs) * 24, "yuv420p")

    monkeypatch.setattr(worker, "_download", fake_download)
    monkeypatch.setattr(worker_module, "probe_video", lambda _path: VideoProbe(Path("source.mp4"), 640, 800, 24.0, 10.0, 240, "yuv420p"))
    monkeypatch.setattr(worker_module, "extract_video_range", fake_extract)
    monkeypatch.setattr(worker_module, "make_blank_video", fake_make_blank)
    monkeypatch.setattr(worker_module, "stitch_videos", fake_stitch)

    audit_scores = iter([0.4, 0.95, 0.95])

    def fake_audit(*_: object, segment_id: str, attempt: int, **__: object) -> dict[str, object]:
        score = next(audit_scores)
        audit_calls.append((segment_id, attempt))
        return {
            "enabled": True,
            "score": score,
            "threshold": 0.8,
            "passed": score >= 0.8,
            "attempt": attempt,
        }

    monkeypatch.setattr(worker_module, "audit_segment_identity", fake_audit)

    result, final_output = worker._run_sequence(
        job_id="sequence-test",
        payload={
            "inputs": [{"id": "driver-a", "role": "driver", "sourceUrl": "https://example.test/a.mp4", "fileName": "a.mp4"}],
        },
        parameters={
            "transparent": False,
            "sequence": {
                "fps": 24,
                "segments": [
                    {"id": "first", "inputId": "driver-a", "sourceStartSeconds": 0, "sourceEndSeconds": 1, "timelineStartSeconds": 0, "timelineEndSeconds": 1},
                    {"id": "adjacent", "inputId": "driver-a", "sourceStartSeconds": 1, "sourceEndSeconds": 2, "timelineStartSeconds": 1, "timelineEndSeconds": 2},
                    {"id": "gapped", "inputId": "driver-a", "sourceStartSeconds": 2, "sourceEndSeconds": 3, "timelineStartSeconds": 3, "timelineEndSeconds": 4},
                ],
            },
        },
        reference_id="reference-test",
    )

    assert final_output.is_file()
    assert calls == [None, None, None, None]
    assert audit_calls == [("adjacent", 0), ("adjacent", 1), ("gapped", 0)]
    assert [item["adjacentToPrevious"] for item in result["segments"]] == [False, True, False]
    assert all(
        item["continuity"]["crossSegmentContinuationUsed"] is False
        for item in result["segments"]
    )
    assert result["continuityPolicy"] == "within-source-segment-only"


def test_success_artifacts_exclude_inputs_removed_after_completion(tmp_path: Path) -> None:
    worker = GenerativeDanceWorker(GenerativeDanceConfig(artifact_root=tmp_path))
    job_id = "artifact-cleanup-test"
    job_root = worker.job_dir(job_id)
    worker._write(job_id, {"id": job_id, "status": "queued"})
    (job_root / "reference.png").write_bytes(b"reference")
    (job_root / "driver-driver-a.mp4").write_bytes(b"driver")
    final_dir = job_root / "00-final"
    final_dir.mkdir()
    (final_dir / "generative-dance-output.mp4").write_bytes(b"render")
    (final_dir / "job-result.json").write_text("{}", encoding="utf-8")

    artifacts = worker._artifacts(job_id, job_root)
    names = {item["name"] for item in artifacts}

    assert names == {"generative-dance-output.mp4", "job-result.json", "job.json"}


def test_reference_prompt_keeps_policy_and_user_description() -> None:
    prompt = build_reference_prompt("a purple cartoon horse wearing a cap")

    assert "single full-body bipedal character" in prompt
    assert "seamless solid contrasting background" in prompt
    assert "purple cartoon horse wearing a cap" in prompt
    assert "extra people" in prompt


def test_canvas_contract_rejects_invalid_anchor() -> None:
    with pytest.raises(ValueError, match="canvas anchor"):
        CanvasContract(anchor_x=1.2).validate()


def test_artifact_store_rejects_paths_outside_root(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "artifacts")

    with pytest.raises(ValueError, match="outside"):
        store.resolve_relative("../secret.txt")


def test_unconfigured_wan_adapter_returns_actionable_error(tmp_path: Path) -> None:
    config = GenerativeDanceConfig(artifact_root=tmp_path)
    adapter = WanAnimate2LiteAdapter(config)

    assert not adapter.configured
    with pytest.raises(AvatarAdapterError, match="not configured"):
        adapter.render(
            segment_id="segment-1",
            reference=object(),  # type: ignore[arg-type]
            driver=object(),  # type: ignore[arg-type]
            output_dir=tmp_path / "render",
        )


def test_service_status_does_not_expose_command_templates(tmp_path: Path) -> None:
    config = GenerativeDanceConfig(
        artifact_root=tmp_path,
        image_command="python private-image-command.py --token SECRET",
        wan_command="python wan.py --token SECRET",
    )
    status = GenerativeDanceService(config).status()

    encoded = json.dumps(status)
    assert "SECRET" not in encoded
    assert status["imageCommandConfigured"] is True
    assert status["wanCommandConfigured"] is True


def test_service_render_forwards_worker_runtime_options(tmp_path: Path) -> None:
    config = GenerativeDanceConfig(artifact_root=tmp_path)
    service = GenerativeDanceService(config)
    reference = AvatarReference(
        id="reference-test",
        description="test reference",
        prompt="test prompt",
        source_image=tmp_path / "reference.png",
        normalized_image=tmp_path / "reference-normalized.png",
        matte_image=None,
        canvas=CanvasContract(),
        metadata_path=tmp_path / "reference.json",
    )
    boundary = BoundaryState(0.0, (0.5, 0.58), (0.1, 0.1, 0.9, 0.95))
    driver = DanceDriver(
        id="driver-test",
        label="test driver",
        source_video=tmp_path / "driver.mp4",
        normalized_video=tmp_path / "driver-normalized.mp4",
        duration_seconds=1.0,
        canvas=CanvasContract(),
        start_boundary=boundary,
        end_boundary=boundary,
        metadata_path=tmp_path / "driver.json",
    )

    class FakeWan:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def render(self, **kwargs: object) -> RenderedSegment:
            self.kwargs = kwargs
            output_dir = kwargs["output_dir"]
            assert isinstance(output_dir, Path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_video = output_dir / "render.mp4"
            metadata_path = output_dir / "render.json"
            output_video.write_bytes(b"test")
            metadata_path.write_text("{}", encoding="utf-8")
            return RenderedSegment(
                id="segment-test",
                driver_id=driver.id,
                reference_id=reference.id,
                output_video=output_video,
                duration_seconds=driver.duration_seconds,
                canvas=driver.canvas,
                metadata_path=metadata_path,
                model_revision="test-model",
                prompt="test prompt",
            )

    fake_wan = FakeWan()
    service.wan = fake_wan  # type: ignore[assignment]
    service.get_reference = lambda _reference_id: reference  # type: ignore[method-assign]
    service.get_driver = lambda _driver_id: driver  # type: ignore[method-assign]

    service.render(
        reference_id=reference.id,
        driver_id=driver.id,
        inference_steps=17,
        text_length=128,
        transparent=False,
    )

    assert fake_wan.kwargs["inference_steps"] == 17
    assert fake_wan.kwargs["text_length"] == 128


def test_native_wan_command_normalizes_probe_fps_to_integer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = GenerativeDanceConfig(
        artifact_root=tmp_path,
        wan_backend="native",
        wan_transformer_checkpoint=tmp_path / "model.gguf",
        wan_official_source=tmp_path / "source",
        wan_t5_checkpoint=tmp_path / "t5.pth",
        wan_t5_tokenizer=tmp_path / "t5-tokenizer",
        wan_clip_checkpoint=tmp_path / "clip.pth",
        wan_clip_tokenizer=tmp_path / "clip-tokenizer",
        wan_vae_checkpoint=tmp_path / "vae.pth",
    )
    adapter = WanAnimate2LiteAdapter(config)

    reference = AvatarReference(
        id="reference-test",
        description="test",
        prompt="test",
        source_image=tmp_path / "reference.png",
        normalized_image=tmp_path / "reference-normalized.png",
        matte_image=None,
        canvas=CanvasContract(),
        metadata_path=tmp_path / "reference.json",
    )
    boundary = BoundaryState(0.0, (0.5, 0.58), (0.1, 0.1, 0.9, 0.95))
    driver = DanceDriver(
        id="driver-test",
        label="test",
        source_video=tmp_path / "driver.mp4",
        normalized_video=tmp_path / "driver-normalized.mp4",
        duration_seconds=1.0,
        canvas=CanvasContract(fps=20.0),
        start_boundary=boundary,
        end_boundary=boundary,
        metadata_path=tmp_path / "driver.json",
    )
    captured: dict[str, object] = {}

    def fake_run_adapter_command(template: object, **kwargs: object) -> None:
        captured.update(kwargs["values"])  # type: ignore[arg-type]
        values = kwargs["values"]  # type: ignore[assignment]
        output = values["output"]  # type: ignore[index]
        assert isinstance(output, Path)
        output.write_bytes(b"render")

    monkeypatch.setattr("autotransition.generative_dance.wan_animate.run_adapter_command", fake_run_adapter_command)
    monkeypatch.setattr(
        "autotransition.generative_dance.wan_animate.probe_video",
        lambda _path: VideoProbe(Path("render.mp4"), 640, 800, 20.0, 1.0, 20, "yuv420p"),
    )

    adapter.render(
        segment_id="segment-test",
        reference=reference,
        driver=driver,
        output_dir=tmp_path / "render",
    )

    assert captured["fps"] == 20
    assert captured["seed"] == 0


def test_driver_composition_plan_records_boundary_review(tmp_path: Path) -> None:
    canvas = CanvasContract()
    boundary_a = BoundaryState(1.0, (0.5, 0.58), (0.2, 0.1, 0.8, 0.95), pose_signature="a", confidence=0.9)
    boundary_b = BoundaryState(0.0, (0.52, 0.58), (0.2, 0.1, 0.8, 0.95), pose_signature="b", confidence=0.8)
    driver_a = DanceDriver("a", "A", Path("a.mp4"), Path("a-normalized.mp4"), 1.0, canvas, boundary_a, boundary_a, Path("a.json"))
    driver_b = DanceDriver("b", "B", Path("b.mp4"), Path("b-normalized.mp4"), 1.0, canvas, boundary_b, boundary_b, Path("b.json"))

    result = build_driver_composition_plan(
        composition_id="chain-1",
        drivers=[driver_a, driver_b],
        config=GenerativeDanceConfig(artifact_root=tmp_path),
        store=ArtifactStore(tmp_path),
    )

    assert result["status"] == "planned"
    assert result["transitions"][0]["strategy"] == "needs-transition-review"
    assert (tmp_path / "driver-compositions" / "chain-1" / "driver-composition.json").is_file()
