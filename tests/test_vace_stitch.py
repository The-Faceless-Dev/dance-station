from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from autotransition.generative_dance.video import VideoProbe
from autotransition.generative_dance.video import transform_alpha_video
from autotransition.vace_stitch.config import VaceStitchConfig
from autotransition.vace_stitch.contracts import BridgeSpec, StitchSequence
from autotransition.vace_stitch.diagnostics import compare_frame_bytes, part_boundaries
from autotransition.vace_stitch.enhancement import VaceVideoStage
from autotransition.vace_stitch.fp8_loader import ScaledFP8Linear
from autotransition.vace_stitch.runtime import VaceRuntime
from autotransition.vace_stitch.video import (
    PreparedVaceInput,
    extract_generated_gap,
    frame_count,
    prepare_firstlastclip,
    solid_video,
)
from autotransition.vace_stitch.worker import VaceStitchWorker


def test_sequence_defaults_to_a_bridge_for_each_boundary_and_a_loop() -> None:
    sequence = StitchSequence.from_parameters(
        {
            "sequence": {
                "segments": [
                    {"id": "a", "inputId": "clip-a"},
                    {"id": "b", "inputId": "clip-b"},
                    {"id": "c", "inputId": "clip-c"},
                ]
            },
            "bridgePrompt": "the character continues dancing",
        },
        default_duration=2.0,
    )

    assert [bridge.before_segment_id for bridge in sequence.bridges] == ["a", "b"]
    assert [bridge.after_segment_id for bridge in sequence.bridges] == ["b", "c"]
    assert sequence.loop_bridge is not None
    assert sequence.loop_bridge.before_segment_id == "c"
    assert sequence.loop_bridge.after_segment_id == "a"


def test_bridge_prompt_overrides_job_prompt() -> None:
    sequence = StitchSequence.from_parameters(
        {
            "sequence": {
                "segments": [{"id": "a", "inputId": "a"}, {"id": "b", "inputId": "b"}],
                "bridges": [{"prompt": "a slower controlled dance transition"}],
                "loop": {"enabled": False},
            },
            "bridgePrompt": "the character continues dancing",
        },
        default_duration=2.0,
    )

    assert sequence.bridges[0].prompt == "a slower controlled dance transition"
    assert sequence.loop_bridge is None


def test_vace_config_exposes_prompt_and_model_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VACE_STITCH_DEFAULT_PROMPT", "continue the same dance naturally")
    monkeypatch.setenv("VACE_STITCH_MODEL_NAME", "vace-1.3B")
    monkeypatch.setenv("VACE_STITCH_SAMPLE_STEPS", "30")

    config = VaceStitchConfig.from_env()

    assert config.default_prompt == "continue the same dance naturally"
    assert config.model_name == "vace-1.3B"
    assert config.sample_steps == 30


def test_vace_config_exposes_scaled_14b_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VACE_STITCH_ENABLED", "true")
    monkeypatch.setenv("VACE_STITCH_MODEL_NAME", "vace-14B")
    monkeypatch.setenv("VACE_STITCH_CHECKPOINT_FILE", "/models/wan-vace-14b/wan2.1_vace_14B_fp8_scaled.safetensors")
    monkeypatch.setenv("VACE_STITCH_LOOP_ENABLED", "true")

    config = VaceStitchConfig.from_env()

    assert config.enabled is True
    assert config.model_name == "vace-14B"
    assert config.checkpoint_file is not None
    assert config.checkpoint_file.name == "wan2.1_vace_14B_fp8_scaled.safetensors"
    assert config.loop_enabled is True


def test_vace_lightx2v_reuses_existing_animate_text_assets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENERATIVE_DANCE_WAN_T5_CHECKPOINT", "/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth")
    monkeypatch.setenv("GENERATIVE_DANCE_WAN_T5_TOKENIZER", "/Wan-AI/umt5-xxl")

    config = VaceStitchConfig.from_env()

    assert config.lightx2v_t5_checkpoint == Path("/Wan-AI/models_t5_umt5-xxl-enc-bf16.pth")
    assert config.lightx2v_t5_tokenizer == Path("/Wan-AI/umt5-xxl")


def test_scaled_fp8_linear_has_a_cpu_fallback() -> None:
    layer = ScaledFP8Linear(4, 3, bias=True)
    layer.weight = torch.nn.Parameter(
        torch.ones((3, 4), dtype=torch.float8_e4m3fn),
        requires_grad=False,
    )
    layer.bias = torch.nn.Parameter(
        torch.zeros(3, dtype=torch.bfloat16),
        requires_grad=False,
    )
    layer.scale_weight = torch.tensor(2.0)

    output = layer(torch.ones((1, 2, 4), dtype=torch.bfloat16))

    assert tuple(output.shape) == (1, 2, 3)
    assert output[0, 0, 0].item() == pytest.approx(8.0, abs=0.1)


def test_vace_config_exposes_gpu_runtime_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VACE_STITCH_ATTENTION_BACKEND", "flash_attention_2")
    monkeypatch.setenv("VACE_STITCH_OFFLOAD_MODEL", "false")
    monkeypatch.setenv("VACE_STITCH_TF32", "true")

    config = VaceStitchConfig.from_env()

    assert config.attention_backend == "flash_attention_2"
    assert config.offload_model is False
    assert config.tf32 is True


def test_realesrgan_scale_is_rendered_as_an_integer_cli_argument(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autotransition.vace_stitch.enhancement as enhancement_module

    config = VaceStitchConfig(
        enhancement_enabled=True,
        enhancement_command="realesrgan_stage.py --scale {scale}",
        enhancement_scale=2.0,
    )
    stage = VaceVideoStage(config, stage="enhancement")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    captured: dict[str, object] = {}

    def fake_run(template: object, *, values: dict[str, object], **_: object) -> None:
        captured.update(values)
        output = values["output"]
        assert isinstance(output, Path)
        output.write_bytes(b"enhanced")

    monkeypatch.setattr(enhancement_module, "run_adapter_command", fake_run)
    monkeypatch.setattr(
        enhancement_module,
        "probe_video",
        lambda path: VideoProbe(path, 128, 128, 24, 1.0, 24, "yuv420p"),
    )
    stage.process(input_video=source, output_dir=tmp_path / "stage", width=64, height=64, fps=24)

    assert stage.command == ("realesrgan_stage.py", "--scale", "{scale}")
    assert captured["scale"] == 2


def test_vace_seed_is_random_when_omitted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("autotransition.vace_stitch.worker.secrets.randbits", lambda bits: 123456789)

    assert VaceStitchWorker._resolve_seed({}) == (123456789, False)
    assert VaceStitchWorker._resolve_seed({"seed": 0}) == (0, True)


def test_native_vace_command_writes_upstream_results_to_job_directory(tmp_path: Path) -> None:
    source_root = tmp_path / "vace"
    script = source_root / "vace" / "vace_wan_inference.py"
    script.parent.mkdir(parents=True)
    script.write_text("# test placeholder\n", encoding="utf-8")

    runtime = VaceRuntime(
        VaceStitchConfig(
            runtime_backend="native",
            source_root=source_root,
            checkpoint_dir=tmp_path / "checkpoint",
        )
    )

    command = runtime.native_command
    save_dir_index = command.index("--save_dir")
    assert command[save_dir_index + 1] == "{output_dir}"


def test_vace_gap_extraction_keeps_model_frame_ownership_before_resampling() -> None:
    source = Path("src/autotransition/vace_stitch/video.py").read_text(encoding="utf-8")
    function_start = source.index("def extract_generated_gap(")
    function_source = source[function_start:]

    assert "model_fps: int" in function_source
    assert "frame_count_value=prepared.gap_frames" in function_source
    assert "fps=model_fps" in function_source


def test_vace_gap_extraction_converts_model_frames_once_at_output_boundary(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    solid_video(source, width=64, height=64, fps=24, frames=48, color="blue")
    prepared = prepare_firstlastclip(
        source,
        source,
        tmp_path / "prepared",
        width=64,
        height=64,
        model_fps=16,
        gap_frames=16,
        context_before_seconds=1.0,
        context_after_seconds=1.0,
        max_window_frames=81,
        background="0x7f7f7f",
    )
    model_output = tmp_path / "model-output.mp4"
    solid_video(
        model_output,
        width=64,
        height=64,
        fps=16,
        frames=prepared.total_frames,
        color="red",
    )

    output = tmp_path / "gap-output.mp4"
    probe = extract_generated_gap(
        model_output,
        output,
        prepared=prepared,
        output_width=64,
        output_height=64,
        output_fps=24,
        model_fps=16,
        background="0x7f7f7f",
    )

    assert prepared.tail_frames == prepared.head_frames
    assert prepared.tail_frames >= 14
    assert prepared.total_frames == 49
    assert prepared.tail_frames + prepared.gap_frames + prepared.head_frames == prepared.total_frames
    assert probe.fps == pytest.approx(24, abs=0.1)
    assert probe.duration_seconds == pytest.approx(prepared.gap_frames / 16, abs=0.1)
    assert frame_count(probe, 24) > prepared.gap_frames


def test_vace_seam_diagnostics_use_exact_part_boundaries() -> None:
    assert part_boundaries([3, 5, 2]) == [
        {"partIndex": 1, "leftFrame": 2, "rightFrame": 3},
        {"partIndex": 2, "leftFrame": 7, "rightFrame": 8},
    ]
    assert compare_frame_bytes(b"abc", b"abc")["changedByteRatio"] == 0.0
    assert compare_frame_bytes(b"abc", b"abd")["changedByteRatio"] > 0


def test_vace_stage_duration_allows_frame_cadence_timestamp_rounding() -> None:
    source = Path("src/autotransition/vace_stitch/worker.py").read_text(encoding="utf-8")
    function_start = source.index("expected_duration = expected_frame_count / float(output_fps)")
    function_source = source[function_start : function_start + 900]

    assert "duration_tolerance = 2.0 / max(quality_fps, 1.0)" in function_source
    # 742 frames at 24 fps become 1483 frames at 48 fps when the first frame
    # is retained and one interpolated frame is inserted between each pair.
    expected_duration = 742 / 24
    actual_duration = 1483 / 48
    assert abs(actual_duration - expected_duration) <= 2 / 48


def test_transform_alpha_video_matches_quality_timeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autotransition.generative_dance.video as video_module

    source = tmp_path / "source-alpha.mov"
    source.write_bytes(b"alpha")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], *, purpose: str) -> None:
        captured["command"] = command
        captured["purpose"] = purpose
        output = Path(command[-1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"transformed")

    monkeypatch.setattr(video_module, "_run_ffmpeg", fake_run)
    monkeypatch.setattr(
        video_module,
        "probe_video",
        lambda path: VideoProbe(path, 1664, 960, 48, 30.895833, 1483, "yuva444p10le", True),
    )

    result = transform_alpha_video(
        source,
        tmp_path / "quality-alpha" / "alpha.mov",
        width=1664,
        height=960,
        fps=48,
        frame_count=1483,
        crf=24,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert "fps=48" in command[command.index("-vf") + 1]
    assert command[command.index("-frames:v") + 1] == "1483"
    assert result.has_alpha is True


def test_vace_worker_processes_full_sequence_as_one_parent_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import autotransition.vace_stitch.worker as worker_module

    config = VaceStitchConfig(
        artifact_root=tmp_path,
        runtime_backend="command",
        runtime_command="fake-vace",
        output_width=64,
        output_height=64,
        output_fps=24,
        model_fps=16,
        max_window_frames=81,
        transparent_default=False,
    )

    class FakeRuntime:
        configured = True

        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def generate(self, **kwargs: object) -> Path:
            self.calls.append(kwargs)
            output_dir = kwargs["output_dir"]
            assert isinstance(output_dir, Path)
            output_dir.mkdir(parents=True, exist_ok=True)
            output = output_dir / "vace-output.mp4"
            output.write_bytes(b"vace-output")
            return output

    runtime = FakeRuntime()
    worker = VaceStitchWorker(config, runtime=runtime)  # type: ignore[arg-type]
    prepare_calls: list[dict[str, object]] = []

    source_probe = VideoProbe(Path("source.mp4"), 64, 64, 24, 1.0, 24, "yuv420p")

    def fake_download(job_id: str, input_id: str, url: str, name: str) -> Path:
        path = worker.job_dir(job_id) / "inputs" / f"{input_id}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"source")
        return path

    def fake_extract(source: Path, output: Path, **_: object) -> VideoProbe:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"range")
        return source_probe

    def fake_normalize(source: Path, output: Path, **_: object) -> VideoProbe:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"normalized")
        return source_probe

    def fake_prepare(_: Path, __: Path, output_dir: Path, *, gap_frames: int, **kwargs: object) -> PreparedVaceInput:
        prepare_calls.append({"gap_frames": gap_frames, **kwargs})
        output_dir.mkdir(parents=True, exist_ok=True)
        source = output_dir / "firstlastclip-source.mp4"
        mask = output_dir / "firstlastclip-mask.mp4"
        source.write_bytes(b"source")
        mask.write_bytes(b"mask")
        return PreparedVaceInput(source, mask, source, source, 8, gap_frames, 8, 8 + gap_frames + 8 + ((1 - (16 + gap_frames)) % 4))

    def fake_extract_gap(model_output: Path, output: Path, **_: object) -> VideoProbe:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"gap")
        return source_probe

    def fake_stitch(inputs: list[Path], output: Path, **_: object) -> VideoProbe:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"stitched")
        return VideoProbe(Path(output), 64, 64, 24, len(inputs), len(inputs) * 24, "yuv420p")

    monkeypatch.setattr(worker, "_download", fake_download)
    monkeypatch.setattr(worker_module, "probe_video", lambda _path: source_probe)
    monkeypatch.setattr(worker_module, "extract_time_range", fake_extract)
    monkeypatch.setattr(worker_module, "normalize_canvas", fake_normalize)
    monkeypatch.setattr(worker_module, "prepare_firstlastclip", fake_prepare)
    monkeypatch.setattr(worker_module, "extract_generated_gap", fake_extract_gap)
    monkeypatch.setattr(worker_module, "stitch_videos", fake_stitch)
    monkeypatch.setattr(worker_module, "analyze_video_seams", lambda *_: [])

    payload = {
        "job_id": "parent-job",
        "inputs": [
            {"id": "clip-a", "role": "sequence_clip", "sourceUrl": "https://example.test/a.mp4", "fileName": "a.mp4"},
            {"id": "clip-b", "role": "sequence_clip", "sourceUrl": "https://example.test/b.mp4", "fileName": "b.mp4"},
        ],
        "parameters": {
            "transparent": False,
            "seed": 123,
            "prompt": "the character continues dancing",
            "modelSize": "480p",
            "sequence": {
                "fps": 24,
                "width": 64,
                "height": 64,
                "segments": [
                    {"id": "a", "inputId": "clip-a", "sourceStartSeconds": 0, "sourceEndSeconds": 1},
                    {"id": "b", "inputId": "clip-b", "sourceStartSeconds": 0, "sourceEndSeconds": 1},
                ],
                "bridges": [{"durationSeconds": 1.0, "prompt": "bridge the dance naturally"}],
                "loop": {"enabled": True, "durationSeconds": 1.0},
            },
        },
    }
    worker._write(
        "parent-job",
        {"id": "parent-job", "status": "queued", "stage": "queued", "progress": 0.0, "attempt": 0},
    )
    worker._run("parent-job", payload)

    result = worker.get("parent-job")
    assert result["status"] == "succeeded"
    assert result["result"]["singleParentJob"] is True
    assert len(result["result"]["bridges"]) == 2
    assert runtime.calls[0]["prompt"] == "bridge the dance naturally"
    assert runtime.calls[1]["prompt"] == config.default_loop_prompt
    assert [call["width"] for call in prepare_calls] == [832, 832]
    assert [call["height"] for call in prepare_calls] == [480, 480]
    assert [call["model_fps"] for call in prepare_calls] == [16, 16]
    assert [call["gap_frames"] for call in prepare_calls] == [16, 16]
    assert result["result"]["seed"] == 123
    assert result["result"]["seedProvided"] is True
    assert [call["seed"] for call in runtime.calls] == [123, 124]
    assert (tmp_path / "jobs" / "parent-job" / "final" / "dance-stitch.mp4").is_file()
