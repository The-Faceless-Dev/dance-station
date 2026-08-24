from __future__ import annotations

from pathlib import Path

import pytest

from autotransition.generative_dance import compose as compose_module
from autotransition.generative_dance.artifacts import ArtifactStore
from autotransition.generative_dance.config import GenerativeDanceConfig
from autotransition.generative_dance.contracts import (
    CanvasContract,
    PlacementTransform,
    RenderedSegment,
)
from autotransition.generative_dance.matting import BiRefNetMattingAdapter
from autotransition.generative_dance.video import VideoProbe


def test_placement_transform_is_normalized_and_serializable() -> None:
    transform = PlacementTransform(translate_x=0.1, translate_y=-0.05, scale=1.2, rotation_degrees=4)

    assert transform.to_dict() == {
        "translate_x": 0.1,
        "translate_y": -0.05,
        "scale": 1.2,
        "rotation_degrees": 4,
    }

    with pytest.raises(ValueError, match="scale"):
        PlacementTransform(scale=0).validate()


def test_canvas_contract_rejects_invalid_floor_and_semantic() -> None:
    with pytest.raises(ValueError, match="floor"):
        CanvasContract(floor_y=1.2).validate()
    with pytest.raises(ValueError, match="semantic"):
        CanvasContract(anchor_semantic="head").validate()


def test_birefnet_command_processes_every_input_frame(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "render.mp4"
    source.write_bytes(b"rgb")
    output_probe = VideoProbe(tmp_path / "birefnet-alpha.webm", 640, 800, 24, 1, 24, "yuva420p", True)
    input_probe = VideoProbe(source, 640, 800, 24, 1, 24, "yuv420p", False)
    seen: dict[str, object] = {}

    def fake_probe(path: Path) -> VideoProbe:
        return input_probe if path == source else output_probe

    def fake_run(command: tuple[str, ...], **kwargs: object) -> None:
        values = kwargs["values"]
        assert isinstance(values, dict)
        seen.update(values)
        output = Path(str(values["output_video"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"alpha")

    monkeypatch.setattr("autotransition.generative_dance.matting.probe_video", fake_probe)
    monkeypatch.setattr("autotransition.generative_dance.matting.run_adapter_command", fake_run)
    config = GenerativeDanceConfig(artifact_root=tmp_path, matte_command=("birefnet", "{input_video}", "{output_video}"))

    result = BiRefNetMattingAdapter(config).process(input_video=source, output_dir=tmp_path / "matte")

    assert result.probe.has_alpha is True
    assert seen["matte_batch_size"] == config.matte_batch_size
    assert json_contains(result.metadata_path, '"framesProcessed": 24')


def test_native_birefnet_command_omits_empty_checkpoint(tmp_path: Path) -> None:
    config = GenerativeDanceConfig(
        artifact_root=tmp_path,
        matte_backend="native",
        matte_model="local-birefnet",
        matte_python="python",
    )

    command = BiRefNetMattingAdapter(config).native_command

    assert "--checkpoint" not in command


def test_birefnet_rejects_output_without_alpha(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source = tmp_path / "render.mp4"
    source.write_bytes(b"rgb")
    probes = iter(
        [
            VideoProbe(source, 640, 800, 24, 1, 24, "yuv420p", False),
            VideoProbe(tmp_path / "matte.webm", 640, 800, 24, 1, 24, "yuv420p", False),
        ]
    )
    monkeypatch.setattr("autotransition.generative_dance.matting.probe_video", lambda _: next(probes))

    def fake_run(command: tuple[str, ...], **kwargs: object) -> None:
        values = kwargs["values"]
        assert isinstance(values, dict)
        output = Path(str(values["output_video"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"opaque")

    monkeypatch.setattr("autotransition.generative_dance.matting.run_adapter_command", fake_run)
    config = GenerativeDanceConfig(artifact_root=tmp_path, matte_command=("birefnet",))

    with pytest.raises(Exception, match="not alpha-capable"):
        BiRefNetMattingAdapter(config).process(input_video=source, output_dir=tmp_path / "matte")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("fps", "frame rate"),
        ("duration", "duration"),
    ],
)
def test_birefnet_rejects_timing_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    source = tmp_path / "render.mp4"
    source.write_bytes(b"rgb")
    input_probe = VideoProbe(source, 640, 800, 24, 1, None, "yuv420p", False)
    output_probe = VideoProbe(
        tmp_path / "matte.webm",
        640,
        800,
        25 if field == "fps" else 24,
        1.5 if field == "duration" else 1,
        None,
        "yuva420p",
        True,
    )
    probes = iter([input_probe, output_probe])
    monkeypatch.setattr("autotransition.generative_dance.matting.probe_video", lambda _: next(probes))

    def fake_run(command: tuple[str, ...], **kwargs: object) -> None:
        values = kwargs["values"]
        assert isinstance(values, dict)
        output = Path(str(values["output_video"]))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"alpha")

    monkeypatch.setattr("autotransition.generative_dance.matting.run_adapter_command", fake_run)
    config = GenerativeDanceConfig(artifact_root=tmp_path, matte_command=("birefnet",))

    with pytest.raises(Exception, match=message):
        BiRefNetMattingAdapter(config).process(input_video=source, output_dir=tmp_path / "matte")


def test_composition_stitches_transparent_segments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    probe = VideoProbe(tmp_path / "video", 640, 800, 24, 1, 24, "yuv420p", False)
    canvas = CanvasContract()
    segments = [
        RenderedSegment(
            id="one",
            driver_id="driver-one",
            reference_id="reference",
            output_video=tmp_path / "one.mp4",
            duration_seconds=1,
            canvas=canvas,
            metadata_path=tmp_path / "one.json",
            model_revision="test",
            prompt="test",
            transparent_video=tmp_path / "one.webm",
        ),
        RenderedSegment(
            id="two",
            driver_id="driver-two",
            reference_id="reference",
            output_video=tmp_path / "two.mp4",
            duration_seconds=1,
            canvas=canvas,
            metadata_path=tmp_path / "two.json",
            model_revision="test",
            prompt="test",
            transparent_video=tmp_path / "two.webm",
        ),
    ]

    def fake_rgb(inputs: list[Path], output: Path, **kwargs: object) -> VideoProbe:
        calls.append("rgb")
        return probe

    def fake_alpha(inputs: list[Path], output: Path, **kwargs: object) -> VideoProbe:
        calls.append("alpha")
        return probe

    def fake_preview(source: Path, output: Path, **kwargs: object) -> VideoProbe:
        calls.append("preview")
        return probe

    def fake_encode(source: Path, output: Path, **kwargs: object) -> VideoProbe:
        calls.append("encode")
        return probe

    monkeypatch.setattr(compose_module, "stitch_videos", fake_rgb)
    monkeypatch.setattr(compose_module, "stitch_transparent_videos", fake_alpha)
    monkeypatch.setattr(compose_module, "encode_transparent_video", fake_encode)
    monkeypatch.setattr(compose_module, "make_transparent_preview", fake_preview)
    result = compose_module.build_composition(
        composition_id="composition",
        rendered_segments=segments,
        config=GenerativeDanceConfig(artifact_root=tmp_path),
        store=ArtifactStore(tmp_path),
    )

    assert calls == ["rgb", "alpha", "encode", "preview"]
    assert result.transparent_video == tmp_path / "compositions" / "composition" / "composition-transparent.webm"
    assert result.transparent_preview_video == tmp_path / "compositions" / "composition" / "composition-transparent-preview.mp4"


def json_contains(path: Path, text: str) -> bool:
    return text in path.read_text(encoding="utf-8")
