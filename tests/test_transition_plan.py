from pathlib import Path

from autotransition.config import OutputConfig, TransitionConfig
from autotransition.pipeline import TransitionRequest, create_scaffold_plan


def test_create_scaffold_plan_is_ui_serializable(tmp_path: Path) -> None:
    config = TransitionConfig(
        context_seconds=16.0,
        repaint_overlap_seconds=4.0,
        new_section_seconds=40.0,
        output=OutputConfig(scaffold_dir=tmp_path / "scaffolds"),
        bpm_hint=128.0,
        key_hint="A minor",
        seed=123,
    )
    request = TransitionRequest(
        source_path=Path("clip.wav"),
        caption="continue smoothly",
        config=config,
        transition_id="demo",
    )

    plan = create_scaffold_plan(request)
    data = plan.to_dict()

    assert plan.scaffold_path == tmp_path / "scaffolds" / "demo" / "scaffold.wav"
    assert plan.metadata_path == tmp_path / "scaffolds" / "demo" / "metadata.json"
    assert data["source_path"] == "clip.wav"
    assert data["source_extension"] == ".wav"
    assert data["source_format"] == "WAV"
    assert data["repainting_start_seconds"] == 12.0
    assert data["repainting_end_seconds"] == -1.0
    assert data["bpm_hint"] == 128.0
    assert data["key_hint"] == "A minor"
    assert data["seed"] == 123
