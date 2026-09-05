from autotransition.generative_dance.salad_adapter import (
    MAX_COMPLETION_ARTIFACT_IDS,
    _completion_artifact_ids,
)


def test_completion_artifacts_are_capped_but_primary_outputs_are_kept() -> None:
    uploaded = [
        ({"name": f"diagnostic-{index}.json", "primary": False}, f"id-{index}")
        for index in range(35)
    ]
    uploaded.extend([
        ({"name": "generative-dance-output.mp4", "primary": True}, "id-final"),
        ({"name": "enhancement.mp4", "primary": True}, "id-enhancement"),
        ({"name": "motion-interpolation.mp4", "primary": True}, "id-interpolation"),
    ])

    result = _completion_artifact_ids(uploaded)

    assert len(result) == MAX_COMPLETION_ARTIFACT_IDS
    assert {"id-final", "id-enhancement", "id-interpolation"}.issubset(result)

