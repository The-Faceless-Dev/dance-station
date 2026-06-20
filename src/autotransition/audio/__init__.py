"""Audio helpers."""

from autotransition.audio.formats import DEFAULT_SCAFFOLD_FORMAT, SUPPORTED_INPUT_FORMATS, validate_supported_source
from autotransition.audio.probe import AudioProbe, probe_audio
from autotransition.audio.scaffold import build_repaint_scaffold
from autotransition.audio.compose import build_continuation_composite
from autotransition.audio.selection import build_selection_scaffold

__all__ = [
    "DEFAULT_SCAFFOLD_FORMAT",
    "SUPPORTED_INPUT_FORMATS",
    "AudioProbe",
    "build_continuation_composite",
    "build_repaint_scaffold",
    "build_selection_scaffold",
    "probe_audio",
    "validate_supported_source",
]
