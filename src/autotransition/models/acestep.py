"""ACE-Step repaint runtime adapter."""

from __future__ import annotations

from pathlib import Path

from autotransition.config import RuntimeConfig
from autotransition.models.base import RepaintResult
from autotransition.models.registry import ModelProfile
from autotransition.pipeline import SourceSelectionPlan


class AceStepRuntimeError(RuntimeError):
    """Raised when ACE-Step repaint generation cannot run."""


def ace_step_runtime_available() -> bool:
    from autotransition.runtime.ace_step import runtime_status

    return runtime_status().api_running


class AceStepRepaintAdapter:
    """Adapter boundary for ACE-Step repaint generation."""

    def __init__(self, profile: ModelProfile, model_path: Path, runtime_config: RuntimeConfig | None = None) -> None:
        self.profile = profile
        self.model_path = model_path
        self.runtime_config = runtime_config or RuntimeConfig()

    def repaint(self, plan: SourceSelectionPlan) -> RepaintResult:
        from autotransition.models.acestep_api import AceStepApiClient, AceStepApiError
        from autotransition.runtime.ace_step import runtime_status

        config = self.runtime_config
        status = runtime_status(config)
        if not status.api_running:
            raise AceStepRuntimeError(
                f"{status.message} Run `autotransition setup` once, then start the full app with `autotransition run`."
            )

        try:
            return AceStepApiClient(config).repaint(
                plan=plan,
                profile=self.profile,
                save_dir=Path("data/generated") / plan.transition_id,
            )
        except AceStepApiError as exc:
            raise AceStepRuntimeError(str(exc)) from exc

    def text2music(self, plan: SourceSelectionPlan) -> RepaintResult:
        from autotransition.models.acestep_api import AceStepApiClient, AceStepApiError
        from autotransition.runtime.ace_step import runtime_status

        config = self.runtime_config
        status = runtime_status(config)
        if not status.api_running:
            raise AceStepRuntimeError(
                f"{status.message} Run `autotransition setup` once, then start the full app with `autotransition run`."
            )

        try:
            return AceStepApiClient(config).text2music(
                plan=plan,
                profile=self.profile,
                save_dir=Path("data/generated") / plan.transition_id / "raw",
            )
        except AceStepApiError as exc:
            raise AceStepRuntimeError(str(exc)) from exc
