from pathlib import Path
from types import SimpleNamespace

from autotransition.audio import ffmpeg as ffmpeg_helpers
from autotransition.ui.state import system_status


def test_resolve_ffmpeg_prefers_system_path(monkeypatch) -> None:
    monkeypatch.setattr(ffmpeg_helpers.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    assert ffmpeg_helpers.resolve_ffmpeg() == "/usr/bin/ffmpeg"


def test_resolve_ffmpeg_uses_bundled_package(tmp_path: Path, monkeypatch) -> None:
    bundled = tmp_path / "ffmpeg"
    bundled.write_text("fake", encoding="utf-8")
    monkeypatch.setattr(ffmpeg_helpers.shutil, "which", lambda name: None)
    monkeypatch.setitem(
        __import__("sys").modules,
        "imageio_ffmpeg",
        SimpleNamespace(get_ffmpeg_exe=lambda: str(bundled)),
    )

    assert ffmpeg_helpers.resolve_ffmpeg() == str(bundled)


def test_system_status_uses_resolved_ffmpeg(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("autotransition.ui.state.resolve_ffmpeg", lambda: "bundled-ffmpeg")

    status = system_status(models_dir=tmp_path)

    assert status["ffmpeg_available"] is True
    assert status["ffmpeg_path"] == "bundled-ffmpeg"
