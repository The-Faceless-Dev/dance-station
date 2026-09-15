from pathlib import Path


ENTRYPOINT = Path("containers/qwen-image-edit-worker/entrypoint.sh")
DOCKERFILE = Path("containers/qwen-image-edit-worker/Dockerfile")
README = Path("containers/qwen-image-edit-worker/README.md")


def test_entrypoint_has_provider_visible_startup_diagnostics() -> None:
    source = ENTRYPOINT.read_text(encoding="utf-8")

    for marker in (
        'event "entrypoint_invoked"',
        'event "python_import_probe_started"',
        'event "http_worker_spawned"',
        'event "health_ready"',
        'event "health_probe_timeout"',
        "collect_diagnostics",
        "nvidia-smi -L",
        "torch_cuda_available=",
        '"torchvision"',
        "QWEN_IMAGE_EDIT_STARTUP_LOG_ROOT",
        "QWEN_IMAGE_EDIT_STARTUP_PROBE_SECONDS",
        "tee -a",
    ):
        assert marker in source


def test_startup_diagnostics_are_writable_in_the_production_image() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "QWEN_IMAGE_EDIT_STARTUP_LOG_ROOT=/var/lib/autotransition/qwen-image-edit-startup" in dockerfile
    assert "/var/lib/autotransition/qwen-image-edit-startup" in dockerfile
    assert "chown -R qwen:qwen /var/lib/autotransition" in dockerfile
    assert '"torchvision"' in dockerfile


def test_diagnostics_document_provider_boundary_and_secret_exclusion() -> None:
    readme = README.read_text(encoding="utf-8")
    assert "entrypoint_invoked" in readme
    assert "RunPod fails before invoking" in readme
    assert "callback tokens" in readme
