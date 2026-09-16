from pathlib import Path


RUNTIME = Path("src/autotransition/qwen_image_edit/runtime.py")


def test_preflight_checks_edit_plus_dependencies() -> None:
    source = RUNTIME.read_text(encoding="utf-8")

    assert "import torchvision" in source
    assert "from diffusers import QwenImageEditPlusPipeline" in source
    assert "from transformers import Qwen2VLProcessor" in source
    assert '"torchvision": torchvision.__version__' in source
