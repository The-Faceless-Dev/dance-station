import base64
from pathlib import Path

from PIL import Image

from autotransition.qwen_image.config import QwenImageConfig
from autotransition.qwen_image.contracts import QwenImageRequest, QwenLoRARequest
from autotransition.qwen_image.runtime import QwenImageRuntime


def test_runtime_builds_cuda_q8_command(tmp_path: Path) -> None:
    config = QwenImageConfig(
        runtime_binary=tmp_path / "sd-server",
        diffusion_model=tmp_path / "qwen-image-2512-Q8_0.gguf",
        text_encoder=tmp_path / "Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf",
        vae=tmp_path / "qwen_image_vae.safetensors",
    )
    command = QwenImageRuntime(config).command()
    assert "--backend" in command and command[command.index("--backend") + 1] == "cuda"
    assert "--diffusion-fa" in command
    assert "--offload-to-cpu" in command
    assert "--mmap" in command
    assert "--lora-apply-mode" in command


def test_runtime_writes_native_base64_png(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "source.png"
    Image.new("RGB", (16, 16), (20, 40, 60)).save(image_path, format="PNG")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    config = QwenImageConfig(
        artifact_root=tmp_path,
        runtime_binary=tmp_path / "sd-server",
        diffusion_model=tmp_path / "qwen-image-2512-Q8_0.gguf",
        text_encoder=tmp_path / "Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf",
        vae=tmp_path / "qwen_image_vae.safetensors",
        poll_interval_seconds=0.001,
    )
    runtime = QwenImageRuntime(config)
    responses = iter([
        {"id": "native-1", "status": "queued"},
        {"id": "native-1", "status": "completed", "result": {"images": [{"b64_json": encoded}]}},
    ])
    monkeypatch.setattr(runtime, "ensure_server", lambda: None)
    monkeypatch.setattr(runtime, "_json_request", lambda *args, **kwargs: next(responses))
    output = tmp_path / "output.png"
    result = runtime.generate(QwenImageRequest(prompt="test"), (), output, lambda *_args: None)
    assert output.read_bytes().startswith(b"\x89PNG")
    assert result["nativeJobId"] == "native-1"


def test_runtime_sends_ordered_structured_loras(tmp_path: Path) -> None:
    config = QwenImageConfig(
        runtime_binary=tmp_path / "sd-server",
        diffusion_model=tmp_path / "qwen-image-2512-Q8_0.gguf",
        text_encoder=tmp_path / "Qwen2.5-VL-7B-Instruct-abliterated.Q8_0.gguf",
        vae=tmp_path / "qwen_image_vae.safetensors",
    )
    first = QwenLoRARequest(path=tmp_path / "first.safetensors", scale=0.7)
    second = QwenLoRARequest(path=tmp_path / "second.safetensors", scale=1.1, is_high_noise=True)
    body = QwenImageRuntime(config)._request_body(QwenImageRequest(prompt="test"), (first, second))
    assert body["lora"] == [
        {"path": str(first.path), "multiplier": 0.7, "is_high_noise": False},
        {"path": str(second.path), "multiplier": 1.1, "is_high_noise": True},
    ]
