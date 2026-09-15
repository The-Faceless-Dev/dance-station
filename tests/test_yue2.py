from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

from autotransition.yue2.config import Yue2Config
from autotransition.yue2.contracts import Yue2Request
from autotransition.yue2.runtime import Yue2Runtime


SIDECARS = (
    "sidecars/yue2-model-config.json",
    "sidecars/yue2-generation-config.json",
    "sidecars/yue2-qwen.tiktoken",
    "sidecars/yue2-vae-config.json",
)


def _package(tmp_path: Path) -> Path:
    root = tmp_path / "model"
    (root / "sidecars").mkdir(parents=True)
    (root / "yue2-3b-q8_0.gguf").write_bytes(b"q8")
    (root / "yue2-vae-f16.gguf").write_bytes(b"f16")
    for name in SIDECARS:
        path = root / name
        path.write_text("{}" if path.suffix == ".json" else "tokenizer", encoding="utf-8")
    return root


def _fake_cli(tmp_path: Path) -> Path:
    script = tmp_path / "fake_audiocpp.py"
    script.write_text(
        "import sys, wave\n"
        "args = sys.argv[1:]\n"
        "out = args[args.index('--out') + 1]\n"
        "print('YuE2 inference step 1/1', flush=True)\n"
        "with wave.open(out, 'wb') as f:\n"
        "    f.setnchannels(2); f.setsampwidth(2); f.setframerate(48000); f.writeframes(b'\\x00\\x00' * 4800)\n",
        encoding="utf-8",
    )
    return script


def test_preflight_requires_the_q8_f16_package(tmp_path: Path) -> None:
    root = _package(tmp_path)
    config = Yue2Config(model_root=root, cli_path=sys.executable, gpu_required=False)
    report = config.preflight()
    assert report["ready"] is True
    assert report["profile"] == {"main": "yue2-3b-q8_0.gguf", "vae": "yue2-vae-f16.gguf"}
    assert all(report["sidecars"].values())


def test_runtime_constructs_native_q8_f16_command_and_validates_output(tmp_path: Path) -> None:
    root = _package(tmp_path)
    fake = _fake_cli(tmp_path)
    config = Yue2Config(model_root=root, cli_path=sys.executable, gpu_required=False)
    runtime = Yue2Runtime(config)
    request = Yue2Request(
        lyrics="[Verse]\nA test song.",
        style="English, bright synth pop",
        cot="off",
        num_inference_steps=8,
        seed=123,
        duration_seconds=5,
        guidance_scale=1,
    )
    output = tmp_path / "audio.wav"
    command, resolved = runtime.build_command(request, output)
    assert command[command.index("--family") + 1] == "yue2"
    assert "yue2.model_gguf=yue2-3b-q8_0.gguf" in command
    assert "yue2.vae_gguf=yue2-vae-f16.gguf" in command
    assert resolved["seed"] == 123
    assert resolved["num_inference_steps"] == 8
    assert "--lyrics" in command
    assert "semantic_max_tokens=125" in command

    # Exercise generate with a fake executable while retaining the same output
    # validation used for the real native CLI.
    class FakeRuntime(Yue2Runtime):
        def build_command(self, request: Yue2Request, output_path: Path):
            return [sys.executable, str(fake), "--out", str(output_path)], self.resolve_request(request)

    result = FakeRuntime(config).generate(request, tmp_path / "attempt", lambda *_: None)
    assert result.output_path.is_file()
    with wave.open(str(result.output_path), "rb") as handle:
        assert handle.getframerate() == 48000
        assert handle.getnchannels() == 2
    assert result.metadata["wave"]["durationSeconds"] > 0
    assert json.loads((tmp_path / "attempt" / "runtime-metadata.json").read_text(encoding="utf-8"))["model"] == "yue2-3b-q8_0.gguf"
