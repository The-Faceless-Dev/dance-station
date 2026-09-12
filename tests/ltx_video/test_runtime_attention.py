from __future__ import annotations

from types import SimpleNamespace

from autotransition.ltx_video.runtime import _ScopedPromptEncoder, _configure_fast_attention, _gemma_attention_context


class _FakeCuda:
    def __init__(self) -> None:
        self.flash = False
        self.memory_efficient = False
        self.math = False
        self.cudnn = False

    def is_available(self) -> bool:
        return True

    def enable_flash_sdp(self, value: bool) -> None:
        self.flash = value

    def enable_mem_efficient_sdp(self, value: bool) -> None:
        self.memory_efficient = value

    def enable_math_sdp(self, value: bool) -> None:
        self.math = value

    def enable_cudnn_sdp(self, value: bool) -> None:
        self.cudnn = value

    def flash_sdp_enabled(self) -> bool:
        return self.flash

    def mem_efficient_sdp_enabled(self) -> bool:
        return self.memory_efficient

    def math_sdp_enabled(self) -> bool:
        return self.math

    def cudnn_sdp_enabled(self) -> bool:
        return self.cudnn


class _FakeKernel:
    def __init__(self, state: dict[str, object], backend: object) -> None:
        self.state = state
        self.backend = backend

    def __enter__(self):  # type: ignore[no-untyped-def]
        self.state["active"] = self.backend
        return self

    def __exit__(self, *_args):  # type: ignore[no-untyped-def]
        self.state["active"] = None


def _fake_torch() -> tuple[SimpleNamespace, dict[str, object]]:
    cuda = _FakeCuda()
    state: dict[str, object] = {"active": None}
    math_backend = object()

    class _Attention:
        SDPBackend = SimpleNamespace(MATH=math_backend)

        @staticmethod
        def sdpa_kernel(backends):  # type: ignore[no-untyped-def]
            assert backends == [math_backend]
            return _FakeKernel(state, math_backend)

    torch = SimpleNamespace(
        cuda=cuda,
        nn=SimpleNamespace(attention=_Attention),
        backends=SimpleNamespace(cuda=cuda),
    )
    return torch, state


def test_video_policy_is_flash_only_when_enforced() -> None:
    torch, _state = _fake_torch()

    result = _configure_fast_attention(torch, enforce=True)

    assert result == {
        "cuda": True,
        "enforced": True,
        "flashSdp": True,
        "memoryEfficientSdp": False,
        "mathSdp": False,
        "cudnnSdp": False,
    }


def test_gemma_math_policy_is_scoped_and_exits_cleanly() -> None:
    torch, state = _fake_torch()

    with _gemma_attention_context(torch) as policy:
        assert policy["backend"] == "sdpa_math"
        assert policy["isolated"] is True
        assert state["active"] is not None

    assert state["active"] is None


def test_prompt_encoder_restores_video_policy_after_encoding() -> None:
    torch, state = _fake_torch()
    _configure_fast_attention(torch, enforce=True)

    class _Encoder:
        def __call__(self, prompts):  # type: ignore[no-untyped-def]
            assert prompts == ["test"]
            assert state["active"] is not None
            return ("encoded",)

    encoder = _ScopedPromptEncoder(_Encoder(), torch, enforce_fast_attention=True, progress=lambda *_args: None)

    assert encoder(["test"]) == ("encoded",)
    assert state["active"] is None
    assert torch.cuda.flash is True
    assert torch.cuda.math is False
