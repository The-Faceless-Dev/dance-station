"""Project-owned Wan-Animate-2 runtime primitives.

The web application and composition code do not import this module. It is an
inference-side adapter that loads the official Wan-Animate-2 transformer with
quantized weights kept in the format selected by the worker profile. Q6 GGUF
layers are dequantized on demand; the official INT8 ConvRot checkpoint uses
the fused ``comfy-kitchen`` CUDA kernel without a CPU fallback.

The official Wan-Animate-2 source tree is supplied separately with
``--official-source`` (or ``WAN_ANIMATE_2_SOURCE``). Keeping that boundary
explicit makes the runtime replaceable without coupling the application to a
workflow host or a particular model tier.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


_TRITON_Q6_KERNEL: Any = None
LOGGER = logging.getLogger("wan-animate-2")


def _configure_cuda_linker(cache_root: Path) -> None:
    """Make Triton's ``-lcuda`` link work when only the driver soname exists."""

    if sys.platform != "linux":
        return

    library_dirs = (
        Path("/usr/local/cuda/lib64"),
        Path("/usr/lib/x86_64-linux-gnu"),
        Path("/lib/x86_64-linux-gnu"),
    )
    if any((directory / "libcuda.so").exists() for directory in library_dirs):
        return

    driver_library = next(
        (
            directory / "libcuda.so.1"
            for directory in library_dirs
            if (directory / "libcuda.so.1").exists()
        ),
        None,
    )
    if driver_library is None:
        LOGGER.warning("cuda_linker_alias_unavailable driver library libcuda.so.1 was not found")
        return

    alias_root = cache_root / "cuda-link"
    alias_root.mkdir(parents=True, exist_ok=True)
    alias = alias_root / "libcuda.so"
    try:
        if not alias.exists():
            alias.symlink_to(driver_library)
    except OSError as exc:
        raise WanAnimate2RuntimeError(
            f"could not create writable CUDA linker alias {alias}: {exc}"
        ) from exc

    existing_library_path = os.getenv("LIBRARY_PATH", "")
    library_path = str(alias_root)
    if existing_library_path:
        library_path = os.pathsep.join((library_path, existing_library_path))
    os.environ["LIBRARY_PATH"] = library_path
    LOGGER.info(
        "cuda_linker_ready driver=%s alias=%s library_path=%s",
        driver_library,
        alias,
        alias_root,
    )


def prepare_runtime_cache_dirs() -> dict[str, str]:
    """Select writable per-user compiler/model caches before CUDA imports."""

    requested_root = os.getenv("XDG_CACHE_HOME", "").strip()
    candidates = [Path(requested_root)] if requested_root else []
    if Path("/home/wan").is_dir():
        candidates.append(Path("/home/wan/.cache"))
    candidates.append(Path("/tmp/wan-animate-cache"))

    cache_root: Path | None = None
    for candidate in candidates:
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write-test"
            probe.write_text("ok", encoding="ascii")
            probe.unlink(missing_ok=True)
            cache_root = candidate
            break
        except OSError:
            continue
    if cache_root is None:
        attempted = ", ".join(str(candidate) for candidate in candidates)
        raise WanAnimate2RuntimeError(
            f"no writable runtime cache directory; tried: {attempted}"
        )

    paths = {
        "XDG_CACHE_HOME": cache_root,
        "TRITON_CACHE_DIR": cache_root / "triton",
        "TORCHINDUCTOR_CACHE_DIR": cache_root / "torchinductor",
        "HF_HOME": cache_root / "huggingface",
        "TRANSFORMERS_CACHE": cache_root / "huggingface" / "transformers",
        "TORCH_HOME": cache_root / "torch",
        "MPLCONFIGDIR": cache_root / "matplotlib",
    }
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    if Path("/home/wan").is_dir():
        os.environ["HOME"] = "/home/wan"
    _configure_cuda_linker(cache_root)
    LOGGER.info(
        "runtime_cache_ready root=%s triton=%s torchinductor=%s",
        cache_root,
        paths["TRITON_CACHE_DIR"],
        paths["TORCHINDUCTOR_CACHE_DIR"],
    )
    return {name: str(path) for name, path in paths.items()}


def _triton_q6_kernel() -> Any:
    """Create the Q6_K decoder only in runtimes that opt into CUDA dequantization."""

    global _TRITON_Q6_KERNEL
    if _TRITON_Q6_KERNEL is not None:
        return _TRITON_Q6_KERNEL

    import triton
    import triton.language as tl

    @triton.jit
    def decode_q6_k(raw, raw_half, output, ncols, blocks_per_row, BLOCK: tl.constexpr):
        block_id = tl.program_id(0)
        row = block_id // blocks_per_row
        block_in_row = block_id - row * blocks_per_row
        values = tl.arange(0, BLOCK)
        ip = values // 128
        local = values % 128
        group = local // 32
        il = local % 32

        ql_index = 64 * ip + il + tl.where((group & 1) != 0, 32, 0)
        qh_index = 128 + 32 * ip + il
        scale_index = 192 + 8 * ip + (il // 16) + group * 2
        raw_base = block_id * 210

        ql = tl.load(raw + raw_base + ql_index).to(tl.int32)
        qh = tl.load(raw + raw_base + qh_index).to(tl.int32)
        scales = tl.load(raw + raw_base + scale_index).to(tl.int32)
        scales = tl.where(scales >= 128, scales - 256, scales)
        shift = group * 2
        low_nibble = tl.where(group < 2, ql & 0xF, ql >> 4)
        quant = low_nibble + (((qh >> shift) & 3) << 4)
        scale = tl.load(raw_half + block_id * 105 + 104).to(tl.float32)
        decoded = scale * scales.to(tl.float32) * (quant.to(tl.float32) - 32.0)

        output_offset = row * ncols + block_in_row * 256 + values
        tl.store(output + output_offset, decoded)

    _TRITON_Q6_KERNEL = decode_q6_k
    return _TRITON_Q6_KERNEL


def _materialize_q6_cuda_raw(raw: Any, shape: tuple[int, ...], dtype: Any) -> Any:
    """Decode one Q6_K tensor from CUDA-resident bytes."""

    import torch

    if len(shape) != 2 or shape[1] % 256 != 0:
        raise WanAnimate2RuntimeError(
            f"Q6_K CUDA decoder expects a 2D matrix with a 256-wide inner dimension; got {shape}"
        )

    if raw.numel() % 210 != 0:
        raise WanAnimate2RuntimeError(
            f"Q6_K tensor byte length is not block aligned: {raw.numel()}"
        )
    if not raw.is_cuda:
        raw = raw.to(device="cuda", non_blocking=True)
    raw_half = raw.view(torch.float16)
    output = torch.empty(shape, device="cuda", dtype=dtype)
    blocks_per_row = shape[1] // 256
    kernel = _triton_q6_kernel()
    kernel[(raw.numel() // 210,)](
        raw,
        raw_half,
        output,
        shape[1],
        blocks_per_row,
        BLOCK=256,
        num_warps=4,
    )
    del raw, raw_half
    return output


def _materialize_q6_cuda(tensor: Any, shape: tuple[int, ...], dtype: Any) -> Any:
    """Decode one Q6_K tensor on CUDA without retaining its FP16 form."""

    import torch

    raw_cpu = torch.from_numpy(tensor.data).reshape(-1)
    try:
        return _materialize_q6_cuda_raw(raw_cpu, shape, dtype)
    finally:
        del raw_cpu


class WanAnimate2RuntimeError(RuntimeError):
    """Raised when the GGUF model cannot be loaded for the official runtime."""


@dataclass(frozen=True)
class WanAnimate2TransformerSpec:
    """Transformer settings from the official distilled inference config."""

    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_len: int = 512
    in_dim: int = 36
    dim: int = 5120
    ffn_dim: int = 13824
    freq_dim: int = 256
    text_dim: int = 4096
    out_dim: int = 16
    num_heads: int = 40
    num_layers: int = 40
    window_size: tuple[int, int] = (-1, -1)
    qk_norm: bool = True
    cross_attn_norm: bool = True
    eps: float = 1e-6
    use_img_emb: bool = True
    refer_offset_t: int = 1
    refer_offset_h: int = 0
    refer_offset_w: int = -1
    refer_stride: int = 1
    sparse_type: int = 0
    use_context_parallel: bool = False
    log_scale: float = -1.3

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "patch_size": self.patch_size,
            "text_len": self.text_len,
            "in_dim": self.in_dim,
            "dim": self.dim,
            "ffn_dim": self.ffn_dim,
            "freq_dim": self.freq_dim,
            "text_dim": self.text_dim,
            "out_dim": self.out_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "window_size": self.window_size,
            "qk_norm": self.qk_norm,
            "cross_attn_norm": self.cross_attn_norm,
            "eps": self.eps,
            "use_img_emb": self.use_img_emb,
            "refer_offset_t": self.refer_offset_t,
            "refer_offset_h": self.refer_offset_h,
            "refer_offset_w": self.refer_offset_w,
            "refer_stride": self.refer_stride,
            "sparse_type": self.sparse_type,
            "use_context_parallel": self.use_context_parallel,
            "log_scale": self.log_scale,
        }


def canonical_tensor_name(name: str) -> str:
    """Map raw Wan GGUF block names to the official module names."""

    if name.startswith("blocks."):
        parts = name.split(".")
        parts.insert(2, "block")
        return ".".join(parts)
    return name


class GGUFWeightStore:
    """Memory-mapped GGUF tensors with official-shape metadata."""

    def __init__(self, path: Path, *, model_type: str = "animate2") -> None:
        if model_type != "animate2":
            raise WanAnimate2RuntimeError(
                f"unsupported Wan GGUF model_type={model_type!r}; expected 'animate2'"
            )
        try:
            import gguf
        except ImportError as exc:  # pragma: no cover - runtime environment only
            raise WanAnimate2RuntimeError(
                "GGUF support is missing; install the Wan-Animate runtime dependencies"
            ) from exc

        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise WanAnimate2RuntimeError(f"GGUF checkpoint not found: {self.path}")
        self.gguf = gguf
        self.reader = gguf.GGUFReader(str(self.path), "r")
        self.metadata = self._read_simple_metadata()
        self._tensors: dict[str, Any] = {}
        self._raw_names: dict[str, str] = {}
        self._gpu_raw: dict[str, Any] = {}
        for tensor in self.reader.tensors:
            key = canonical_tensor_name(tensor.name)
            if key in self._tensors:
                raise WanAnimate2RuntimeError(f"duplicate GGUF tensor after mapping: {key}")
            self._tensors[key] = tensor
            self._raw_names[key] = tensor.name

    def _read_simple_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, field in self.reader.fields.items():
            try:
                if len(field.types) != 1:
                    continue
                value_type = field.types[0]
                if value_type == self.gguf.GGUFValueType.STRING:
                    result[name] = str(field.parts[field.data[-1]], "utf-8")
                elif value_type == self.gguf.GGUFValueType.INT32:
                    result[name] = int(field.parts[field.data[-1]])
                elif value_type == self.gguf.GGUFValueType.FLOAT32:
                    result[name] = float(field.parts[field.data[-1]])
                elif value_type == self.gguf.GGUFValueType.BOOL:
                    result[name] = bool(field.parts[field.data[-1]])
            except (IndexError, TypeError, ValueError):
                continue
        return result

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def __len__(self) -> int:
        return len(self._tensors)

    def raw_name(self, name: str) -> str:
        return self._raw_names[name]

    def tensor(self, name: str) -> Any:
        try:
            return self._tensors[name]
        except KeyError as exc:
            raise WanAnimate2RuntimeError(f"GGUF tensor is missing: {name}") from exc

    def shape(self, name: str) -> tuple[int, ...]:
        tensor = self.tensor(name)
        field = self.reader.get_field(f"comfy.gguf.orig_shape.{tensor.name}")
        if field is not None:
            return tuple(int(field.parts[index][0]) for index in field.data)
        return tuple(int(value) for value in reversed(tensor.shape))

    def tensor_type_name(self, name: str) -> str:
        tensor_type = self.tensor(name).tensor_type
        return getattr(tensor_type, "name", repr(tensor_type))

    def configure_gpu_raw_cache(self, device: str) -> None:
        """Cache quantized Q6 bytes on CUDA only when the device has room."""

        if not device.startswith("cuda"):
            return
        policy = os.getenv("WAN_GGUF_GPU_RAW_CACHE", "auto").strip().lower()
        if policy in {"0", "false", "no", "off"}:
            LOGGER.info("gguf_raw_cache=disabled policy=%s", policy)
            return

        import torch

        q6_type = self.gguf.GGMLQuantizationType.Q6_K
        q6_tensors = [tensor for tensor in self.reader.tensors if tensor.tensor_type == q6_type]
        total_bytes = sum(int(tensor.data.nbytes) for tensor in q6_tensors)
        free_bytes, total_device_bytes = torch.cuda.mem_get_info(device=device)
        if policy in {"auto", "1", "true", "yes", "on"}:
            reserve_mb = int(os.getenv("WAN_GGUF_GPU_RAW_RESERVE_MB", "2048"))
            if total_bytes > max(0, free_bytes - reserve_mb * 1024 * 1024):
                message = (
                    "Q6 raw GPU cache cannot fit the configured VRAM reserve: "
                    f"need={total_bytes / 1024**3:.2f}GiB free={free_bytes / 1024**3:.2f}GiB "
                    f"device={total_device_bytes / 1024**3:.2f}GiB reserve={reserve_mb}MiB"
                )
                LOGGER.error("gguf_raw_cache=disabled reason=%s policy=%s", message, policy)
                if policy in {"1", "true", "yes", "on"}:
                    raise WanAnimate2RuntimeError(message)
                warnings.warn(message, RuntimeWarning)
                return
            try:
                for tensor in q6_tensors:
                    key = canonical_tensor_name(tensor.name)
                    self._gpu_raw[key] = torch.from_numpy(tensor.data.reshape(-1)).to(
                        device=device, non_blocking=True
                    )
            except Exception:
                self._gpu_raw.clear()
                torch.cuda.empty_cache()
                LOGGER.exception(
                    "gguf_raw_cache=failed requested_bytes=%s free_bytes=%s policy=%s",
                    total_bytes,
                    free_bytes,
                    policy,
                )
                if policy in {"1", "true", "yes", "on"}:
                    raise
                warnings.warn(
                    "Q6 raw GPU cache allocation failed; using per-tensor CUDA dequantization",
                    RuntimeWarning,
                )
                return
            LOGGER.info(
                "gguf_raw_cache=enabled bytes=%.2fGiB tensors=%s free_before=%.2fGiB "
                "device_total=%.2fGiB reserve=%sMiB",
                total_bytes / 1024**3,
                len(q6_tensors),
                free_bytes / 1024**3,
                total_device_bytes / 1024**3,
                reserve_mb,
            )

    def materialize(self, name: str, *, device: str = "cpu", dtype: Any = None) -> Any:
        """Materialize one non-linear tensor for model construction."""

        import torch

        tensor = self.tensor(name)
        shape = self.shape(name)
        qtype = tensor.tensor_type
        requested_backend = os.getenv("WAN_GGUF_DEQUANT_BACKEND", "auto").strip().lower()
        use_triton = (
            device.startswith("cuda")
            and requested_backend in {"auto", "triton", "cuda"}
            and qtype == self.gguf.GGMLQuantizationType.Q6_K
        )
        if use_triton:
            try:
                cached_raw = self._gpu_raw.get(name)
                if cached_raw is not None:
                    return _materialize_q6_cuda_raw(cached_raw, shape, dtype or torch.float16)
                return _materialize_q6_cuda(tensor, shape, dtype or torch.float16)
            except Exception as exc:
                if requested_backend in {"triton", "cuda"}:
                    raise
                warnings.warn(
                    f"Q6_K CUDA dequantization unavailable for {name}; using CPU fallback: {exc}",
                    RuntimeWarning,
                )
        if qtype in {
            self.gguf.GGMLQuantizationType.F32,
            self.gguf.GGMLQuantizationType.F16,
        }:
            with __import__("warnings").catch_warnings():
                __import__("warnings").filterwarnings(
                    "ignore", message="The given NumPy array is not writable"
                )
                result = torch.from_numpy(tensor.data).reshape(shape).clone()
        else:
            result = torch.from_numpy(self.gguf.quants.dequantize(tensor.data, qtype)).reshape(shape)
        if dtype is not None:
            result = result.to(dtype=dtype)
        return result.to(device=device)

    def validate_against(self, expected_parameters: dict[str, Any]) -> dict[str, Any]:
        """Validate names and shapes before allocating model weights."""

        expected = set(expected_parameters)
        actual = set(self._tensors)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        mismatches = [
            {
                "name": name,
                "expected": tuple(int(value) for value in expected_parameters[name].shape),
                "actual": self.shape(name),
            }
            for name in sorted(expected & actual)
            if tuple(int(value) for value in expected_parameters[name].shape) != self.shape(name)
        ]
        if missing or extra or mismatches:
            raise WanAnimate2RuntimeError(
                "Wan-Animate-2 GGUF does not match the official transformer: "
                f"missing={len(missing)} extra={len(extra)} shape_mismatches={len(mismatches)}"
            )
        return {
            "tensorCount": len(self),
            "parameterCount": len(expected_parameters),
            "missing": missing,
            "extra": extra,
            "shapeMismatches": mismatches,
        }


def _raw_safetensors_name(model_name: str) -> str:
    """Map the instantiated model name back to the official checkpoint name."""

    parts = model_name.split(".")
    if len(parts) >= 4 and parts[0] == "blocks" and parts[1].isdigit() and parts[2] == "block":
        parts.pop(2)
    return ".".join(parts)


class SafeTensorWeightStore:
    """Read the official INT8 ConvRot checkpoint without an eager fp16 copy."""

    def __init__(self, path: Path) -> None:
        try:
            from safetensors import safe_open
        except ImportError as exc:  # pragma: no cover - runtime environment only
            raise WanAnimate2RuntimeError(
                "safetensors support is missing; install the Wan-Animate runtime dependencies"
            ) from exc

        self.path = path.expanduser().resolve()
        if not self.path.is_file():
            raise WanAnimate2RuntimeError(f"Wan-Animate-2 safetensors checkpoint not found: {self.path}")
        self.reader = safe_open(str(self.path), framework="pt", device="cpu")
        self.metadata = dict(self.reader.metadata() or {})
        self._keys = set(self.reader.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._keys

    def __len__(self) -> int:
        return sum(
            not name.endswith((".weight_scale", ".comfy_quant"))
            for name in self._keys
        )

    def keys(self) -> set[str]:
        return set(self._keys)

    def tensor(self, name: str) -> Any:
        try:
            return self.reader.get_tensor(name)
        except Exception as exc:
            raise WanAnimate2RuntimeError(
                f"Wan-Animate-2 safetensors tensor is missing or unreadable: {name}"
            ) from exc

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(int(value) for value in self.reader.get_slice(name).get_shape())

    def quantization(self, weight_name: str) -> dict[str, Any] | None:
        if not weight_name.endswith(".weight"):
            raise WanAnimate2RuntimeError(
                f"INT8 ConvRot weight name must end in .weight: {weight_name}"
            )
        marker_name = f"{weight_name[:-len('.weight')]}.comfy_quant"
        if marker_name not in self:
            return None
        raw = self.tensor(marker_name).detach().cpu().numpy().tobytes()
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise WanAnimate2RuntimeError(
                f"invalid ConvRot quantization metadata for {weight_name}"
            ) from exc
        if not isinstance(value, dict):
            raise WanAnimate2RuntimeError(
                f"ConvRot quantization metadata for {weight_name} is not an object"
            )
        return value

    def materialize(self, name: str, *, device: str, dtype: Any) -> Any:
        import torch

        value = self.tensor(name)
        if value.dtype.is_floating_point and dtype is not None:
            value = value.to(dtype=dtype)
        return value.to(device=device, non_blocking=True).contiguous()

    def validate_against(self, expected_parameters: dict[str, Any]) -> dict[str, Any]:
        """Validate the official state-dict names and shapes before loading."""

        expected_raw = {
            _raw_safetensors_name(name): parameter
            for name, parameter in expected_parameters.items()
        }
        actual = {
            name
            for name in self._keys
            if not name.endswith((".weight_scale", ".comfy_quant"))
        }
        missing = sorted(set(expected_raw) - actual)
        extra = sorted(actual - set(expected_raw))
        mismatches = [
            {
                "name": model_name,
                "checkpointName": raw_name,
                "expected": tuple(int(value) for value in parameter.shape),
                "actual": self.shape(raw_name),
            }
            for model_name, (raw_name, parameter) in sorted(
                ((name, (_raw_safetensors_name(name), parameter)) for name, parameter in expected_parameters.items()),
                key=lambda item: item[0],
            )
            if raw_name in actual
            and tuple(int(value) for value in parameter.shape) != self.shape(raw_name)
        ]
        if missing or extra or mismatches:
            raise WanAnimate2RuntimeError(
                "Wan-Animate-2 INT8 ConvRot does not match the official transformer: "
                f"missing={len(missing)} extra={len(extra)} shape_mismatches={len(mismatches)}"
            )
        quantized = sum(
            name.endswith(".weight")
            and f"{name[:-len('.weight')]}.comfy_quant" in self
            for name in actual
        )
        return {
            "tensorCount": len(self),
            "parameterCount": len(expected_parameters),
            "quantizedLinearCount": quantized,
            "modelFormat": "int8_convrot",
            "missing": missing,
            "extra": extra,
            "shapeMismatches": mismatches,
            "metadata": self.metadata,
        }


def _validate_convrot_cuda(device: str) -> dict[str, Any]:
    """Require the fused CUDA ConvRot path before allocating the model."""

    if not device.startswith("cuda"):
        raise WanAnimate2RuntimeError(
            "the INT8 ConvRot Wan-Animate profile requires CUDA; CPU execution is disabled"
        )
    try:
        import torch
        import comfy_kitchen
        from comfy_kitchen.backends import cuda as comfy_cuda
    except ImportError as exc:  # pragma: no cover - runtime environment only
        raise WanAnimate2RuntimeError(
            "the INT8 ConvRot profile requires the comfy-kitchen CUDA runtime"
        ) from exc
    if not torch.cuda.is_available():
        raise WanAnimate2RuntimeError("the INT8 ConvRot profile requires an available CUDA device")
    backends = comfy_kitchen.list_backends()
    cuda_status = backends.get("cuda") or {}
    if not cuda_status.get("available"):
        raise WanAnimate2RuntimeError(
            "comfy-kitchen CUDA backend is unavailable; refusing the eager/CPU fallback: "
            f"{cuda_status}"
        )
    try:
        comfy_kitchen.set_backend_priority(["cuda"])
        with comfy_kitchen.use_backend("cuda"):
            probe_input = torch.zeros((2, 256), device=device, dtype=torch.bfloat16)
            probe_weight = torch.zeros((1, 256), device=device, dtype=torch.int8)
            probe_scale = torch.ones((1, 1), device=device, dtype=torch.float32)
            # comfy-kitchen only registers the public dispatcher entry when
            # its optional cuBLASLt capability is present. The CUDA module's
            # implementation is still available for the ConvRot kernels and
            # must be called directly; never fall back to eager/CPU execution.
            probe_output = comfy_cuda.int8_linear(
                probe_input,
                probe_weight,
                probe_scale,
                out_dtype=torch.bfloat16,
                convrot=True,
                convrot_groupsize=256,
            )
            torch.cuda.synchronize(device)
            if tuple(probe_output.shape) != (2, 1):
                raise RuntimeError(f"unexpected ConvRot probe shape: {tuple(probe_output.shape)}")
            del probe_input, probe_weight, probe_scale, probe_output
    except Exception as exc:
        raise WanAnimate2RuntimeError(
            "comfy-kitchen CUDA ConvRot probe failed; refusing a slower fallback"
        ) from exc
    LOGGER.info(
        "stage=convrot_cuda_ready backend=cuda implementation=cuda.int8_linear "
        "probe=passed device=%s dispatcher=%s",
        device,
        cuda_status,
    )
    return {
        "backend": "cuda",
        "implementation": "cuda.int8_linear",
        "probe": "passed",
        "backends": backends,
    }


def _make_adapter_linear(weight: Any, bias: Any, *, device: str) -> Any:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional

    class AdapterLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_features = int(weight.shape[1])
            self.out_features = int(weight.shape[0])
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
            self._lightx2v_adapter_names: list[tuple[str, str, float]] = []
            self.register_buffer("_lightx2v_bias_delta", None, persistent=False)

        def add_lightx2v_adapter(self, down: Any, up: Any, scale: float) -> None:
            index = len(self._lightx2v_adapter_names)
            down_name = f"_lightx2v_down_{index}"
            up_name = f"_lightx2v_up_{index}"
            self.register_buffer(down_name, down, persistent=False)
            self.register_buffer(up_name, up, persistent=False)
            self._lightx2v_adapter_names.append((down_name, up_name, float(scale)))

        def add_lightx2v_bias(self, value: Any, scale: float = 1.0) -> None:
            value = value * float(scale)
            if self._lightx2v_bias_delta is None:
                self._lightx2v_bias_delta = value
            else:
                self._lightx2v_bias_delta.add_(value)

        def forward(self, input_tensor: Any) -> Any:
            input_value = input_tensor.to(dtype=self.weight.dtype)
            result = functional.linear(input_value, self.weight, self.bias)
            for down_name, up_name, scale in self._lightx2v_adapter_names:
                down = getattr(self, down_name)
                up = getattr(self, up_name)
                adapter_result = functional.linear(functional.linear(input_value, down), up)
                result = result + adapter_result.to(dtype=result.dtype) * scale
            if self._lightx2v_bias_delta is not None:
                result = result + self._lightx2v_bias_delta.to(
                    device=result.device, dtype=result.dtype
                )
            return result

    return AdapterLinear()


def _make_convrot_linear(
    qweight: Any,
    weight_scale: Any,
    bias: Any,
    *,
    device: str,
    convrot_groupsize: int,
) -> Any:
    import torch
    import torch.nn as nn

    class ConvRotLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.in_features = int(qweight.shape[1])
            self.out_features = int(qweight.shape[0])
            self.register_buffer("qweight", qweight.contiguous(), persistent=False)
            self.register_buffer("weight_scale", weight_scale.float().contiguous(), persistent=False)
            self.register_buffer(
                "bias",
                bias.contiguous() if bias is not None else None,
                persistent=False,
            )
            self.convrot_groupsize = int(convrot_groupsize)
            self._lightx2v_adapter_names: list[tuple[str, str, float]] = []
            self.register_buffer("_lightx2v_bias_delta", None, persistent=False)

        def add_lightx2v_adapter(self, down: Any, up: Any, scale: float) -> None:
            index = len(self._lightx2v_adapter_names)
            down_name = f"_lightx2v_down_{index}"
            up_name = f"_lightx2v_up_{index}"
            self.register_buffer(down_name, down, persistent=False)
            self.register_buffer(up_name, up, persistent=False)
            self._lightx2v_adapter_names.append((down_name, up_name, float(scale)))

        def add_lightx2v_bias(self, value: Any, scale: float = 1.0) -> None:
            value = value * float(scale)
            if self._lightx2v_bias_delta is None:
                self._lightx2v_bias_delta = value
            else:
                self._lightx2v_bias_delta.add_(value)

        def forward(self, input_tensor: Any) -> Any:
            from comfy_kitchen.backends import cuda as comfy_cuda

            compute_dtype = (
                torch.bfloat16
                if input_tensor.is_cuda and input_tensor.dtype == torch.float32
                else input_tensor.dtype
            )
            linear_input = input_tensor.to(dtype=compute_dtype)
            # Use the CUDA implementation directly. The public comfy-kitchen
            # dispatcher omits int8_linear when optional cuBLASLt is absent,
            # although this CUDA implementation remains usable for ConvRot.
            result = comfy_cuda.int8_linear(
                linear_input,
                self.qweight,
                self.weight_scale,
                self.bias,
                out_dtype=compute_dtype,
                convrot=True,
                convrot_groupsize=self.convrot_groupsize,
            )
            for down_name, up_name, scale in self._lightx2v_adapter_names:
                down = getattr(self, down_name)
                up = getattr(self, up_name)
                adapter_input = linear_input.to(dtype=down.dtype)
                adapter_result = torch.nn.functional.linear(
                    torch.nn.functional.linear(adapter_input, down), up
                )
                result = result + adapter_result.to(dtype=result.dtype) * scale
            if self._lightx2v_bias_delta is not None:
                result = result + self._lightx2v_bias_delta.to(
                    device=result.device, dtype=result.dtype
                )
            return result

    return ConvRotLinear()


class GGUFLinear:
    """Lazy linear operator that dequantizes a single GGUF weight on demand."""

    def __init__(self, store: GGUFWeightStore, weight_name: str, bias_name: str | None, in_features: int, out_features: int) -> None:
        self.store = store
        self.weight_name = weight_name
        self.bias_name = bias_name
        self.in_features = in_features
        self.out_features = out_features

    def as_module(self) -> Any:
        import torch
        import torch.nn as nn

        store = self.store
        weight_name = self.weight_name
        bias_name = self.bias_name
        in_features = self.in_features
        out_features = self.out_features

        class LazyLinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.weight_name = weight_name
                self.bias_name = bias_name
                if bias_name is not None:
                    self.register_buffer("bias", store.materialize(bias_name), persistent=False)
                else:
                    self.bias = None
                self._lightx2v_adapter_names: list[tuple[str, str, float]] = []
                self.register_buffer("_lightx2v_bias_delta", None, persistent=False)

            def add_lightx2v_adapter(self, down: Any, up: Any, scale: float) -> None:
                """Attach a low-rank adapter without materializing a Q6 base weight."""

                index = len(self._lightx2v_adapter_names)
                down_name = f"_lightx2v_down_{index}"
                up_name = f"_lightx2v_up_{index}"
                self.register_buffer(down_name, down, persistent=False)
                self.register_buffer(up_name, up, persistent=False)
                self._lightx2v_adapter_names.append((down_name, up_name, float(scale)))

            def add_lightx2v_bias(self, value: Any, scale: float = 1.0) -> None:
                value = value * float(scale)
                if self._lightx2v_bias_delta is None:
                    self._lightx2v_bias_delta = value
                else:
                    self._lightx2v_bias_delta.add_(value)

            def forward(self, input_tensor: Any) -> Any:
                import torch
                import torch.nn.functional as functional

                requested_dtype = os.getenv("WAN_GGUF_DEQUANT_DTYPE", "auto").strip().lower()
                dtype_names = {
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                }
                if requested_dtype in {"", "auto", "input"}:
                    # The official Wan blocks normalize through FP32 before
                    # their linear projections. Keeping that transient dtype
                    # would also make every cached reference K/V tensor FP32.
                    # CUDA inference uses the runtime's lower-precision dtype
                    # for those projections; CPU keeps the input dtype.
                    dequant_dtype = (
                        torch.bfloat16
                        if input_tensor.is_cuda and input_tensor.dtype == torch.float32
                        else input_tensor.dtype
                    )
                else:
                    try:
                        dequant_dtype = dtype_names[requested_dtype]
                    except KeyError as exc:
                        raise WanAnimate2RuntimeError(
                            "WAN_GGUF_DEQUANT_DTYPE must be auto, float16, or bfloat16"
                        ) from exc

                weight = store.materialize(
                    self.weight_name,
                    device=str(input_tensor.device),
                    dtype=dequant_dtype,
                )
                linear_input = input_tensor
                if linear_input.dtype != weight.dtype:
                    linear_input = linear_input.to(dtype=weight.dtype)
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input_tensor.device, dtype=weight.dtype)
                result = functional.linear(linear_input, weight, bias)
                for down_name, up_name, scale in self._lightx2v_adapter_names:
                    down = getattr(self, down_name)
                    up = getattr(self, up_name)
                    adapter_input = linear_input
                    if adapter_input.dtype != down.dtype:
                        adapter_input = adapter_input.to(dtype=down.dtype)
                    adapter_result = functional.linear(
                        functional.linear(adapter_input, down),
                        up,
                    )
                    result = result + adapter_result.to(dtype=result.dtype) * scale
                    del adapter_result
                if self._lightx2v_bias_delta is not None:
                    result = result + self._lightx2v_bias_delta.to(
                        device=result.device,
                        dtype=result.dtype,
                    )
                del linear_input, weight
                return result

        return LazyLinear()


def _official_module(source_root: Path) -> type[Any]:
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise WanAnimate2RuntimeError(f"official Wan-Animate-2 source not found: {source_root}")
    source_value = str(source_root)
    if source_value not in sys.path:
        sys.path.insert(0, source_value)
    try:
        module: ModuleType = importlib.import_module("wanxiang.wanxiang_animate_2_arch")
        return module.WanxiangAnimate2Transformer
    except (ImportError, AttributeError) as exc:
        raise WanAnimate2RuntimeError(
            "could not import WanxiangAnimate2Transformer from the official source tree"
        ) from exc


def _resolve_parent(root: Any, dotted_name: str) -> tuple[Any, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def _replace_parameter(root: Any, name: str, value: Any) -> None:
    import torch.nn as nn

    parent, attribute = _resolve_parent(root, name)
    setattr(parent, attribute, nn.Parameter(value, requires_grad=False))


def _build_meta_transformer(transformer_class: type[Any], spec: WanAnimate2TransformerSpec) -> Any:
    import torch

    with torch.device("meta"):
        return transformer_class(**spec.as_kwargs())


def _normalize_lightx2v_target(name: str) -> str:
    """Map the official LoRA names to the project runtime's block names."""

    parts = name.split(".")
    if parts and parts[0] == "diffusion_model":
        parts = parts[1:]
    if len(parts) >= 3 and parts[0] == "blocks" and parts[2] != "block":
        parts.insert(2, "block")
    return ".".join(parts)


def _resolve_value(root: Any, name: str) -> Any:
    value = root
    for part in name.split("."):
        value = value[int(part)] if part.isdigit() else getattr(value, part)
    return value


def _lightx2v_tensor_name(prefix: str, suffix: str) -> str:
    return f"{prefix}.{suffix}"


def _apply_lightx2v(
    model: Any,
    checkpoint: Path,
    *,
    strength: float,
    device: str,
    replaced_linears: dict[str, Any],
) -> dict[str, Any]:
    """Apply the official Wan-Animate-2 LightX2V adapter to a loaded model.

    Standard LoRA deltas are evaluated as ``(x @ down.T) @ up.T`` in each
    target linear, while the small direct ``diff``/``diff_b`` tensors are
    applied once during model loading. The same adapter contract is used for
    both the existing Q6 runtime and the native INT8 ConvRot runtime.
    """

    if not checkpoint.is_file():
        raise WanAnimate2RuntimeError(
            f"LightX2V is enabled but its checkpoint was not found: {checkpoint}"
        )
    if not device.startswith("cuda"):
        raise WanAnimate2RuntimeError(
            "LightX2V acceleration requires a CUDA device; refusing the CPU path"
        )
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - runtime environment only
        raise WanAnimate2RuntimeError(
            "LightX2V requires safetensors to be installed"
        ) from exc

    import torch

    if strength <= 0 or strength > 2:
        raise WanAnimate2RuntimeError(
            f"LightX2V strength must be greater than 0 and at most 2; got {strength}"
        )

    adapter_count = 0
    diff_count = 0
    diff_bias_count = 0
    loaded_keys: set[str] = set()
    with safe_open(str(checkpoint), framework="pt", device="cpu") as reader:
        keys = set(reader.keys())
        for key in sorted(keys):
            if not key.endswith(".lora_down.weight"):
                continue
            prefix = key[: -len(".lora_down.weight")]
            up_key = _lightx2v_tensor_name(prefix, "lora_up.weight")
            if up_key not in keys:
                raise WanAnimate2RuntimeError(
                    f"LightX2V adapter is incomplete: {key} has no {up_key}"
                )
            target = _normalize_lightx2v_target(prefix)
            module = replaced_linears.get(target)
            if module is None or not hasattr(module, "add_lightx2v_adapter"):
                raise WanAnimate2RuntimeError(
                    f"LightX2V target is not a loaded Wan linear: {prefix} -> {target}"
                )
            down = reader.get_tensor(key)
            up = reader.get_tensor(up_key)
            if down.ndim != 2 or up.ndim != 2 or down.shape[0] != up.shape[1]:
                raise WanAnimate2RuntimeError(
                    f"LightX2V adapter has invalid shapes for {prefix}: "
                    f"down={tuple(down.shape)} up={tuple(up.shape)}"
                )
            if down.shape[1] != module.in_features or up.shape[0] != module.out_features:
                raise WanAnimate2RuntimeError(
                    f"LightX2V adapter shape mismatch for {prefix}: "
                    f"down={tuple(down.shape)} up={tuple(up.shape)} "
                    f"module=({module.out_features}, {module.in_features})"
                )
            module.add_lightx2v_adapter(
                down.to(device=device, dtype=torch.bfloat16),
                up.to(device=device, dtype=torch.bfloat16),
                strength,
            )
            loaded_keys.update((key, up_key))
            adapter_count += 1

        with torch.no_grad():
            for key in sorted(keys):
                if key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight"):
                    continue
                if key.endswith(".diff_b"):
                    prefix = key[: -len(".diff_b")]
                    target = _normalize_lightx2v_target(prefix)
                    module = _resolve_value(model, target)
                    value = reader.get_tensor(key).to(device=device, dtype=torch.bfloat16)
                    if hasattr(module, "add_lightx2v_bias"):
                        module.add_lightx2v_bias(value, strength)
                    else:
                        bias = getattr(module, "bias", None)
                        if bias is None or tuple(bias.shape) != tuple(value.shape):
                            raise WanAnimate2RuntimeError(
                                f"LightX2V bias target is not compatible: {prefix} -> {target}"
                            )
                        bias.add_(value.to(dtype=bias.dtype) * strength)
                    loaded_keys.add(key)
                    diff_bias_count += 1
                    continue
                if key.endswith(".diff"):
                    prefix = key[: -len(".diff")]
                    target = _normalize_lightx2v_target(prefix)
                    value = reader.get_tensor(key).to(device=device, dtype=torch.bfloat16)
                    parameter = None
                    for parameter_name in (target, f"{target}.weight"):
                        parameter = dict(model.named_parameters()).get(parameter_name)
                        if parameter is not None:
                            break
                    if parameter is None or tuple(parameter.shape) != tuple(value.shape):
                        raise WanAnimate2RuntimeError(
                            f"LightX2V direct-diff target is not compatible: "
                            f"{prefix} -> {target} shape={tuple(value.shape)}"
                        )
                    parameter.add_(value.to(dtype=parameter.dtype) * strength)
                    loaded_keys.add(key)
                    diff_count += 1

        unhandled = sorted(keys - loaded_keys)
        if unhandled:
            raise WanAnimate2RuntimeError(
                "LightX2V checkpoint contains unsupported keys: "
                + ", ".join(unhandled[:8])
                + (" ..." if len(unhandled) > 8 else "")
            )

    LOGGER.info(
        "stage=lightx2v_ready checkpoint=%s strength=%.4f lora_adapters=%s direct_diffs=%s "
        "direct_bias_diffs=%s device=%s",
        checkpoint,
        strength,
        adapter_count,
        diff_count,
        diff_bias_count,
        device,
    )
    return {
        "enabled": True,
        "checkpoint": str(checkpoint),
        "strength": float(strength),
        "adapterCount": adapter_count,
        "directDiffCount": diff_count,
        "directBiasDiffCount": diff_bias_count,
        "device": device,
    }


def load_transformer(
    checkpoint: Path,
    official_source: Path,
    *,
    device: str = "cuda",
    spec: WanAnimate2TransformerSpec | None = None,
    lightx2v_checkpoint: Path | None = None,
    lightx2v_strength: float = 1.0,
) -> tuple[Any, Any, dict[str, Any]]:
    """Load the official transformer using the checkpoint's explicit format."""

    import torch
    import torch.nn as nn

    prepare_runtime_cache_dirs()
    spec = spec or WanAnimate2TransformerSpec(
        log_scale=float(os.getenv("WAN_ANIMATE_LOG_SCALE", "-1.3"))
    )
    if checkpoint.suffix.lower() in {".safetensors", ".safetensor"}:
        return _load_int8_convrot_transformer(
            checkpoint,
            official_source,
            device=device,
            spec=spec,
            lightx2v_checkpoint=lightx2v_checkpoint,
            lightx2v_strength=lightx2v_strength,
        )
    store = GGUFWeightStore(checkpoint, model_type="animate2")
    store.configure_gpu_raw_cache(device)
    transformer_class = _official_module(official_source)
    model = _build_meta_transformer(transformer_class, spec)
    validation = store.validate_against(dict(model.named_parameters()))

    replaced_linears: dict[str, Any] = {}
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        weight_name = f"{module_name}.weight" if module_name else "weight"
        bias_name = f"{module_name}.bias" if module_name and module.bias is not None else None
        replacement = GGUFLinear(
            store,
            weight_name,
            bias_name,
            module.in_features,
            module.out_features,
        ).as_module()
        parent, attribute = _resolve_parent(model, module_name)
        setattr(parent, attribute, replacement)
        replaced_linears[module_name] = replacement

    for parameter_name, parameter in list(model.named_parameters()):
        if not parameter.is_meta:
            continue
        _replace_parameter(model, parameter_name, store.materialize(parameter_name))

    model = model.to(device=device).eval().requires_grad_(False)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise WanAnimate2RuntimeError("Wan-Animate-2 model still contains meta parameters after loading")
    if lightx2v_checkpoint is not None:
        validation["lightx2v"] = _apply_lightx2v(
            model,
            lightx2v_checkpoint,
            strength=lightx2v_strength,
            device=device,
            replaced_linears=replaced_linears,
        )
    else:
        validation["lightx2v"] = {"enabled": False}
    return model, store, validation


def _load_int8_convrot_transformer(
    checkpoint: Path,
    official_source: Path,
    *,
    device: str,
    spec: WanAnimate2TransformerSpec,
    lightx2v_checkpoint: Path | None,
    lightx2v_strength: float,
) -> tuple[Any, SafeTensorWeightStore, dict[str, Any]]:
    """Load Comfy's official INT8 ConvRot layout through comfy-kitchen only."""

    import torch
    import torch.nn as nn

    kernel_report = _validate_convrot_cuda(device)
    store = SafeTensorWeightStore(checkpoint)
    transformer_class = _official_module(official_source)
    with torch.device("meta"):
        model = transformer_class(**spec.as_kwargs())
    validation = store.validate_against(dict(model.named_parameters()))
    validation["convrotKernel"] = kernel_report

    replaced_linears: dict[str, Any] = {}
    loaded_quantized = 0
    loaded_full_precision = 0
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        weight_name = f"{module_name}.weight" if module_name else "weight"
        bias_name = f"{module_name}.bias" if module_name and module.bias is not None else None
        raw_weight_name = _raw_safetensors_name(weight_name)
        quantization = store.quantization(raw_weight_name)
        if quantization is not None:
            if quantization.get("format") != "int8_tensorwise":
                raise WanAnimate2RuntimeError(
                    f"unsupported INT8 quantization format for {raw_weight_name}: {quantization}"
                )
            if quantization.get("convrot") is not True:
                raise WanAnimate2RuntimeError(
                    f"INT8 checkpoint is not ConvRot-enabled for {raw_weight_name}"
                )
            group_size = int(quantization.get("convrot_groupsize", 0))
            if group_size != 256:
                raise WanAnimate2RuntimeError(
                    f"unsupported ConvRot group size for {raw_weight_name}: {group_size}"
                )
            scale_name = f"{raw_weight_name[:-len('.weight')]}.weight_scale"
            if scale_name not in store:
                raise WanAnimate2RuntimeError(
                    f"INT8 ConvRot weight has no scale tensor: {raw_weight_name}"
                )
            bias = (
                store.materialize(_raw_safetensors_name(bias_name), device=device, dtype=torch.bfloat16)
                if bias_name is not None
                else None
            )
            replacement = _make_convrot_linear(
                store.materialize(raw_weight_name, device=device, dtype=torch.int8),
                store.materialize(scale_name, device=device, dtype=torch.float32),
                bias,
                device=device,
                convrot_groupsize=group_size,
            )
            loaded_quantized += 1
        else:
            weight = store.materialize(raw_weight_name, device=device, dtype=torch.bfloat16)
            bias = (
                store.materialize(_raw_safetensors_name(bias_name), device=device, dtype=torch.bfloat16)
                if bias_name is not None
                else None
            )
            replacement = _make_adapter_linear(weight, bias, device=device)
            loaded_full_precision += 1
        parent, attribute = _resolve_parent(model, module_name)
        setattr(parent, attribute, replacement)
        replaced_linears[module_name] = replacement

    for parameter_name, parameter in list(model.named_parameters()):
        if not parameter.is_meta:
            continue
        raw_name = _raw_safetensors_name(parameter_name)
        _replace_parameter(
            model,
            parameter_name,
            store.materialize(raw_name, device=device, dtype=torch.bfloat16),
        )

    model = model.to(device=device).eval().requires_grad_(False)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise WanAnimate2RuntimeError(
            "Wan-Animate-2 INT8 ConvRot model still contains meta parameters after loading"
        )
    validation.update(
        {
            "quantizedLinearCount": loaded_quantized,
            "fullPrecisionLinearCount": loaded_full_precision,
        }
    )
    if lightx2v_checkpoint is not None:
        validation["lightx2v"] = _apply_lightx2v(
            model,
            lightx2v_checkpoint,
            strength=lightx2v_strength,
            device=device,
            replaced_linears=replaced_linears,
        )
    else:
        validation["lightx2v"] = {"enabled": False}
    LOGGER.info(
        "stage=int8_convrot_transformer_ready checkpoint=%s quantized_linears=%s "
        "full_precision_linears=%s device=%s",
        checkpoint,
        loaded_quantized,
        loaded_full_precision,
        device,
    )
    return model, store, validation


def inspect_checkpoint(checkpoint: Path, official_source: Path) -> dict[str, Any]:
    """Validate a GGUF or INT8 ConvRot tensor contract without loading weights."""

    import torch

    is_safetensors = checkpoint.suffix.lower() in {".safetensors", ".safetensor"}
    store = (
        SafeTensorWeightStore(checkpoint)
        if is_safetensors
        else GGUFWeightStore(checkpoint, model_type="animate2")
    )
    transformer_class = _official_module(official_source)
    model = _build_meta_transformer(
        transformer_class,
        WanAnimate2TransformerSpec(
            log_scale=float(os.getenv("WAN_ANIMATE_LOG_SCALE", "-1.3"))
        ),
    )
    validation = store.validate_against(dict(model.named_parameters()))
    validation.update(
        {
            "checkpoint": str(checkpoint.expanduser().resolve()),
            "modelType": "animate2",
            "modelFormat": "int8_convrot" if is_safetensors else "gguf",
            "metadata": store.metadata,
            "cudaAvailable": torch.cuda.is_available(),
        }
    )
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate or load a Wan-Animate-2 GGUF checkpoint.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--official-source", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--load", action="store_true", help="Materialize the non-quantized model state and load the runtime.")
    args = parser.parse_args()
    import os

    source_value = args.official_source or os.environ.get("WAN_ANIMATE_2_SOURCE")
    if not source_value:
        parser.error("--official-source or WAN_ANIMATE_2_SOURCE is required")
    source = Path(source_value)
    if args.load:
        _, store, validation = load_transformer(args.model, source, device=args.device)
        print({**validation, "loaded": True, "tensorCount": len(store)})
    else:
        print(inspect_checkpoint(args.model, source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
