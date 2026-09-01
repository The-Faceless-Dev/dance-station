"""Project-owned Wan-Animate-2 GGUF runtime primitives.

The web application and composition code do not import this module. It is an
inference-side adapter that loads the official Wan-Animate-2 transformer with
quantized weights kept memory-mapped on CPU. Each quantized linear layer is
dequantized only for the duration of its forward pass.

The official Wan-Animate-2 source tree is supplied separately with
``--official-source`` (or ``WAN_ANIMATE_2_SOURCE``). Keeping that boundary
explicit makes the runtime replaceable without coupling the application to a
workflow host or a particular model tier.
"""

from __future__ import annotations

import argparse
import importlib
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

            def forward(self, input_tensor: Any) -> Any:
                import torch.nn.functional as functional

                weight = store.materialize(self.weight_name, device=str(input_tensor.device), dtype=input_tensor.dtype)
                bias = self.bias
                if bias is not None:
                    bias = bias.to(device=input_tensor.device, dtype=input_tensor.dtype)
                result = functional.linear(input_tensor, weight, bias)
                del weight
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


def load_transformer(
    checkpoint: Path,
    official_source: Path,
    *,
    device: str = "cuda",
    spec: WanAnimate2TransformerSpec = WanAnimate2TransformerSpec(),
) -> tuple[Any, GGUFWeightStore, dict[str, Any]]:
    """Load the official transformer with project-owned GGUF offload."""

    import torch
    import torch.nn as nn

    prepare_runtime_cache_dirs()
    store = GGUFWeightStore(checkpoint, model_type="animate2")
    store.configure_gpu_raw_cache(device)
    transformer_class = _official_module(official_source)
    model = _build_meta_transformer(transformer_class, spec)
    validation = store.validate_against(dict(model.named_parameters()))

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

    for parameter_name, parameter in list(model.named_parameters()):
        if not parameter.is_meta:
            continue
        _replace_parameter(model, parameter_name, store.materialize(parameter_name))

    model = model.to(device=device).eval().requires_grad_(False)
    if any(parameter.is_meta for parameter in model.parameters()):
        raise WanAnimate2RuntimeError("Wan-Animate-2 model still contains meta parameters after loading")
    return model, store, validation


def inspect_checkpoint(checkpoint: Path, official_source: Path) -> dict[str, Any]:
    """Validate the Q3/Q8 tensor contract without allocating model weights."""

    import torch

    store = GGUFWeightStore(checkpoint, model_type="animate2")
    transformer_class = _official_module(official_source)
    model = _build_meta_transformer(transformer_class, WanAnimate2TransformerSpec())
    validation = store.validate_against(dict(model.named_parameters()))
    validation.update(
        {
            "checkpoint": str(checkpoint.expanduser().resolve()),
            "modelType": "animate2",
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
