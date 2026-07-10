"""INT8 inference machinery for K2.

Eligible linear layers support two INT8 layouts. ``blockwise`` stores FP32
scales per 128x128 weight and activation block. ``rowwise`` stores one FP32
scale for the complete weight tensor and dynamically uses one scale per
flattened activation row. GEMMs run through :mod:`krea2.kernels.int8`.

Norms, embeddings, modulations and attention (torch SDPA) stay in bf16.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from krea2.kernels import int8 as kernels

INT8_K_BLOCK = kernels.K_BLOCK
INT8_M_BLOCK = kernels.M_BLOCK
INT8_N_BLOCK = kernels.N_BLOCK
QUANTIZATION_TYPES = ("blockwise", "rowwise")


def quantize_int8_weight_blocks(w: Tensor) -> tuple[Tensor, Tensor]:
    """(N, K) float -> int8 and fp32 scales shaped (N//128, K//128)."""
    return kernels.weight_block_quant(w, n_block=INT8_N_BLOCK, k_block=INT8_K_BLOCK)


def quantize_int8_weight_tensorwise(w: Tensor) -> tuple[Tensor, Tensor]:
    """(N, K) float -> int8 and one scalar FP32 scale."""
    return kernels.weight_tensorwise_quant(w)


def quantize_int8_activation_blocks(x: Tensor) -> tuple[Tensor, Tensor]:
    """(..., K) float -> int8 and fp32 scales shaped (ceil(M/128), K//128)."""
    return kernels.activation_block_quant(x, m_block=INT8_M_BLOCK, k_block=INT8_K_BLOCK)


def int8_gemm(
    xq: Tensor,
    x_scale: Tensor,
    wq: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """x @ w.T with int8 inputs and 128x128 A/B block scales.

    xq: (M, K) int8, x_scale: (ceil(M/128), K//128) f32,
    wq: (N, K) int8, w_scale: (N//128, K//128) f32.
    """
    return kernels.matmul_int8_block2d(
        xq,
        wq.t(),
        x_scale,
        w_scale,
        bias=bias,
        m_block=INT8_M_BLOCK,
        k_block=INT8_K_BLOCK,
        n_block=INT8_N_BLOCK,
        out_dtype=out_dtype,
    )


def int8_linear(
    x: Tensor,
    wq: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    quantization_type: str = "blockwise",
) -> Tensor:
    """bf16/fp32 x @ int8 weight.T using the selected scale geometry."""
    lead = x.shape[:-1]
    out_features, in_features = wq.shape
    assert x.shape[-1] == in_features, (
        f"incompatible input/weight shapes: {tuple(x.shape)} and {tuple(wq.shape)}"
    )
    if quantization_type == "rowwise":
        return kernels.int8_mm_rowwise(x, wq, w_scale, bias, out_dtype=out_dtype)
    assert quantization_type == "blockwise", quantization_type
    xq, x_scale = quantize_int8_activation_blocks(x)
    y = int8_gemm(
        xq.reshape(-1, in_features),
        x_scale,
        wq,
        w_scale,
        bias=bias,
        out_dtype=out_dtype,
    )
    return y.view(*lead, out_features)


def int8_lora_linear(
    x: Tensor,
    wq: Tensor,
    w_scale: Tensor,
    bias: Tensor | None,
    lora_a: Tensor,
    lora_b: Tensor,
    lora_scale: float,
    out_dtype: torch.dtype = torch.bfloat16,
    quantization_type: str = "blockwise",
) -> Tensor:
    """bf16/fp32 X @ frozen INT8 W.T + FP32 LoRA -> bf16.

    The LoRA master weights stay fp32, but the custom kernels cast them to
    bf16 for the small-rank matmuls used in forward and backward.
    """
    op = (
        kernels.int8_mm_fused_lora_rowwise
        if quantization_type == "rowwise"
        else kernels.int8_mm_fused_lora
    )
    assert quantization_type in QUANTIZATION_TYPES, quantization_type
    return op(
        x,
        wq,
        w_scale,
        bias,
        lora_a,
        lora_b,
        lora_scale,
        out_dtype=out_dtype,
    )


class INT8Linear(nn.Module):
    """Drop-in nn.Linear replacement using blockwise or rowwise INT8 GEMMs.

    Weights are int8 with one scale per 128x128 (N, K) block. The normal
    forward path takes bf16/fp32 activations, quantizes them with one scale per
    128x128 (M, K) block, then returns a bf16 GEMM result.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        out_dtype: torch.dtype = torch.bfloat16,
        quantization_type: str = "blockwise",
    ):
        super().__init__()
        if quantization_type not in QUANTIZATION_TYPES:
            raise ValueError(
                f"quantization_type must be one of {QUANTIZATION_TYPES}, "
                f"got {quantization_type!r}"
            )
        assert in_features % INT8_K_BLOCK == 0, (
            f"INT8 K-block GEMM needs in_features divisible by {INT8_K_BLOCK}, "
            f"got {in_features}"
        )
        assert out_features % INT8_N_BLOCK == 0, (
            f"INT8 weight blocks need out_features divisible by {INT8_N_BLOCK}, "
            f"got {out_features}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.quantization_type = quantization_type
        self.register_buffer(
            "weight",
            torch.empty(out_features, in_features, dtype=torch.int8, device=device),
        )
        scale_shape = (
            ()
            if quantization_type == "rowwise"
            else (out_features // INT8_N_BLOCK, in_features // INT8_K_BLOCK)
        )
        self.register_buffer(
            "weight_scale",
            torch.empty(scale_shape, dtype=torch.float32, device=device),
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=out_dtype, device=device),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

    @classmethod
    @torch.no_grad()
    def from_linear(
        cls,
        linear: nn.Linear,
        device=None,
        quantization_type: str = "blockwise",
    ) -> "INT8Linear":
        device = device or linear.weight.device
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device="meta",
            quantization_type=quantization_type,
        )
        quantize = (
            quantize_int8_weight_tensorwise
            if quantization_type == "rowwise"
            else quantize_int8_weight_blocks
        )
        q, scale = quantize(linear.weight.detach().to(device))
        mod.weight = q
        mod.weight_scale = scale
        if linear.bias is not None:
            mod.bias = nn.Parameter(
                linear.bias.detach().to(device=device, dtype=mod.out_dtype),
                requires_grad=False,
            )
        return mod

    def forward(self, x: Tensor) -> Tensor:
        return int8_linear(
            x,
            self.weight,
            self.weight_scale,
            bias=self.bias,
            out_dtype=self.out_dtype,
            quantization_type=self.quantization_type,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, "
            f"quantization_type={self.quantization_type}, "
            f"m_block={INT8_M_BLOCK}, n_block={INT8_N_BLOCK}, "
            f"k_block={INT8_K_BLOCK}"
        )


class LinearLoraINT8(INT8Linear):
    """INT8 frozen linear with trainable FP32 LoRA adapters.

    Base weights are 128x128 block-quantized INT8 buffers. LoRA uses
    A[rank, in_features] and B[out_features, rank] FP32 master parameters.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 32,
        alpha: float | None = None,
        bias: bool = True,
        device=None,
        out_dtype: torch.dtype = torch.bfloat16,
        quantization_type: str = "blockwise",
    ):
        assert rank in (32, 64), f"supported LoRA ranks are 32 and 64, got {rank}"
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            out_dtype=out_dtype,
            quantization_type=quantization_type,
        )
        self.rank = rank
        self.lora_alpha = float(rank if alpha is None else alpha)
        self.lora_scale = self.lora_alpha / float(rank)
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.empty(out_features, rank, device=device, dtype=torch.float32)
        )
        self.reset_lora_parameters()

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @classmethod
    def from_int8_linear(
        cls,
        linear: INT8Linear,
        rank: int = 32,
        alpha: float | None = None,
    ) -> "LinearLoraINT8":
        mod = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            device=linear.weight.device,
            out_dtype=linear.out_dtype,
            quantization_type=linear.quantization_type,
        )
        mod.weight = linear.weight
        mod.weight_scale = linear.weight_scale
        if linear.bias is not None:
            mod.bias = nn.Parameter(linear.bias.detach(), requires_grad=False)
        return mod

    def forward(self, x: Tensor) -> Tensor:
        return int8_lora_linear(
            x,
            self.weight,
            self.weight_scale,
            self.bias,
            self.lora_A,
            self.lora_B,
            self.lora_scale,
            out_dtype=self.out_dtype,
            quantization_type=self.quantization_type,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, alpha={self.lora_alpha:g}, "
            f"bias={self.bias is not None}, "
            f"quantization_type={self.quantization_type}, "
            f"m_block={INT8_M_BLOCK}, n_block={INT8_N_BLOCK}, "
            f"k_block={INT8_K_BLOCK}"
        )


def _target_linears(root: nn.Module, skip: tuple[str, ...]):
    for name, module in root.named_modules():
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if child.in_features % INT8_K_BLOCK or child.out_features % INT8_N_BLOCK:
                continue
            if any(full == s or full.startswith(s + ".") for s in skip):
                continue
            yield module, child_name, full, child


def swap_linears_meta(
    root: nn.Module,
    skip: tuple[str, ...] = (),
    quantization_type: str = "blockwise",
) -> list[str]:
    """Replace eligible nn.Linear submodules of a meta-device model with empty
    INT8Linear shells, to be filled by load_int8_state_dict. Returns names."""
    swapped = []
    for module, child_name, full, child in _target_linears(root, skip):
        setattr(
            module,
            child_name,
            INT8Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device="meta",
                quantization_type=quantization_type,
            ),
        )
        swapped.append(full)
    return swapped


@torch.no_grad()
def convert_linears_int8(
    root: nn.Module,
    device="cuda",
    skip: tuple[str, ...] = (),
    quantization_type: str = "blockwise",
) -> list[str]:
    """Quantize eligible nn.Linear submodules to INT8Linear on `device`, in place.

    For already-materialized models (e.g. the Hugging Face text encoder), each
    weight is quantized on the GPU and the original is freed immediately.
    """
    converted = []
    for module, child_name, full, child in _target_linears(root, skip):
        setattr(
            module,
            child_name,
            INT8Linear.from_linear(
                child, device=device, quantization_type=quantization_type
            ),
        )
        converted.append(full)
    return converted


def add_lora_to_int8_blocks(
    model: nn.Module,
    rank: int = 32,
    alpha: float | None = None,
) -> list[str]:
    """Replace INT8Linear modules inside main DiT blocks with LinearLoraINT8."""
    converted = []
    if not hasattr(model, "blocks"):
        return converted
    for block_idx, block in enumerate(model.blocks):
        prefix = f"blocks.{block_idx}"
        for name, module in list(block.named_modules()):
            for child_name, child in list(module.named_children()):
                if not isinstance(child, INT8Linear) or isinstance(
                    child, LinearLoraINT8
                ):
                    continue
                full = (
                    f"{prefix}.{name}.{child_name}"
                    if name
                    else f"{prefix}.{child_name}"
                )
                setattr(
                    module,
                    child_name,
                    LinearLoraINT8.from_int8_linear(child, rank=rank, alpha=alpha),
                )
                converted.append(full)
    return converted


def lora_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, LinearLoraINT8):
            yield module.lora_A
            yield module.lora_B


def _canonical_lora_module_name(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 4 and parts[0] == "blocks" and parts[2] == "block":
        return ".".join([parts[0], parts[1], *parts[3:]])
    return name


def _canonical_lora_key(key: str) -> str:
    module_name, param_name = key.rsplit(".", 1)
    return f"{_canonical_lora_module_name(module_name)}.{param_name}"


def _lora_module_aliases(model: nn.Module) -> dict[str, LinearLoraINT8]:
    aliases = {}
    for name, module in model.named_modules():
        if not isinstance(module, LinearLoraINT8):
            continue
        for alias in {name, _canonical_lora_module_name(name)}:
            if alias in aliases and aliases[alias] is not module:
                raise RuntimeError(f"duplicate LoRA module alias: {alias}")
            aliases[alias] = module
    return aliases


def lora_state_tensors(model: nn.Module) -> dict[str, Tensor]:
    tensors = {}
    for name, module in model.named_modules():
        if isinstance(module, LinearLoraINT8):
            canonical = _canonical_lora_module_name(name)
            for param_name, tensor in (
                ("lora_A", module.lora_A),
                ("lora_B", module.lora_B),
            ):
                key = f"{canonical}.{param_name}"
                if key in tensors:
                    raise RuntimeError(f"duplicate LoRA tensor key: {key}")
                tensors[key] = tensor.detach().float().cpu()
    return tensors


def save_lora_adapters(
    model: nn.Module,
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> None:
    md = {} if metadata is None else {str(k): str(v) for k, v in metadata.items()}
    save_file(lora_state_tensors(model), str(path), metadata=md)


def load_lora_state_tensors(
    model: nn.Module, tensors: dict[str, Tensor], strict: bool = True
) -> None:
    """Load adapter tensors from an in-memory training checkpoint."""
    canonical_tensors = {}
    for key, tensor in tensors.items():
        canonical_key = _canonical_lora_key(key)
        if canonical_key in canonical_tensors:
            raise RuntimeError(f"duplicate LoRA tensor key after normalization: {key}")
        canonical_tensors[canonical_key] = tensor
    expected = set(lora_state_tensors(model))
    found = set(canonical_tensors)
    if strict and expected != found:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise RuntimeError(f"LoRA adapter mismatch: missing={missing}, extra={extra}")
    modules = _lora_module_aliases(model)
    with torch.no_grad():
        for key, tensor in canonical_tensors.items():
            module_name, param_name = key.rsplit(".", 1)
            module = modules.get(module_name)
            if module is None:
                if strict:
                    raise RuntimeError(f"LoRA adapter target not found: {module_name}")
                continue
            target = getattr(module, param_name)
            target.copy_(tensor.to(device=target.device, dtype=target.dtype))


def load_lora_adapters(model: nn.Module, path: str | Path, strict: bool = True) -> None:
    load_lora_state_tensors(model, load_file(str(path), device="cpu"), strict=strict)


@torch.no_grad()
def load_int8_state_dict(
    model: nn.Module, path: str, device="cuda", dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    """Stream a safetensors checkpoint into a meta model prepared with
    swap_linears_meta: INT8Linear weights are quantized on `device` as they are
    read, everything else is loaded as `dtype`. Peak host memory is one tensor,
    not the whole checkpoint."""
    int8_weights = {
        f"{name}.weight": module
        for name, module in model.named_modules()
        if isinstance(module, INT8Linear)
    }
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key in int8_weights:
                module = int8_weights[key]
                quantize = (
                    quantize_int8_weight_tensorwise
                    if module.quantization_type == "rowwise"
                    else quantize_int8_weight_blocks
                )
                q, scale = quantize(tensor.to(device))
                state[key] = q
                state[key[: -len("weight")] + "weight_scale"] = scale
            else:
                state[key] = tensor.to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True, assign=True)
    return model
