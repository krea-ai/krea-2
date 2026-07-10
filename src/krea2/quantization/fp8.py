"""FP8 inference machinery for K2.

Only linear layers are converted: weights are stored as fp8 e4m3 with per-
output-channel (rowwise) f32 scales, activations are dynamically quantized
per row (per token) by the fused :mod:`krea2.kernels.fp8` Triton kernels,
and the GEMM runs through torch.nn.functional.scaled_mm. Norms, embeddings,
modulations and attention (torch SDPA) stay in bf16.

On SM 8.9 (Ada) torch routes RowWise-scaled GEMMs to a CUTLASS kernel that is
slower than plain bf16 matmul, while the TensorWise cuBLASLt path runs at full
fp8 throughput (~2x bf16). fp8_gemm therefore defaults on Ada to running
scaled_mm with unit scales and applying the rowwise dequant (+ bias) in a small
Triton epilogue — numerically identical to native RowWise scaling. On SM 9.0+
it uses native RowWise scaled_mm. Override with K2_FP8_NATIVE_ROWWISE=0/1.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from safetensors import safe_open
from torch import Tensor
from torch.nn.functional import ScalingType

from krea2.kernels import fp8 as kernels
from krea2.models.transformer import (
    Attention,
    SingleStreamBlock,
    SwiGLU,
    TextFusionBlock,
    attention,
    ropeapply,
)

FP8_DTYPE = kernels.FP8_DTYPE


def _default_native_rowwise() -> bool:
    env = os.environ.get("K2_FP8_NATIVE_ROWWISE")
    if env is not None:
        return env.lower() not in ("0", "false")
    if not torch.cuda.is_available():
        return True
    return torch.cuda.get_device_capability() >= (9, 0)


_NATIVE_ROWWISE = _default_native_rowwise()

_UNIT_SCALES: dict[torch.device, Tensor] = {}


def _unit_scale(device: torch.device) -> Tensor:
    scale = _UNIT_SCALES.get(device)
    if scale is None:
        scale = _UNIT_SCALES[device] = torch.ones((), device=device)
    return scale


def quantize_fp8_rowwise(w: Tensor) -> tuple[Tensor, Tensor]:
    """(..., K) float -> (fp8 (..., K), f32 scale (..., 1)); w ~= q.float() * scale.

    On CUDA this runs the Triton kernel: fp32 math on the fly with no fp32
    materialization of w — the eager path transiently needs ~3 fp32 copies,
    which OOMs when quantizing 200 MiB weights on a nearly-full card.
    """
    if w.is_cuda:
        return kernels.rowwise_quant(w)
    w = w.float()
    scale = w.abs().amax(dim=-1, keepdim=True).clamp_(min=kernels.SCALE_EPS)
    scale = scale / kernels.FP8_MAX
    q = (w / scale).clamp_(-kernels.FP8_MAX, kernels.FP8_MAX).to(FP8_DTYPE)
    return q, scale


def fp8_gemm(
    xq: Tensor,
    x_scale: Tensor,
    wq: Tensor,
    w_scale: Tensor,
    bias: Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> Tensor:
    """x @ w.T with fp8 inputs and rowwise scales via torch.nn.functional.scaled_mm.

    xq: (M, K) fp8, x_scale: (M, 1) f32, wq: (N, K) fp8, w_scale: (N, 1) f32.
    """
    if _NATIVE_ROWWISE:
        return F.scaled_mm(
            xq,
            wq.t(),
            x_scale,
            ScalingType.RowWise,
            w_scale.t(),
            ScalingType.RowWise,
            bias=bias,
            output_dtype=out_dtype,
        )
    one = _unit_scale(xq.device)
    y = F.scaled_mm(
        xq,
        wq.t(),
        one,
        ScalingType.TensorWise,
        one,
        ScalingType.TensorWise,
        output_dtype=out_dtype,
    )
    return kernels.rowwise_dequant_(y, x_scale, w_scale, bias)


class FP8Linear(nn.Module):
    """Drop-in nn.Linear replacement running the GEMM in fp8 via F.scaled_mm.

    Weights are fp8 e4m3 with per-output-channel scales; the input is
    dynamically quantized per row (Triton kernel) unless the caller already
    holds a quantized input, in which case use forward_quantized.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        out_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        assert in_features % 16 == 0 and out_features % 16 == 0, (
            f"scaled_mm needs dims divisible by 16, got {in_features}x{out_features}"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.out_dtype = out_dtype
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=FP8_DTYPE, device=device),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.empty(out_features, 1, dtype=torch.float32, device=device),
            requires_grad=False,
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
    def from_linear(cls, linear: nn.Linear, device=None) -> "FP8Linear":
        device = device or linear.weight.device
        mod = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            device="meta",
        )
        q, scale = quantize_fp8_rowwise(linear.weight.detach().to(device))
        mod.weight = nn.Parameter(q, requires_grad=False)
        mod.weight_scale = nn.Parameter(scale, requires_grad=False)
        if linear.bias is not None:
            mod.bias = nn.Parameter(
                linear.bias.detach().to(device=device, dtype=mod.out_dtype),
                requires_grad=False,
            )
        return mod

    def forward_quantized(self, xq: Tensor, x_scale: Tensor) -> Tensor:
        lead = xq.shape[:-1]
        y = fp8_gemm(
            xq.reshape(-1, self.in_features),
            x_scale.reshape(-1, 1),
            self.weight,
            self.weight_scale,
            self.bias,
            self.out_dtype,
        )
        return y.view(*lead, self.out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_quantized(*kernels.rowwise_quant(x))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def _target_linears(root: nn.Module, skip: tuple[str, ...]):
    for name, module in root.named_modules():
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if not isinstance(child, nn.Linear):
                continue
            if child.in_features % 16 or child.out_features % 16:
                continue
            if any(full == s or full.startswith(s + ".") for s in skip):
                continue
            yield module, child_name, full, child


def swap_linears_meta(root: nn.Module, skip: tuple[str, ...] = ()) -> list[str]:
    """Replace eligible nn.Linear submodules of a meta-device model with empty
    FP8Linear shells, to be filled by load_fp8_state_dict. Returns their names."""
    swapped = []
    for module, child_name, full, child in _target_linears(root, skip):
        setattr(
            module,
            child_name,
            FP8Linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                device="meta",
            ),
        )
        swapped.append(full)
    return swapped


@torch.no_grad()
def convert_linears_fp8(
    root: nn.Module, device="cuda", skip: tuple[str, ...] = ()
) -> list[str]:
    """Quantize eligible nn.Linear submodules to FP8Linear on `device`, in place.

    For already-materialized models (e.g. the Hugging Face text encoder); each
    weight is quantized on the GPU and the original is freed immediately.
    """
    converted = []
    for module, child_name, full, child in _target_linears(root, skip):
        setattr(module, child_name, FP8Linear.from_linear(child, device=device))
        converted.append(full)
    return converted


@torch.no_grad()
def load_fp8_state_dict(
    model: nn.Module, path: str, device="cuda", dtype: torch.dtype = torch.bfloat16
) -> nn.Module:
    """Stream a safetensors checkpoint into a meta model prepared with
    swap_linears_meta: FP8Linear weights are quantized on `device` as they are
    read, everything else is loaded as `dtype`. Peak host memory is one tensor,
    not the whole checkpoint."""
    fp8_weights = {
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, FP8Linear)
    }
    state = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key in fp8_weights:
                q, scale = quantize_fp8_rowwise(tensor.to(device))
                state[key] = q
                state[key[: -len("weight")] + "weight_scale"] = scale
            else:
                state[key] = tensor.to(device=device, dtype=dtype)
    model.load_state_dict(state, strict=True, assign=True)
    return model


class FP8Attention(Attention):
    """mmdit.Attention over a pre-quantized input: the four input projections
    share one quantization, and the output-gate sigmoid is fused with the wo
    input quantization. Attention itself is torch SDPA in bf16."""

    def forward(
        self,
        xq: Tensor,
        x_scale: Tensor,
        freqs: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        q = self.wq.forward_quantized(xq, x_scale)
        k = self.wk.forward_quantized(xq, x_scale)
        v = self.wv.forward_quantized(xq, x_scale)
        gate = self.gate.forward_quantized(xq, x_scale)

        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)
        x = attention(q, k, v, mask=mask, gqa=self.gqa)
        oq, o_scale = kernels.mul_sigmoid_quant(x, gate)
        return self.wo.forward_quantized(oq, o_scale)


class FP8SwiGLU(SwiGLU):
    """mmdit.SwiGLU over a pre-quantized input; silu(gate) * up is fused with
    the down-projection input quantization."""

    def forward(self, xq: Tensor, x_scale: Tensor) -> Tensor:
        gate = self.gate.forward_quantized(xq, x_scale)
        up = self.up.forward_quantized(xq, x_scale)
        hq, h_scale = kernels.silu_mul_quant(gate, up)
        return self.down.forward_quantized(hq, h_scale)


class FP8SingleStreamBlock(SingleStreamBlock):
    """mmdit.SingleStreamBlock with RMSNorm + modulation fused into the fp8
    input quantization of attention and MLP."""

    def forward(
        self, x: Tensor, vec: Tensor, freqs: Tensor, mask: Tensor | None = None
    ) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        aq, a_scale = kernels.rmsnorm_quant(
            x, self.prenorm.scale, self.prenorm.eps, prescale, preshift
        )
        x = x + pregate * self.attn(aq, a_scale, freqs, mask)
        mq, m_scale = kernels.rmsnorm_quant(
            x, self.postnorm.scale, self.postnorm.eps, postscale, postshift
        )
        x = x + postgate * self.mlp(mq, m_scale)
        return x


class FP8TextFusionBlock(TextFusionBlock):
    """mmdit.TextFusionBlock with RMSNorm fused into the fp8 quantization."""

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        aq, a_scale = kernels.rmsnorm_quant(x, self.prenorm.scale, self.prenorm.eps)
        x = x + self.attn(aq, a_scale, None, mask)
        mq, m_scale = kernels.rmsnorm_quant(x, self.postnorm.scale, self.postnorm.eps)
        x = x + self.mlp(mq, m_scale)
        return x


class FP8TimestepMLP(nn.Sequential):
    """tmlp (Linear, GELU, Linear) with GELU fused into the second quantization."""

    def forward(self, x: Tensor) -> Tensor:
        h = self[0](x)
        hq, h_scale = kernels.gelu_quant(h)
        return self[2].forward_quantized(hq, h_scale)


class FP8TextMLP(nn.Sequential):
    """txtmlp (RMSNorm, Linear, GELU, Linear) with the norm and GELU fused
    into the fp8 quantizations."""

    def forward(self, x: Tensor) -> Tensor:
        xq, x_scale = kernels.rmsnorm_quant(x, self[0].scale, self[0].eps)
        h = self[1].forward_quantized(xq, x_scale)
        hq, h_scale = kernels.gelu_quant(h)
        return self[3].forward_quantized(hq, h_scale)


class FP8TimestepProj(nn.Sequential):
    """tproj (GELU, Linear) with GELU fused into the quantization."""

    def forward(self, x: Tensor) -> Tensor:
        xq, x_scale = kernels.gelu_quant(x)
        return self[1].forward_quantized(xq, x_scale)


def _all_fp8(*modules: nn.Module) -> bool:
    return all(isinstance(m, FP8Linear) for m in modules)


def apply_fused_forwards(model: nn.Module) -> nn.Module:
    """Rebind a SingleStreamDiT's block forwards to the fused fp8 versions,
    module by module: only modules whose linears are all FP8Linear are
    swapped; anything left in bf16 (via swap skips) keeps its original
    forward. Pure class swap — no parameters move and the state-dict layout
    is unchanged."""
    for module in model.modules():
        kind = type(module)
        if kind is Attention and _all_fp8(
            module.wq, module.wk, module.wv, module.gate, module.wo
        ):
            module.__class__ = FP8Attention
        elif kind is SwiGLU and _all_fp8(module.gate, module.up, module.down):
            module.__class__ = FP8SwiGLU
    for module in model.modules():
        kind = type(module)
        fused_children = (
            isinstance(module, nn.Module)
            and isinstance(getattr(module, "attn", None), FP8Attention)
            and isinstance(getattr(module, "mlp", None), FP8SwiGLU)
        )
        if kind is SingleStreamBlock and fused_children:
            module.__class__ = FP8SingleStreamBlock
        elif kind is TextFusionBlock and fused_children:
            module.__class__ = FP8TextFusionBlock
    if _all_fp8(model.tmlp[0], model.tmlp[2]):
        model.tmlp.__class__ = FP8TimestepMLP
    if _all_fp8(model.txtmlp[1], model.txtmlp[3]):
        model.txtmlp.__class__ = FP8TextMLP
    if _all_fp8(model.tproj[1]):
        model.tproj.__class__ = FP8TimestepProj
    return model
