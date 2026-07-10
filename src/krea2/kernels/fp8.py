"""Fused producer + rowwise FP8 quantization Triton kernels for K2 inference.

Every linear layer running in FP8 needs its input dynamically quantized with
per-row (per-token) scales. Doing that as separate eager ops costs several
extra passes over the activation. Each kernel here fuses the elementwise op
that *produces* a linear's input with the quantization itself, writing the
fp8 e4m3 tensor and the f32 row scales in one pass:

  - rmsnorm_quant:     RMSNorm (+ optional DiT scale/shift modulation) + quant
  - gelu_quant:        tanh-approx GELU + quant
  - silu_mul_quant:    silu(gate) * up (SwiGLU inner op) + quant
  - mul_sigmoid_quant: x * sigmoid(gate) (attention out-gate) + quant
  - rowwise_quant:     plain quantization (no producer)
  - rowwise_dequant_:  GEMM epilogue: y *= xs[row] * ws[col] (+ bias), in place.
                       Used because on SM 8.9 torch's RowWise scaled_mm kernel is
                       slower than bf16; running scaled_mm with unit scales and
                       dequantizing here keeps rowwise numerics at full fp8 speed.

All kernels compute in f32 and expect row-major inputs (last dim contiguous).
Scales follow the same convention as torch._scaled_mm / F.scaled_mm rowwise
scaling: real_value = q.float() * scale, scale = amax(|row|) / 448.

The module lives under ``krea2.kernels`` so its normal package import cannot
shadow the third-party ``triton`` package used below.
"""

import torch
import triton.language as tl

import triton

FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = float(torch.finfo(FP8_DTYPE).max)  # 448.0
SCALE_EPS = 1e-12

# Producer op codes for _fused_quant_kernel.
_OP_IDENTITY = 0
_OP_GELU_TANH = 1
_OP_SILU_MUL = 2
_OP_MUL_SIGMOID = 3

# Kernel-visible constants (triton only allows tl.constexpr(...) globals).
_FMAX = tl.constexpr(FP8_MAX)
_EPS = tl.constexpr(SCALE_EPS)
# 2 * sqrt(2 / pi), for the tanh GELU written in sigmoid form:
# 0.5 * x * (1 + tanh(z)) == x * sigmoid(2 * z)
_GELU_2C = tl.constexpr(1.5957691216057308)


@triton.jit
def _produce(a, b, OP: tl.constexpr):
    if OP == 0:  # identity
        return a
    elif OP == 1:  # GELU(approximate="tanh")
        return a * tl.sigmoid(_GELU_2C * (a + 0.044715 * a * a * a))
    elif OP == 2:  # silu(a) * b
        return a * tl.sigmoid(a) * b
    else:  # a * sigmoid(b)
        return a * tl.sigmoid(b)


@triton.jit
def _fused_quant_kernel(
    A,
    B,
    Q,
    S,
    D,
    OP: tl.constexpr,
    HAS_B: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    """One program per row: q[row] = fp8(op(a, b)[row] / s), s = amax / 448."""
    row = tl.program_id(0).to(tl.int64)
    a_row = A + row * D
    b_row = B + row * D
    q_row = Q + row * D

    if NCHUNK == 1:
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        a = tl.load(a_row + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_B:
            b = tl.load(b_row + cols, mask=mask, other=0.0).to(tl.float32)
        else:
            b = a
        v = _produce(a, b, OP)
        scale = tl.maximum(tl.max(tl.abs(v)), _EPS) / _FMAX
        q = tl.minimum(tl.maximum(v / scale, -_FMAX), _FMAX)
        tl.store(q_row + cols, q.to(tl.float8e4nv), mask=mask)
    else:
        # Two passes with recompute: amax needs the whole row before any store.
        amax = 0.0
        for i in tl.static_range(NCHUNK):
            cols = i * BLOCK + tl.arange(0, BLOCK)
            mask = cols < D
            a = tl.load(a_row + cols, mask=mask, other=0.0).to(tl.float32)
            if HAS_B:
                b = tl.load(b_row + cols, mask=mask, other=0.0).to(tl.float32)
            else:
                b = a
            amax = tl.maximum(amax, tl.max(tl.abs(_produce(a, b, OP))))
        scale = tl.maximum(amax, _EPS) / _FMAX
        for i in tl.static_range(NCHUNK):
            cols = i * BLOCK + tl.arange(0, BLOCK)
            mask = cols < D
            a = tl.load(a_row + cols, mask=mask, other=0.0).to(tl.float32)
            if HAS_B:
                b = tl.load(b_row + cols, mask=mask, other=0.0).to(tl.float32)
            else:
                b = a
            v = _produce(a, b, OP)
            q = tl.minimum(tl.maximum(v / scale, -_FMAX), _FMAX)
            tl.store(q_row + cols, q.to(tl.float8e4nv), mask=mask)
    tl.store(S + row, scale)


@triton.jit
def _rmsnorm_quant_kernel(
    X,
    W,
    MSC,
    MSH,
    Q,
    S,
    D,
    mod_stride,
    rows_per_batch,
    eps,
    HAS_MOD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """q[row] = fp8(((1 + msc[b]) * rmsnorm(x[row]) * (w + 1) + msh[b]) / s).

    Matches mmdit.RMSNorm (weight stored as scale, applied as scale + 1) and
    the (1 + scale) * x + shift modulation of SingleStreamBlock; msc/msh are
    per-batch vectors, b = row // rows_per_batch.
    """
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * D + cols, mask=mask, other=0.0).to(tl.float32)
    rms = tl.rsqrt(tl.sum(x * x) / D + eps)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32) + 1.0
    y = x * rms * w
    if HAS_MOD:
        b = row // rows_per_batch
        msc = tl.load(MSC + b * mod_stride + cols, mask=mask, other=0.0).to(tl.float32)
        msh = tl.load(MSH + b * mod_stride + cols, mask=mask, other=0.0).to(tl.float32)
        y = (1.0 + msc) * y + msh

    scale = tl.maximum(tl.max(tl.abs(y)), _EPS) / _FMAX
    q = tl.minimum(tl.maximum(y / scale, -_FMAX), _FMAX)
    tl.store(Q + row * D + cols, q.to(tl.float8e4nv), mask=mask)
    tl.store(S + row, scale)


@triton.jit
def _rowwise_dequant_kernel(
    Y,
    XS,
    WS,
    BIAS,
    N,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """In place: y[row, col] = y[row, col] * xs[row] * ws[col] (+ bias[col])."""
    pid = tl.program_id(0).to(tl.int64)
    nblocks = tl.cdiv(N, BLOCK)
    row = pid // nblocks
    cols = (pid % nblocks) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N

    ptr = Y + row * N + cols
    y = tl.load(ptr, mask=mask, other=0.0).to(tl.float32)
    xs = tl.load(XS + row)
    ws = tl.load(WS + cols, mask=mask, other=0.0)
    y = y * (xs * ws)
    if HAS_BIAS:
        y += tl.load(BIAS + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(ptr, y.to(Y.dtype.element_ty), mask=mask)


def _flatten(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1, x.shape[-1])
    return x if x.stride(-1) == 1 else x.contiguous()


def _launch_fused(op: int, a: torch.Tensor, b: torch.Tensor | None):
    a2 = _flatten(a)
    if b is not None:
        assert b.shape == a.shape, f"{b.shape} != {a.shape}"
        b2 = _flatten(b)
    else:
        b2 = a2
    m, d = a2.shape
    q = torch.empty(m, d, device=a2.device, dtype=FP8_DTYPE)
    s = torch.empty(m, 1, device=a2.device, dtype=torch.float32)
    block = min(triton.next_power_of_2(d), 8192)
    _fused_quant_kernel[(m,)](
        a2,
        b2,
        q,
        s,
        d,
        OP=op,
        HAS_B=b is not None,
        BLOCK=block,
        NCHUNK=triton.cdiv(d, block),
        num_warps=8 if block >= 4096 else 4,
    )
    return q.view(*a.shape), s.view(*a.shape[:-1], 1)


def rowwise_quant(x: torch.Tensor):
    """x -> (fp8, row scales)."""
    return _launch_fused(_OP_IDENTITY, x, None)


def gelu_quant(x: torch.Tensor):
    """GELU(approximate="tanh")(x) -> (fp8, row scales)."""
    return _launch_fused(_OP_GELU_TANH, x, None)


def silu_mul_quant(gate: torch.Tensor, up: torch.Tensor):
    """silu(gate) * up -> (fp8, row scales)."""
    return _launch_fused(_OP_SILU_MUL, gate, up)


def mul_sigmoid_quant(x: torch.Tensor, gate: torch.Tensor):
    """x * sigmoid(gate) -> (fp8, row scales)."""
    return _launch_fused(_OP_MUL_SIGMOID, x, gate)


def rmsnorm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    mod_scale: torch.Tensor | None = None,
    mod_shift: torch.Tensor | None = None,
):
    """(1 + mod_scale) * rmsnorm(x) * (weight + 1) + mod_shift -> (fp8, row scales).

    mod_scale/mod_shift are (B, 1, D) per-batch vectors (may be non-contiguous
    views, e.g. chunks of the tvec projection); x is (..., D) with the leading
    dims collapsing to B * rows_per_batch.
    """
    x2 = _flatten(x)
    m, d = x2.shape
    block = triton.next_power_of_2(d)
    assert block <= 16384, f"rmsnorm_quant supports D <= 16384, got {d}"
    assert weight.shape == (d,)

    has_mod = mod_scale is not None
    if has_mod:
        assert mod_shift is not None
        assert mod_scale.shape[-1] == d and mod_scale.stride(-1) == 1
        assert mod_shift.stride() == mod_scale.stride()
        assert m % mod_scale.shape[0] == 0
        rows_per_batch = m // mod_scale.shape[0]
        mod_stride = mod_scale.stride(0)
    else:
        mod_scale = mod_shift = weight  # unused dummy pointers
        rows_per_batch, mod_stride = m, 0

    q = torch.empty(m, d, device=x2.device, dtype=FP8_DTYPE)
    s = torch.empty(m, 1, device=x2.device, dtype=torch.float32)
    _rmsnorm_quant_kernel[(m,)](
        x2,
        weight,
        mod_scale,
        mod_shift,
        q,
        s,
        d,
        mod_stride,
        rows_per_batch,
        eps,
        HAS_MOD=has_mod,
        BLOCK=block,
        num_warps=8 if block >= 4096 else 4,
    )
    return q.view(*x.shape), s.view(*x.shape[:-1], 1)


def rowwise_dequant_(
    y: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """In place: y * x_scale (per row) * w_scale (per col) + bias. Returns y."""
    assert y.dim() == 2 and y.stride(-1) == 1
    m, n = y.shape
    assert x_scale.numel() == m and w_scale.numel() == n
    block = 1024
    _rowwise_dequant_kernel[(m * triton.cdiv(n, block),)](
        y,
        x_scale,
        w_scale,
        bias if bias is not None else w_scale,  # dummy pointer when unused
        n,
        HAS_BIAS=bias is not None,
        BLOCK=block,
    )
    return y


if __name__ == "__main__":
    # Self-test against eager torch references.
    import torch.nn.functional as F

    torch.manual_seed(0)
    dev = "cuda"

    def ref_quant(v):
        s = v.abs().amax(dim=-1, keepdim=True).clamp(min=SCALE_EPS) / FP8_MAX
        return (v / s).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE), s

    def check(name, got_q, got_s, ref):
        deq = got_q.float() * got_s
        err = (deq - ref).abs().max() / ref.abs().max()
        # fp8 e4m3 has 3 mantissa bits: elementwise rounding error <= 2^-4 * amax
        assert err < 2**-4, f"{name}: rel err {err:.4f}"
        rq, rs = ref_quant(ref)
        srel = (got_s - rs).abs().max() / rs.abs().max()
        print(
            f"{name:18s} dequant rel err {err.item():.5f}  "
            f"scale mismatch {srel.item():.2e}"
        )

    for d in (256, 2560, 6144, 6912, 9728, 16384):
        x = torch.randn(3, 517, d, device=dev, dtype=torch.bfloat16)
        g = torch.randn_like(x)
        xf, gf = x.float(), g.float()
        check(f"identity d={d}", *rowwise_quant(x), xf)
        check(f"gelu d={d}", *gelu_quant(x), F.gelu(xf, approximate="tanh"))
        check(f"silu_mul d={d}", *silu_mul_quant(x, g), F.silu(xf) * gf)
        check(f"mul_sigmoid d={d}", *mul_sigmoid_quant(x, g), xf * torch.sigmoid(gf))

    # rmsnorm (+ modulation), against mmdit's RMSNorm formula
    for d, mod in ((2560, False), (6144, True)):
        batch, length = 2, 517
        x = torch.randn(batch, length, d, device=dev, dtype=torch.bfloat16)
        w = torch.randn(d, device=dev, dtype=torch.bfloat16) * 0.1
        ref = F.rms_norm(x.float(), (d,), eps=1e-5, weight=w.float() + 1.0)
        args = ()
        if mod:
            vec = torch.randn(batch, 1, 6 * d, device=dev, dtype=torch.bfloat16)
            msc, msh = vec.chunk(6, dim=-1)[:2]  # non-contiguous views
            ref = (1.0 + msc.float()) * ref + msh.float()
            args = (msc, msh)
        check(f"rmsnorm mod={mod}", *rmsnorm_quant(x, w, 1e-5, *args), ref)

    # dequant epilogue
    for n, bias in ((6144, True), (16384, False)):
        m = 517
        y = torch.randn(m, n, device=dev, dtype=torch.bfloat16)
        xs = torch.rand(m, 1, device=dev) + 0.5
        ws = torch.rand(n, 1, device=dev) + 0.5
        bt = torch.randn(n, device=dev, dtype=torch.bfloat16) if bias else None
        ref = y.float() * xs * ws.t() + (bt.float() if bias else 0.0)
        got = rowwise_dequant_(y.clone(), xs, ws, bt).float()
        err = (got - ref).abs().max() / ref.abs().max()
        assert err < 1e-2, err
        print(f"dequant n={n} bias={bias}  rel err {err.item():.5f}")

    print("all kernel self-tests passed")
