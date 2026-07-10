"""Blockwise and rowwise INT8 quantization/GEMM kernels for K2.

The K2 INT8 path uses:

  - activations: int8 with one fp32 scale per 128x128 (M, K) block
  - weights: int8 with one fp32 scale per 128x128 (N, K) block
  - GEMM: C[bf16] = dequant(A[int8] @ B[int8].T) with matching 128x128
    activation and weight scales

The optional rowwise path instead uses one activation scale per flattened
token row and one scale for the complete weight tensor. Both forward and dX
backward stay on INT8 tensor cores; backward requantizes grad_output rowwise.

Scale convention:
    real_value = q.float() * scale
    scale = amax(abs(values)) / 127

The module lives under ``krea2.kernels`` so its normal package import cannot
shadow the third-party ``triton`` package used below.
"""

import torch
import triton.language as tl
from torch.library import triton_op, wrap_triton

import triton
from triton import Config, autotune, cdiv, heuristics, jit

K_BLOCK = 128
M_BLOCK = 128
N_BLOCK = 128
INT8_MAX = 127.0
SCALE_EPS = 1e-8

# Kernel-visible constants (triton only allows tl.constexpr(...) globals).
_QMAX = tl.constexpr(INT8_MAX)
_EPS = tl.constexpr(SCALE_EPS)


@triton.jit
def _round_clamp_int8(x):
    q = tl.where(x >= 0.0, x + 0.5, x - 0.5)
    return tl.minimum(tl.maximum(q, -_QMAX), _QMAX)


@triton.jit
def _rowwise_quant_kernel(
    X,
    Q,
    S,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    NCHUNK: tl.constexpr,
):
    """One program per row: q[row] = int8(x[row] / s), s = amax / 127."""
    row = tl.program_id(0).to(tl.int64)
    x_row = X + row * D
    q_row = Q + row * D

    if NCHUNK == 1:
        cols = tl.arange(0, BLOCK)
        mask = cols < D
        x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
        scale = tl.maximum(tl.max(tl.abs(x)), _EPS) / _QMAX
        q = _round_clamp_int8(x / scale)
        tl.store(q_row + cols, q.to(tl.int8), mask=mask)
    else:
        amax = 0.0
        for i in tl.static_range(NCHUNK):
            cols = i * BLOCK + tl.arange(0, BLOCK)
            mask = cols < D
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            amax = tl.maximum(amax, tl.max(tl.abs(x)))
        scale = tl.maximum(amax, _EPS) / _QMAX
        for i in tl.static_range(NCHUNK):
            cols = i * BLOCK + tl.arange(0, BLOCK)
            mask = cols < D
            x = tl.load(x_row + cols, mask=mask, other=0.0).to(tl.float32)
            q = _round_clamp_int8(x / scale)
            tl.store(q_row + cols, q.to(tl.int8), mask=mask)
    tl.store(S + row, scale)


@triton.jit
def _weight_block_quant_kernel(
    W,
    Q,
    S,
    N: tl.constexpr,
    K: tl.constexpr,
    N_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
):
    """One program per 128x128 weight block; S shape is [N/128, K/128]."""
    nb = tl.program_id(0).to(tl.int64)
    kg = tl.program_id(1).to(tl.int64)
    rows = nb * N_BLOCK + tl.arange(0, N_BLOCK)
    cols = kg * K_BLOCK + tl.arange(0, K_BLOCK)

    w = tl.load(W + rows[:, None] * K + cols[None, :]).to(tl.float32)
    amax = tl.max(tl.max(tl.abs(w), axis=0), axis=0)
    scale = tl.maximum(amax, _EPS) / _QMAX
    q = _round_clamp_int8(w / scale)
    tl.store(Q + rows[:, None] * K + cols[None, :], q.to(tl.int8))
    tl.store(S + nb * (K // K_BLOCK) + kg, scale)


@triton.jit
def _activation_block_quant_kernel(
    X,
    Q,
    S,
    M: tl.constexpr,
    K: tl.constexpr,
    M_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
):
    """One program per activation 128x128 block; S shape is [ceil(M/128), K/128]."""
    mb = tl.program_id(0).to(tl.int64)
    kg = tl.program_id(1).to(tl.int64)
    rows = mb * M_BLOCK + tl.arange(0, M_BLOCK)
    cols = kg * K_BLOCK + tl.arange(0, K_BLOCK)
    mask = rows[:, None] < M

    x = tl.load(X + rows[:, None] * K + cols[None, :], mask=mask, other=0.0).to(
        tl.float32
    )
    amax = tl.max(tl.max(tl.abs(x), axis=0), axis=0)
    scale = tl.maximum(amax, _EPS) / _QMAX
    q = _round_clamp_int8(x / scale)
    tl.store(Q + rows[:, None] * K + cols[None, :], q.to(tl.int8), mask=mask)
    tl.store(S + mb * (K // K_BLOCK) + kg, scale)


@triton.jit
def _tensor_absmax_kernel(X, AMAX, NUMEL, BLOCK: tl.constexpr):
    """First pass of tensorwise weight quantization."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + offsets, mask=offsets < NUMEL, other=0.0).to(tl.float32)
    local_max = tl.max(tl.abs(x))
    tl.atomic_max(AMAX, local_max, sem="relaxed")


@triton.jit
def _tensor_quant_kernel(X, Q, SCALE, NUMEL, BLOCK: tl.constexpr):
    """Second pass of tensorwise weight quantization."""
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < NUMEL
    x = tl.load(X + offsets, mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(SCALE)
    q = _round_clamp_int8(x / scale)
    tl.store(Q + offsets, q.to(tl.int8), mask=mask)


def _blockwise_mm_configs():
    configs = []
    for bm, bn, ns, nw in [
        (128, 256, 3, 8),
        (256, 128, 3, 8),
        (128, 128, 3, 4),
        (128, 128, 4, 4),
        (128, 64, 4, 4),
        (64, 128, 4, 4),
        (64, 64, 4, 4),
        (64, 64, 2, 4),
        (32, 128, 4, 4),
    ]:
        configs.append(
            Config(
                {"BLOCK_M": bm, "BLOCK_N": bn, "GROUP_M": 8, "SPLIT_K": 1},
                num_stages=ns,
                num_warps=nw,
            )
        )
        for split_k in (2, 4, 8):
            configs.append(
                Config(
                    {
                        "BLOCK_M": bm,
                        "BLOCK_N": bn,
                        "GROUP_M": 8,
                        "SPLIT_K": split_k,
                    },
                    num_stages=ns,
                    num_warps=nw,
                )
            )
    return configs


def _rowwise_mm_configs():
    return [
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8},
            num_stages=3,
            num_warps=8,
        ),
        Config(
            {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
        Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8},
            num_stages=4,
            num_warps=4,
        ),
    ]


@jit
def _swizzle_tile(
    pid,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    return pid_m, pid_n


@autotune(
    configs=_blockwise_mm_configs(),
    key=[
        "M",
        "N",
        "K",
        "M_BLOCK",
        "K_BLOCK",
        "N_BLOCK",
        "stride_asm",
        "stride_asg",
        "stride_bsn",
        "stride_bsg",
    ],
)
@heuristics(
    {
        "EVEN_SPLIT_K": lambda args: (
            (args["K"] // args["K_BLOCK"]) % args["SPLIT_K"] == 0
        ),
    }
)
@jit
def _int8_block2d_kernel(
    A,
    B,
    C,
    ASCALE,
    BSCALE,
    BIAS,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_asg,
    stride_bsn,
    stride_bsg,
    M_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = _swizzle_tile(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    pid_k = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_k * K_BLOCK + tl.arange(0, K_BLOCK)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    m_mask = rm < M
    n_mask = rn < N
    groups = K // K_BLOCK

    facc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    g = pid_k
    for _ in range(0, tl.cdiv(K // K_BLOCK, SPLIT_K)):
        if EVEN_SPLIT_K:
            a = tl.load(A)
            b = tl.load(B)
            k_mask = True
        else:
            k_mask = g < groups
            a = tl.load(A, mask=k_mask, other=0)
            b = tl.load(B, mask=k_mask, other=0)
        iacc = tl.dot(a, b, out_dtype=tl.int32)
        a_s = tl.load(
            ASCALE + (rm // M_BLOCK) * stride_asm + g * stride_asg,
            mask=m_mask & k_mask,
            other=0.0,
        )
        b_s = tl.load(
            BSCALE + (rn // N_BLOCK) * stride_bsn + g * stride_bsg,
            mask=n_mask & k_mask,
            other=0.0,
        )
        facc += iacc.to(tl.float32) * a_s[:, None] * b_s[None, :]
        A += K_BLOCK * SPLIT_K * stride_ak
        B += K_BLOCK * SPLIT_K * stride_bk
        g += SPLIT_K

    c = facc
    if HAS_BIAS:
        bias = tl.load(BIAS + rn, mask=n_mask, other=0.0).to(tl.float32)
        if SPLIT_K == 1:
            c += bias[None, :]
        else:
            c += tl.where(pid_k == 0, bias[None, :], 0.0)

    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = m_mask[:, None] & n_mask[None, :]
    c = c.to(C.dtype.element_ty)
    if SPLIT_K == 1:
        tl.store(C, c, mask=mask)
    else:
        tl.atomic_add(C, c, sem="relaxed", mask=mask)


@autotune(
    configs=_blockwise_mm_configs(),
    key=[
        "M",
        "N",
        "K",
        "R",
        "M_BLOCK",
        "K_BLOCK",
        "N_BLOCK",
        "stride_asm",
        "stride_asg",
        "stride_bsn",
        "stride_bsg",
    ],
)
@heuristics(
    {
        "EVEN_SPLIT_K": lambda args: (
            (args["K"] // args["K_BLOCK"]) % args["SPLIT_K"] == 0
        ),
    }
)
@jit
def _int8_block2d_lora_kernel(
    A,
    B,
    C,
    ASCALE,
    BSCALE,
    BIAS,
    O_LORA,
    LORA_B,
    LORA_SCALE: tl.constexpr,
    M,
    N,
    K,
    R: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_asg,
    stride_bsn,
    stride_bsg,
    stride_om,
    stride_or,
    stride_lbn,
    stride_lbr,
    M_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    N_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    EVEN_SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_m, pid_n = _swizzle_tile(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    pid_k = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_k * K_BLOCK + tl.arange(0, K_BLOCK)
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    m_mask = rm < M
    n_mask = rn < N
    groups = K // K_BLOCK

    facc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    g = pid_k
    for _ in range(0, tl.cdiv(K // K_BLOCK, SPLIT_K)):
        if EVEN_SPLIT_K:
            a = tl.load(A)
            b = tl.load(B)
            k_mask = True
        else:
            k_mask = g < groups
            a = tl.load(A, mask=k_mask, other=0)
            b = tl.load(B, mask=k_mask, other=0)
        iacc = tl.dot(a, b, out_dtype=tl.int32)
        a_s = tl.load(
            ASCALE + (rm // M_BLOCK) * stride_asm + g * stride_asg,
            mask=m_mask & k_mask,
            other=0.0,
        )
        b_s = tl.load(
            BSCALE + (rn // N_BLOCK) * stride_bsn + g * stride_bsg,
            mask=n_mask & k_mask,
            other=0.0,
        )
        facc += iacc.to(tl.float32) * a_s[:, None] * b_s[None, :]
        A += K_BLOCK * SPLIT_K * stride_ak
        B += K_BLOCK * SPLIT_K * stride_bk
        g += SPLIT_K

    c = facc
    add_epilogue = SPLIT_K == 1 or pid_k == 0
    if HAS_BIAS:
        bias = tl.load(BIAS + rn, mask=n_mask, other=0.0).to(tl.float32)
        c += tl.where(add_epilogue, bias[None, :], 0.0)
    if add_epilogue:
        rr = tl.arange(0, R)
        o = tl.load(
            O_LORA + rm[:, None] * stride_om + rr[None, :] * stride_or,
            mask=m_mask[:, None],
            other=0.0,
        )
        lb = tl.load(
            LORA_B + rn[None, :] * stride_lbn + rr[:, None] * stride_lbr,
            mask=n_mask[None, :],
            other=0.0,
        )
        c += tl.dot(o, lb, out_dtype=tl.float32) * LORA_SCALE

    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = m_mask[:, None] & n_mask[None, :]
    c = c.to(C.dtype.element_ty)
    if SPLIT_K == 1:
        tl.store(C, c, mask=mask)
    else:
        tl.atomic_add(C, c, sem="relaxed", mask=mask)


@autotune(
    configs=_rowwise_mm_configs(),
    key=["M", "N", "K", "stride_bk", "stride_bn"],
)
@jit
def _int8_rowwise_tensor_weight_kernel(
    A,
    B,
    C,
    ASCALE,
    BSCALE,
    BIAS,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Rowwise A / tensorwise B INT8 GEMM with fused BF16 epilogue."""
    pid_m, pid_n = _swizzle_tile(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        remaining = K - k0 * BLOCK_K
        a = tl.load(a_ptrs, mask=rk[None, :] < remaining, other=0)
        b = tl.load(b_ptrs, mask=rk[:, None] < remaining, other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    a_s = tl.load(ASCALE + rm, mask=rm < M, other=0.0)
    c = acc.to(tl.float32) * a_s[:, None] * tl.load(BSCALE)
    if HAS_BIAS:
        bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
        c += bias[None, :]
    C = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, c.to(C.dtype.element_ty), mask=mask)


@autotune(
    configs=_rowwise_mm_configs(),
    key=["M", "N", "K", "R", "stride_bk", "stride_bn"],
)
@jit
def _int8_rowwise_tensor_weight_lora_kernel(
    A,
    B,
    C,
    ASCALE,
    BSCALE,
    BIAS,
    O_LORA,
    LORA_B,
    LORA_SCALE: tl.constexpr,
    M,
    N,
    K,
    R: tl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_om,
    stride_or,
    stride_lbn,
    stride_lbr,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """Rowwise/tensorwise INT8 GEMM with a fused LoRA output epilogue."""
    pid_m, pid_n = _swizzle_tile(tl.program_id(0), M, N, BLOCK_M, BLOCK_N, GROUP_M)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    a_ptrs = A + ram[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = B + rk[:, None] * stride_bk + rbn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k0 in range(0, tl.cdiv(K, BLOCK_K)):
        remaining = K - k0 * BLOCK_K
        a = tl.load(a_ptrs, mask=rk[None, :] < remaining, other=0)
        b = tl.load(b_ptrs, mask=rk[:, None] < remaining, other=0)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    a_s = tl.load(ASCALE + rm, mask=rm < M, other=0.0)
    c = acc.to(tl.float32) * a_s[:, None] * tl.load(BSCALE)
    if HAS_BIAS:
        bias = tl.load(BIAS + rn, mask=rn < N, other=0.0).to(tl.float32)
        c += bias[None, :]

    rr = tl.arange(0, R)
    o = tl.load(
        O_LORA + rm[:, None] * stride_om + rr[None, :] * stride_or,
        mask=(rm < M)[:, None],
        other=0.0,
    )
    lb = tl.load(
        LORA_B + rn[None, :] * stride_lbn + rr[:, None] * stride_lbr,
        mask=(rn < N)[None, :],
        other=0.0,
    )
    c += tl.dot(o, lb, out_dtype=tl.float32) * LORA_SCALE

    C = C + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(C, c.to(C.dtype.element_ty), mask=mask)


@triton.jit
def _lora_project_kernel(
    X,
    A,
    OUT,
    M: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    stride_xm,
    stride_xk,
    stride_ar,
    stride_ak,
    stride_om,
    stride_or,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rr = tl.arange(0, R)
    rk = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, R), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + rk
        x = tl.load(
            X + rm[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(rm[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        a = tl.load(
            A + rr[:, None] * stride_ar + k[None, :] * stride_ak,
            mask=k[None, :] < K,
            other=0.0,
        )
        acc += tl.dot(x, tl.trans(a), out_dtype=tl.float32)
    tl.store(
        OUT + rm[:, None] * stride_om + rr[None, :] * stride_or,
        acc.to(OUT.dtype.element_ty),
        mask=rm[:, None] < M,
    )


@triton.jit
def _int8_weight_t_dx_kernel(
    DY,
    WQ,
    WSCALE,
    DX,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_dym,
    stride_dyn,
    stride_wn,
    stride_wk,
    stride_wsn,
    stride_wsk,
    stride_dxm,
    stride_dxk,
    N_BLOCK: tl.constexpr,
    K_BLOCK: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rn = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n = n0 + rn
        dy = tl.load(
            DY + rm[:, None] * stride_dym + n[None, :] * stride_dyn,
            mask=(rm[:, None] < M) & (n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        wq = tl.load(
            WQ + n[:, None] * stride_wn + rk[None, :] * stride_wk,
            mask=(n[:, None] < N) & (rk[None, :] < K),
            other=0,
        ).to(tl.float32)
        ws = tl.load(
            WSCALE
            + (n[:, None] // N_BLOCK) * stride_wsn
            + (rk[None, :] // K_BLOCK) * stride_wsk,
            mask=(n[:, None] < N) & (rk[None, :] < K),
            other=0.0,
        )
        acc += tl.dot(dy, wq * ws, out_dtype=tl.float32)
    tl.store(
        DX + rm[:, None] * stride_dxm + rk[None, :] * stride_dxk,
        acc.to(DX.dtype.element_ty),
        mask=(rm[:, None] < M) & (rk[None, :] < K),
    )


@triton.jit
def _lora_down_kernel(
    DY,
    LORA_B,
    TMP,
    M: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    stride_dym,
    stride_dyn,
    stride_lbn,
    stride_lbr,
    stride_tm,
    stride_tr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = tl.arange(0, BLOCK_N)
    rr = tl.arange(0, R)
    acc = tl.zeros((BLOCK_M, R), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n = n0 + rn
        dy = tl.load(
            DY + rm[:, None] * stride_dym + n[None, :] * stride_dyn,
            mask=(rm[:, None] < M) & (n[None, :] < N),
            other=0.0,
        ).to(tl.float32)
        lb = tl.load(
            LORA_B + n[:, None] * stride_lbn + rr[None, :] * stride_lbr,
            mask=n[:, None] < N,
            other=0.0,
        ).to(tl.float32)
        acc += tl.dot(dy, lb, out_dtype=tl.float32)
    tl.store(
        TMP + rm[:, None] * stride_tm + rr[None, :] * stride_tr,
        acc,
        mask=rm[:, None] < M,
    )


@triton.jit
def _lora_dx_add_kernel(
    TMP,
    LORA_A,
    DX,
    LORA_SCALE: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    stride_tm,
    stride_tr,
    stride_ar,
    stride_ak,
    stride_dxm,
    stride_dxk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_k = tl.program_id(1).to(tl.int64)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rr = tl.arange(0, R)
    tmp = tl.load(
        TMP + rm[:, None] * stride_tm + rr[None, :] * stride_tr,
        mask=rm[:, None] < M,
        other=0.0,
    )
    la = tl.load(
        LORA_A + rr[:, None] * stride_ar + rk[None, :] * stride_ak,
        mask=rk[None, :] < K,
        other=0.0,
    ).to(tl.float32)
    add = tl.dot(tmp, la, out_dtype=tl.float32) * LORA_SCALE
    old = tl.load(
        DX + rm[:, None] * stride_dxm + rk[None, :] * stride_dxk,
        mask=(rm[:, None] < M) & (rk[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    tl.store(
        DX + rm[:, None] * stride_dxm + rk[None, :] * stride_dxk,
        (old + add).to(DX.dtype.element_ty),
        mask=(rm[:, None] < M) & (rk[None, :] < K),
    )


@triton.jit
def _lora_dA_kernel(
    TMP,
    X,
    DA,
    LORA_SCALE: tl.constexpr,
    M: tl.constexpr,
    K: tl.constexpr,
    R: tl.constexpr,
    stride_tm,
    stride_tr,
    stride_xm,
    stride_xk,
    stride_dar,
    stride_dak,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_k = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    rk = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rr = tl.arange(0, R)
    tmp = tl.load(
        TMP + rm[None, :] * stride_tm + rr[:, None] * stride_tr,
        mask=rm[None, :] < M,
        other=0.0,
    )
    x = tl.load(
        X + rm[:, None] * stride_xm + rk[None, :] * stride_xk,
        mask=(rm[:, None] < M) & (rk[None, :] < K),
        other=0.0,
    ).to(tl.float32)
    acc = tl.dot(tmp, x, out_dtype=tl.float32) * LORA_SCALE
    tl.atomic_add(
        DA + rr[:, None] * stride_dar + rk[None, :] * stride_dak,
        acc,
        sem="relaxed",
        mask=rk[None, :] < K,
    )


@triton.jit
def _lora_dB_kernel(
    DY,
    LORA_OUT,
    DB,
    LORA_SCALE: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    R: tl.constexpr,
    stride_dym,
    stride_dyn,
    stride_om,
    stride_or,
    stride_dbn,
    stride_dbr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0).to(tl.int64)
    pid_m = tl.program_id(1).to(tl.int64)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rr = tl.arange(0, R)
    dy = tl.load(
        DY + rm[None, :] * stride_dym + rn[:, None] * stride_dyn,
        mask=(rn[:, None] < N) & (rm[None, :] < M),
        other=0.0,
    ).to(tl.float32)
    o = tl.load(
        LORA_OUT + rm[:, None] * stride_om + rr[None, :] * stride_or,
        mask=rm[:, None] < M,
        other=0.0,
    ).to(tl.float32)
    acc = tl.dot(dy, o, out_dtype=tl.float32) * LORA_SCALE
    tl.atomic_add(
        DB + rn[:, None] * stride_dbn + rr[None, :] * stride_dbr,
        acc,
        sem="relaxed",
        mask=rn[:, None] < N,
    )


def _flatten(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(-1, x.shape[-1])
    return x if x.stride(-1) == 1 else x.contiguous()


def _quantize_int8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q = torch.round(x.float() / scale).clamp_(-INT8_MAX, INT8_MAX)
    return q.to(torch.int8)


def rowwise_quant(x: torch.Tensor):
    """x -> (int8, row scales)."""
    x2 = _flatten(x)
    m, d = x2.shape
    if not x2.is_cuda:
        scale = x2.float().abs().amax(dim=1).clamp_min(SCALE_EPS) / INT8_MAX
        q = _quantize_int8(x2, scale[:, None])
        return q.view(*x.shape), scale
    q, s = _rowwise_quant_op(x2)
    return q.view(*x.shape), s


@triton_op("krea2_int8::rowwise_quant", mutates_args={})
def _rowwise_quant_op(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, d = x.shape
    q = torch.empty((m, d), device=x.device, dtype=torch.int8)
    scale = torch.empty((m,), device=x.device, dtype=torch.float32)
    block = min(triton.next_power_of_2(d), 8192)
    wrap_triton(_rowwise_quant_kernel)[(m,)](
        x,
        q,
        scale,
        d,
        BLOCK=block,
        NCHUNK=cdiv(d, block),
        num_warps=8 if block >= 4096 else 4,
    )
    return q, scale


def weight_block_quant(x: torch.Tensor, n_block: int = N_BLOCK, k_block: int = K_BLOCK):
    """(N, K) weight -> int8 and fp32 scales shaped (N//128, K//128)."""
    x2 = _flatten(x)
    n, k = x2.shape
    assert n_block == N_BLOCK, f"K2 INT8 weights use n_block={N_BLOCK}"
    assert k_block == K_BLOCK, f"K2 INT8 weights use k_block={K_BLOCK}"
    assert n % n_block == 0, f"N={n} must be divisible by n_block={n_block}"
    assert k % k_block == 0, f"K={k} must be divisible by k_block={k_block}"

    if not x2.is_cuda:
        xb = x2.float().view(n // n_block, n_block, k // k_block, k_block)
        scale = xb.abs().amax(dim=(1, 3)).clamp_min(SCALE_EPS) / INT8_MAX
        q = _quantize_int8(xb, scale[:, None, :, None]).view(n, k)
        return q.view(*x.shape), scale

    q = torch.empty(n, k, device=x2.device, dtype=torch.int8)
    scale = torch.empty(
        n // n_block, k // k_block, device=x2.device, dtype=torch.float32
    )
    _weight_block_quant_kernel[(n // n_block, k // k_block)](
        x2,
        q,
        scale,
        n,
        k,
        N_BLOCK=n_block,
        K_BLOCK=k_block,
        num_warps=8,
    )
    return q.view(*x.shape), scale


def weight_tensorwise_quant(x: torch.Tensor):
    """Weight tensor -> INT8 with one FP32 scale, without a full FP32 copy."""
    x2 = _flatten(x)
    if not x2.is_cuda:
        scale = x2.float().abs().amax().clamp_min(SCALE_EPS) / INT8_MAX
        return _quantize_int8(x2, scale).view(*x.shape), scale.reshape(())

    numel = x2.numel()
    scale = torch.zeros((), device=x2.device, dtype=torch.float32)
    reduction_block = 4096
    _tensor_absmax_kernel[(cdiv(numel, reduction_block),)](
        x2,
        scale,
        numel,
        BLOCK=reduction_block,
        num_warps=8,
    )
    scale.div_(INT8_MAX).clamp_min_(SCALE_EPS)
    q = torch.empty_like(x2, dtype=torch.int8)
    # Keeping this tile modest avoids creating a very large SSA program while
    # still giving the one-time checkpoint conversion plenty of bandwidth.
    pointwise_block = 8192
    _tensor_quant_kernel[(cdiv(numel, pointwise_block),)](
        x2,
        q,
        scale,
        numel,
        BLOCK=pointwise_block,
        num_warps=8,
    )
    return q.view(*x.shape), scale


@triton_op("krea2_int8::activation_block_quant", mutates_args={})
def _activation_block_quant_op(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, k = x.shape
    mb = cdiv(m, M_BLOCK)
    g = k // K_BLOCK
    q = torch.empty((m, k), device=x.device, dtype=torch.int8)
    scale = torch.empty((mb, g), device=x.device, dtype=torch.float32)
    wrap_triton(_activation_block_quant_kernel)[(mb, g)](
        x,
        q,
        scale,
        m,
        k,
        M_BLOCK=M_BLOCK,
        K_BLOCK=K_BLOCK,
        num_warps=8,
    )
    return q, scale


def activation_block_quant(
    x: torch.Tensor, m_block: int = M_BLOCK, k_block: int = K_BLOCK
):
    """(..., K) activation -> int8 and fp32 scales shaped (ceil(M/128), K/128).

    The last dimension is K; all leading dimensions are flattened into M.
    """
    x2 = _flatten(x)
    m, k = x2.shape
    assert m_block == M_BLOCK, f"K2 INT8 activations use m_block={M_BLOCK}"
    assert k_block == K_BLOCK, f"K2 INT8 activations use k_block={K_BLOCK}"
    assert k % k_block == 0, f"K={k} must be divisible by k_block={k_block}"
    mb = cdiv(m, m_block)
    g = k // k_block

    if not x2.is_cuda:
        xp = torch.zeros(mb * m_block, k, device=x2.device, dtype=torch.float32)
        xp[:m] = x2.float()
        xb = xp.view(mb, m_block, g, k_block)
        scale = xb.abs().amax(dim=(1, 3)).clamp_min(SCALE_EPS) / INT8_MAX
        qp = _quantize_int8(xb, scale[:, None, :, None]).view(mb * m_block, k)
        return qp[:m].contiguous().view(*x.shape), scale

    q, scale = _activation_block_quant_op(x2)
    return q.view(*x.shape), scale


def _check_common(a, b, b_scale):
    assert a.dtype == torch.int8 and b.dtype == torch.int8
    assert a.dim() == 2 and b.dim() == 2 and a.shape[1] == b.shape[0], (
        f"incompatible {tuple(a.shape)} @ {tuple(b.shape)}"
    )
    assert a.stride(1) == 1, "A must be row-major (K contiguous)"
    assert b.stride(0) == 1 or b.stride(1) == 1
    m, k = a.shape
    n = b.shape[1]
    return m, n, k


def _grid(m, n):
    return lambda meta: (
        cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]),
        meta["SPLIT_K"],
    )


def _rowwise_grid(m, n):
    return lambda meta: (cdiv(m, meta["BLOCK_M"]) * cdiv(n, meta["BLOCK_N"]),)


@triton_op("krea2_int8::matmul_int8_block2d", mutates_args={})
def _matmul_int8_block2d_op(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    c = torch.empty((m, n), device=a.device, dtype=out_dtype)
    c.zero_()
    wrap_triton(_int8_block2d_kernel)[_grid(m, n)](
        a,
        b,
        c,
        a_scale,
        b_scale,
        bias if bias is not None else b_scale,
        m,
        n,
        k,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        a_scale.stride(0),
        a_scale.stride(1),
        b_scale.stride(0),
        b_scale.stride(1),
        M_BLOCK=M_BLOCK,
        K_BLOCK=K_BLOCK,
        N_BLOCK=N_BLOCK,
        HAS_BIAS=bias is not None,
    )
    return c


def matmul_int8_block2d(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    m_block: int = M_BLOCK,
    k_block: int = K_BLOCK,
    n_block: int = N_BLOCK,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """C = dequant(A @ B) with 128x128 activation and weight scales.

    a_scale is shaped (ceil(M / 128), K / 128). b_scale is shaped
    (N / 128, K / 128), matching a row-major (N, K) weight tensor.
    """
    m, n, k = _check_common(a, b, b_scale)
    assert m_block == M_BLOCK, f"K2 INT8 activations use m_block={M_BLOCK}"
    assert k_block == K_BLOCK, f"K2 INT8 inference uses k_block={K_BLOCK}"
    assert n_block == N_BLOCK, f"K2 INT8 weights use n_block={N_BLOCK}"
    assert k % k_block == 0, f"K={k} must be a multiple of k_block={k_block}"
    assert n % n_block == 0, f"N={n} must be a multiple of n_block={n_block}"
    g = k // k_block
    assert a_scale.shape == (cdiv(m, m_block), g) and a_scale.dtype == torch.float32
    assert b_scale.shape == (n // n_block, g) and b_scale.dtype == torch.float32
    if bias is not None:
        assert bias.shape == (n,), (
            f"bias must have shape ({n},), got {tuple(bias.shape)}"
        )

    return _matmul_int8_block2d_op(a, b, a_scale, b_scale, bias, out_dtype)


def _dequant_weight_blocks(wq: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    n, k = wq.shape
    scale = w_scale.repeat_interleave(N_BLOCK, dim=0).repeat_interleave(K_BLOCK, dim=1)
    return wq.float() * scale[:n, :k]


@triton_op("krea2_int8::lora_project", mutates_args={})
def _lora_project_op(x: torch.Tensor, lora_a_bf16: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    r = lora_a_bf16.shape[0]
    o = torch.empty((m, r), device=x.device, dtype=torch.bfloat16)
    wrap_triton(_lora_project_kernel)[(cdiv(m, 32),)](
        x,
        lora_a_bf16,
        o,
        m,
        k,
        r,
        x.stride(0),
        x.stride(1),
        lora_a_bf16.stride(0),
        lora_a_bf16.stride(1),
        o.stride(0),
        o.stride(1),
        BLOCK_M=32,
        BLOCK_K=64,
        num_warps=4,
    )
    return o


@triton_op("krea2_int8::mm_fused_lora", mutates_args={})
def _int8_mm_fused_lora_op(
    x: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m, k = x.shape
    n = wq.shape[0]
    r = lora_a.shape[0]

    if not x.is_cuda:
        w = _dequant_weight_blocks(wq, w_scale)
        out = x.float() @ w.t()
        if bias is not None:
            out = out + bias.float()
        a = lora_a.to(torch.bfloat16)
        b = lora_b.to(torch.bfloat16)
        o = x.to(torch.bfloat16) @ a.t()
        out = out + float(lora_scale) * (o @ b.t()).float()
        return out.to(out_dtype)

    xq, x_scale = _activation_block_quant_op(x)
    lora_a_bf16 = lora_a.to(torch.bfloat16).contiguous()
    lora_b_bf16 = lora_b.to(torch.bfloat16).contiguous()
    x_lora = x.to(torch.bfloat16).contiguous()
    o_lora = _lora_project_op(x_lora, lora_a_bf16)
    c = torch.empty((m, n), device=x.device, dtype=out_dtype)
    c.zero_()
    wrap_triton(_int8_block2d_lora_kernel)[_grid(m, n)](
        xq,
        wq.t(),
        c,
        x_scale,
        w_scale,
        bias if bias is not None else w_scale,
        o_lora,
        lora_b_bf16,
        float(lora_scale),
        m,
        n,
        k,
        r,
        xq.stride(0),
        xq.stride(1),
        wq.t().stride(0),
        wq.t().stride(1),
        c.stride(0),
        c.stride(1),
        x_scale.stride(0),
        x_scale.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        o_lora.stride(0),
        o_lora.stride(1),
        lora_b_bf16.stride(0),
        lora_b_bf16.stride(1),
        M_BLOCK=M_BLOCK,
        K_BLOCK=K_BLOCK,
        N_BLOCK=N_BLOCK,
        HAS_BIAS=bias is not None,
    )
    return c


@triton_op("krea2_int8::mm_fused_lora_backward", mutates_args={})
def _int8_mm_fused_lora_backward_op(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m, n = grad_out.shape
    k = x.shape[1]
    r = lora_a.shape[0]

    if not grad_out.is_cuda:
        w = _dequant_weight_blocks(wq, w_scale)
        go = grad_out.float()
        a = lora_a.to(torch.bfloat16)
        b = lora_b.to(torch.bfloat16)
        xb = x.to(torch.bfloat16)
        tmp = go.to(torch.bfloat16) @ b
        o = xb @ a.t()
        dx = go @ w + float(lora_scale) * (tmp.float() @ a.float())
        da = float(lora_scale) * (tmp.float().t() @ xb.float())
        db = float(lora_scale) * (go.to(torch.bfloat16).t().float() @ o.float())
        return dx.to(x.dtype), da, db

    go = grad_out if grad_out.stride(-1) == 1 else grad_out.contiguous()
    x2 = x if x.stride(-1) == 1 else x.contiguous()
    lora_a_bf16 = lora_a.to(torch.bfloat16).contiguous()
    lora_b_bf16 = lora_b.to(torch.bfloat16).contiguous()

    dx = torch.empty((m, k), device=x.device, dtype=x.dtype)
    wrap_triton(_int8_weight_t_dx_kernel)[(cdiv(m, 32), cdiv(k, 64))](
        go,
        wq,
        w_scale,
        dx,
        m,
        n,
        k,
        go.stride(0),
        go.stride(1),
        wq.stride(0),
        wq.stride(1),
        w_scale.stride(0),
        w_scale.stride(1),
        dx.stride(0),
        dx.stride(1),
        N_BLOCK=N_BLOCK,
        K_BLOCK=K_BLOCK,
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
    )

    tmp = torch.empty((m, r), device=x.device, dtype=torch.float32)
    wrap_triton(_lora_down_kernel)[(cdiv(m, 32),)](
        go,
        lora_b_bf16,
        tmp,
        m,
        n,
        r,
        go.stride(0),
        go.stride(1),
        lora_b_bf16.stride(0),
        lora_b_bf16.stride(1),
        tmp.stride(0),
        tmp.stride(1),
        BLOCK_M=32,
        BLOCK_N=64,
        num_warps=4,
    )
    wrap_triton(_lora_dx_add_kernel)[(cdiv(m, 32), cdiv(k, 64))](
        tmp,
        lora_a_bf16,
        dx,
        float(lora_scale),
        m,
        k,
        r,
        tmp.stride(0),
        tmp.stride(1),
        lora_a_bf16.stride(0),
        lora_a_bf16.stride(1),
        dx.stride(0),
        dx.stride(1),
        BLOCK_M=32,
        BLOCK_K=64,
        num_warps=4,
    )

    da = torch.empty((r, k), device=x.device, dtype=torch.float32)
    da.zero_()
    xb = x2.to(torch.bfloat16).contiguous()
    wrap_triton(_lora_dA_kernel)[(cdiv(k, 64), cdiv(m, 64))](
        tmp,
        xb,
        da,
        float(lora_scale),
        m,
        k,
        r,
        tmp.stride(0),
        tmp.stride(1),
        xb.stride(0),
        xb.stride(1),
        da.stride(0),
        da.stride(1),
        BLOCK_M=64,
        BLOCK_K=64,
        num_warps=4,
    )

    o_lora = _lora_project_op(xb, lora_a_bf16)
    db = torch.empty((n, r), device=x.device, dtype=torch.float32)
    db.zero_()
    wrap_triton(_lora_dB_kernel)[(cdiv(n, 32), cdiv(m, 64))](
        go,
        o_lora,
        db,
        float(lora_scale),
        m,
        n,
        r,
        go.stride(0),
        go.stride(1),
        o_lora.stride(0),
        o_lora.stride(1),
        db.stride(0),
        db.stride(1),
        BLOCK_M=64,
        BLOCK_N=32,
        num_warps=4,
    )
    return dx, da, db


def _int8_mm_fused_lora_setup_context(ctx, inputs, output):
    x, wq, w_scale, bias, lora_a, lora_b, lora_scale, out_dtype = inputs
    ctx.save_for_backward(x, wq, w_scale, lora_a, lora_b)
    ctx.lora_scale = float(lora_scale)


def _int8_mm_fused_lora_backward(ctx, grad_out):
    x, wq, w_scale, lora_a, lora_b = ctx.saved_tensors
    dx, da, db = _int8_mm_fused_lora_backward_op(
        grad_out.contiguous(),
        x,
        wq,
        w_scale,
        lora_a,
        lora_b,
        ctx.lora_scale,
    )
    return dx, None, None, None, da, db, None, None


_int8_mm_fused_lora_op.register_autograd(
    _int8_mm_fused_lora_backward,
    setup_context=_int8_mm_fused_lora_setup_context,
)


def int8_mm_fused_lora(
    x: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_scale: float,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """bf16/fp32 X @ int8 W.T + LoRA, with custom backward.

    LoRA parameters are stored as fp32 master weights and cast to bf16 inside
    the forward/backward kernels.
    """
    x2 = _flatten(x)
    m, k = x2.shape
    n = wq.shape[0]
    r = lora_a.shape[0]
    assert r in (32, 64), f"supported LoRA ranks are 32 and 64, got {r}"
    assert lora_a.shape == (r, k)
    assert lora_b.shape == (n, r)
    assert wq.shape == (n, k)
    assert w_scale.shape == (n // N_BLOCK, k // K_BLOCK)
    if bias is not None:
        assert bias.shape == (n,)
    y = _int8_mm_fused_lora_op(
        x2, wq, w_scale, bias, lora_a, lora_b, float(lora_scale), out_dtype
    )
    return y.view(*x.shape[:-1], n)


def _dequant_weight_tensorwise(wq: torch.Tensor, w_scale: torch.Tensor) -> torch.Tensor:
    return wq.float() * w_scale.float()


@triton_op("krea2_int8::mm_rowwise", mutates_args={})
def _int8_mm_rowwise_op(
    x: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Rowwise-X/tensorwise-W INT8 linear forward."""
    m, k = x.shape
    n = wq.shape[0]
    if not x.is_cuda:
        xq, x_scale = rowwise_quant(x)
        out = (xq.float() * x_scale[:, None]) @ _dequant_weight_tensorwise(
            wq, w_scale
        ).t()
        if bias is not None:
            out += bias.float()
        return out.to(out_dtype)

    xq, x_scale = _rowwise_quant_op(x)
    c = torch.empty((m, n), device=x.device, dtype=out_dtype)
    wt = wq.t()
    wrap_triton(_int8_rowwise_tensor_weight_kernel)[_rowwise_grid(m, n)](
        xq,
        wt,
        c,
        x_scale,
        w_scale,
        bias if bias is not None else w_scale,
        m,
        n,
        k,
        xq.stride(0),
        xq.stride(1),
        wt.stride(0),
        wt.stride(1),
        c.stride(0),
        c.stride(1),
        HAS_BIAS=bias is not None,
    )
    return c


@triton_op("krea2_int8::mm_rowwise_backward", mutates_args={})
def _int8_mm_rowwise_backward_op(
    grad_out: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
) -> torch.Tensor:
    """Approximate dX using a fresh rowwise quantization of grad_output."""
    m, n = grad_out.shape
    k = wq.shape[1]
    if not grad_out.is_cuda:
        gq, g_scale = rowwise_quant(grad_out)
        return (
            (gq.float() * g_scale[:, None]) @ _dequant_weight_tensorwise(wq, w_scale)
        ).to(grad_out.dtype)

    go = grad_out if grad_out.stride(-1) == 1 else grad_out.contiguous()
    gq, g_scale = _rowwise_quant_op(go)
    dx = torch.empty((m, k), device=go.device, dtype=go.dtype)
    # Here Wq is already laid out (reduction=N, output=K), so no transpose is
    # needed. The same tensor-core kernel serves forward and frozen-base dX.
    wrap_triton(_int8_rowwise_tensor_weight_kernel)[_rowwise_grid(m, k)](
        gq,
        wq,
        dx,
        g_scale,
        w_scale,
        w_scale,
        m,
        k,
        n,
        gq.stride(0),
        gq.stride(1),
        wq.stride(0),
        wq.stride(1),
        dx.stride(0),
        dx.stride(1),
        HAS_BIAS=False,
    )
    return dx


def _int8_mm_rowwise_setup_context(ctx, inputs, output):
    _x, wq, w_scale, _bias, _out_dtype = inputs
    ctx.save_for_backward(wq, w_scale)


def _int8_mm_rowwise_backward(ctx, grad_out):
    wq, w_scale = ctx.saved_tensors
    dx = _int8_mm_rowwise_backward_op(grad_out.contiguous(), wq, w_scale)
    return dx, None, None, None, None


_int8_mm_rowwise_op.register_autograd(
    _int8_mm_rowwise_backward,
    setup_context=_int8_mm_rowwise_setup_context,
)


def int8_mm_rowwise(
    x: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """bf16/fp32 X @ tensorwise-int8 W.T with rowwise X and dY quantization."""
    x2 = _flatten(x)
    n, k = wq.shape
    assert x2.shape[1] == k
    assert wq.dtype == torch.int8
    assert w_scale.dtype == torch.float32 and w_scale.numel() == 1
    if bias is not None:
        assert bias.shape == (n,)
    y = _int8_mm_rowwise_op(x2, wq, w_scale, bias, out_dtype)
    return y.view(*x.shape[:-1], n)


@triton_op("krea2_int8::mm_fused_lora_rowwise", mutates_args={})
def _int8_mm_fused_lora_rowwise_op(
    x: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_scale: float,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """Rowwise-X/tensorwise-W INT8 forward with a fused LoRA epilogue."""
    m, k = x.shape
    n = wq.shape[0]
    r = lora_a.shape[0]
    if not x.is_cuda:
        xq, x_scale = rowwise_quant(x)
        out = (xq.float() * x_scale[:, None]) @ _dequant_weight_tensorwise(
            wq, w_scale
        ).t()
        if bias is not None:
            out += bias.float()
        o = x.to(torch.bfloat16) @ lora_a.to(torch.bfloat16).t()
        out += float(lora_scale) * (o @ lora_b.to(torch.bfloat16).t()).float()
        return out.to(out_dtype)

    x2 = x if x.stride(-1) == 1 else x.contiguous()
    xq, x_scale = _rowwise_quant_op(x2)
    a = lora_a.to(torch.bfloat16).contiguous()
    b = lora_b.to(torch.bfloat16).contiguous()
    o = _lora_project_op(x2.to(torch.bfloat16), a)
    c = torch.empty((m, n), device=x.device, dtype=out_dtype)
    wt = wq.t()
    wrap_triton(_int8_rowwise_tensor_weight_lora_kernel)[_rowwise_grid(m, n)](
        xq,
        wt,
        c,
        x_scale,
        w_scale,
        bias if bias is not None else w_scale,
        o,
        b,
        float(lora_scale),
        m,
        n,
        k,
        r,
        xq.stride(0),
        xq.stride(1),
        wt.stride(0),
        wt.stride(1),
        c.stride(0),
        c.stride(1),
        o.stride(0),
        o.stride(1),
        b.stride(0),
        b.stride(1),
        HAS_BIAS=bias is not None,
    )
    return c


@triton_op("krea2_int8::mm_fused_lora_rowwise_backward", mutates_args={})
def _int8_mm_fused_lora_rowwise_backward_op(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused rowwise base dX plus BF16 LoRA dX/dA/dB."""
    m, n = grad_out.shape
    k = x.shape[1]
    if not grad_out.is_cuda:
        gq, g_scale = rowwise_quant(grad_out)
        go_deq = gq.float() * g_scale[:, None]
        go = grad_out.float()
        a = lora_a.to(torch.bfloat16)
        b = lora_b.to(torch.bfloat16)
        xb = x.to(torch.bfloat16)
        tmp = grad_out.to(torch.bfloat16) @ b
        o = xb @ a.t()
        dx = go_deq @ _dequant_weight_tensorwise(wq, w_scale)
        dx += float(lora_scale) * (tmp.float() @ a.float())
        da = float(lora_scale) * (tmp.float().t() @ xb.float())
        db = float(lora_scale) * (go.to(torch.bfloat16).t().float() @ o.float())
        return dx.to(x.dtype), da, db

    go = grad_out if grad_out.stride(-1) == 1 else grad_out.contiguous()
    x2 = x if x.stride(-1) == 1 else x.contiguous()
    a = lora_a.to(torch.bfloat16).contiguous()
    b = lora_b.to(torch.bfloat16).contiguous()
    gq, g_scale = _rowwise_quant_op(go)

    # Rank-32/64 LoRA GEMMs are faster through cuBLAS than the former atomic
    # Triton reductions on Ada. Keep their operands/results in BF16 until the
    # FP32 master-gradient boundary.
    tmp = torch.mm(go.to(torch.bfloat16), b)
    dx = torch.empty((m, k), device=x.device, dtype=x.dtype)
    wrap_triton(_int8_rowwise_tensor_weight_kernel)[_rowwise_grid(m, k)](
        gq,
        wq,
        dx,
        g_scale,
        w_scale,
        w_scale,
        m,
        k,
        n,
        gq.stride(0),
        gq.stride(1),
        wq.stride(0),
        wq.stride(1),
        dx.stride(0),
        dx.stride(1),
        HAS_BIAS=False,
    )
    dx.addmm_(tmp, a, beta=1.0, alpha=float(lora_scale))

    xb = x2.to(torch.bfloat16).contiguous()
    da = torch.mm(tmp.t(), xb).float().mul_(float(lora_scale))
    o = torch.mm(xb, a.t())
    db = torch.mm(go.to(torch.bfloat16).t(), o).float().mul_(float(lora_scale))
    return dx, da, db


def _int8_mm_fused_lora_rowwise_setup_context(ctx, inputs, output):
    x, wq, w_scale, _bias, lora_a, lora_b, lora_scale, _out_dtype = inputs
    ctx.save_for_backward(x, wq, w_scale, lora_a, lora_b)
    ctx.lora_scale = float(lora_scale)


def _int8_mm_fused_lora_rowwise_backward(ctx, grad_out):
    x, wq, w_scale, lora_a, lora_b = ctx.saved_tensors
    dx, da, db = _int8_mm_fused_lora_rowwise_backward_op(
        grad_out.contiguous(),
        x,
        wq,
        w_scale,
        lora_a,
        lora_b,
        ctx.lora_scale,
    )
    return dx, None, None, None, da, db, None, None


_int8_mm_fused_lora_rowwise_op.register_autograd(
    _int8_mm_fused_lora_rowwise_backward,
    setup_context=_int8_mm_fused_lora_rowwise_setup_context,
)


def int8_mm_fused_lora_rowwise(
    x: torch.Tensor,
    wq: torch.Tensor,
    w_scale: torch.Tensor,
    bias: torch.Tensor | None,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    lora_scale: float,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Rowwise-X/tensorwise-W INT8 linear + LoRA with custom backward."""
    x2 = _flatten(x)
    n, k = wq.shape
    r = lora_a.shape[0]
    assert r in (32, 64), f"supported LoRA ranks are 32 and 64, got {r}"
    assert x2.shape[1] == k and lora_a.shape == (r, k)
    assert lora_b.shape == (n, r)
    assert wq.dtype == torch.int8
    assert w_scale.dtype == torch.float32 and w_scale.numel() == 1
    if bias is not None:
        assert bias.shape == (n,)
    y = _int8_mm_fused_lora_rowwise_op(
        x2, wq, w_scale, bias, lora_a, lora_b, float(lora_scale), out_dtype
    )
    return y.view(*x.shape[:-1], n)
