# INT8 Inference

`scripts/inference_int8.py` runs Krea 2 with eligible linear layers quantized to INT8.
It mirrors the low-memory FP8 inference path while using the custom Triton INT8
GEMM kernels in `src/krea2/kernels/int8.py`.

```bash
uv run scripts/inference_int8.py "a fox walking in the snow" \
    --checkpoint oss_raw --steps 52 --cfg 3.5 --width 1024 --height 1024
```

Two quantization modes are available. Training defaults to `rowwise`; the
lower-level module and inference defaults remain `blockwise` for compatibility.

- `--quantization-type blockwise` uses 128x128
  activation `(M,K)` and weight `(N,K)` scales. Its GEMM autotunes split-K
  values 1, 2, 4, and 8.
- `--quantization-type rowwise` uses one FP32 scale per flattened activation
  row and one scalar FP32 scale for the complete weight tensor. It uses custom
  INT8 tensor-core kernels for both forward and frozen-base `dX`.
- Both modes write BF16 output and keep attention itself in BF16 torch SDPA.

The DiT keeps `first`, `last`, `tmlp`, `tproj`, and `txtmlp` in bf16. These
layers are quality-sensitive in the FP8 path as well, and they are not the
dominant GEMM cost. Attention and MLP linears inside the main DiT blocks and
text fusion blocks use INT8 when both their input and output dimensions are
divisible by 128.

The current INT8 inference path does not use fused norm/GELU/SwiGLU quantizers.
Each converted linear receives a bf16 input tensor, quantizes that input with a
plain activation block quantization kernel, multiplies by pre-quantized int8
weights, and returns a bf16 result.

The runtime activation quantization and GEMMs launch custom Triton kernels.
Triton compilation and autotuning happen on the first use of a new shape;
compiled artifacts are cached under `~/.cache/krea-2` by default. Override the
location with `KREA2_CACHE_DIR`.

By default, the Qwen text encoder also converts eligible linear layers to INT8.
Use `--bf16-text-encoder` to keep text encoder weights in bf16 and host-offload
them between encodes:

```bash
uv run scripts/inference_int8.py "a fox walking in the snow" \
    --checkpoint oss_raw --steps 52 --cfg 3.5 --bf16-text-encoder \
    --quantization-type rowwise
```

The implementation requires CUDA and Triton. The VAE remains bf16 and uses the
lean decode optimization from `vae_opt.py`.

## LoRA Inference

INT8 LoRA adapters saved by `scripts/train.py` can be loaded directly:

```bash
uv run scripts/inference_int8.py "triggerword portrait under soft window light" \
    --checkpoint oss_raw --width 512 --height 512 \
    --quantization-type rowwise \
    --lora runs/character_sft/lora_latest.safetensors
```

The loader inserts `LinearLoraINT8` into the same main DiT block linears used
during training, infers `rank`, `lora_alpha`, and saved `lora_scale` from the
adapter metadata, then loads the LoRA tensors. Use `--lora-scale` only as an
extra inference-time multiplier on top of the saved adapter scale:

```bash
uv run scripts/inference_int8.py "triggerword cinematic close-up" \
    --width 512 --height 512 \
    --lora runs/character_draft/lora_latest.safetensors \
    --lora-scale 0.8
```

Adapter tensor names are normalized on load, so checkpoints saved from a
training model with activation-checkpoint wrappers are compatible with the
unwrapped inference model.

## LoRA Training

`scripts/train.py` provides a 512x512 training path for FP32 LoRA adapters
on top of the frozen INT8 DiT. It supports DRaFT-K reward fine-tuning and
supervised flow-matching from an `image_path,prompt` CSV. LoRA is inserted only
inside the main MMDiT blocks, with ranks `32` and `64`.

See `draft_int8_lora.md` for commands, reward API, adapter format, and
kernel test instructions.

## Rowwise training kernel

For a flattened activation `X[M,K]` and row-major linear weight `W[N,K]`, the
rowwise mode stores:

```text
Xq[M,K] int8, X_scale[M] fp32
Wq[N,K] int8, W_scale scalar fp32
X ~= Xq * X_scale[:, None]
W ~= Wq * W_scale
```

Forward fuses INT8 GEMM dequantization, optional bias, and the rank-32/64 LoRA
output into one BF16 store. Backward does not requantize the saved forward
activation for the frozen base. It freshly quantizes `grad_output` rowwise and
runs `Gq @ Wq` on INT8 tensor cores. The saved BF16 activation is used only by
the LoRA `dA`; rank-small LoRA `dX`, `dA`, and `dB` use BF16 cuBLAS GEMMs and
return FP32 master gradients.
