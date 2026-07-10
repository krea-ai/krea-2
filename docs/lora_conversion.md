# ComfyUI LoRA conversion

The training script writes compact adapter keys in the native Krea 2 namespace:

```text
blocks.0.attn.wq.lora_A
blocks.0.attn.wq.lora_B
```

Krea 2 LoRAs loaded by ComfyUI use the native model path under a
`diffusion_model` prefix and PEFT-style parameter suffixes:

```text
diffusion_model.blocks.0.attn.wq.lora_A.weight
diffusion_model.blocks.0.attn.wq.lora_B.weight
```

Convert a training checkpoint with:

```bash
uv run scripts/convert_lora_to_comfyui.py \
  runs/character/draft/lora_latest.safetensors \
  runs/character/draft/character_comfyui.safetensors
```

The destination is optional. If omitted, the converter writes
`<source>_comfyui.safetensors` beside the source. BF16 is the default and
matches common Krea 2 ComfyUI LoRAs. `--dtype float16` and `--dtype float32`
are also available.

Our runtime applies each adapter as

```text
(lora_alpha / rank) * lora_scale * (X @ A^T @ B^T)
```

The reference ComfyUI file does not contain per-module `alpha` tensors. The
converter therefore folds the complete multiplier into every `lora_B` tensor.
This makes the converted delta identical when `lora_alpha` differs from rank
or `lora_scale` differs from one. The source values and folded multiplier are
recorded as provenance in `conversion_info`, not as active output scale fields.
For the normal rank-32, alpha-32, scale-1 configuration, the tensors are
unchanged apart from their dtype.

The converter validates every A/B pair, rank, matrix shape, output key, and
output dtype. It intentionally rejects other LoRA namespaces instead of
silently producing a partially loadable file. Training metadata that can
contain local paths or prompt files is not copied; model, rank, step, trigger,
and conversion metadata are retained. The converter only includes trained
modules, so current adapters contain the 28 main transformer blocks but do not
synthesize LoRAs for untrained text-fusion blocks.

Useful options:

```bash
# Validate and display the effective scale without writing.
uv run scripts/convert_lora_to_comfyui.py adapter.safetensors --dry-run

# Replace an existing converted file and set its displayed metadata name.
uv run scripts/convert_lora_to_comfyui.py adapter.safetensors output.safetensors \
  --overwrite --name my-character
```
