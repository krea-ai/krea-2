# Inference

The CLI inference scripts show a progress bar during the denoising sampler:

- `scripts/inference.py`
- `scripts/inference_fp8.py`
- `scripts/inference_int8.py`

`scripts/inference_int8.py` also accepts LoRA adapters saved by
`scripts/train.py`:

```bash
uv run scripts/inference_int8.py \
  "triggerword portrait under soft window light" \
  --width 512 --height 512 \
  --lora runs/character_sft/lora_latest.safetensors
```

The adapter metadata supplies rank, alpha, and saved scale. `--lora-scale`
applies an extra inference-time multiplier.

For programmatic use, pass `progress=False` to `sample(...)` to disable the
progress bar.

After decoding, the sampler prints synchronized latencies:

- `initialization`: latent creation, text conditioning, mask/position setup, and timestep setup
- `warmup`: one discarded model forward at the first denoising timestep
- `denoising`: the measured Euler denoising loop after warmup
- `vae`: the `ae.decode(...)` call

For programmatic use, pass `report_latency=False` to `sample(...)` to suppress
the latency line.

The training compiler uses fixed-shape Inductor graphs without CUDA graph
capture. This avoids replay-lifetime hazards with activation checkpointing and
the custom autograd operators.
