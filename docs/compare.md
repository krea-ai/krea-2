# Controlled SFT versus DRaFT-K comparison

`scripts/compare.py` runs one controlled character experiment with a shared first 500
SFT updates. One branch receives 50 DRaFT-K updates; the other restores the
complete step-500 SFT state and continues to global step 1000.

```bash
export OSS_RAW=/models/krea2-raw.safetensors
export DEEPINFRA_KEY=your_key

uv run --extra train --extra face-reward scripts/compare.py \
  /path/to/character/images \
  --output-dir runs/comparisons/character_name \
  --trigger-word character_token
```

The trigger word is optional. The default experiment uses rank 32, batch size
1, 64 DRaFT training prompts, eight held-out evaluation prompts, 20 denoising
steps, and CFG 4.5.

## Fairness contract

- Both branches share the identical SFT updates from global steps 1 through
  500.
- The SFT continuation restores LoRA tensors, AdamW moments, Torch CPU/CUDA
  RNG, the cached SFT tensors, and the exact shuffled-data cursor.
- DRaFT-K starts from the same step-500 LoRA but intentionally uses a fresh
  AdamW optimizer.
- The eight evaluation prompts are excluded from DRaFT training.
- Every evaluation uses the same effective prompts, ordering, image seeds,
  resolution, sampler settings, and empty negative prompt.
- Counts are fixed at 500+50 versus 1000. Measured synchronized optimization
  time is reported rather than used as a stopping condition.

There are no intermediate validations. Images are generated only before SFT,
after shared SFT, after DRaFT-K, and after the continued SFT run. Per-step
DRaFT training images and the trainer's single final sample are disabled.

## Outputs

```text
output/
  data/
    dataset.csv
    all_prompts.txt
    draft_train_prompts.txt
    evaluation_prompts.txt
  shared_sft_500/
    lora_latest.safetensors
    training_state_step_000500.pt
    training_summary.json
    validation/step_000000/...
    validation/step_000500/...
  draft_50/
    lora_latest.safetensors
    training_summary.json
    validation/step_000050/...
  sft_1000/
    lora_latest.safetensors
    training_state_step_001000.pt
    training_summary.json
    validation/step_001000/...
  comparison_grid.png
  experiment_plan.json
  experiment_results.json
  metrics.csv
```

The annotated grid has `SFT 500 + DRaFT-K 50` on the first row and `SFT 1000`
on the second. Individual images and prompt sidecars remain available.

`experiment_results.json` reports synchronized optimization time, stage wall
time, latency percentiles, throughput, face-detection rate, detected-face
identity similarity, and the fallback-aware training reward. `metrics.csv`
contains per-image values.

Every training stage has a command/dependency signature. Re-running the same
command reuses complete stages; `--force` runs all three again. Training-state
files are deliberately self-contained and can be large because they contain
AdamW moments and cached text embeddings.

AVIF and valid extensionless images are supported, including the current
contents of `test_characters/`.
