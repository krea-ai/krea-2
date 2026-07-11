# Controlled SFT versus DRaFT-LV comparison

`scripts/compare.py` runs one controlled character experiment with a shared first 500
SFT updates. One branch receives 60 DRaFT-LV updates; the other restores the
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
1, 64 DRaFT training prompts, ten held-out evaluation prompts, one LV sample,
a 12-step training sampler, 20-step evaluation, DRaFT LR 1e-4, CFG 4.5, and
seed 42. DRaFT optimizes QKVO LoRAs only. Every fourth update uses an
independent trajectory pair in place of the correlated LV auxiliary sample.

## Fairness contract

- Both branches share the identical SFT updates from global steps 1 through
  500.
- The SFT continuation restores LoRA tensors, AdamW moments, Torch CPU/CUDA
  RNG, the cached SFT tensors, and the exact shuffled-data cursor.
- DRaFT-LV starts from the same step-500 LoRA but intentionally uses a fresh
  AdamW optimizer.
- Frozen attention-gate and MLP LoRAs retain their exact SFT-500 values and are
  still present in the final adapter.
- The ten evaluation prompts are excluded from DRaFT training.
- Every evaluation uses the same effective prompts, ordering, image seeds,
  resolution, sampler settings, and empty negative prompt.
- Counts are fixed at 500+60 versus 1000. Measured synchronized optimization
time is reported rather than used as a stopping condition.
First-step compilation is excluded with `--timing-warmup-steps 1`; validation,
initialization, and checkpoint I/O remain outside synchronized step timing.

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
  draft_60/
    lora_latest.safetensors
    training_summary.json
    validation/step_000060/...
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

The annotated grid has `SFT 500 + DRaFT-LV 60` on the first row and `SFT 1000`
on the second. Individual images and prompt sidecars remain available.

`experiment_results.json` reports synchronized optimization time, stage wall
time, latency percentiles, throughput, face-detection rate, detected-face
identity similarity, and the fallback-aware training reward. `metrics.csv`
contains per-image values. It also reports nearest-reference similarity, the
nearest-versus-centroid gap, reference-assignment entropy, and maximum
assignment probability so automatic evaluation can expose reference-view
collapse that centroid identity alone misses.

Every training stage has a command/dependency signature. Re-running the same
command reuses complete stages; `--force` runs all three again. Training-state
files are deliberately self-contained and can be large because they contain
AdamW moments and cached text embeddings.

When changing only DRaFT hyperparameters, combine `--reuse-sft-branches` with
`--force`. The runner verifies the SFT-500 and SFT-1000 adapters, states,
summaries, and validation images independently. It explicitly reuses each
complete branch and rebuilds an incomplete one, while `--force` reruns DRaFT
plus the final grid and metrics.

AVIF and valid extensionless images are supported, including the current
contents of `test_characters/`.
