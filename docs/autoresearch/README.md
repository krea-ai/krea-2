# Character-training autoresearch

## Objective

Maximize mean antelopev2 identity similarity on held-out generations while
minimizing pure optimization time. Every promoted method is tested on the
Julia Jacklin and Tommy Guerrero reference folders with seed 42 and ten fixed
held-out prompts. Initialization, compilation warmup, captioning, validation,
metric computation, and checkpoint I/O are excluded from the training budget.
No method may exceed 900 seconds of measured SFT + DRaFT optimization.

Measurements in this log use one RTX 4090 (24,564 MiB, driver 595.58.03),
Torch 2.13.0+cu130, CUDA 13.0, and Triton 3.7.1. During a warmed LV update the
GPU sustains 100% SM utilization in a five-second `nvidia-smi dmon` sample.

## Reproduction

The runner trains SFT once, then produces a fixed-interval DRaFT learning
curve. A 100-step curve includes the former 50-step baseline without a second
training run because validation uses explicit generators and does not perturb
training RNG state.

```bash
export OSS_RAW=/path/to/krea2-raw.safetensors

uv run --extra train --extra face-reward scripts/autoresearch.py \
  ../test_characters/julia_jacklin \
  --output-dir ../expirements/autoresearch/julia_jacklin \
  --variant lv1_sampler12_lr2e4 \
  --sft-steps 500 --draft-steps 120 --validation-every 20 \
  --draft-k 1 --draft-lv-samples 1 --denoising-steps 12 --draft-lr 0.0002 \
  --validation-size 10 --seed 42 \
  --face-model-dir ../antelopev2
```

Run the identical command for `../test_characters/tommy_guerrero`. Stage
signatures make the dataset, SFT adapter, and completed variants resumable.

## Measurement contract

- Primary metric: mean cosine identity similarity over all ten validation
  generations. A missed face still counts in the denominator through the
  reported detection rate and must not be silently dropped when comparing
  methods.
- Validation prompts are the final ten items of a frozen 74-prompt pool and
  are disjoint from the 64 DRaFT training prompts.
- Validation uses 512×512, 20 denoising steps, CFG 4.5, and seeds beginning at
  200042.
- Training sampler length is independent of evaluation. The runner always
  passes `--validation-steps 20`, including when DRaFT uses 12 or 16 steps.
- `timing_warmup_steps=1` excludes first-step Inductor/Triton compilation.
  `training_summary.json` retains every raw step latency and both total and
  budgeted optimization time.
- A result is promoted only after it improves the two-character mean without
  exceeding 900 seconds on either character.

## Research log

| ID | Change | Status | Rationale |
| --- | --- | --- | --- |
| B0 | SFT 500 + DRaFT-K 1, 20 sampler steps | Complete | Step 80 improves the two-character mean from the former step-50 value 0.4651 to 0.5175. |
| R0 | Unpinned Torch CUDA 12.8 → latest CUDA 13.0 index | Complete | Torch 2.13/Triton 3.7 passes the CPU and CUDA suites; first-step compilation is explicitly excluded. |
| C1 | Fuse conditional/unconditional no-grad CFG passes | Rejected | 6.34 s/step on Julia versus 5.44 s/step baseline; batch-2 kernels outweighed launch savings. |
| P0 | Decouple training and validation sampler lengths | Complete | An audit found that early 12/16-step experiments also evaluated at that length. Their LoRAs are valid, but scores are quarantined until fixed 20-step re-evaluation. |
| S0 | Reduce the DRaFT training sampler to 12 or 16 steps | Complete | Twelve steps is the fastest useful base for LV; evaluation remains fixed at 20 steps. |
| LV1 | One DRaFT-LV last-step perturbation | Promoted | Supplies two reward gradients at about 1.20× the sampler-12 update cost. It improves both characters and the complete measured Pareto frontier. |
| M1 | Disable DiT/VAE checkpoint recomputation where 24 GB permits | Rejected | Disabling DiT checkpointing OOMs a 24 GB card; disabling only VAE checkpointing is within measurement noise. |
| B1 | DRaFT batch size 2 | Rejected | Uses 18.9 GiB but improves sample throughput by only about 4.7%; LV is the stronger two-sample construction. |
| S1 | Increase LV learning rate from 5e-5 to 1e-4 | Superseded | Improves both characters at every matched 20-step checkpoint through step 100; 2e-4 moves the frontier again. |
| S2 | Increase LV learning rate from 1e-4 to 2e-4 | Promoted | Improves both characters through the 120-step budget boundary; this is the release default. |
| S3 | Increase LV learning rate from 2e-4 to 4e-4 | Fast option | Improves the 20/40-step frontier, but Tommy regresses after step 40 and its maximum is below 2e-4/120. |

Raw machine-readable curves live beside experiment artifacts; consolidated
results and rejected variants will be added here as measurements complete.

## Baseline measurements

The primary mean assigns zero similarity to a generation where antelopev2 does
not detect a face. The historical detected-only mean is retained in the JSON
artifacts but is not used to select a method.

| Method/checkpoint | Julia similarity/time | Tommy similarity/time | Mean similarity | Mean time |
| --- | ---: | ---: | ---: | ---: |
| Baseline: SFT 500 + DRaFT 50, train/eval 20 | 0.4907 / 542.4 s | 0.4395 / 630.6 s | 0.4651 | 586.5 s |
| Baseline: SFT 500 + DRaFT 80, train/eval 20 | 0.5571 / 705.7 s | 0.4779 / 831.7 s | 0.5175 | 768.7 s |
| **LV: SFT 500 + DRaFT 60, train 12/eval 20** | **0.5279 / 560.7 s** | **0.4709 / 582.0 s** | **0.4994** | **571.3 s** |
| **LV: SFT 500 + DRaFT 80, train 12/eval 20** | **0.6146 / 660.6 s** | **0.5072 / 682.2 s** | **0.5609** | **671.4 s** |
| **LV: SFT 500 + DRaFT 120, train 12/eval 20** | **0.6742 / 860.8 s** | **0.5635 / 882.2 s** | **0.6189** | **871.5 s** |
| **LV 1e-4: SFT 500 + DRaFT 60, train 12/eval 20** | **0.6356 / 560.9 s** | **0.5132 / 581.5 s** | **0.5744** | **571.2 s** |
| **LV 1e-4: SFT 500 + DRaFT 80, train 12/eval 20** | **0.6703 / 661.1 s** | **0.5625 / 681.6 s** | **0.6164** | **671.3 s** |
| **LV 1e-4: SFT 500 + DRaFT 100, train 12/eval 20** | **0.7651 / 761.1 s** | **0.6173 / 781.5 s** | **0.6912** | **771.3 s** |
| **LV 2e-4: SFT 500 + DRaFT 20, train 12/eval 20** | **0.5228 / 360.6 s** | **0.4244 / 381.7 s** | **0.4736** | **371.2 s** |
| **LV 2e-4: SFT 500 + DRaFT 40, train 12/eval 20** | **0.5661 / 460.8 s** | **0.5415 / 482.1 s** | **0.5538** | **471.5 s** |
| **LV 2e-4: SFT 500 + DRaFT 60, train 12/eval 20** | **0.6671 / 561.0 s** | **0.6046 / 582.5 s** | **0.6358** | **571.8 s** |
| **LV 2e-4: SFT 500 + DRaFT 80, train 12/eval 20** | **0.6966 / 661.1 s** | **0.6918 / 682.6 s** | **0.6942** | **671.8 s** |
| **LV 2e-4: SFT 500 + DRaFT 100, train 12/eval 20** | **0.7921 / 756.6 s** | **0.7018 / 777.6 s** | **0.7470** | **767.1 s** |
| **LV 2e-4: SFT 500 + DRaFT 120, train 12/eval 20** | **0.8249 / 855.9 s** | **0.7292 / 877.0 s** | **0.7770** | **866.4 s** |
| **LV 4e-4: SFT 500 + DRaFT 20, train 12/eval 20** | **0.5571 / 360.6 s** | **0.4904 / 381.7 s** | **0.5237** | **371.1 s** |
| **LV 4e-4: SFT 500 + DRaFT 40, train 12/eval 20** | **0.6654 / 460.8 s** | **0.6278 / 481.6 s** | **0.6466** | **471.2 s** |

At the near-original runtime, LV-60 is 2.6% faster on average and improves
mean similarity by 7.4%. At 80 reward updates, LV is faster on each character
and improves the mean by 8.4%. Using the available budget, LV-120 improves the
mean by 19.6% over the 80-step baseline and by 33.1% over the original
500+50 target. These base-rate LV rows established the method before the
learning-rate ablation moved the frontier again.

Increasing the LV learning rate to 1e-4 strictly dominates the 5e-5 LV curve
at every matched checkpoint. The promoted 100-update configuration improves
mean similarity by 33.6% over the strongest 20-step baseline while taking only
2.6 seconds more on average. It remains 128.7 and 118.5 seconds below the
15-minute cap for Julia and Tommy respectively.

At 2e-4, only 20 DRaFT updates exceed the original 500+50 mean while cutting
average training time by 36.7%. The 80-update endpoint improves mean similarity
by 49.3% over the original target and remains a strict quality/time improvement
over the strongest old baseline. Using the full budget, step 120 reaches 0.7770
mean similarity with 100% detection: 67.1% above the original target and 50.1%
above the old 80-update baseline. Julia and Tommy finish 44.1 and 23.0 seconds
below the cap, leaving fewer than five Tommy updates and no useful room for a
phase reallocation.

For latency-sensitive runs, 4e-4/40 reaches 0.6466 mean similarity with 100%
face detection in 471.2 seconds average. Do not extend that rate past 40
updates: Tommy falls from 0.6278 at step 40 to 0.6206 at step 60. The release
default remains 2e-4/120 because it has the highest cross-character mean.

The 20-step curve is monotonic through step 80 on the all-generation metric;
Julia reaches 0.5785 at step 90, while Tommy's 90-step projection would exceed
the budget because its early reward calls frequently use fallback crops.

The first sampler-length runs accidentally used the shorter sampler for both
training and validation. Those scores are intentionally excluded from the
comparison table. Re-evaluate any saved endpoint without retraining:

```bash
uv run --extra train --extra face-reward scripts/evaluate_autoresearch.py \
  ../test_characters/julia_jacklin \
  --lora ../expirements/autoresearch/julia_jacklin/variants/VARIANT/lora_latest.safetensors \
  --prompts ../expirements/autoresearch/julia_jacklin/data/evaluation_prompts.txt \
  --output-dir ../expirements/autoresearch/julia_jacklin/variants/VARIANT/evaluation_20step \
  --face-model-dir ../antelopev2
```

## Literature basis

- Clark et al., [Directly Fine-Tuning Diffusion Models on Differentiable
  Rewards](https://arxiv.org/abs/2309.17400), motivates truncated DRaFT-K,
  K=1, and the last-step perturbations used by DRaFT-LV.
- Chen et al., [ID-Aligner](https://arxiv.org/abs/2404.15449), independently
  supports identity-consistency reward feedback for face-preserving image
  personalization.
- Prabhudesai et al., [Aligning Text-to-Image Diffusion Models with Reward
  Backpropagation](https://arxiv.org/abs/2310.03739), supports LoRA plus
  checkpointed reward backpropagation as the memory baseline.
- Hang et al., [Efficient Diffusion Training via Min-SNR
  Weighting](https://arxiv.org/abs/2303.09556), motivates a later SFT timestep
  weighting ablation if direct-reward allocation alone does not dominate.
