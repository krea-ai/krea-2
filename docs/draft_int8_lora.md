# DRaFT-K INT8 LoRA Training

`scripts/train.py` trains LoRA adapters on top of the frozen INT8 Krea
2 DiT. Base linear weights stay INT8; only FP32 LoRA master weights are
optimized. Training defaults to the faster `rowwise`
activation/tensorwise-weight layout and retains the compatibility `blockwise`
layout as an explicit option.

V1 supports 512x512 training only.

## Objectives

Character SFT uses supervised flow matching on an `image_path,prompt` CSV. A
trigger word is prepended to each prompt before text encoding:

```bash
uv run --extra train scripts/train.py \
  --objective sft \
  --csv ../test_images/labels.csv \
  --validation-csv ../test_images/labels.csv \
  --validation-step 100 \
  --validation-size 4 \
  --trigger-word "triggerword" \
  --cache-latents \
  --rank 32 \
  --quantization-type rowwise \
  --output-dir runs/character_sft
```

`--cache-latents` is SFT-only. Before training starts, it encodes all training
images to VAE latents and all triggered prompts to text embeddings, stores both
in CPU RAM, then moves the text encoder and unneeded VAE modules to CPU so
training VRAM is used by the DiT and LoRA optimizer state. Without periodic
validation the whole VAE is offloaded; validation keeps only its decoder on
GPU.

DRaFT-K reward training backpropagates through the last `K` denoising steps and
the VAE decode. It reads prompts from a text file with one prompt per line and
can resume from the SFT adapter:

```bash
uv run --extra train --extra face-reward scripts/train.py \
  --objective draft \
  --prompts ../test_images/generated_prompts.txt \
  --validation-prompts ../test_images/generated_prompts.txt \
  --validation-step 100 \
  --validation-size 4 \
  --trigger-word "triggerword" \
  --resume-lora runs/character_sft/lora_latest.safetensors \
  --reward krea2.rewards.face:FaceSimilarityReward \
  --reward-init-kwargs '{"reference_images":"../test_images","model_dir":"../antelopev2"}' \
  --rank 32 \
  --draft-k 1 \
  --steps 28 \
  --cfg 4.5 \
  --output-dir runs/reward_lora
```

The reward object is loaded from `module:object`. If the object is a class it is
constructed with `--reward-init-kwargs`. During training it is called as:

```python
reward(image, prompt, **reward_kwargs)
```

`image` is a tensor shaped `[1, C, H, W]` with values clamped to `[-1, 1]`.
Pass `--reward-kwargs '{"key": "value"}'` to provide extra call kwargs.

`krea2.rewards.face:FaceSimilarityReward` uses local antelopev2 ONNX files.
Detection runs without gradients. Recognition is ported to PyTorch so the loss
backpropagates through the aligned crop into the generated image. Reference
images with no detected face are skipped. For multi-face reference photos the
largest face is selected by default, avoiding the previous behavior that
discarded the whole image because of a small background face.

The reward uses a normalized identity prototype blended with a smooth
nearest-reference score. Generated multi-face images are matched by identity,
not merely detector confidence, and a second face resembling the target ID is
penalized. If detection fails, three differentiable upper-center crop
hypotheses are passed through the recognition network. A no-face image now has
a useful image gradient instead of the former constant `-2` zero-gradient
reward. Relevant constructor controls are `reference_face_policy`,
`nearest_reference_weight`, `nearest_temperature`, `no_face_penalty`, and
`duplicate_identity_weight`.

Both SFT and DRaFT-K use the same high-noise time shift. The resolution-derived
Krea shift is increased by `--high-noise-shift 0.5` before applying the
logistic time transform. At 512x512 this changes the effective shift from
approximately `0.58125` to `1.08125`; `t=1` remains noise and both endpoints
remain fixed. Set `--high-noise-shift 0` for the legacy schedule. The value is
stored in adapter metadata and is also used by the final training sample.

For DRaFT-K, prompt text embeddings are cached automatically before training.
The Qwen text encoder is then moved to CPU, and the unused VAE encode-side
modules (`encoder`, `quant_conv` when present) are moved to CPU while the VAE
decoder remains on GPU for differentiable reward training.

At the end of SFT and DRaFT-K training, the trainer samples one prompt and saves
a 512x512 generated image plus a prompt sidecar in `--output-dir`:

```text
sft_final_sample_step_001000.png
sft_final_sample_step_001000.txt
draft_final_sample_step_001000.png
draft_final_sample_step_001000.txt
```

For SFT the prompt is sampled from `--validation-csv` when provided, otherwise
from `--csv`. For DRaFT-K it is sampled from `--validation-prompts` when
provided, otherwise from `--prompts`. `--final-sample-seed` fixes both prompt
selection and generation seed; `--skip-final-sample` disables this step.

## Fixed validation generations

Set `--validation-step N` to generate a comparable validation set before the
first optimization step and after every `N` completed training steps. The
default `0` disables periodic validation. `--validation-size M` controls the
number of generated images and defaults to 4.

The trainer randomly chooses the `M` prompts once at startup with a local,
deterministic generator. `--validation-prompts` supplies a fixed text file for
either SFT or DRaFT-K. Otherwise SFT/flow use `--validation-csv` or `--csv`,
while DRaFT-K uses `--prompts`. SFT and DRaFT-K apply the configured trigger
word. If `M` is larger than the prompt source, shuffled permutations are
repeated. Prompt text conditioning is cached before the text encoder can be
offloaded, so every validation pass reuses the same prompts without restoring
that encoder.

Generation also reuses fixed per-image seeds and the training high-noise
schedule. Images are generated one at a time to keep peak validation VRAM
independent of `--validation-size`. Each checkpoint directory contains an
image and prompt sidecar for every fixed item:

```text
validation/step_000000/image_000.png
validation/step_000000/image_000.txt
validation/step_000100/image_000.png
validation/step_000100/image_000.txt
```

For cached SFT with validation enabled, the VAE encode-side modules and text
encoder are offloaded while the VAE decoder stays on GPU. Validation generation
runs outside the reported `step_time`; the logged number measures the training
step only. The interval, set size, prompt source/indices, and validation seed
are stored in adapter metadata.

For controlled experiments, `--validation-at-start` and
`--validation-at-end` request only those boundary evaluations while
`--validation-step 0` keeps intermediate validation disabled. Pass
`--validation-seed` to reuse exactly the same prompt ordering and image seeds
across separate SFT and DRaFT processes.

## Exact SFT continuation

For cached SFT, `--save-training-state PATH` writes a self-contained state with
the LoRA tensors, AdamW state, CPU/CUDA RNG, cached latents/text, sampler
permutation, cursor, and global step. Resume it with:

```bash
uv run --extra train scripts/train.py \
  --objective sft --csv labels.csv --cache-latents \
  --resume-training-state training_state_step_000500.pt \
  --train-steps 500 --output-dir runs/sft_1000
```

`--train-steps` is the number of additional updates, so this example finishes
at global step 1000. State save/resume requires `--num-workers 0` and rejects
changes to the dataset or compatibility-critical training configuration.
`--resume-training-state` and `--resume-lora` are mutually exclusive.

Use `--draft-image-every 0` to disable per-step DRaFT training PNGs when their
I/O should not be included in a timing comparison.

During DRaFT-K training, every generated training image is also saved before
reward evaluation:

```text
draft_step_images/step_000001_00.png
draft_step_images/step_000001_00.txt
```

Training logs include `step_time` and `steps_per_second` at step 1 and every 10
steps. Set `--log-every 1` for per-step profiling. `--save-every 0` disables
intermediate adapter checkpoints; DRaFT training images and enabled validation
generations are still saved for reward/debug inspection.

All fixed-shape objectives default to `--compile-mode default`. Blocks are
compiled after LoRA insertion and before activation-checkpoint wrapping, so
Inductor fuses normalization, modulation, residual, and pointwise work while
the custom INT8 operators remain opaque graph nodes. DRaFT-K creates two stable
specializations: one for its early `torch.no_grad()` denoising calls and one for
the gradient-carrying tail. Use `--compile-mode none` for debugging.
CUDA-graph-oriented compile modes are not offered: checkpoint recomputation
and the custom autograd operators can release temporaries differently during
recording and replay.

Flow-matching training uses the CSV images as real data:

```bash
uv run --extra train scripts/train.py \
  --objective flow \
  --csv images.csv \
  --rank 32 \
  --output-dir runs/flow_lora
```

The flow loss follows Krea sampling's time direction, where `t=1` is noise and
`t=0` is data: `MSE(v_theta(x_t, t), x_0 - x_1)` with
`x_t = t * x_0 + (1 - t) * x_1`. Here `x_0` is sampled noise and `x_1` is the
encoded image latent.

Adapters trained with the previous opposite convention should be discarded and
retrained; at inference they can leave the denoising trajectory close to noise.

## Input Formats

The input CSV must contain:

```csv
image_path,prompt
path/to/image.png,a prompt string
```

`image_path` is used by flow matching. DRaFT-K uses the prompt column for online
generation and reward training.

DRaFT-K also supports prompt-only text files:

```text
A man in a tailored charcoal blazer stands near a projection screen...
A man sits at a wooden desk under soft daylight...
```

## LoRA and Kernels

LoRA is inserted only into eligible INT8 linears inside the main MMDiT blocks.
Supported ranks are `32` and `64`. The default LoRA scale is
`alpha / rank = 1.0` because `--lora-alpha` defaults to `rank`.

Forward computes:

```text
Y = X @ W_int8.T + scale * (X @ A.T) @ B.T + bias
```

The fused Triton op keeps the base INT8 epilogue and adds the LoRA contribution
before storing BF16 output. The custom backward returns gradients for `X`,
`A`, and `B`; base INT8 weights, scales, and bias stay frozen.

With `--quantization-type rowwise`, every frozen weight has one scalar FP32
scale and each flattened activation row has one FP32 scale. Backward freshly
quantizes `grad_output` rowwise for the frozen-base `dX = dY @ W`; it does not
requantize the saved forward `X`. LoRA gradients continue to use the saved BF16
activation and FP32 master parameters. Adapter metadata records the selected
quantization type.

The blockwise default remains available for compatibility:

```bash
... scripts/train.py --objective sft --quantization-type blockwise ...
```

On an RTX 4090 (PyTorch 2.12.1+cu130, Triton 3.7.1), the rowwise rank-32
forward+backward microbenchmark for `(M,N,K)=(1536,6144,6144)` measured a
1.14 ms median versus 8.15 ms for the previous blockwise implementation. After
using BF16 cuBLAS for the small-rank LoRA gradients, the layer-only measurement
with a direct BF16 upstream gradient was 0.40 ms forward and 0.42 ms backward.
The final RTX 4090 verification used the real `krea2-raw.safetensors`
checkpoint and the exact cached-SFT loss path at batch 1 and 512x512: cached
latents `[1,16,64,64]`, text embeddings `[1,512,12,2560]`, rank 32, per-block
activation checkpointing, `torch.compile(fullgraph=True)` in default mode, and
both FP32 Adam moment buffers resident. Across 12 warmed runs it measured
126.5 ms for forward plus MSE construction and 305.5 ms backward: 431.9 ms
combined median, 432.8 ms p80, 447.5 ms maximum, and 15.45 GB peak. First-use
Inductor compilation and Triton autotuning are excluded; data loading and the
optimizer update are outside the stated forward/backward measurement.

For the end-to-end DRaFT-K target, the same RTX 4090 system used the real Krea
checkpoint, cached text, the resumed rank-32 character SFT adapter, 20 denoising
steps, `draft_k=1`, CFG 4.5, 512x512 output, differentiable VAE decode,
`FaceSimilarityReward`, training-image PNG save, gradient clipping, and AdamW.
After the first compilation/autotuning iteration, five consecutive steps took
5.742, 5.761, 5.702, 5.665, and 5.678 seconds: 5.702 s median and 5.761 s
maximum. The first iteration took 18.672 s and is intentionally excluded from
steady-state throughput.

In CFG DRaFT training, only the conditional prediction carries gradients. The
unconditional branch runs under `torch.no_grad()`.

## Outputs

Adapters are saved as safetensors:

- `lora_step_000100.safetensors`
- `lora_latest.safetensors`

Files contain only LoRA tensors plus metadata for objective, rank, alpha,
scale, checkpoint name, and target module names.

Pass `--resume-lora path/to/lora_latest.safetensors` to initialize the LoRA
parameters from a previous adapter file before continuing training.

To use a trained adapter in ComfyUI, convert its tensor names, dtype, and LoRA
scale with `scripts/convert_lora_to_comfyui.py`; see
[ComfyUI LoRA conversion](lora_conversion.md).

## Checks

Run:

```bash
uv run --extra train --extra face-reward --extra dev scripts/check_release.py

# Add the complete CUDA kernel and antelopev2 suite.
uv run --extra train --extra face-reward --extra dev \
  scripts/check_release.py --cuda
```
