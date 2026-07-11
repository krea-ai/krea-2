# End-to-end character training

`train_face.py` turns a folder of character reference images into a final SFT
+ DRaFT-K LoRA. It is the recommended quick start:

```bash
export OSS_RAW=/models/krea2-raw.safetensors
export DEEPINFRA_KEY=your_key

uv run --extra train --extra face-reward train_face.py \
  /path/to/reference_images \
  --output-dir runs/my_character \
  --trigger-word optional_character_token
```

Omit `--trigger-word` to leave captions and generated prompts unprefixed.

To avoid DeepInfra, provide an `image_path,prompt` CSV and a text file with one
DRaFT prompt per line:

```bash
uv run --extra train --extra face-reward train_face.py \
  /path/to/reference_images --output-dir runs/my_character \
  --captions /path/to/captions.csv \
  --draft-prompts prompts/draft_woman.txt
```

Relative image paths are resolved from the CSV directory. The repository also
ships `prompts/draft_man.txt`; both supplied files bypass API initialization.

## Custom rewards

`main.py` runs the same image-folder pipeline but requires a custom reward and
does not import face dependencies:

```bash
uv run --extra train main.py /path/to/reference_images \
  --output-dir runs/custom \
  --reward package.module:Reward \
  --reward-init-kwargs '{"constructor":"values"}' \
  --reward-kwargs '{"call":"values"}'
```

The `module:object` target may be a class or callable. Classes are constructed
once with `--reward-init-kwargs`. For each generated image the trainer calls:

```python
reward(image, prompt, **reward_kwargs)
```

`image` has shape `[1, C, H, W]` and values in `[-1, 1]`. The return value must
be a finite scalar tensor connected to the image's autograd graph. A Python
number, constant tensor, NaN, or detached result fails immediately with a
clear error. Reward options are included in stage signatures; secret-like
manifest values are redacted.

## Pipeline stages

1. Discover supported images recursively, including valid extensionless files.
2. Reuse caption sidecars or caption the images through DeepInfra, with a
   visible progress bar.
3. Write an absolute `image_path,prompt` dataset CSV.
4. Generate and cache diverse DRaFT prompts through DeepInfra, also with
   progress reporting.
5. Run cached-latent character SFT with rowwise/tensorwise INT8.
6. Start a clean subprocess, load the SFT LoRA, and run DRaFT-K with the chosen
   differentiable reward.

Separating SFT and DRaFT-K processes releases all GPU state before the next
model load. Both objectives use the shared high-noise schedule.

## Face models

`train_face.py` expects this antelopev2 layout:

```text
antelopev2/
  detection/model.onnx
  recognition/model.onnx
```

Pass `--face-model-dir PATH` or set `ANTELOPEV2_DIR`. Otherwise the wrapper
checks the repository parent, the repository itself, and the user cache.
Missing files are downloaded atomically from
[immich-app/antelopev2](https://huggingface.co/immich-app/antelopev2).
Face discovery and download are exclusive to face-specific commands.

## Defaults and outputs

The short command uses 500 SFT steps followed by 60 DRaFT-LV updates, rank 32,
batch size 1, 64 generated prompts, `draft_k=1`, one LV perturbation, 12
training denoising steps, a 1e-4 DRaFT learning rate, CFG 4.5, seed 42, and
ten fixed validation images generated with 20 steps. Use `--help` for controls
such as step counts, validation, caption/prompt models, regeneration, and
`--force`.

The saturated reward, QKVO restriction, EOT, and prompt-diversity controls make
1e-4 usable without the reference-expression collapse seen in the earlier
unprotected high-rate runs.

During DRaFT the complete SFT adapter remains loaded and saved, but only its
Q/K/V/O LoRA tensors are optimized; attention gating and MLP LoRAs retain their
exact SFT values. Face training uses a saturated centroid reward rather than a
nearest-reference bonus, averages recognition over weak aligned-crop
augmentations, and applies a balanced expression schedule. One quarter of
training prompts leave expression unspecified. A second balanced schedule
covers frontal, three-quarter, profile, high/low, and oblique viewpoints while
leaving one quarter unspecified. Every fourth update replaces
the correlated LV auxiliary sample with an independent trajectory and adds an
aligned eye/mouth diversity term for those unspecified prompts.

```text
runs/my_character/
  pipeline_plan.json
  data/
    captions.json
    dataset.csv
    draft_prompts.txt
  sft/
    lora_latest.safetensors
    validation/...
  draft/
    lora_latest.safetensors
    validation/...
```

Each stage records a code-and-input signature. An identical rerun reuses the
completed stage. A changed dataset, reward implementation, or upstream LoRA
invalidates only the affected work. `--force` reruns training stages.

## Image conversion

Convert a folder recursively before training when desired:

```bash
uv run scripts/convert_images_to_jpg.py /path/to/images
```

Use `--output-dir` for a mirrored destination, `--dry-run` to inspect the plan,
and `--overwrite` or `--delete-originals` deliberately. The project does not
install a separate AVIF dependency; conversion supports formats decoded by the
installed Pillow build.

## Checks

```bash
uv run --extra train --extra face-reward --extra dev \
  scripts/check_release.py
```
