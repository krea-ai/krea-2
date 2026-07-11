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

The short command uses 500 SFT steps followed by 120 DRaFT-LV updates, rank 32,
batch size 1, 64 generated prompts, `draft_k=1`, one last-step LV perturbation,
12 training denoising steps, a 2e-4 DRaFT learning rate, CFG 4.5, seed 42, and
ten fixed validation images generated with 20 steps. Use `--help` for controls
such as step counts, validation, caption/prompt models, regeneration, and
`--force`.

For the measured fast Pareto point, pass `--draft-steps 40 --draft-lr 0.0004`.
It is less stable beyond 40 reward updates, so do not combine that rate with
the default 120-step duration.

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
