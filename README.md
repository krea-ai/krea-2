# Krea 2 (K2)

Krea 2 - an image generation model from [Krea AI](https://www.krea.ai).

---

<p align="center">
<a href="https://docs.krea.ai/api-reference/introduction">API Docs</a> •
<a href="https://huggingface.co/krea/krea-2-raw">Hugging Face (RAW)</a> •
<a href="https://huggingface.co/krea/krea-2-turbo">Hugging Face (TURBO)</a> •
<a href="https://www.krea.ai/blog/krea-2-technical-report">Technical Blog</a>
</p>

<img src="assets/big.png" alt="k2 banner">

<img src="assets/k2.png" alt="k2 banner">

This is the official repository for the open version of Krea 2, an image model trained from scratch focused on creative and stylistic exploration. The repository contains inference code and instructions to run the model.

Krea 2 is the most aesthetic open-source image model available. On quality, Krea 2 is the #1 text-to-image model from an independent lab on Artificial Analysis [\[1\]](https://artificialanalysis.ai/image/leaderboard/text-to-image).

Krea 2 ships as two models. Krea 2 RAW is the base model. It's a pretrained checkpoint with no distillation, so it's diverse and highly malleable, and it's what you should use for fine-tuning, post-training, and LoRA training. Krea 2 Turbo is an 8-step distilled checkpoint built for fast, high-quality text-to-image.

The two models are designed to work together. You train LoRAs on RAW and apply them on Turbo, and the LoRAs trained on RAW will work well on Turbo. **We highly recommend using RAW for training LoRAs and applying them on Turbo for inference.**

## Setup

```bash
uv sync
```

Character training and the differentiable face reward are optional dependency
groups, so inference-only installations stay small:

```bash
uv sync --extra train --extra face-reward
```

Use `uv sync --extra train --extra face-reward --extra dev` for a complete
development environment. PyTorch and torchvision intentionally have no version
constraints: `uv sync` resolves the newest compatible CUDA 13.0 builds
available from the configured PyTorch index instead of pinning a release.

Both [Raw](https://huggingface.co/krea/Krea-2-Raw) and [Turbo](https://huggingface.co/krea/Krea-2-Turbo) safetensor files are available on Hugging Face. After downloading the checkpoints, set the `OSS_RAW` and `OSS_TURBO` environment variables to the paths of the downloaded files.

```bash
export OSS_RAW=...
export OSS_TURBO=...
```

## Usage

The following commands run inference using the two available checkpoints with recommended settings.

### Raw (`oss_raw`)

The base undistilled model. Use the full sampler with classifier-free guidance:
The model has been trained to generate upto 1k resolution.

```bash
uv run scripts/inference.py "a fox walking in the snow" \
    --checkpoint oss_raw --steps 52 --cfg 3.5
```

### Turbo (`oss_turbo`)

Distilled for few-step sampling — run with 8 steps and CFG disabled.
The model can generate images from 1k ~ 2k resolution.

```bash
uv run scripts/inference.py "a fox walking in the snow" \
    --checkpoint oss_turbo --steps 8 --cfg 0.0 --mu 1.15 --width 2048 --height 2048
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `prompt` (positional) | — | Text prompt to generate from. |
| `--steps` | `28` | Number of denoising steps. |
| `--cfg` | `4.5` | Classifier-free guidance scale (`0` disables CFG). |
| `--y1` | `0.5` | Timestep-shift `mu` at min resolution. |
| `--y2` | `1.15` | Timestep-shift `mu` at max resolution. |
| `--mu` | `None` | Pin a constant timestep-shift `mu`, overriding the resolution-derived value. Recommended `1.15` for `oss_turbo`. |
| `--width` / `--height` | `1024` ~ `2048` | Output resolution; padded up to a multiple of 16 if needed. |
| `--num-images` | `1` | Number of images to generate from the prompt. |
| `--seed` | `0` | Base seed; image *i* uses `seed + i`. |
| `--checkpoint` | `oss_raw` | Checkpoint to load (`oss_raw`, `oss_turbo`). Defaults to `$K2_CHECKPOINT`. |
| `--output` | `sample` | Output filename prefix. |

## Character training

Install the training extras and point `OSS_RAW` at Krea 2 RAW:

```bash
uv sync --extra train --extra face-reward
export OSS_RAW=/models/krea2-raw.safetensors
export DEEPINFRA_KEY=your_key

uv run --extra train --extra face-reward train_face.py IMAGES \
  --output-dir runs/my_character --trigger-word optional_token
```

`IMAGES` is searched recursively for JPG, PNG, WebP, BMP, or AVIF files. The
trigger word is optional, and missing antelopev2 models are downloaded
automatically. The default is SFT 500 + QKVO DRaFT-LV 60.

### Dataset files

Automatic mode captions the image folder and creates 64 DRaFT prompts. For
local data, captions use CSV and DRaFT prompts use one prompt per line:

```csv
image_path,prompt
images/01.jpg,A woman standing in a softly lit studio.
```

Relative image paths are resolved from the CSV location. As another option,
place a one-line caption beside every image (`01.jpg` + `01.txt`). Ready-made
DRaFT prompt files are included at [`prompts/draft_woman.txt`](prompts/draft_woman.txt)
and [`prompts/draft_man.txt`](prompts/draft_man.txt).

To train without DeepInfra, provide both local files:

```bash
uv run --extra train --extra face-reward train_face.py IMAGES \
  --output-dir runs/my_character \
  --captions /path/to/captions.csv \
  --draft-prompts prompts/draft_woman.txt
```

### ComfyUI conversion

```bash
uv run scripts/convert_lora_to_comfyui.py \
  runs/my_character/draft/lora_latest.safetensors \
  runs/my_character/comfyui_lora.safetensors
```

For a custom differentiable reward, use `main.py`; individual objectives are
available through `scripts/train.py`.


## Documentation

- [Prompting Guide](docs/prompting.md)
- [Safety Guide](docs/safety.md)
- [Inference Notes](docs/inference.md)
- [INT8 Inference](docs/int8.md)
- [Project Architecture](docs/architecture.md)
- [End-to-end Character Training](docs/character_pipeline.md)
- [SFT versus DRaFT-K Comparison](docs/compare.md)
- [Character Training Results](docs/results.md)
- [ComfyUI LoRA Conversion](docs/lora_conversion.md)

## Inference

You can run our open source models on the following platforms.

- [ComfyUI](https://github.com/comfy-org/comfyui)
- [Fal](https://fal.ai/models/fal-ai/krea-2/turbo)
- [SGLang](https://docs.sglang.io/cookbook/diffusion/Krea/Krea-2)


## Finetuning Krea 2

For finetuning Krea 2, we highly recommend that you **train a LoRA on the Raw model and apply it to the Turbo model** for inference.
We recommend using the following providers and open source tools for finetuning Krea 2.

- [Huggingface Diffusers](https://github.com/huggingface/diffusers)
- [Ostris AI toolkit](https://github.com/ostris/ai-toolkit)
- [Fal](https://fal.ai)
  - [Training](https://fal.ai/models/fal-ai/krea-2-trainer)
  - [Inference](https://fal.ai/models/fal-ai/krea-2/turbo/lora)
- [Kohya (musubi tuner)]( https://github.com/kohya-ss/musubi-tuner)

## FAQ

**Which model I should use?**

Use the **Turbo** model for fast inference with high quality results. The **Raw** model is an undistilled checkpoint without any step / cfg guidance distillation and posttraining. It is a highly finetunable base model that can be used to train LoRAs for the Turbo model as well as posttraining research. In short, **TRAIN on Raw and RUN on Turbo**.

**What license is this model released under?**

The source code in this repository is released under Apache-2.0. The Krea 2
model weights are a separate work governed by the
[Krea 2 Community License Agreement](https://www.krea.ai/krea-2-licensing).
Review that agreement for the terms that apply to the weights. For a commercial
license, contact [opensource@krea.ai](mailto:opensource@krea.ai). See
[`NOTICE`](NOTICE) for the distinction and third-party notices.

## Citation
```
@misc{krea-2-2026,
    author={Sangwu Lee, Erwann Millon, Le Zhuo, Matthew Newton, Andrei Filatov, Abhinay Devarinti, Dazhi Zhong, Avram Djordjevic, Gabriel Menezes, Will Beddow, Titus Ebbecke, Mihai Petrescu, Owen Fahey, Gian Saß, Felix Gil, Victor Perez},
    title={{Krea 2}},
    year={2026},
    howpublished={\url{https://www.krea.ai/blog/krea-2-technical-report}},
}
```
