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
uv run inference.py "a fox walking in the snow" \
    --checkpoint oss_raw --steps 52 --cfg 3.5
```

### Turbo (`oss_turbo`)

Distilled for few-step sampling — run with 8 steps and CFG disabled.
The model can generate images from 1k ~ 2k resolution.

```bash
uv run inference.py "a fox walking in the snow" \
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


## Documentation

- [Prompting Guide](docs/prompting.md)
- [Safety Guide](docs/safety.md)

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

Both model weights are under our [community license](https://www.krea.ai/krea-2-licensing) with permissive use. To purchase a commercial license, please contact us at [opensource@krea.ai](mailto:opensource@krea.ai).

## Citation

If you use Krea 2 in your work, please cite:

```bibtex
@misc{krea2_2026,
  title        = {Krea 2 Technical Report},
  author       = {Krea AI},
  year         = {2026},
  howpublished = {\url{https://www.krea.ai/blog/krea-2-technical-report}}
}
```
