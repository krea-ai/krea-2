# Krea 2 Character Training

Krea 2 - an image generation model from [Krea AI](https://www.krea.ai/).

This repository is a fork of
[krea-ai/krea-2](https://github.com/krea-ai/krea-2), focused on fast character
LoRA training with SFT and DRaFT-K.

## Character training

Install the training dependencies and point `OSS_RAW` at the Krea 2 RAW
checkpoint:

```bash
uv sync --extra train --extra face-reward
export OSS_RAW=/models/krea2-raw.safetensors
export DEEPINFRA_KEY=your_key

uv run --extra train --extra face-reward train_face.py IMAGES \
  --output-dir runs/my_character --trigger-word optional_token
```

`IMAGES` is searched recursively for JPG, PNG, WebP, BMP, or AVIF files. The
trigger word is optional. Missing antelopev2 face models are downloaded
automatically. Training defaults to 500 SFT updates followed by 60 QKVO-only
DRaFT-LV updates.

### Dataset formats

Automatic mode captions the image folder and creates 64 DRaFT prompts. For
offline training, captions use an `image_path,prompt` CSV:

```csv
image_path,prompt
images/01.jpg,A woman standing in a softly lit studio.
```

Relative image paths are resolved from the CSV location. Alternatively, place
a one-line caption beside every image (`01.jpg` and `01.txt`). DRaFT prompt
files contain one prompt per line.

Ready-made prompt files are included for
[women](prompts/draft_woman.txt) and [men](prompts/draft_man.txt).

### Training without DeepInfra

Provide local captions and prompts to bypass both DeepInfra stages:

```bash
uv run --extra train --extra face-reward train_face.py IMAGES \
  --output-dir runs/my_character \
  --captions /path/to/captions.csv \
  --draft-prompts prompts/draft_woman.txt
```

For a custom differentiable reward, use `main.py`. Individual SFT, DRaFT-K,
and flow objectives are available through `scripts/train.py`.

### ComfyUI conversion

```bash
uv run scripts/convert_lora_to_comfyui.py \
  runs/my_character/draft/lora_latest.safetensors \
  runs/my_character/comfyui_lora.safetensors
```

## Setup

For a complete development environment:

```bash
uv sync --extra train --extra face-reward --extra dev
```

PyTorch and torchvision intentionally have no version constraints. Download
Krea 2 RAW and Turbo from Hugging Face and set their paths as needed:

```bash
export OSS_RAW=/models/krea2-raw.safetensors
export OSS_TURBO=/models/krea2-turbo.safetensors
```

## License

The source code in this repository is released under Apache-2.0. The Krea 2
model weights are governed separately by the
[Krea 2 Community License Agreement](https://www.krea.ai/krea-2-licensing).
See [`NOTICE`](NOTICE) for third-party notices and licensing distinctions.

The bundled InsightFace model weights also have separate usage terms; review
their license before commercial use.

## Citation

```bibtex
@misc{krea-2-2026,
  author={Sangwu Lee, Erwann Millon, Le Zhuo, Matthew Newton, Andrei Filatov,
          Abhinay Devarinti, Dazhi Zhong, Avram Djordjevic, Gabriel Menezes,
          Will Beddow, Titus Ebbecke, Mihai Petrescu, Owen Fahey, Gian Saß,
          Felix Gil, Victor Perez},
  title={{Krea 2}},
  year={2026},
  howpublished={\url{https://www.krea.ai/blog/krea-2-technical-report}},
}
```

Thanks to [Krea AI](https://www.krea.ai/) and the
[krea-ai/krea-2](https://github.com/krea-ai/krea-2) contributors for creating
and releasing the model.
