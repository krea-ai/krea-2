# Architecture

Krea 2 uses a standard installed `src` package. The root contains only the two
end-to-end training entry points; lower-level and research commands live under
`scripts/`.

```text
krea-2/
  main.py                         generic custom-reward pipeline
  train_face.py                   face-specialized pipeline
  src/krea2/
    models/                       transformer, conditioner, autoencoder
    inference/                    BF16/FP8/INT8 pipelines and sampling
    quantization/                 FP8, INT8, LoRA, VAE optimization
    kernels/                      normally imported Triton kernels
    training/                     data, state, objectives, validation, lifecycle
    preprocessing/                image discovery, captioning, prompts
    rewards/                      reward implementations and face-model manager
    experiments/                  controlled comparison and metrics
  scripts/                        thin executable adapters and converters
  tests/                          pytest CPU/CUDA coverage
  docs/
```

`uv sync` installs `krea2`, so runtime modules and tests use the same package
imports. There is no local package named `triton`; custom kernels import
normally from `krea2.kernels` and coexist with the upstream Triton runtime.

## Boundaries

- Model definitions do not parse command-line arguments.
- Inference builders resolve checkpoints and assemble reusable pipelines;
  Click modules only adapt arguments.
- The trainer separates cached data, exact state persistence, objectives,
  validation, and the optimization lifecycle.
- End-to-end orchestration invokes SFT and DRaFT-K in separate subprocesses.
- Generic orchestration accepts a `RewardSpec` and never imports antelopev2,
  OpenCV, ONNX Runtime, or InsightFace.
- `train_face.py` is the boundary that resolves antelopev2 and injects
  `FaceSimilarityReward` plus the reference image paths.

## Script inventory

| Command | Purpose |
| --- | --- |
| `main.py` | Caption, SFT, then DRaFT-K with a required custom reward. |
| `train_face.py` | The same pipeline with face-model management and face reward. |
| `scripts/train.py` | Direct SFT, DRaFT-K, or flow-matching trainer. |
| `scripts/compare.py` | Controlled hybrid-versus-SFT character experiment. |
| `scripts/inference*.py` | BF16, FP8, and INT8 image generation. |
| `scripts/caption_images_deepinfra.py` | Standalone caption adapter. |
| `scripts/generate_character_prompts_deepinfra.py` | Standalone prompt adapter. |
| `scripts/convert_images_to_jpg.py` | Recursive image-to-JPEG conversion. |
| `scripts/convert_lora_to_comfyui.py` | Native adapter to ComfyUI conversion. |
| `scripts/check_release.py` | Lock, lint, tests, imports, and CLI release gate. |

## Checkpoints and resuming

Native LoRA tensor names and metadata remain compatible with the former flat
implementation. Exact SFT state contains LoRA tensors, optimizer moments,
global step, CPU/CUDA RNG, sampler permutation/cursor, cached SFT tensors, and
compatibility metadata. `--resume-training-state` restores the entire state;
`--resume-lora` intentionally restores only the adapter.

## Optional dependencies

The base install supports inference. `train` adds torchvision, `face-reward`
adds face and ONNX dependencies, and `dev` adds pytest and Ruff. Torch packages
are unconstrained and resolve to the latest compatible CUDA 12.8 builds; the
lockfile is a reproducible environment snapshot, not a source constraint.

## Release gate

```bash
uv sync --extra train --extra face-reward --extra dev
uv run scripts/check_release.py
uv run scripts/check_release.py --cuda
```

The default gate checks the lockfile, Ruff, CPU tests, installed imports, and
every CLI's `--help`. `--cuda` adds blockwise/rowwise rank-32/64 LoRA kernels,
compiled checkpointing, full save/load paths, face-reward gradients, and ONNX
recognition parity.
