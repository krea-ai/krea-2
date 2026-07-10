"""INT8 inference for Krea 2 (K2).

Same CLI as inference.py, but the DiT and, by default, the text encoder load
eligible linear layers as int8. Each INT8 linear takes a bf16 activation,
quantizes it with 128x128 (M, K) activation blocks, then runs a matching
128x128 activation/weight block GEMM through :mod:`krea2.kernels.int8`.

The DiT keeps first/last and timestep/text conditioning MLPs in bf16, matching
the FP8 inference path's quality-sensitive skips. Attention runs in bf16
through torch SDPA. The VAE remains bf16 and uses the lean decode optimization
from :mod:`krea2.quantization.vae`.

--bf16-text-encoder keeps the text encoder weights in bf16 instead: they park
in pinned host memory and are copied to the GPU only while encoding.
"""

# ruff: noqa: E402  # Configure allocator/cache environment before torch imports.

import os
from pathlib import Path

# Must be set before the first CUDA allocation: with large resident weights on
# a 24 GB card, allocator fragmentation can otherwise cost substantial memory.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Keep Triton artifacts in a stable cache across process starts.
_cache_root = os.path.expanduser(os.environ.get("KREA2_CACHE_DIR", "~/.cache/krea-2"))
os.makedirs(_cache_root, exist_ok=True)
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, "triton"))

import click
import torch
from safetensors import safe_open

from krea2.inference.bf16 import checkpoints, qwen3_vl_4b, single_mmdit_large_wide
from krea2.inference.sampling import sample
from krea2.models.autoencoder import QwenAutoencoder
from krea2.models.conditioner import Qwen3VLConditionerLowMem
from krea2.models.transformer import SingleStreamDiT
from krea2.quantization.int8 import (
    LinearLoraINT8,
    add_lora_to_int8_blocks,
    convert_linears_int8,
    load_int8_state_dict,
    load_lora_adapters,
    swap_linears_meta,
)
from krea2.quantization.vae import optimize_vae_decode


def _adapter_metadata(path: str | os.PathLike) -> dict:
    """Read LoRA rank/scale metadata, falling back to tensor shapes."""
    path = Path(path)
    with safe_open(str(path), framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
        keys = list(f.keys())
        lora_a_keys = sorted(key for key in keys if key.endswith(".lora_A"))
        if not lora_a_keys:
            raise ValueError(f"{path} does not contain any LoRA A tensors")

        if "rank" in metadata:
            rank = int(metadata["rank"])
        else:
            rank = int(f.get_tensor(lora_a_keys[0]).shape[0])
        if rank not in (32, 64):
            raise ValueError(f"supported LoRA ranks are 32 and 64, got {rank}")

        alpha = float(metadata.get("lora_alpha", rank))
        checkpoint_scale = float(metadata.get("lora_scale", 1.0))
        targets = [line for line in metadata.get("targets", "").splitlines() if line]
        if not targets:
            targets = [key[: -len(".lora_A")] for key in lora_a_keys]

    return {
        "rank": rank,
        "alpha": alpha,
        "checkpoint_scale": checkpoint_scale,
        "targets": targets,
        "metadata": metadata,
    }


def load_lora_into_int8_model(
    model: torch.nn.Module,
    path: str | os.PathLike,
    extra_scale: float = 1.0,
    strict: bool = True,
) -> dict:
    """Insert INT8 LoRA modules and load an adapter safetensors file.

    The effective scale is:
        alpha / rank * checkpoint lora_scale * extra_scale
    """
    config = _adapter_metadata(path)
    converted = add_lora_to_int8_blocks(
        model,
        rank=config["rank"],
        alpha=config["alpha"],
    )
    if not converted:
        raise RuntimeError("no INT8 DiT block linears were converted to LoRA")
    effective_multiplier = float(config["checkpoint_scale"]) * float(extra_scale)
    for module in model.modules():
        if isinstance(module, LinearLoraINT8):
            module.lora_scale *= effective_multiplier
    load_lora_adapters(model, path, strict=strict)
    config["converted"] = converted
    config["effective_scale"] = (
        float(config["alpha"]) / float(config["rank"]) * effective_multiplier
    )
    return config


def build_int8_pipeline(
    mmdit_config=single_mmdit_large_wide,
    text_encoder_config=qwen3_vl_4b,
    checkpoint="oss_raw",
    device="cuda",
    dtype=torch.bfloat16,
    bf16_text_encoder=False,
    lora=None,
    lora_scale=1.0,
    quantization_type="blockwise",
):
    """Build the autoencoder, text encoder and INT8 MMDiT on `device`.

    The DiT is built on meta, eligible linears are swapped for INT8Linear
    shells, and the checkpoint is streamed straight into int8 on the GPU.
    `first`, `last`, `tmlp`, `tproj` and `txtmlp` stay in bf16 for the same
    quality and stability reasons as the FP8 path.
    """
    with torch.device("meta"):
        mmdit = SingleStreamDiT(mmdit_config)
    swap_linears_meta(
        mmdit,
        skip=("first", "last", "tmlp", "tproj", "txtmlp"),
        quantization_type=quantization_type,
    )
    load_int8_state_dict(mmdit, checkpoints[checkpoint], device=device, dtype=dtype)
    if lora is not None:
        config = load_lora_into_int8_model(mmdit, lora, extra_scale=lora_scale)
        click.echo(
            "loaded LoRA "
            f"{lora} rank={config['rank']} alpha={config['alpha']:g} "
            f"scale={config['effective_scale']:g} targets={len(config['targets'])}"
        )
        metadata = config.get("metadata", {})
        adapter_quantization = metadata.get("quantization_type")
        if adapter_quantization and adapter_quantization != quantization_type:
            click.echo(
                "warning: adapter was trained with quantization_type="
                f"{adapter_quantization}, but inference is using {quantization_type}"
            )
        if metadata.get("objective") in {"sft", "flow"} and (
            metadata.get("flow_convention") != "krea_t1_noise_t0_data"
        ):
            click.echo(
                "warning: this SFT/flow LoRA adapter lacks the corrected Krea "
                "flow_convention metadata; adapters trained before the "
                "t=1-noise/t=0-data fix can leave generations as noise."
            )
    mmdit.eval().requires_grad_(False)
    encoder = Qwen3VLConditionerLowMem(
        text_encoder_config.model_id,
        text_encoder_config.max_length,
        select_layers=text_encoder_config.select_layers,
        device=device,
        dtype=dtype,
        bf16_weights=bf16_text_encoder,
        quantize_linears=lambda module: convert_linears_int8(
            module,
            device=device,
            skip=("lm_head",),
            quantization_type=quantization_type,
        ),
    )

    ae = QwenAutoencoder()
    optimize_vae_decode(ae.ae)
    ae = ae.to(device=device, dtype=dtype).eval().requires_grad_(False)

    return mmdit, ae, encoder


@click.command(help="Generate images with Krea 2 (K2) using an INT8 DiT.")
@click.argument("prompt")
@click.option(
    "--steps", default=28, show_default=True, help="number of denoising steps"
)
@click.option(
    "--cfg",
    default=4.5,
    show_default=True,
    help="classifier-free guidance scale (0 disables CFG)",
)
@click.option(
    "--y1",
    default=0.5,
    show_default=True,
    help="timestep-shift mu at min resolution",
)
@click.option(
    "--y2",
    default=1.15,
    show_default=True,
    help="timestep-shift mu at max resolution",
)
@click.option(
    "--width",
    default="1024",
    show_default=True,
    type=click.Choice(["512", "1024"]),
    help="fixed square output width",
)
@click.option(
    "--height",
    default="1024",
    show_default=True,
    type=click.Choice(["512", "1024"]),
    help="fixed square output height",
)
@click.option(
    "--num-images",
    default=1,
    show_default=True,
    help="number of images to generate from the prompt",
)
@click.option(
    "--seed", default=0, show_default=True, help="base seed; image i uses seed + i"
)
@click.option(
    "--checkpoint",
    envvar="K2_CHECKPOINT",
    default="oss_raw",
    show_default=True,
    type=click.Choice(list(checkpoints)),
)
@click.option(
    "--mu",
    default=None,
    help="timestep-shift mu",
    type=float,
)
@click.option(
    "--output", default="sample", show_default=True, help="output filename prefix"
)
@click.option(
    "--lora",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="LoRA adapter safetensors file produced by scripts/train.py",
)
@click.option(
    "--lora-scale",
    default=1.0,
    show_default=True,
    type=float,
    help="extra multiplier applied on top of the adapter's saved LoRA scale",
)
@click.option(
    "--bf16-text-encoder",
    is_flag=True,
    default=False,
    show_default=True,
    help="keep text encoder weights in bf16 (host-offloaded between encodes) "
    "instead of int8",
)
@click.option(
    "--quantization-type",
    default="blockwise",
    show_default=True,
    type=click.Choice(["blockwise", "rowwise"]),
    help="INT8 scale geometry; rowwise uses rowwise activations and one weight scale",
)
def main(
    prompt,
    steps,
    cfg,
    y1,
    y2,
    width,
    height,
    num_images,
    seed,
    checkpoint,
    output,
    lora,
    lora_scale,
    mu,
    bf16_text_encoder,
    quantization_type,
):
    width, height = int(width), int(height)
    if width != height:
        raise click.ClickException(
            "INT8 inference currently supports only square 512 or 1024 outputs"
        )
    if lora is None and lora_scale != 1.0:
        raise click.ClickException("--lora-scale requires --lora")
    dit, ae, encoder = build_int8_pipeline(
        checkpoint=checkpoint,
        bf16_text_encoder=bf16_text_encoder,
        lora=lora,
        lora_scale=lora_scale,
        quantization_type=quantization_type,
    )

    images = sample(
        dit,
        ae,
        encoder,
        [prompt] * num_images,
        width=width,
        height=height,
        steps=steps,
        guidance=cfg,
        seed=seed,
        y1=y1,
        y2=y2,
        mu=mu,
    )
    for i, image in enumerate(images):
        out = f"{output}_{i}.png"
        image.save(out)
        click.echo(f"saved {out}")


if __name__ == "__main__":
    main()
