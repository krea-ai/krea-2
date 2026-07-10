"""FP8 inference for Krea 2 (K2).

Same CLI as inference.py, but the DiT and (by default) the text encoder load
their linear layers as fp8 e4m3 (rowwise scales) and run them through
torch.nn.functional.scaled_mm; in the DiT the quantization is fused into the
preceding op (RMSNorm / GELU / SwiGLU / attention out-gate) by the Triton
kernels in :mod:`krea2.kernels.fp8`. Attention runs in bf16 through torch
SDPA in both models. The VAE is unchanged (bf16, lean single-frame decode).

--bf16-text-encoder keeps the text encoder weights in bf16 instead: they park
in pinned host memory and are copied to the GPU only while encoding, since
bf16 encoder (~8 GiB) + fp8 DiT (~12 GiB) + activations exceed a 24 GB card.

On a 24 GB card fp8 is what makes K2 fit at all: the bf16 DiT checkpoint
alone is 26 GB.
"""

import os

# Must be set before the first CUDA allocation: with ~20+ GiB resident on a
# 24 GB card, allocator fragmentation (reserved-but-unallocated blocks) is
# worth up to a GiB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import click
import torch

from krea2.inference.bf16 import checkpoints, qwen3_vl_4b, single_mmdit_large_wide
from krea2.inference.sampling import sample
from krea2.models.autoencoder import QwenAutoencoder
from krea2.models.conditioner import Qwen3VLConditionerLowMem
from krea2.models.transformer import SingleStreamDiT
from krea2.quantization.fp8 import (
    apply_fused_forwards,
    convert_linears_fp8,
    load_fp8_state_dict,
    swap_linears_meta,
)
from krea2.quantization.vae import optimize_vae_decode


def build_fp8_pipeline(
    mmdit_config=single_mmdit_large_wide,
    text_encoder_config=qwen3_vl_4b,
    checkpoint="oss_raw",
    device="cuda",
    dtype=torch.bfloat16,
    bf16_text_encoder=False,
):
    """Build the autoencoder, text encoder (fp8, or bf16 host-offloaded with
    bf16_text_encoder=True), and fp8 MMDiT on `device`.

    The DiT is built on meta, its Attention/SwiGLU/t-mlp linears are swapped
    for FP8Linear shells, and the checkpoint is streamed straight into fp8 on
    the GPU (the 26 GB bf16 checkpoint never materializes in memory). `first`
    and `last` stay in bf16: boundary layers are the most quantization-
    sensitive and contribute nothing to runtime.
    """
    # DiT first: quantize-on-load wants headroom, and the encoder is the
    # single biggest contiguous block — allocate it after the DiT's many small
    # ones to keep fragmentation down.
    #
    # Besides the boundary layers, the timestep/text conditioning MLPs stay in
    # bf16: their outputs modulate every block ((1 + scale) * x + shift and the
    # residual gates), so quantization noise there is applied multiplicatively
    # to the whole stream 28 times — measured as the largest single fp8 error
    # contributor — while their GEMMs are a rounding error of the runtime
    # (M = batch or text length only).
    with torch.device("meta"):
        mmdit = SingleStreamDiT(mmdit_config)
    swap_linears_meta(mmdit, skip=("first", "last", "tmlp", "tproj", "txtmlp"))
    apply_fused_forwards(mmdit)
    load_fp8_state_dict(mmdit, checkpoints[checkpoint], device=device, dtype=dtype)
    mmdit.eval().requires_grad_(False)

    encoder = Qwen3VLConditionerLowMem(
        text_encoder_config.model_id,
        text_encoder_config.max_length,
        select_layers=text_encoder_config.select_layers,
        device=device,
        dtype=dtype,
        bf16_weights=bf16_text_encoder,
        quantize_linears=lambda module: convert_linears_fp8(
            module, device=device, skip=("lm_head",)
        ),
    )

    ae = QwenAutoencoder()
    # With ~17 GiB of weights resident, the stock decode (fp32 norm/upsample
    # upcasts, temporal padding, per-conv feature cache) peaks 4+ GiB at 1024px
    # and OOMs a 24 GiB card; the lean single-frame decode peaks ~1.3 GiB.
    optimize_vae_decode(ae.ae)
    ae = ae.to(device=device, dtype=dtype).eval().requires_grad_(False)

    return mmdit, ae, encoder


@click.command(help="Generate images with Krea 2 (K2) using an FP8 DiT.")
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
@click.option("--width", default=1024, show_default=True)
@click.option("--height", default=1024, show_default=True)
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
    "--bf16-text-encoder",
    is_flag=True,
    default=False,
    show_default=True,
    help="keep text encoder weights in bf16 (host-offloaded between encodes) "
    "instead of fp8",
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
    mu,
    bf16_text_encoder,
):
    dit, ae, encoder = build_fp8_pipeline(
        checkpoint=checkpoint, bf16_text_encoder=bf16_text_encoder
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
