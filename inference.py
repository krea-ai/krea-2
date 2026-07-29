import os

import click
import torch
from safetensors.torch import load_file

from autoencoder import QwenAutoencoder
from encoder import Qwen3VLConditioner, TextEncoderConfig
from mmdit import SingleMMDiTConfig, SingleStreamDiT
from sampling import sample

single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)

qwen3_vl_4b = TextEncoderConfig(model_id="Qwen/Qwen3-VL-4B-Instruct")
checkpoints = {
    "oss_raw": os.environ.get("OSS_RAW"),
    "oss_turbo": os.environ.get("OSS_TURBO"),
}


def _pipeline(
    mmdit_config=single_mmdit_large_wide,
    text_encoder_config=qwen3_vl_4b,
    checkpoint="oss_raw",
    device="cuda",
    dtype=torch.bfloat16,
):
    """Build the autoencoder, text encoder, and MMDiT, load weights, and move to GPU."""
    ae = QwenAutoencoder()
    encoder = Qwen3VLConditioner(
        text_encoder_config.model_id,
        text_encoder_config.max_length,
        select_layers=text_encoder_config.select_layers,
    )

    # Build on meta, load to passed device
    with torch.device("meta"):
        mmdit = SingleStreamDiT(mmdit_config)

    ckpt = checkpoints[checkpoint]
    mmdit.load_state_dict(load_file(ckpt), strict=True, assign=True)
    mmdit = mmdit.to(device=device, dtype=dtype).eval().requires_grad_(False)
    ae = ae.to(device=device, dtype=dtype).eval().requires_grad_(False)
    encoder = encoder.to(device=device, dtype=dtype).eval().requires_grad_(False)

    return mmdit, ae, encoder


@click.command(help="Generate images with Krea 2 (K2).")
@click.argument("prompt")
@click.option(
    "--steps", default=52, show_default=True, help="number of denoising steps"
)
@click.option(
    "--cfg",
    default=3.5,
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
def main(
    prompt, steps, cfg, y1, y2, width, height, num_images, seed, checkpoint, output, mu
):
    dit, ae, encoder = _pipeline(checkpoint=checkpoint)

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
