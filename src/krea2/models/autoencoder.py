import torch
from einops import rearrange
from torch import Tensor, nn


class QwenAutoencoder(nn.Module):
    """qwen-ae-f8-16c: the Qwen-Image VAE (f8, 16 latent channels)."""

    def __init__(self):
        super().__init__()
        from diffusers import AutoencoderKLQwenImage

        self.ae = AutoencoderKLQwenImage.from_pretrained(
            "Qwen/Qwen-Image", subfolder="vae"
        )
        self.compression = 8
        self.channels = 16
        config = getattr(self.ae, "config", self.ae)
        latents_mean = getattr(config, "latents_mean")
        latents_std = getattr(config, "latents_std")
        self.register_buffer(
            "latents_mean", torch.tensor(latents_mean).view(1, -1, 1, 1, 1)
        )
        self.register_buffer(
            "latents_std", torch.tensor(latents_std).view(1, -1, 1, 1, 1)
        )

    def decode(self, x: Tensor) -> Tensor:
        x = rearrange(x, "b c h w -> b c 1 h w")
        x = (x * self.latents_std) + self.latents_mean
        return rearrange(self.ae.decode(x).sample, "b c 1 h w -> b c h w")
