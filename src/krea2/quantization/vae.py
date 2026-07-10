"""Memory-lean single-frame decode for the Qwen-Image VAE — no tiling.

diffusers' AutoencoderKLQwenImage is a Wan-style *video* VAE; decoding one
image drags in machinery that transiently costs GiBs at 1024px+:

  - QwenImageRMS_norm upcasts the whole feature map to fp32 (`x.float()`)
    and divides in fp32: ~3 full-map fp32 temporaries, ~0.4 GiB each at
    96x1024x1024.
  - QwenImageUpsample runs nearest-exact interpolation through fp32 — a pure
    index gather that needs no precision — ~1.2 GiB transient at the last
    upsample (192ch fp32 at 1024px).
  - Every QwenImageCausalConv3d F.pads two all-zero causal history frames
    into a fresh tensor and convolves a 3-frame temporal window: one extra
    full-map copy and 3x the useful FLOPs per conv.
  - _decode runs the decoder with feat_cache enabled (video chunking state):
    every conv clone()s its full input into the cache — useless for T=1 and
    over 1 GiB held simultaneously at full resolution.

For T=1 all of this is removable without changing the math.
optimize_vae_decode(ae.ae) rebinds the module forwards (class swaps on
instances; the diffusers library itself is untouched) and skips the feature
cache. Multi-frame (video) inputs fall back to the stock implementation.
"""

import types

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders import autoencoder_kl_qwenimage as _qi
from diffusers.models.autoencoders.autoencoder_kl_qwenimage import (
    AutoencoderKLQwenImage,
    QwenImageCausalConv3d,
    QwenImageRMS_norm,
    QwenImageUpsample,
)


class LeanCausalConv3d(QwenImageCausalConv3d):
    def forward(self, x, cache_x=None):
        if cache_x is None and x.shape[2] == 1:
            # Single frame: the causal temporal padding is all zeros, so only
            # the last temporal slice of the kernel can contribute. Run as a
            # 2D conv with native spatial padding: no F.pad copy, no FLOPs
            # spent on zero frames. Exact for any temporal kernel size.
            y = F.conv2d(
                x.squeeze(2),
                self.weight[:, :, -1],
                self.bias,
                stride=self.stride[1:],
                padding=(self._padding[2], self._padding[0]),
                dilation=self.dilation[1:],
                groups=self.groups,
            )
            return y.unsqueeze(2)
        return super().forward(x, cache_x)


class LeanRMSNorm(QwenImageRMS_norm):
    def forward(self, x):
        dim = 1 if self.channel_first else -1
        # F.normalize(x.float()) semantics with an on-the-fly fp32 reduction:
        # only the (B, 1, ...) norm is fp32, the full map never leaves bf16.
        norm = torch.linalg.vector_norm(x, dim=dim, keepdim=True, dtype=torch.float32)
        mult = (self.scale / norm.clamp_min_(1e-12)).to(x.dtype)
        return x * mult * self.gamma + self.bias


class LeanUpsample(QwenImageUpsample):
    def forward(self, x):
        # nearest-exact is a pure index gather; the fp32 round-trip of the
        # stock forward changes nothing but allocates 3x the map in fp32.
        return nn.Upsample.forward(self, x)


def _lean_decode(self, z: torch.Tensor, return_dict: bool = True):
    if z.shape[2] != 1 or self.use_tiling:
        return AutoencoderKLQwenImage._decode(self, z, return_dict=return_dict)
    # Stock _decode threads feat_cache through the decoder, making every conv
    # clone its input into video-chunking state that a single frame never uses.
    x = self.post_quant_conv(z)
    out = torch.clamp(self.decoder(x), min=-1.0, max=1.0)
    if not return_dict:
        return (out,)
    return _qi.DecoderOutput(sample=out)


_LEAN_CLASSES = {
    QwenImageCausalConv3d: LeanCausalConv3d,
    QwenImageRMS_norm: LeanRMSNorm,
    QwenImageUpsample: LeanUpsample,
}


def optimize_vae_decode(vae: AutoencoderKLQwenImage) -> AutoencoderKLQwenImage:
    """Make single-frame decode fit without tiling. Class swaps on module
    instances plus an instance-level _decode override; no parameters move."""
    for module in vae.modules():
        lean = _LEAN_CLASSES.get(type(module))
        if lean is not None:
            module.__class__ = lean
    vae._decode = types.MethodType(_lean_decode, vae)
    return vae
