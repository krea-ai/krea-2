"""Small CPU correctness tests for training objectives."""

from types import SimpleNamespace

import torch
import torch.nn as nn

from krea2.training.objectives import draft_sample_images


class TinyFlow(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.1, dtype=torch.bfloat16))
        self.config = SimpleNamespace(patch=2)

    def forward(self, *, img, context, t, pos, mask):
        del context, t, pos, mask
        return img * self.scale


class TinyAutoencoder(nn.Module):
    channels = 4
    compression = 64

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.bfloat16))

    def decode(self, latents):
        return latents[:, :3] * self.scale


def test_draft_lv_adds_detached_last_step_samples():
    model = TinyFlow()
    ae = TinyAutoencoder()
    embeddings = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    masks = torch.ones(1, 2, dtype=torch.bool)

    images = draft_sample_images(
        model,
        ae,
        ["portrait"],
        text_embeddings=embeddings,
        text_masks=masks,
        negative_text_embeddings=embeddings,
        negative_text_masks=masks,
        steps=2,
        draft_k=1,
        guidance=4.5,
        seed=42,
        y1=0.5,
        y2=1.15,
        mu=None,
        checkpoint_vae=False,
        lv_samples=1,
    )

    assert images.shape == (2, 3, 8, 8)
    images.float().mean().backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad)
