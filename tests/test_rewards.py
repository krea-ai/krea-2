"""Generic differentiable reward contract tests."""

import pytest
import torch

from krea2.training.objectives import reward_loss


def test_reward_must_return_connected_finite_tensor():
    images = torch.randn(1, 3, 4, 4, requires_grad=True)

    def connected(image, prompt, multiplier):
        assert prompt == "prompt"
        return image.mean() * multiplier

    loss, values = reward_loss(
        connected,
        images,
        ["prompt"],
        {"multiplier": 2.0},
    )
    loss.backward()
    assert values.shape == (1,)
    assert images.grad is not None

    with pytest.raises(TypeError, match="torch.Tensor"):
        reward_loss(lambda *_args, **_kwargs: 1.0, images, ["prompt"], {})
    with pytest.raises(ValueError, match="not connected"):
        reward_loss(
            lambda *_args, **_kwargs: torch.tensor(1.0),
            images,
            ["prompt"],
            {},
        )
    with pytest.raises(ValueError, match="non-finite"):
        reward_loss(
            lambda image, *_args, **_kwargs: image.mean() * torch.inf,
            images,
            ["prompt"],
            {},
        )
