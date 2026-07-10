"""Exact, self-contained SFT training state persistence."""

from pathlib import Path

import torch
from torch import nn

from krea2.quantization.int8 import lora_state_tensors
from krea2.training.data import CachedSFTDataset, StatefulShuffleBatchSampler

TRAINING_STATE_VERSION = 1


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def write_training_state(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    dataset: CachedSFTDataset,
    sampler: StatefulShuffleBatchSampler,
    global_step: int,
    compatibility: dict,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format_version": TRAINING_STATE_VERSION,
        "global_step": int(global_step),
        "compatibility": compatibility,
        "lora": lora_state_tensors(model),
        "optimizer": _cpu_tree(optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "sampler": sampler.state_dict(),
        "cached_sft": {
            "latents": dataset.latents,
            "text_embeddings": dataset.text_embeddings,
            "text_masks": dataset.text_masks,
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return path


def load_training_state(path: str | Path) -> dict:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or state.get("format_version") != TRAINING_STATE_VERSION
    ):
        raise ValueError(f"unsupported training state: {path}")
    required = {
        "global_step",
        "compatibility",
        "lora",
        "optimizer",
        "torch_rng_state",
        "cuda_rng_states",
        "sampler",
        "cached_sft",
    }
    missing = required - state.keys()
    if missing:
        raise ValueError(f"training state is missing: {sorted(missing)}")
    return state


def validate_training_state(state: dict, expected: dict) -> None:
    actual = state["compatibility"]
    mismatches = {
        key: (actual.get(key), value)
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        details = ", ".join(
            f"{key}: saved={saved!r}, requested={requested!r}"
            for key, (saved, requested) in sorted(mismatches.items())
        )
        raise ValueError(f"incompatible training state ({details})")


def restore_training_rng(state: dict) -> None:
    torch.set_rng_state(state["torch_rng_state"].cpu())
    cuda_states = state["cuda_rng_states"]
    if cuda_states and torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA device count changed since the training state was saved"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
