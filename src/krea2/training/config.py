"""Typed configuration shared by full-pipeline entry points."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RewardSpec:
    """Import and call configuration for a differentiable DRaFT-K reward."""

    target: str
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    call_kwargs: dict[str, Any] = field(default_factory=dict)
    dependencies: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        module, separator, object_name = self.target.partition(":")
        if not separator or not module or not object_name:
            raise ValueError("reward must be in package.module:object format")
        if not isinstance(self.init_kwargs, dict) or not isinstance(
            self.call_kwargs, dict
        ):
            raise TypeError("reward kwargs must be dictionaries")


@dataclass(frozen=True)
class PipelineConfig:
    images_dir: Path
    output_dir: Path
    trigger_word: str | None = None
    checkpoint: str = "oss_raw"
    rank: int = 32
    batch_size: int = 1
    sft_steps: int = 500
    draft_steps: int = 60
    prompt_count: int = 64
    sft_lr: float = 1e-4
    draft_lr: float = 5e-5
    draft_k: int = 1
    draft_lv_samples: int = 1
    denoising_steps: int = 12
    validation_steps: int = 20
    cfg: float = 4.5
    validation_step: int = 100
    validation_size: int = 10
    seed: int = 42
    caption_model: str = "Qwen/Qwen3.6-35B-A3B"
    prompt_model: str = "Qwen/Qwen3.6-35B-A3B"
    recaption: bool = False
    regenerate_prompts: bool = False
    force: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    """Resolved inputs for the two low-level training subprocesses."""

    python: str
    dataset_csv: Path
    draft_prompts: Path
    sft_output_dir: Path
    draft_output_dir: Path
    reward: RewardSpec
    trigger_word: str | None = None
    checkpoint: str = "oss_raw"
    rank: int = 32
    batch_size: int = 1
    sft_steps: int = 500
    draft_steps: int = 60
    sft_lr: float = 1e-4
    draft_lr: float = 5e-5
    draft_k: int = 1
    draft_lv_samples: int = 1
    denoising_steps: int = 12
    validation_steps: int = 20
    cfg: float = 4.5
    validation_step: int = 100
    validation_size: int = 10
    seed: int = 42
