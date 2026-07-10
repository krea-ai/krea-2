"""Datasets and deterministic sampling for LoRA training."""

import csv
import hashlib
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


class ImagePromptDataset(Dataset):
    def __init__(self, csv_path: str | Path, size: int = 512):
        self.csv_path = Path(csv_path)
        self.size = size
        with self.csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            if (
                "image_path" not in reader.fieldnames
                or "prompt" not in reader.fieldnames
            ):
                raise ValueError("CSV must contain image_path,prompt columns")
            self.rows = list(reader)
        if not self.rows:
            raise ValueError(f"{self.csv_path} is empty")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = Path(row["image_path"])
        if not path.is_absolute():
            path = self.csv_path.parent / path
        image = Image.open(path).convert("RGB")
        image = TF.resize(image, self.size, interpolation=InterpolationMode.BICUBIC)
        image = TF.center_crop(image, [self.size, self.size])
        tensor = TF.to_tensor(image) * 2.0 - 1.0
        return tensor, row["prompt"]


def read_csv_prompts(csv_path: str | Path) -> list[str]:
    path = Path(csv_path)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if "prompt" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a prompt column")
        prompts = [
            row["prompt"].strip() for row in reader if row.get("prompt", "").strip()
        ]
    if not prompts:
        raise ValueError(f"{path} does not contain any prompts")
    return prompts


class PromptDataset(Dataset):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.prompts = [
            line.strip() for line in self.path.read_text().splitlines() if line.strip()
        ]
        if not self.prompts:
            raise ValueError(f"{self.path} does not contain any prompts")

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


class CachedSFTDataset(Dataset):
    def __init__(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        text_masks: torch.Tensor,
    ):
        if not (latents.shape[0] == text_embeddings.shape[0] == text_masks.shape[0]):
            raise ValueError("cached latents/text tensors must have the same length")
        self.latents = latents.detach().cpu().contiguous()
        self.text_embeddings = text_embeddings.detach().cpu().contiguous()
        self.text_masks = text_masks.detach().cpu().bool().contiguous()

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        return self.latents[idx], self.text_embeddings[idx], self.text_masks[idx]


class CachedDraftPromptDataset(Dataset):
    def __init__(
        self,
        prompts: list[str],
        text_embeddings: torch.Tensor,
        text_masks: torch.Tensor,
        negative_text_embeddings: torch.Tensor,
        negative_text_masks: torch.Tensor,
    ):
        if not (
            len(prompts)
            == text_embeddings.shape[0]
            == text_masks.shape[0]
            == negative_text_embeddings.shape[0]
            == negative_text_masks.shape[0]
        ):
            raise ValueError(
                "cached draft prompts/text tensors must have the same length"
            )
        self.prompts = list(prompts)
        self.text_embeddings = text_embeddings.detach().cpu().contiguous()
        self.text_masks = text_masks.detach().cpu().bool().contiguous()
        self.negative_text_embeddings = (
            negative_text_embeddings.detach().cpu().contiguous()
        )
        self.negative_text_masks = (
            negative_text_masks.detach().cpu().bool().contiguous()
        )

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {
            "prompt": self.prompts[idx],
            "text_embeddings": self.text_embeddings[idx],
            "text_masks": self.text_masks[idx],
            "negative_text_embeddings": self.negative_text_embeddings[idx],
            "negative_text_masks": self.negative_text_masks[idx],
        }


class StatefulShuffleBatchSampler(Sampler[list[int]]):
    """Infinite shuffled batches with an exactly serializable cursor."""

    def __init__(self, dataset_size: int, batch_size: int, seed: int):
        if dataset_size <= 0 or batch_size <= 0 or batch_size > dataset_size:
            raise ValueError("invalid dataset or batch size")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.permutation = torch.empty(0, dtype=torch.int64)
        self.cursor = 0
        self.epoch = 0

    def __iter__(self):
        while True:
            if self.cursor + self.batch_size > self.permutation.numel():
                self.permutation = torch.randperm(
                    self.dataset_size, generator=self.generator
                )
                self.cursor = 0
                self.epoch += 1
            batch = self.permutation[
                self.cursor : self.cursor + self.batch_size
            ].tolist()
            self.cursor += self.batch_size
            yield batch

    def __len__(self):
        return 2**31

    def state_dict(self) -> dict:
        return {
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "generator_state": self.generator.get_state(),
            "permutation": self.permutation.clone(),
            "cursor": self.cursor,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state: dict) -> None:
        if state["dataset_size"] != self.dataset_size:
            raise ValueError("sampler dataset size changed")
        if state["batch_size"] != self.batch_size:
            raise ValueError("sampler batch size changed")
        permutation = state["permutation"].to(dtype=torch.int64, device="cpu")
        if permutation.numel() not in {0, self.dataset_size}:
            raise ValueError("invalid sampler permutation")
        cursor = int(state["cursor"])
        if cursor < 0 or cursor > permutation.numel():
            raise ValueError("invalid sampler cursor")
        self.generator.set_state(state["generator_state"].cpu())
        self.permutation = permutation.clone()
        self.cursor = cursor
        self.epoch = int(state["epoch"])


def _resolved_image_path(dataset: ImagePromptDataset, row: dict) -> Path:
    path = Path(row["image_path"])
    return path if path.is_absolute() else dataset.csv_path.parent / path


def sft_dataset_signature(dataset: ImagePromptDataset) -> str:
    """Content signature for captions, ordering, and every training image."""
    digest = hashlib.sha256(dataset.csv_path.read_bytes())
    for row in dataset.rows:
        path = _resolved_image_path(dataset, row).resolve()
        digest.update(str(path).encode())
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
