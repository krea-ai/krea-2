"""CPU caches and conditioner device lifecycle for training."""

import click
import torch
from torch import nn
from torch.utils.data import DataLoader

from krea2.training.data import (
    CachedDraftPromptDataset,
    CachedSFTDataset,
    ImagePromptDataset,
    PromptDataset,
)
from krea2.training.objectives import encode_latents
from krea2.training.validation import apply_trigger_word


@torch.no_grad()
def build_sft_cache(
    dataset: ImagePromptDataset,
    ae,
    encoder,
    *,
    trigger_word: str | None,
    batch_size: int,
    num_workers: int,
) -> CachedSFTDataset:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    latent_chunks = []
    text_chunks = []
    mask_chunks = []
    with click.progressbar(loader, length=len(loader), label="caching SFT") as bar:
        for images, prompts in bar:
            prompts = apply_trigger_word(list(prompts), trigger_word)
            latents = encode_latents(ae, images)
            txt, txtmask = encoder(prompts)
            latent_chunks.append(latents.detach().cpu())
            text_chunks.append(txt.detach().cpu())
            mask_chunks.append(txtmask.detach().cpu())
    return CachedSFTDataset(
        torch.cat(latent_chunks, dim=0),
        torch.cat(text_chunks, dim=0),
        torch.cat(mask_chunks, dim=0),
    )


@torch.no_grad()
def build_draft_text_cache(
    dataset: PromptDataset,
    encoder,
    *,
    trigger_word: str | None,
    batch_size: int,
    num_workers: int,
) -> CachedDraftPromptDataset:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    prompts_all: list[str] = []
    text_chunks = []
    mask_chunks = []
    neg_text_chunks = []
    neg_mask_chunks = []
    with click.progressbar(
        loader, length=len(loader), label="caching DRaFT text"
    ) as bar:
        for prompts in bar:
            prompts = apply_trigger_word(list(prompts), trigger_word)
            negatives = [""] * len(prompts)
            txt, txtmask = encoder(prompts)
            untxt, untxtmask = encoder(negatives)
            prompts_all.extend(prompts)
            text_chunks.append(txt.detach().cpu())
            mask_chunks.append(txtmask.detach().cpu())
            neg_text_chunks.append(untxt.detach().cpu())
            neg_mask_chunks.append(untxtmask.detach().cpu())
    return CachedDraftPromptDataset(
        prompts_all,
        torch.cat(text_chunks, dim=0),
        torch.cat(mask_chunks, dim=0),
        torch.cat(neg_text_chunks, dim=0),
        torch.cat(neg_mask_chunks, dim=0),
    )


def offload_conditioners_to_cpu(ae, encoder) -> None:
    ae.to("cpu")
    encoder.to("cpu")
    if hasattr(encoder, "_device"):
        encoder._device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def offload_text_encoder_to_cpu(encoder) -> None:
    encoder.to("cpu")
    if hasattr(encoder, "_device"):
        encoder._device = torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def restore_text_encoder_to_device(encoder, device) -> None:
    device = torch.device(device)
    encoder.to(device).eval()
    if hasattr(encoder, "_device"):
        encoder._device = device


def offload_vae_encoder_to_cpu(ae) -> list[str]:
    moved = []
    vae = getattr(ae, "ae", ae)
    for name in ("encoder", "quant_conv"):
        module = getattr(vae, name, None)
        if isinstance(module, nn.Module):
            module.to("cpu")
            moved.append(name)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return moved


def restore_conditioners_to_device(ae, encoder, device) -> None:
    device = torch.device(device)
    ae.to(device=device, dtype=torch.bfloat16).eval()
    encoder.to(device).eval()
    if hasattr(encoder, "_device"):
        encoder._device = device
