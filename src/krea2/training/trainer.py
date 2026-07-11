"""DRaFT-K and flow-matching LoRA training on the Krea 2 INT8 DiT.

The base DiT weights stay frozen in 128x128 block INT8. Only FP32 LoRA
adapters inside the main MMDiT blocks are optimized.
"""

import importlib
import json
import time
from pathlib import Path

import click
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from krea2.inference.bf16 import checkpoints
from krea2.inference.int8 import build_int8_pipeline
from krea2.inference.sampling import sample as sample_images
from krea2.quantization.int8 import (
    LORA_TARGETS,
    LinearLoraINT8,
    add_lora_to_int8_blocks,
    load_lora_adapters,
    load_lora_state_tensors,
    lora_parameters,
    save_lora_adapters,
    set_lora_trainable,
)
from krea2.training.cache import (
    build_draft_text_cache,
    build_sft_cache,
    offload_conditioners_to_cpu,
    offload_text_encoder_to_cpu,
    offload_vae_encoder_to_cpu,
    restore_conditioners_to_device,
    restore_text_encoder_to_device,
)
from krea2.training.data import (
    CachedSFTDataset,
    ImagePromptDataset,
    PromptDataset,
    StatefulShuffleBatchSampler,
    sft_dataset_signature,
)
from krea2.training.objectives import (
    cached_flow_loss,
    draft_sample_images,
    flow_loss,
    high_noise_schedule_mu,
    reward_loss,
    save_draft_step_images,
)
from krea2.training.state import (
    load_training_state,
    restore_training_rng,
    validate_training_state,
    write_training_state,
)
from krea2.training.validation import (
    apply_trigger_word,
    build_validation_encoder,
    choose_final_sample_prompt,
    choose_validation_prompts,
    save_validation_images,
)


class CheckpointBlock(nn.Module):
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block

    def forward(self, x, vec, freqs, mask=None):
        if not torch.is_grad_enabled():
            return self.block(x, vec, freqs, mask)

        def run(x_, vec_, freqs_, mask_):
            return self.block(x_, vec_, freqs_, mask_)

        return checkpoint(
            run,
            x,
            vec,
            freqs,
            mask,
            use_reentrant=False,
            preserve_rng_state=False,
        )


def apply_block_checkpointing(model: nn.Module) -> nn.Module:
    for i, block in enumerate(model.blocks):
        if not isinstance(block, CheckpointBlock):
            model.blocks[i] = CheckpointBlock(block)
    return model


def compile_training_blocks(model: nn.Module) -> nn.Module:
    """Compile fixed-shape block math without CUDA graphs.

    ``mode="default"`` lets Inductor fuse normalization, modulation, residual,
    and pointwise work around the opaque INT8 custom ops. CUDA-graph modes are
    deliberately not exposed here: activation checkpoint recomputation plus
    the custom autograd ops can release temporary tensors at different points
    during graph recording and replay.
    """
    for block in model.blocks:
        block.compile(fullgraph=True)
    return model


def load_reward(spec: str, init_kwargs: dict | None = None):
    module_name, sep, object_name = spec.partition(":")
    if not sep:
        raise ValueError("reward must be in module:object format")
    module = importlib.import_module(module_name)
    obj = getattr(module, object_name)
    init_kwargs = {} if init_kwargs is None else init_kwargs
    return obj(**init_kwargs) if isinstance(obj, type) or init_kwargs else obj


def parse_json_dict(text: str | None) -> dict:
    if not text:
        return {}
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("JSON value must be an object")
    return value


def sync_if_cuda(device) -> None:
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def save_step(model, output_dir: Path, step: int, metadata: dict[str, str]):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"lora_step_{step:06d}.safetensors"
    save_lora_adapters(model, path, metadata={**metadata, "step": str(step)})
    save_lora_adapters(model, output_dir / "lora_latest.safetensors", metadata)
    return path


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def save_training_summary(
    output_dir: Path,
    *,
    objective: str,
    initial_step: int,
    final_step: int,
    step_times: list[float],
    timing_warmup_steps: int = 0,
    peak_cuda_memory_bytes: int | None = None,
) -> Path:
    timing_warmup_steps = min(
        max(int(timing_warmup_steps), 0), max(len(step_times) - 1, 0)
    )
    measured = step_times[timing_warmup_steps:]
    total = sum(measured)
    summary = {
        "objective": objective,
        "initial_step": initial_step,
        "final_step": final_step,
        "updates": len(step_times),
        "timed_updates": len(measured),
        "timing_warmup_steps": timing_warmup_steps,
        "optimization_seconds": total,
        "total_step_seconds": sum(step_times),
        "steps_per_second": len(measured) / max(total, 1e-12),
        "step_times_seconds": step_times,
        "peak_cuda_memory_bytes": peak_cuda_memory_bytes,
        "latency_seconds": {
            "p20": _percentile(measured, 0.2),
            "median": _percentile(measured, 0.5),
            "p80": _percentile(measured, 0.8),
        },
    }
    path = output_dir / "training_summary.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def save_final_sample(
    model,
    ae,
    encoder,
    *,
    prompt: str,
    output_dir: Path,
    objective: str,
    train_steps: int,
    steps: int,
    cfg: float,
    seed: int,
    y1: float,
    y2: float,
    mu: float | None,
    high_noise_shift: float = 0.5,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_mu = high_noise_schedule_mu(
        1024,
        minres=256,
        maxres=1280,
        y1=y1,
        y2=y2,
        mu=mu,
        compression=8,
        patch=2,
        high_noise_shift=high_noise_shift,
    )
    image = sample_images(
        model,
        ae,
        encoder,
        [prompt],
        negative_prompts=[""],
        device=next(model.parameters()).device,
        dtype=torch.bfloat16,
        width=512,
        height=512,
        steps=steps,
        guidance=cfg,
        seed=seed,
        y1=y1,
        y2=y2,
        mu=sample_mu,
        progress=False,
        report_latency=False,
    )[0]
    stem = f"{objective}_final_sample_step_{train_steps:06d}"
    image_path = output_dir / f"{stem}.png"
    prompt_path = output_dir / f"{stem}.txt"
    image.save(image_path)
    prompt_path.write_text(prompt + "\n")
    return image_path


@click.command(
    help="Train Krea 2 INT8 LoRA adapters with SFT, DRaFT-K, or flow matching."
)
@click.option("--objective", type=click.Choice(["draft", "flow", "sft"]), required=True)
@click.option("--csv", "csv_path", default=None, type=click.Path(exists=True))
@click.option("--prompts", "prompts_path", default=None, type=click.Path(exists=True))
@click.option("--validation-csv", default=None, type=click.Path(exists=True))
@click.option("--validation-prompts", default=None, type=click.Path(exists=True))
@click.option(
    "--validation-step",
    default=0,
    show_default=True,
    type=int,
    help="generate the fixed validation set every N training steps; 0 disables",
)
@click.option(
    "--validation-size",
    default=4,
    show_default=True,
    type=int,
    help="number of fixed random validation prompts/images",
)
@click.option(
    "--validation-steps",
    default=None,
    type=click.IntRange(1),
    help="validation denoising steps; defaults to --steps",
)
@click.option(
    "--validation-at-start",
    is_flag=True,
    default=False,
    help="generate the fixed validation set before this invocation's updates",
)
@click.option(
    "--validation-at-end",
    is_flag=True,
    default=False,
    help="generate the fixed validation set after the final update",
)
@click.option(
    "--validation-seed",
    default=None,
    type=int,
    help="fixed validation selection/generation seed; default is seed + 200000",
)
@click.option("--trigger-word", default="", show_default=True)
@click.option(
    "--cache-latents",
    is_flag=True,
    default=False,
    help="SFT only: cache VAE latents and text embeddings in CPU RAM before training",
)
@click.option("--reward", default=None, help="reward loader in module:object format")
@click.option(
    "--reward-init-kwargs",
    default=None,
    help="JSON object passed to reward constructors",
)
@click.option(
    "--reward-kwargs", default=None, help="JSON object passed to reward calls"
)
@click.option("--rank", default=32, show_default=True, type=click.Choice(["32", "64"]))
@click.option("--lora-alpha", default=None, type=float)
@click.option("--lora-scale", default=1.0, show_default=True, type=float)
@click.option(
    "--lora-target",
    default="all",
    show_default=True,
    type=click.Choice(sorted(LORA_TARGETS)),
    help="LoRA tensors optimized in this stage; adapters still save all tensors",
)
@click.option("--draft-k", default=1, show_default=True, type=int)
@click.option(
    "--draft-lv-samples",
    default=0,
    show_default=True,
    type=click.IntRange(0),
    help="additional re-noised one-step reward gradients for DRaFT-LV",
)
@click.option(
    "--fuse-no-grad-cfg",
    is_flag=True,
    help="batch detached conditional/unconditional DRaFT CFG model calls",
)
@click.option(
    "--checkpoint-dit/--no-checkpoint-dit",
    default=True,
    show_default=True,
    help="activation-checkpoint DiT blocks during training",
)
@click.option(
    "--checkpoint-vae/--no-checkpoint-vae",
    default=True,
    show_default=True,
    help="activation-checkpoint the differentiable DRaFT VAE decode",
)
@click.option(
    "--draft-image-every",
    default=1,
    show_default=True,
    type=int,
    help="save generated DRaFT training images every N steps; 0 disables",
)
@click.option(
    "--draft-diversity-every",
    default=0,
    show_default=True,
    type=click.IntRange(0),
    help="generate an independent reward/diversity pair every N DRaFT steps",
)
@click.option("--steps", default=28, show_default=True, help="denoising steps")
@click.option("--cfg", default=4.5, show_default=True, help="CFG scale for DRaFT")
@click.option("--train-steps", default=1000, show_default=True)
@click.option("--batch-size", default=1, show_default=True)
@click.option("--lr", default=1e-4, show_default=True)
@click.option("--weight-decay", default=0.1, show_default=True)
@click.option("--max-grad-norm", default=1.0, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option("--final-sample-seed", default=None, type=int)
@click.option("--skip-final-sample", is_flag=True, default=False)
@click.option("--save-every", default=100, show_default=True)
@click.option("--log-every", default=10, show_default=True, type=int)
@click.option(
    "--timing-warmup-steps",
    default=0,
    show_default=True,
    type=click.IntRange(0),
    help="exclude the first N update latencies from reported optimization time",
)
@click.option("--output-dir", default="draft_int8_lora", show_default=True)
@click.option("--resume-lora", default=None, type=click.Path(exists=True))
@click.option(
    "--save-training-state",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
    help="SFT only: save an exact resumable state after the final update",
)
@click.option(
    "--resume-training-state",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="SFT only: resume LoRA, AdamW, RNG, cache, sampler, and global step",
)
@click.option(
    "--checkpoint",
    envvar="K2_CHECKPOINT",
    default="oss_raw",
    show_default=True,
    type=click.Choice(list(checkpoints)),
)
@click.option("--bf16-text-encoder", is_flag=True, default=False)
@click.option(
    "--compile-mode",
    default="default",
    show_default=True,
    type=click.Choice(["none", "default"]),
    help="compile fixed-shape DiT blocks without CUDA graphs",
)
@click.option(
    "--quantization-type",
    default="rowwise",
    show_default=True,
    type=click.Choice(["blockwise", "rowwise"]),
    help="INT8 scale geometry; rowwise is optimized for LoRA forward/backward",
)
@click.option("--y1", default=0.5, show_default=True)
@click.option("--y2", default=1.15, show_default=True)
@click.option("--mu", default=None, type=float)
@click.option(
    "--high-noise-shift",
    default=0.5,
    show_default=True,
    type=float,
    help="additive time-shift applied to both SFT sampling and DRaFT denoising",
)
@click.option("--num-workers", default=0, show_default=True)
def main(
    objective,
    csv_path,
    prompts_path,
    validation_csv,
    validation_prompts,
    validation_step,
    validation_size,
    validation_steps,
    validation_at_start,
    validation_at_end,
    validation_seed,
    trigger_word,
    cache_latents,
    reward,
    reward_init_kwargs,
    reward_kwargs,
    rank,
    lora_alpha,
    lora_scale,
    lora_target,
    draft_k,
    draft_lv_samples,
    fuse_no_grad_cfg,
    checkpoint_dit,
    checkpoint_vae,
    draft_image_every,
    draft_diversity_every,
    steps,
    cfg,
    train_steps,
    batch_size,
    lr,
    weight_decay,
    max_grad_norm,
    seed,
    final_sample_seed,
    skip_final_sample,
    save_every,
    log_every,
    timing_warmup_steps,
    output_dir,
    resume_lora,
    save_training_state,
    resume_training_state,
    checkpoint,
    bf16_text_encoder,
    compile_mode,
    quantization_type,
    y1,
    y2,
    mu,
    high_noise_shift,
    num_workers,
):
    if objective == "draft" and reward is None:
        raise click.ClickException("--reward is required for --objective draft")
    if objective == "draft" and prompts_path is None:
        raise click.ClickException("--prompts is required for --objective draft")
    if objective in {"flow", "sft"} and csv_path is None:
        raise click.ClickException(f"--csv is required for --objective {objective}")
    if validation_csv is not None and validation_prompts is not None:
        raise click.ClickException(
            "--validation-csv and --validation-prompts are mutually exclusive"
        )
    if objective == "draft" and validation_csv is not None:
        raise click.ClickException(
            "--validation-csv is only used for image CSV objectives"
        )
    if cache_latents and objective != "sft":
        raise click.ClickException(
            "--cache-latents is only supported for --objective sft"
        )
    if validation_step < 0:
        raise click.ClickException("--validation-step must be non-negative")
    validation_enabled = validation_step > 0 or validation_at_start or validation_at_end
    if validation_enabled and validation_size <= 0:
        raise click.ClickException(
            "--validation-size must be positive when validation is enabled"
        )
    if train_steps <= 0:
        raise click.ClickException("--train-steps must be positive")
    if steps <= 0:
        raise click.ClickException("--steps must be positive")
    validation_steps = steps if validation_steps is None else validation_steps
    if batch_size <= 0:
        raise click.ClickException("--batch-size must be positive")
    if objective == "draft" and not 1 <= draft_k <= steps:
        raise click.ClickException("--draft-k must be between 1 and --steps")
    if draft_lv_samples and (objective != "draft" or draft_k != 1):
        raise click.ClickException("--draft-lv-samples requires DRaFT with --draft-k 1")
    if draft_image_every < 0:
        raise click.ClickException("--draft-image-every must be non-negative")
    if draft_diversity_every and objective != "draft":
        raise click.ClickException("--draft-diversity-every is DRaFT-only")
    if resume_lora is not None and resume_training_state is not None:
        raise click.ClickException(
            "--resume-lora and --resume-training-state are mutually exclusive"
        )
    if (save_training_state is not None or resume_training_state is not None) and (
        objective != "sft" or not cache_latents
    ):
        raise click.ClickException(
            "training-state save/resume requires --objective sft --cache-latents"
        )
    if (save_training_state is not None or resume_training_state is not None) and (
        num_workers != 0
    ):
        raise click.ClickException(
            "exact training-state save/resume requires --num-workers 0"
        )
    rank = int(rank)
    resume_state = (
        load_training_state(resume_training_state)
        if resume_training_state is not None
        else None
    )
    initial_step = 0 if resume_state is None else int(resume_state["global_step"])
    final_step = initial_step + int(train_steps)
    torch.manual_seed(seed)
    reward_init_kw = parse_json_dict(reward_init_kwargs)
    reward_obj = load_reward(reward, reward_init_kw) if objective == "draft" else None
    reward_kw = parse_json_dict(reward_kwargs)

    model, ae, encoder = build_int8_pipeline(
        checkpoint=checkpoint,
        bf16_text_encoder=bf16_text_encoder,
        quantization_type=quantization_type,
    )
    model.eval()
    ae.eval()
    encoder.eval()
    targets = add_lora_to_int8_blocks(model, rank=rank, alpha=lora_alpha)
    for module in model.modules():
        if isinstance(module, LinearLoraINT8):
            module.lora_scale *= float(lora_scale)
    if resume_lora is not None:
        load_lora_adapters(model, resume_lora, strict=True)
    elif resume_state is not None:
        load_lora_state_tensors(model, resume_state["lora"], strict=True)
    trainable_targets = set_lora_trainable(model, lora_target)
    if compile_mode != "none":
        compile_training_blocks(model)
    if checkpoint_dit:
        apply_block_checkpointing(model)
    params = list(lora_parameters(model))
    if not params:
        raise click.ClickException("no INT8 linears were converted to LoRA")
    opt = torch.optim.AdamW(
        params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay
    )
    if resume_state is not None:
        opt.load_state_dict(resume_state["optimizer"])
    train_device = next(model.parameters()).device

    fixed_validation_prompts = []
    fixed_validation_indices = []
    fixed_validation_source = None
    validation_encoder = None
    validation_seed = (
        int(seed) + 200_000 if validation_seed is None else int(validation_seed)
    )
    if validation_enabled:
        (
            fixed_validation_prompts,
            fixed_validation_indices,
            fixed_validation_source,
        ) = choose_validation_prompts(
            objective=objective,
            csv_path=csv_path,
            prompts_path=prompts_path,
            validation_csv=validation_csv,
            validation_prompts=validation_prompts,
            trigger_word=trigger_word,
            size=validation_size,
            seed=validation_seed,
        )
        validation_encoder = build_validation_encoder(
            encoder, fixed_validation_prompts, train_device
        )
        click.echo(
            f"fixed {len(fixed_validation_prompts)} validation prompts from "
            f"{fixed_validation_source} with indices={fixed_validation_indices}"
        )

    cached_sft = False
    dataset_signature = None
    offloaded_vae_modules: list[str] = []
    if objective == "draft":
        prompt_dataset = PromptDataset(prompts_path)
        dataset = build_draft_text_cache(
            prompt_dataset,
            encoder,
            trigger_word=trigger_word,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        offload_text_encoder_to_cpu(encoder)
        offloaded_vae_modules = offload_vae_encoder_to_cpu(ae)
        moved = ", ".join(offloaded_vae_modules) if offloaded_vae_modules else "none"
        click.echo(
            f"cached {len(dataset)} DRaFT prompts in CPU RAM, moved text encoder "
            f"to CPU, and moved VAE encode modules to CPU: {moved}"
        )
    else:
        image_dataset = ImagePromptDataset(csv_path, size=512)
        dataset_signature = sft_dataset_signature(image_dataset)
        if cache_latents:
            if resume_state is None:
                dataset = build_sft_cache(
                    image_dataset,
                    ae,
                    encoder,
                    trigger_word=trigger_word,
                    batch_size=batch_size,
                    num_workers=num_workers,
                )
            else:
                cache = resume_state["cached_sft"]
                dataset = CachedSFTDataset(
                    cache["latents"],
                    cache["text_embeddings"],
                    cache["text_masks"],
                )
                if len(dataset) != len(image_dataset):
                    raise click.ClickException(
                        "cached SFT dataset length does not match the current CSV"
                    )
            cached_sft = True
            if validation_enabled:
                offload_text_encoder_to_cpu(encoder)
                offloaded_vae_modules = offload_vae_encoder_to_cpu(ae)
                moved = (
                    ", ".join(offloaded_vae_modules)
                    if offloaded_vae_modules
                    else "none"
                )
                click.echo(
                    f"cached {len(dataset)} SFT samples in CPU RAM, moved text "
                    f"encoder and VAE encode modules to CPU: {moved}"
                )
            else:
                offload_conditioners_to_cpu(ae, encoder)
                click.echo(
                    f"cached {len(dataset)} SFT samples in CPU RAM and moved "
                    "VAE/text encoder to CPU"
                )
        else:
            dataset = image_dataset
    if batch_size > len(dataset):
        raise click.ClickException(
            f"--batch-size ({batch_size}) exceeds dataset size ({len(dataset)})"
        )
    state_compatibility = {
        "objective": objective,
        "checkpoint": checkpoint,
        "rank": rank,
        "lora_alpha": float(rank if lora_alpha is None else lora_alpha),
        "lora_scale": float(lora_scale),
        "quantization_type": quantization_type,
        "batch_size": int(batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "max_grad_norm": float(max_grad_norm),
        "seed": int(seed),
        "y1": float(y1),
        "y2": float(y2),
        "mu": None if mu is None else float(mu),
        "high_noise_shift": float(high_noise_shift),
        "dataset_signature": dataset_signature,
        "dataset_size": len(dataset),
    }
    if resume_state is not None:
        try:
            validate_training_state(resume_state, state_compatibility)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    exact_sampler = None
    if save_training_state is not None or resume_state is not None:
        exact_sampler = StatefulShuffleBatchSampler(
            len(dataset), batch_size, seed=seed + 1_000_000
        )
        if resume_state is not None:
            try:
                exact_sampler.load_state_dict(resume_state["sampler"])
            except ValueError as exc:
                raise click.ClickException(str(exc)) from exc
        loader = DataLoader(
            dataset,
            batch_sampler=exact_sampler,
            num_workers=0,
            pin_memory=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
            pin_memory=True,
        )
    data_iter = iter(loader)
    metadata = {
        "objective": objective,
        "rank": str(rank),
        "lora_alpha": str(rank if lora_alpha is None else lora_alpha),
        "lora_scale": str(lora_scale),
        "checkpoint": checkpoint,
        "targets": "\n".join(targets),
        "trainable_targets": "\n".join(trainable_targets),
        "lora_target": lora_target,
        "trigger_word": trigger_word,
        "cache_latents": str(cache_latents),
        "cache_text": str(objective == "draft"),
        "vae_encoder_offloaded": ",".join(offloaded_vae_modules),
        "flow_convention": "krea_t1_noise_t0_data",
        "quantization_type": quantization_type,
        "compile_mode": compile_mode,
        "lr": str(lr),
        "draft_k": str(draft_k),
        "draft_lv_samples": str(draft_lv_samples),
        "draft_diversity_every": str(draft_diversity_every),
        "denoising_steps": str(steps),
        "cfg": str(cfg),
        "high_noise_shift": str(high_noise_shift),
        "validation_step": str(validation_step),
        "validation_size": str(validation_size),
        "validation_steps": str(validation_steps),
        "validation_seed": str(validation_seed),
        "validation_at_start": str(validation_at_start),
        "validation_at_end": str(validation_at_end),
        "initial_step": str(initial_step),
        "final_step": str(final_step),
    }
    if csv_path is not None:
        metadata["csv"] = str(csv_path)
    if prompts_path is not None:
        metadata["prompts"] = str(prompts_path)
    if validation_csv is not None:
        metadata["validation_csv"] = str(validation_csv)
    if validation_prompts is not None:
        metadata["validation_prompts"] = str(validation_prompts)
    if fixed_validation_source is not None:
        metadata["fixed_validation_source"] = str(fixed_validation_source)
        metadata["fixed_validation_indices"] = ",".join(
            map(str, fixed_validation_indices)
        )
    output_dir = Path(output_dir)

    validated_steps: set[int] = set()
    if validation_step > 0 or validation_at_start:
        paths = save_validation_images(
            model,
            ae,
            validation_encoder,
            fixed_validation_prompts,
            output_dir=output_dir,
            step=initial_step,
            steps=validation_steps,
            cfg=cfg,
            seed=validation_seed,
            y1=y1,
            y2=y2,
            mu=mu,
            high_noise_shift=high_noise_shift,
        )
        validated_steps.add(initial_step)
        click.echo(f"saved start validation images to {paths[0].parent}")

    if resume_state is not None:
        try:
            restore_training_rng(resume_state)
        except ValueError as exc:
            raise click.ClickException(str(exc)) from exc

    last_saved_step = None
    last_saved_path = None
    step_times = []
    if train_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(train_device)
    with click.progressbar(range(1, train_steps + 1), label="training") as bar:
        for local_step in bar:
            step = initial_step + local_step
            sync_if_cuda(train_device)
            step_start = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            if objective == "draft":
                images = None
                prompts = list(batch["prompt"])
                text_embeddings = batch["text_embeddings"]
                text_masks = batch["text_masks"]
                negative_text_embeddings = batch["negative_text_embeddings"]
                negative_text_masks = batch["negative_text_masks"]
            elif cached_sft:
                images = None
                prompts = None
                latents, text_embeddings, text_masks = batch
            else:
                images, prompts = batch
                prompts = list(prompts)
                if objective == "sft":
                    prompts = apply_trigger_word(prompts, trigger_word)

            opt.zero_grad(set_to_none=True)
            if objective == "draft":
                prompt_list = list(prompts)
                diversity_step = (
                    draft_diversity_every > 0
                    and step % draft_diversity_every == 0
                    and hasattr(reward_obj, "pairwise_reward")
                )
                main_lv_samples = 0 if diversity_step else draft_lv_samples
                out_images = draft_sample_images(
                    model,
                    ae,
                    prompt_list,
                    text_embeddings=text_embeddings,
                    text_masks=text_masks,
                    negative_text_embeddings=negative_text_embeddings,
                    negative_text_masks=negative_text_masks,
                    steps=steps,
                    draft_k=draft_k,
                    guidance=cfg,
                    seed=seed + step * batch_size,
                    y1=y1,
                    y2=y2,
                    mu=mu,
                    high_noise_shift=high_noise_shift,
                    fuse_no_grad_cfg=fuse_no_grad_cfg,
                    checkpoint_vae=checkpoint_vae,
                    lv_samples=main_lv_samples,
                )
                reward_prompts = prompt_list * (main_lv_samples + 1)
                pair_indices = None
                if diversity_step:
                    independent = draft_sample_images(
                        model,
                        ae,
                        prompt_list,
                        text_embeddings=text_embeddings,
                        text_masks=text_masks,
                        negative_text_embeddings=negative_text_embeddings,
                        negative_text_masks=negative_text_masks,
                        steps=steps,
                        draft_k=draft_k,
                        guidance=cfg,
                        seed=seed + 100_000_000 + step * batch_size,
                        y1=y1,
                        y2=y2,
                        mu=mu,
                        high_noise_shift=high_noise_shift,
                        fuse_no_grad_cfg=fuse_no_grad_cfg,
                        checkpoint_vae=checkpoint_vae,
                        lv_samples=0,
                    )
                    independent_offset = out_images.shape[0]
                    out_images = torch.cat((out_images, independent), dim=0)
                    reward_prompts.extend(prompt_list)
                    pair_indices = [
                        (index, independent_offset + index)
                        for index in range(len(prompt_list))
                    ]
                if draft_image_every > 0 and step % draft_image_every == 0:
                    save_draft_step_images(out_images, reward_prompts, output_dir, step)
                loss, rewards = reward_loss(
                    reward_obj,
                    out_images,
                    reward_prompts,
                    reward_kw,
                    pair_indices=pair_indices,
                )
            elif cached_sft:
                loss = cached_flow_loss(
                    model,
                    latents,
                    text_embeddings,
                    text_masks,
                    y1=y1,
                    y2=y2,
                    mu=mu,
                    high_noise_shift=high_noise_shift,
                )
                rewards = None
            else:
                loss = flow_loss(
                    model,
                    ae,
                    encoder,
                    images,
                    prompts,
                    y1=y1,
                    y2=y2,
                    mu=mu,
                    high_noise_shift=high_noise_shift,
                )
                rewards = None

            loss.backward()
            if max_grad_norm and max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            opt.step()
            sync_if_cuda(train_device)
            step_time = time.perf_counter() - step_start
            step_times.append(step_time)
            steps_per_second = 1.0 / max(step_time, 1e-12)

            if local_step == 1 or (log_every > 0 and step % log_every == 0):
                if rewards is None:
                    click.echo(
                        f"step={step} loss={loss.detach().item():.6f} "
                        f"step_time={step_time:.3f}s "
                        f"steps_per_second={steps_per_second:.3f}"
                    )
                else:
                    click.echo(
                        f"step={step} loss={loss.detach().item():.6f} "
                        f"reward={rewards.mean().item():.6f} "
                        f"step_time={step_time:.3f}s "
                        f"steps_per_second={steps_per_second:.3f}"
                    )
            if save_every > 0 and step % save_every == 0:
                last_saved_path = save_step(model, output_dir, step, metadata)
                last_saved_step = step
                click.echo(f"saved {last_saved_path}")
            if validation_step > 0 and step % validation_step == 0:
                opt.zero_grad(set_to_none=True)
                paths = save_validation_images(
                    model,
                    ae,
                    validation_encoder,
                    fixed_validation_prompts,
                    output_dir=output_dir,
                    step=step,
                    steps=validation_steps,
                    cfg=cfg,
                    seed=validation_seed,
                    y1=y1,
                    y2=y2,
                    mu=mu,
                    high_noise_shift=high_noise_shift,
                )
                validated_steps.add(step)
                click.echo(f"saved validation images to {paths[0].parent}")

    if last_saved_step == final_step:
        path = last_saved_path
    else:
        path = save_step(model, output_dir, final_step, metadata)
    click.echo(f"saved {path}")

    summary_path = save_training_summary(
        output_dir,
        objective=objective,
        initial_step=initial_step,
        final_step=final_step,
        step_times=step_times,
        timing_warmup_steps=timing_warmup_steps,
        peak_cuda_memory_bytes=(
            torch.cuda.max_memory_allocated(train_device)
            if train_device.type == "cuda"
            else None
        ),
    )
    click.echo(f"saved {summary_path}")

    if save_training_state is not None:
        if not isinstance(dataset, CachedSFTDataset) or exact_sampler is None:
            raise click.ClickException("training state requires the cached SFT sampler")
        state_path = write_training_state(
            save_training_state,
            model=model,
            optimizer=opt,
            dataset=dataset,
            sampler=exact_sampler,
            global_step=final_step,
            compatibility=state_compatibility,
        )
        click.echo(f"saved {state_path}")

    if validation_at_end and final_step not in validated_steps:
        opt.zero_grad(set_to_none=True)
        paths = save_validation_images(
            model,
            ae,
            validation_encoder,
            fixed_validation_prompts,
            output_dir=output_dir,
            step=final_step,
            steps=validation_steps,
            cfg=cfg,
            seed=validation_seed,
            y1=y1,
            y2=y2,
            mu=mu,
            high_noise_shift=high_noise_shift,
        )
        validated_steps.add(final_step)
        click.echo(f"saved end validation images to {paths[0].parent}")

    if objective in {"draft", "sft"} and not skip_final_sample:
        if cached_sft:
            restore_conditioners_to_device(ae, encoder, train_device)
        elif objective == "draft":
            restore_text_encoder_to_device(encoder, train_device)
        sample_seed = (
            int(final_sample_seed)
            if final_sample_seed is not None
            else int(seed + final_step * max(batch_size, 1) + 100_000)
        )
        prompt, prompt_index, prompt_source = choose_final_sample_prompt(
            objective=objective,
            csv_path=csv_path,
            prompts_path=prompts_path,
            validation_csv=validation_csv,
            validation_prompts=validation_prompts,
            trigger_word=trigger_word,
            seed=sample_seed,
        )
        sample_path = save_final_sample(
            model,
            ae,
            encoder,
            prompt=prompt,
            output_dir=output_dir,
            objective=objective,
            train_steps=final_step,
            steps=steps,
            cfg=cfg,
            seed=sample_seed,
            y1=y1,
            y2=y2,
            mu=mu,
            high_noise_shift=high_noise_shift,
        )
        click.echo(
            f"saved final sample {sample_path} "
            f"(prompt_index={prompt_index}, prompt_source={prompt_source})"
        )


if __name__ == "__main__":
    main()
