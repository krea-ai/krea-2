#!/usr/bin/env python3
"""Evaluate one character LoRA with the fixed autoresearch protocol."""

from __future__ import annotations

import gc
import json
from pathlib import Path

import click
import torch

from krea2.experiments import comparison
from krea2.inference.int8 import build_int8_pipeline
from krea2.rewards.face import FaceSimilarityReward
from krea2.rewards.face_models import ensure_face_models, locate_face_model_dir
from krea2.training.pipeline import discover_images, validate_base_checkpoint
from krea2.training.trainer import compile_training_blocks
from krea2.training.validation import (
    build_validation_encoder,
    choose_validation_prompts,
    save_validation_images,
)


@click.command(help="Evaluate a LoRA at fixed seed 42 with ten held-out prompts.")
@click.argument(
    "images_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--lora",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--prompts",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--face-model-dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--step", default=0, show_default=True, type=click.IntRange(0))
def main(
    images_dir: Path,
    lora: Path,
    prompts: Path,
    output_dir: Path,
    face_model_dir: Path | None,
    step: int,
) -> None:
    source_values = [
        line.strip() for line in prompts.read_text().splitlines() if line.strip()
    ]
    if len(source_values) != 10:
        raise click.ClickException(
            f"expected exactly 10 prompts, found {len(source_values)}"
        )
    values, prompt_indices, _ = choose_validation_prompts(
        objective="draft",
        csv_path=None,
        prompts_path=prompts,
        validation_csv=None,
        validation_prompts=prompts,
        trigger_word=None,
        size=10,
        seed=200_042,
    )
    references = discover_images(images_dir.resolve(), exclude_dir=output_dir.resolve())
    if not references:
        raise click.ClickException(f"no reference images found under {images_dir}")
    face_dir = ensure_face_models(locate_face_model_dir(face_model_dir))
    validate_base_checkpoint("oss_raw")

    model, ae, encoder = build_int8_pipeline(
        checkpoint="oss_raw",
        lora=lora,
        quantization_type="rowwise",
    )
    compile_training_blocks(model)
    device = next(model.parameters()).device
    cached_encoder = build_validation_encoder(encoder, values, device)
    paths = save_validation_images(
        model,
        ae,
        cached_encoder,
        values,
        output_dir=output_dir.resolve(),
        step=step,
        steps=20,
        cfg=4.5,
        seed=200_042,
        y1=0.5,
        y2=1.15,
        mu=None,
        high_noise_shift=0.5,
    )

    del cached_encoder, encoder, ae, model
    gc.collect()
    torch.cuda.empty_cache()
    reward = FaceSimilarityReward(reference_images=references, model_dir=face_dir)
    metrics, records = comparison.score_image_set("evaluation", paths, reward)
    result = {
        "protocol": {
            "seed": 42,
            "generation_seed": 200_042,
            "validation_size": 10,
            "steps": 20,
            "cfg": 4.5,
            "prompt_indices": prompt_indices,
        },
        "lora": str(lora.resolve()),
        "metrics": metrics,
        "images": records,
    }
    output = output_dir.resolve() / "evaluation_metrics.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    click.echo(f"mean_face_similarity={metrics['mean_face_similarity']:.6f}")
    click.echo(f"wrote {output}")


if __name__ == "__main__":
    main()
