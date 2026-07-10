"""Face-specialized full training entry point."""

from __future__ import annotations

from pathlib import Path

import click

from krea2.rewards.face_models import (
    FACE_MODEL_FILES,
    ensure_face_models,
    locate_face_model_dir,
)
from krea2.training.config import RewardSpec
from krea2.training.pipeline import (
    discover_images,
    pipeline_config_from_options,
    pipeline_options,
    run_pipeline,
)


@click.command(help="Train a character LoRA with the antelopev2 face reward.")
@pipeline_options
@click.option(
    "--face-model-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="antelopev2 root; auto-detected or downloaded when omitted",
)
def main(face_model_dir: Path | None, **options) -> None:
    config = pipeline_config_from_options(**options)
    images = discover_images(config.images_dir, exclude_dir=config.output_dir)
    if not images:
        raise click.ClickException(
            f"no supported images found under {config.images_dir}"
        )
    model_dir = ensure_face_models(locate_face_model_dir(face_model_dir))
    click.echo(f"using antelopev2 models from {model_dir}")
    reward = RewardSpec(
        target="krea2.rewards.face:FaceSimilarityReward",
        init_kwargs={
            "reference_images": [str(path) for path in images],
            "model_dir": str(model_dir),
        },
        dependencies=tuple(model_dir / spec.relative_path for spec in FACE_MODEL_FILES),
    )
    run_pipeline(config, reward)
