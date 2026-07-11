"""Face-identity training research runner with fixed held-out evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from krea2.experiments import comparison
from krea2.preprocessing import prompting
from krea2.rewards.face import FaceSimilarityReward
from krea2.rewards.face_models import ensure_face_models, locate_face_model_dir
from krea2.training import pipeline

ROOT = Path(__file__).resolve().parents[3]
TRAINER = ROOT / "scripts" / "train.py"
TRAINING_CODE = tuple(sorted((ROOT / "src" / "krea2").rglob("*.py"))) + (TRAINER,)


def write_prompts(path: Path, values: list[str]) -> None:
    content = "\n".join(values) + "\n"
    if not path.is_file() or path.read_text(encoding="utf-8") != content:
        prompting.write_prompts(path, values)


def prepare_protocol(
    images_dir: Path,
    output_dir: Path,
    *,
    prompt_count: int,
    validation_size: int,
    seed: int,
    caption_model: str,
    prompt_model: str,
) -> tuple[list[Path], Path, Path, Path]:
    data_dir = output_dir / "data"
    images = pipeline.discover_images(images_dir, exclude_dir=output_dir)
    if not images:
        raise click.ClickException(f"no images found under {images_dir}")
    captions = pipeline.prepare_captions(
        images,
        data_dir / "captions.json",
        model_name=caption_model,
    )
    dataset_csv = data_dir / "dataset.csv"
    pipeline.write_dataset_csv(dataset_csv, images, captions)
    pool_path = data_dir / "all_prompts.txt"
    pool = pipeline.prepare_draft_prompts(
        dataset_csv,
        pool_path,
        count=prompt_count + validation_size,
        seed=seed + 10_000,
        model_name=prompt_model,
        regenerate=False,
    )
    training, validation = comparison.split_prompt_pool(
        pool, prompt_count, validation_size
    )
    draft_prompts = data_dir / "draft_train_prompts.txt"
    validation_prompts = data_dir / "evaluation_prompts.txt"
    write_prompts(draft_prompts, training)
    write_prompts(validation_prompts, validation)
    return images, dataset_csv, draft_prompts, validation_prompts


def common_args(
    *,
    train_steps: int,
    batch_size: int,
    lr: float,
    seed: int,
    rank: int,
) -> list[str]:
    return [
        "--rank",
        str(rank),
        "--train-steps",
        str(train_steps),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
        "--checkpoint",
        "oss_raw",
        "--seed",
        str(seed),
        "--quantization-type",
        "rowwise",
        "--compile-mode",
        "default",
        "--skip-final-sample",
        "--draft-image-every",
        "0",
        "--num-workers",
        "0",
        "--timing-warmup-steps",
        "1",
    ]


def score_learning_curve(
    *,
    output_dir: Path,
    sft_dir: Path,
    draft_dir: Path,
    images: list[Path],
    face_model_dir: Path,
    validation_size: int,
    training_denoising_steps: int,
) -> Path:
    sft_summary = json.loads((sft_dir / "training_summary.json").read_text())
    draft_summary = json.loads((draft_dir / "training_summary.json").read_text())
    warmup = int(draft_summary.get("timing_warmup_steps", 0))
    step_times = draft_summary["step_times_seconds"]
    reward = FaceSimilarityReward(
        reference_images=images,
        model_dir=face_model_dir,
    )
    rows = []
    for directory in sorted((draft_dir / "validation").glob("step_*")):
        step = int(directory.name.removeprefix("step_"))
        paths = comparison.validation_images(directory, validation_size)
        metrics, _ = comparison.score_image_set(f"step_{step}", paths, reward)
        measured_draft = sum(step_times[warmup:step]) if step else 0.0
        total = float(sft_summary["optimization_seconds"]) + measured_draft
        rows.append(
            {
                "draft_step": step,
                "pure_training_seconds": total,
                "within_15_minute_budget": total <= 900.0,
                "face_detection_rate": metrics["face_detection_rate"],
                "mean_face_similarity": metrics["mean_face_similarity"],
                "face_similarity_all": metrics["identity_similarity_all"],
                "detected_face_similarity": metrics["identity_similarity"],
            }
        )
    result = {
        "protocol": {
            "seed": 42,
            "validation_size": validation_size,
            "training_denoising_steps": training_denoising_steps,
            "validation_denoising_steps": 20,
        },
        "sft_summary": sft_summary,
        "draft_summary": draft_summary,
        "learning_curve": rows,
    }
    output = output_dir / "learning_curve.json"
    comparison.write_json(output, result)
    return output


@click.command(help="Run a resumable SFT + DRaFT identity learning curve.")
@click.argument(
    "images_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--variant", default="lv1_sampler12_lr2e4", show_default=True)
@click.option("--sft-steps", default=500, show_default=True, type=click.IntRange(1))
@click.option("--draft-steps", default=120, show_default=True, type=click.IntRange(1))
@click.option("--prompt-count", default=64, show_default=True, type=click.IntRange(1))
@click.option(
    "--validation-size", default=10, show_default=True, type=click.IntRange(1)
)
@click.option(
    "--validation-every", default=20, show_default=True, type=click.IntRange(1)
)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--rank", default=32, show_default=True, type=click.Choice([32, 64]))
@click.option("--sft-lr", default=1e-4, show_default=True, type=float)
@click.option("--draft-lr", default=2e-4, show_default=True, type=float)
@click.option("--draft-k", default=1, show_default=True, type=click.IntRange(1))
@click.option(
    "--draft-lv-samples", default=1, show_default=True, type=click.IntRange(0)
)
@click.option(
    "--denoising-steps", default=12, show_default=True, type=click.IntRange(1)
)
@click.option("--cfg", default=4.5, show_default=True, type=float)
@click.option(
    "--draft-batch-size", default=1, show_default=True, type=click.IntRange(1)
)
@click.option("--face-model-dir", type=click.Path(file_okay=False, path_type=Path))
@click.option("--caption-model", default=pipeline.DEFAULT_CAPTION_MODEL)
@click.option("--prompt-model", default=pipeline.DEFAULT_PROMPT_MODEL)
@click.option("--reward-init-kwargs", default="{}", help="extra face reward JSON")
@click.option("--fuse-no-grad-cfg", is_flag=True)
@click.option("--checkpoint-dit/--no-checkpoint-dit", default=True)
@click.option("--checkpoint-vae/--no-checkpoint-vae", default=True)
@click.option(
    "--reuse-sft-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="use an existing fixed SFT stage even when research code changes",
)
@click.option("--force", is_flag=True)
def main(
    images_dir: Path,
    output_dir: Path,
    variant: str,
    sft_steps: int,
    draft_steps: int,
    prompt_count: int,
    validation_size: int,
    validation_every: int,
    seed: int,
    rank: int,
    sft_lr: float,
    draft_lr: float,
    draft_k: int,
    draft_lv_samples: int,
    denoising_steps: int,
    cfg: float,
    draft_batch_size: int,
    face_model_dir: Path | None,
    caption_model: str,
    prompt_model: str,
    reward_init_kwargs: str,
    fuse_no_grad_cfg: bool,
    checkpoint_dit: bool,
    checkpoint_vae: bool,
    reuse_sft_dir: Path | None,
    force: bool,
) -> None:
    if seed != 42 or validation_size != 10:
        raise click.ClickException(
            "autoresearch protocol requires seed=42 and validation-size=10"
        )
    if draft_k > denoising_steps:
        raise click.ClickException("--draft-k cannot exceed --denoising-steps")
    try:
        extra_reward = json.loads(reward_init_kwargs)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid reward JSON: {exc}") from exc
    if not isinstance(extra_reward, dict):
        raise click.ClickException("--reward-init-kwargs must be a JSON object")

    images_dir = images_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint = pipeline.validate_base_checkpoint("oss_raw")
    face_dir = ensure_face_models(locate_face_model_dir(face_model_dir))
    images, dataset_csv, draft_prompts, validation_prompts = prepare_protocol(
        images_dir,
        output_dir,
        prompt_count=prompt_count,
        validation_size=validation_size,
        seed=seed,
        caption_model=caption_model,
        prompt_model=prompt_model,
    )

    sft_dir = (
        reuse_sft_dir.resolve()
        if reuse_sft_dir is not None
        else output_dir / f"sft_{sft_steps}_seed_{seed}"
    )
    sft_command = [
        sys.executable,
        str(TRAINER),
        "--objective",
        "sft",
        "--csv",
        str(dataset_csv),
        "--cache-latents",
        "--output-dir",
        str(sft_dir),
        "--save-every",
        str(sft_steps),
        "--validation-step",
        "0",
        *common_args(
            train_steps=sft_steps,
            batch_size=1,
            lr=sft_lr,
            seed=seed,
            rank=rank,
        ),
    ]
    if reuse_sft_dir is None:
        comparison.run_stage(
            "autoresearch SFT",
            sft_command,
            output_dir=sft_dir,
            expected=[
                sft_dir / "lora_latest.safetensors",
                sft_dir / "training_summary.json",
            ],
            dependencies=[dataset_csv, base_checkpoint, *TRAINING_CODE],
            force=force,
        )
    elif not all(
        (sft_dir / name).is_file()
        for name in ("lora_latest.safetensors", "training_summary.json")
    ):
        raise click.ClickException(f"incomplete reused SFT stage: {sft_dir}")
    else:
        click.echo(f"autoresearch SFT: explicitly reusing {sft_dir}")

    draft_dir = output_dir / "variants" / variant
    reward_init = {
        "reference_images": [str(path) for path in images],
        "model_dir": str(face_dir),
        **extra_reward,
    }
    draft_command = [
        sys.executable,
        str(TRAINER),
        "--objective",
        "draft",
        "--prompts",
        str(draft_prompts),
        "--resume-lora",
        str(sft_dir / "lora_latest.safetensors"),
        "--reward",
        "krea2.rewards.face:FaceSimilarityReward",
        "--reward-init-kwargs",
        json.dumps(reward_init, separators=(",", ":")),
        "--output-dir",
        str(draft_dir),
        "--draft-k",
        str(draft_k),
        "--draft-lv-samples",
        str(draft_lv_samples),
        "--steps",
        str(denoising_steps),
        "--cfg",
        str(cfg),
        "--save-every",
        str(validation_every),
        "--validation-step",
        str(validation_every),
        "--validation-size",
        str(validation_size),
        "--validation-steps",
        "20",
        "--validation-prompts",
        str(validation_prompts),
        "--validation-seed",
        str(seed + 200_000),
        "--validation-at-end",
        *common_args(
            train_steps=draft_steps,
            batch_size=draft_batch_size,
            lr=draft_lr,
            seed=seed + 1,
            rank=rank,
        ),
    ]
    if fuse_no_grad_cfg:
        draft_command.append("--fuse-no-grad-cfg")
    if not checkpoint_dit:
        draft_command.append("--no-checkpoint-dit")
    if not checkpoint_vae:
        draft_command.append("--no-checkpoint-vae")
    validation_steps = sorted(
        {*range(0, draft_steps + 1, validation_every), draft_steps}
    )
    expected = [
        draft_dir / "lora_latest.safetensors",
        draft_dir / "training_summary.json",
    ]
    for step in validation_steps:
        expected.extend(
            comparison.expected_validation_images(
                draft_dir / "validation" / f"step_{step:06d}", validation_size
            )
        )
    comparison.run_stage(
        f"autoresearch DRaFT {variant}",
        draft_command,
        output_dir=draft_dir,
        expected=expected,
        dependencies=[
            draft_prompts,
            validation_prompts,
            sft_dir / "lora_latest.safetensors",
            *TRAINING_CODE,
        ],
        force=force,
    )
    result = score_learning_curve(
        output_dir=draft_dir,
        sft_dir=sft_dir,
        draft_dir=draft_dir,
        images=images,
        face_model_dir=face_dir,
        validation_size=validation_size,
        training_denoising_steps=denoising_steps,
    )
    click.echo(f"wrote {result}")


if __name__ == "__main__":
    main()
