"""Compare 500-step SFT + DRaFT-K against an exact 1000-step SFT run."""

from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

import click
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from krea2.preprocessing import prompting as prompt_script
from krea2.rewards.face import FaceSimilarityReward, tensor_to_bgr
from krea2.rewards.face_models import (
    FACE_MODEL_FILES,
    ensure_face_models,
    locate_face_model_dir,
)
from krea2.training import pipeline

ROOT = Path(__file__).resolve().parents[3]
TRAINER = ROOT / "scripts" / "train.py"
TRAINING_CODE = [
    TRAINER,
    ROOT / "src" / "krea2" / "quantization" / "int8.py",
    ROOT / "src" / "krea2" / "inference" / "int8.py",
    ROOT / "src" / "krea2" / "models" / "conditioner.py",
    ROOT / "src" / "krea2" / "inference" / "sampling.py",
    ROOT / "src" / "krea2" / "rewards" / "face.py",
    ROOT / "src" / "krea2" / "kernels" / "int8.py",
]


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_prompts(path: Path, prompts: list[str]) -> None:
    content = "\n".join(prompts) + "\n"
    if path.is_file() and path.read_text(encoding="utf-8") == content:
        return
    prompt_script.write_prompts(path, prompts)


def split_prompt_pool(
    prompts: list[str], training_count: int, validation_size: int
) -> tuple[list[str], list[str]]:
    if len(prompts) != training_count + validation_size:
        raise ValueError("prompt pool has the wrong size")
    training = prompts[:training_count]
    evaluation = prompts[training_count:]
    train_keys = {prompt_script.prompt_key(prompt) for prompt in training}
    eval_keys = {prompt_script.prompt_key(prompt) for prompt in evaluation}
    if train_keys & eval_keys or len(eval_keys) != validation_size:
        raise ValueError("generated training/evaluation prompts are not disjoint")
    return training, evaluation


def file_dependencies(paths) -> dict[str, dict[str, int]]:
    return {
        str(Path(path).resolve()): pipeline.file_fingerprint(Path(path).resolve())
        for path in paths
    }


def append_trigger(command: list[str], trigger_word: str | None) -> None:
    if trigger_word:
        command.extend(["--trigger-word", trigger_word])


def common_training_args(
    *,
    rank: int,
    batch_size: int,
    train_steps: int,
    lr: float,
    checkpoint: str,
    seed: int,
    validation_prompts: Path,
    validation_size: int,
    validation_seed: int,
    denoising_steps: int,
    cfg: float,
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
        checkpoint,
        "--seed",
        str(seed),
        "--save-every",
        str(train_steps),
        "--validation-step",
        "0",
        "--validation-size",
        str(validation_size),
        "--validation-prompts",
        str(validation_prompts),
        "--validation-seed",
        str(validation_seed),
        "--steps",
        str(denoising_steps),
        "--cfg",
        str(cfg),
        "--quantization-type",
        "rowwise",
        "--compile-mode",
        "default",
        "--skip-final-sample",
        "--num-workers",
        "0",
    ]


def build_experiment_commands(
    *,
    python: str,
    dataset_csv: Path,
    draft_prompts: Path,
    evaluation_prompts: Path,
    reference_images: list[Path],
    face_model_dir: Path,
    shared_dir: Path,
    draft_dir: Path,
    sft_dir: Path,
    trigger_word: str | None,
    checkpoint: str,
    rank: int,
    batch_size: int,
    shared_sft_steps: int,
    draft_steps: int,
    total_sft_steps: int,
    sft_lr: float,
    draft_lr: float,
    draft_k: int,
    denoising_steps: int,
    cfg: float,
    validation_size: int,
    seed: int,
) -> tuple[list[str], list[str], list[str]]:
    continuation_steps = total_sft_steps - shared_sft_steps
    validation_seed = seed + 200_000
    shared_state = shared_dir / f"training_state_step_{shared_sft_steps:06d}.pt"
    final_state = sft_dir / f"training_state_step_{total_sft_steps:06d}.pt"

    shared = [
        python,
        str(TRAINER),
        "--objective",
        "sft",
        "--csv",
        str(dataset_csv),
        "--cache-latents",
        "--output-dir",
        str(shared_dir),
        "--save-training-state",
        str(shared_state),
        "--validation-at-start",
        "--validation-at-end",
        *common_training_args(
            rank=rank,
            batch_size=batch_size,
            train_steps=shared_sft_steps,
            lr=sft_lr,
            checkpoint=checkpoint,
            seed=seed,
            validation_prompts=evaluation_prompts,
            validation_size=validation_size,
            validation_seed=validation_seed,
            denoising_steps=denoising_steps,
            cfg=cfg,
        ),
    ]
    append_trigger(shared, trigger_word)

    reward_init = json.dumps(
        {
            "reference_images": [str(path) for path in reference_images],
            "model_dir": str(face_model_dir),
        },
        separators=(",", ":"),
    )
    draft = [
        python,
        str(TRAINER),
        "--objective",
        "draft",
        "--prompts",
        str(draft_prompts),
        "--resume-lora",
        str(shared_dir / "lora_latest.safetensors"),
        "--reward",
        "krea2.rewards.face:FaceSimilarityReward",
        "--reward-init-kwargs",
        reward_init,
        "--draft-k",
        str(draft_k),
        "--draft-image-every",
        "0",
        "--output-dir",
        str(draft_dir),
        "--validation-at-end",
        *common_training_args(
            rank=rank,
            batch_size=batch_size,
            train_steps=draft_steps,
            lr=draft_lr,
            checkpoint=checkpoint,
            seed=seed + 1,
            validation_prompts=evaluation_prompts,
            validation_size=validation_size,
            validation_seed=validation_seed,
            denoising_steps=denoising_steps,
            cfg=cfg,
        ),
    ]
    append_trigger(draft, trigger_word)

    continued = [
        python,
        str(TRAINER),
        "--objective",
        "sft",
        "--csv",
        str(dataset_csv),
        "--cache-latents",
        "--resume-training-state",
        str(shared_state),
        "--save-training-state",
        str(final_state),
        "--output-dir",
        str(sft_dir),
        "--validation-at-end",
        *common_training_args(
            rank=rank,
            batch_size=batch_size,
            train_steps=continuation_steps,
            lr=sft_lr,
            checkpoint=checkpoint,
            seed=seed,
            validation_prompts=evaluation_prompts,
            validation_size=validation_size,
            validation_seed=validation_seed,
            denoising_steps=denoising_steps,
            cfg=cfg,
        ),
    ]
    append_trigger(continued, trigger_word)
    return shared, draft, continued


def run_stage(
    name: str,
    command: list[str],
    *,
    output_dir: Path,
    expected: list[Path],
    dependencies: list[Path],
    force: bool,
) -> float:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".compare-stage.json"
    signature = {
        "command": command,
        "dependencies": file_dependencies(dependencies),
    }
    if not force and marker.is_file() and all(path.exists() for path in expected):
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if previous and previous.get("signature") == signature:
            click.echo(f"{name}: already complete")
            return float(previous.get("wall_seconds", 0.0))

    click.echo(f"{name}: {shlex.join(command)}")
    log_path = output_dir / "stage.log"
    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            click.echo(line, nl=False)
        return_code = process.wait()
    wall_seconds = time.perf_counter() - start
    if return_code:
        raise click.ClickException(f"{name} failed with exit code {return_code}")
    missing = [path for path in expected if not path.exists()]
    if missing:
        raise click.ClickException(f"{name} did not produce: {missing}")
    write_json(
        marker,
        {
            "signature": signature,
            "wall_seconds": wall_seconds,
        },
    )
    return wall_seconds


def validation_images(directory: Path, count: int) -> list[Path]:
    paths = [directory / f"image_{index:03d}.png" for index in range(count)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise click.ClickException(f"missing validation images: {missing}")
    return paths


def expected_validation_images(directory: Path, count: int) -> list[Path]:
    return [directory / f"image_{index:03d}.png" for index in range(count)]


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def create_grid(
    draft_images: list[Path],
    sft_images: list[Path],
    output: Path,
    *,
    hybrid_label: str = "SFT 500 +\nDRaFT-K 50",
    sft_label: str = "SFT 1000",
) -> None:
    if len(draft_images) != len(sft_images):
        raise ValueError("comparison rows must have equal lengths")
    cell = 512
    gutter = 260
    header = 48
    canvas = Image.new(
        "RGB", (gutter + cell * len(draft_images), header + cell * 2), "white"
    )
    draw = ImageDraw.Draw(canvas)
    header_font = _font(22)
    label_font = _font(24)
    for index in range(len(draft_images)):
        text = f"Prompt {index + 1}"
        x = gutter + index * cell + cell // 2
        box = draw.textbbox((0, 0), text, font=header_font)
        draw.text(
            (x - (box[2] - box[0]) // 2, 12), text, fill="black", font=header_font
        )
    labels = [hybrid_label, sft_label]
    rows = [draft_images, sft_images]
    for row_index, (label, images) in enumerate(zip(labels, rows)):
        y = header + row_index * cell
        draw.multiline_text(
            (20, y + cell // 2 - 35),
            label,
            fill="black",
            font=label_font,
            spacing=8,
        )
        for column, path in enumerate(images):
            with Image.open(path) as image:
                image = image.convert("RGB").resize(
                    (cell, cell), Image.Resampling.LANCZOS
                )
                canvas.paste(image, (gutter + column * cell, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    with Image.open(path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1.0
    return tensor.to(device)


def score_image_set(
    name: str,
    paths: list[Path],
    reward: FaceSimilarityReward,
) -> tuple[dict, list[dict]]:
    rows = []
    detected_scores = []
    reward_values = []
    for index, path in enumerate(paths):
        tensor = image_tensor(path, reward.device)
        faces = reward.detect_faces(tensor_to_bgr(tensor))
        crops = []
        for face in faces:
            try:
                crops.append(reward.crop_tensor(tensor, face))
            except (RuntimeError, ValueError):
                continue
        identity = None
        if crops:
            with torch.no_grad():
                embeddings = reward.encode_faces(torch.cat(crops, dim=0))
                identity = float(reward.identity_scores(embeddings).max().item())
            detected_scores.append(identity)
        with torch.no_grad():
            training_reward = float(reward(tensor.unsqueeze(0)).item())
        reward_values.append(training_reward)
        rows.append(
            {
                "set": name,
                "image_index": index,
                "image_path": str(path),
                "detected": bool(crops),
                "face_count": len(faces),
                "identity_similarity": identity,
                "training_reward": training_reward,
            }
        )

    def stats(values: list[float]) -> dict | None:
        if not values:
            return None
        tensor = torch.tensor(values, dtype=torch.float64)
        return {
            "mean": float(tensor.mean()),
            "std": float(tensor.std(unbiased=False)),
            "min": float(tensor.min()),
            "max": float(tensor.max()),
        }

    summary = {
        "images": len(paths),
        "face_detection_rate": len(detected_scores) / max(len(paths), 1),
        "identity_similarity": stats(detected_scores),
        "training_reward": stats(reward_values),
    }
    return summary, rows


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "set",
        "image_index",
        "image_path",
        "detected",
        "face_count",
        "identity_similarity",
        "training_reward",
    ]
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def runtime_metadata() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except subprocess.CalledProcessError:
        commit, dirty = None, None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": gpu,
    }


@click.command(help="Compare SFT 500 + DRaFT-K 50 against an exact SFT 1000 run.")
@click.argument(
    "images_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--output-dir", required=True, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--trigger-word", default=None)
@click.option(
    "--shared-sft-steps", default=500, show_default=True, type=click.IntRange(1)
)
@click.option("--draft-steps", default=50, show_default=True, type=click.IntRange(1))
@click.option(
    "--total-sft-steps", default=1000, show_default=True, type=click.IntRange(2)
)
@click.option("--prompt-count", default=64, show_default=True, type=click.IntRange(1))
@click.option("--validation-size", default=8, show_default=True, type=click.IntRange(1))
@click.option("--rank", default=32, show_default=True, type=click.Choice([32, 64]))
@click.option("--batch-size", default=1, show_default=True, type=click.IntRange(1))
@click.option(
    "--sft-lr",
    default=1e-4,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option(
    "--draft-lr",
    default=5e-5,
    show_default=True,
    type=click.FloatRange(min=0, min_open=True),
)
@click.option("--draft-k", default=1, show_default=True, type=click.IntRange(1))
@click.option(
    "--denoising-steps", default=20, show_default=True, type=click.IntRange(1)
)
@click.option("--cfg", default=4.5, show_default=True, type=float)
@click.option("--seed", default=0, show_default=True, type=int)
@click.option(
    "--checkpoint",
    envvar="K2_CHECKPOINT",
    default="oss_raw",
    show_default=True,
    type=click.Choice(["oss_raw", "oss_turbo"]),
)
@click.option(
    "--caption-model", default=pipeline.DEFAULT_CAPTION_MODEL, show_default=True
)
@click.option(
    "--prompt-model", default=pipeline.DEFAULT_PROMPT_MODEL, show_default=True
)
@click.option(
    "--face-model-dir", default=None, type=click.Path(file_okay=False, path_type=Path)
)
@click.option("--recaption", is_flag=True)
@click.option("--regenerate-prompts", is_flag=True)
@click.option("--force", is_flag=True)
def main(
    images_dir: Path,
    output_dir: Path,
    trigger_word: str | None,
    shared_sft_steps: int,
    draft_steps: int,
    total_sft_steps: int,
    prompt_count: int,
    validation_size: int,
    rank: int,
    batch_size: int,
    sft_lr: float,
    draft_lr: float,
    draft_k: int,
    denoising_steps: int,
    cfg: float,
    seed: int,
    checkpoint: str,
    caption_model: str,
    prompt_model: str,
    face_model_dir: Path | None,
    recaption: bool,
    regenerate_prompts: bool,
    force: bool,
) -> None:
    if total_sft_steps <= shared_sft_steps:
        raise click.ClickException("--total-sft-steps must exceed --shared-sft-steps")
    if draft_k > denoising_steps:
        raise click.ClickException("--draft-k cannot exceed --denoising-steps")

    images_dir = images_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trigger_word = pipeline.normalize_trigger_word(trigger_word)
    images = pipeline.discover_images(images_dir, exclude_dir=output_dir)
    if not images:
        raise click.ClickException(f"no supported images found under {images_dir}")
    if batch_size > len(images):
        raise click.ClickException("--batch-size exceeds the reference image count")
    base_checkpoint = pipeline.validate_base_checkpoint(checkpoint)

    data_dir = output_dir / "data"
    dataset_csv = data_dir / "dataset.csv"
    captions = pipeline.prepare_captions(
        images,
        data_dir / "captions.json",
        model_name=caption_model,
        recaption=recaption,
    )
    pipeline.write_dataset_csv(dataset_csv, images, captions)

    all_prompts_path = data_dir / "all_prompts.txt"
    all_prompts = pipeline.prepare_draft_prompts(
        dataset_csv,
        all_prompts_path,
        count=prompt_count + validation_size,
        seed=seed + 10_000,
        model_name=prompt_model,
        regenerate=regenerate_prompts,
    )
    try:
        draft_prompt_values, evaluation_prompt_values = split_prompt_pool(
            all_prompts, prompt_count, validation_size
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    draft_prompts = data_dir / "draft_train_prompts.txt"
    evaluation_prompts = data_dir / "evaluation_prompts.txt"
    write_prompts(draft_prompts, draft_prompt_values)
    write_prompts(evaluation_prompts, evaluation_prompt_values)

    face_dir = ensure_face_models(locate_face_model_dir(face_model_dir))
    shared_dir = output_dir / f"shared_sft_{shared_sft_steps}"
    draft_dir = output_dir / f"draft_{draft_steps}"
    sft_dir = output_dir / f"sft_{total_sft_steps}"
    shared_state = shared_dir / f"training_state_step_{shared_sft_steps:06d}.pt"
    final_state = sft_dir / f"training_state_step_{total_sft_steps:06d}.pt"
    commands = build_experiment_commands(
        python=sys.executable,
        dataset_csv=dataset_csv,
        draft_prompts=draft_prompts,
        evaluation_prompts=evaluation_prompts,
        reference_images=images,
        face_model_dir=face_dir,
        shared_dir=shared_dir,
        draft_dir=draft_dir,
        sft_dir=sft_dir,
        trigger_word=trigger_word,
        checkpoint=checkpoint,
        rank=rank,
        batch_size=batch_size,
        shared_sft_steps=shared_sft_steps,
        draft_steps=draft_steps,
        total_sft_steps=total_sft_steps,
        sft_lr=sft_lr,
        draft_lr=draft_lr,
        draft_k=draft_k,
        denoising_steps=denoising_steps,
        cfg=cfg,
        validation_size=validation_size,
        seed=seed,
    )
    shared_command, draft_command, continued_command = commands
    plan = {
        "images_dir": str(images_dir),
        "images": [str(path) for path in images],
        "trigger_word": trigger_word,
        "held_out_evaluation_prompts": evaluation_prompt_values,
        "commands": {
            "shared_sft": shared_command,
            "draft_branch": draft_command,
            "continued_sft_branch": continued_command,
        },
        "runtime": runtime_metadata(),
    }
    write_json(output_dir / "experiment_plan.json", plan)

    common_dependencies = [
        dataset_csv,
        evaluation_prompts,
        base_checkpoint,
        *TRAINING_CODE,
    ]
    shared_wall = run_stage(
        "shared SFT",
        shared_command,
        output_dir=shared_dir,
        expected=[
            shared_dir / "lora_latest.safetensors",
            shared_state,
            shared_dir / "training_summary.json",
            *expected_validation_images(
                shared_dir / "validation" / "step_000000", validation_size
            ),
            *expected_validation_images(
                shared_dir / "validation" / f"step_{shared_sft_steps:06d}",
                validation_size,
            ),
        ],
        dependencies=common_dependencies,
        force=force,
    )
    draft_wall = run_stage(
        "DRaFT-K branch",
        draft_command,
        output_dir=draft_dir,
        expected=[
            draft_dir / "lora_latest.safetensors",
            draft_dir / "training_summary.json",
            *expected_validation_images(
                draft_dir / "validation" / f"step_{draft_steps:06d}",
                validation_size,
            ),
        ],
        dependencies=[
            draft_prompts,
            evaluation_prompts,
            shared_dir / "lora_latest.safetensors",
            *TRAINING_CODE,
            *(face_dir / spec.relative_path for spec in FACE_MODEL_FILES),
        ],
        force=force,
    )
    continued_wall = run_stage(
        "continued SFT branch",
        continued_command,
        output_dir=sft_dir,
        expected=[
            sft_dir / "lora_latest.safetensors",
            final_state,
            sft_dir / "training_summary.json",
            *expected_validation_images(
                sft_dir / "validation" / f"step_{total_sft_steps:06d}",
                validation_size,
            ),
        ],
        dependencies=[
            shared_state,
            dataset_csv,
            evaluation_prompts,
            base_checkpoint,
            *TRAINING_CODE,
        ],
        force=force,
    )

    image_sets = {
        "base": validation_images(
            shared_dir / "validation" / "step_000000", validation_size
        ),
        "shared_sft": validation_images(
            shared_dir / "validation" / f"step_{shared_sft_steps:06d}",
            validation_size,
        ),
        "sft_plus_draft": validation_images(
            draft_dir / "validation" / f"step_{draft_steps:06d}", validation_size
        ),
        "total_sft": validation_images(
            sft_dir / "validation" / f"step_{total_sft_steps:06d}", validation_size
        ),
    }
    grid_path = output_dir / "comparison_grid.png"
    create_grid(
        image_sets["sft_plus_draft"],
        image_sets["total_sft"],
        grid_path,
        hybrid_label=f"SFT {shared_sft_steps} +\nDRaFT-K {draft_steps}",
        sft_label=f"SFT {total_sft_steps}",
    )

    reward = FaceSimilarityReward(reference_images=images, model_dir=face_dir)
    metric_summaries = {}
    metric_rows = []
    for name, paths in image_sets.items():
        summary, rows = score_image_set(name, paths, reward)
        metric_summaries[name] = summary
        metric_rows.extend(rows)
    write_metrics_csv(output_dir / "metrics.csv", metric_rows)

    summaries = {
        name: json.loads((directory / "training_summary.json").read_text())
        for name, directory in {
            "shared_sft": shared_dir,
            "draft_branch": draft_dir,
            "continued_sft_branch": sft_dir,
        }.items()
    }
    shared_seconds = summaries["shared_sft"]["optimization_seconds"]
    hybrid_seconds = shared_seconds + summaries["draft_branch"]["optimization_seconds"]
    pure_seconds = (
        shared_seconds + summaries["continued_sft_branch"]["optimization_seconds"]
    )
    results = {
        "training": summaries,
        "totals": {
            "sft_plus_draft_seconds": hybrid_seconds,
            "pure_sft_seconds": pure_seconds,
            "hybrid_over_pure_time_ratio": hybrid_seconds / max(pure_seconds, 1e-12),
        },
        "stage_wall_seconds": {
            "shared_sft": shared_wall,
            "draft_branch": draft_wall,
            "continued_sft_branch": continued_wall,
        },
        "face_metrics": metric_summaries,
        "comparison_grid": str(grid_path),
    }
    write_json(output_dir / "experiment_results.json", results)
    click.echo(f"comparison complete: {grid_path}")


if __name__ == "__main__":
    main()
