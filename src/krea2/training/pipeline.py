"""End-to-end character LoRA training from a folder of reference images.

The pipeline captions input images, writes the SFT CSV, creates a fixed set of
diverse DRaFT prompts, runs SFT, and resumes that adapter for DRaFT-K.
Each training phase runs in a child process so all model/GPU state is released
between stages.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

import click

from krea2.preprocessing import captioning as caption_script
from krea2.preprocessing import prompting as prompt_script
from krea2.training.config import PipelineConfig, RewardSpec, TrainingConfig

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TRAINER_PATH = SCRIPTS_DIR / "train.py"
CAPTION_SCRIPT = Path(caption_script.__file__).resolve()
PROMPT_SCRIPT = Path(prompt_script.__file__).resolve()
IMAGE_EXTENSIONS = caption_script.IMAGE_EXTS
CAPTION_CACHE_VERSION = 1
DEFAULT_CAPTION_MODEL = caption_script.DEFAULT_MODEL
DEFAULT_PROMPT_MODEL = prompt_script.DEFAULT_MODEL


def normalize_trigger_word(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def discover_images(images_dir: Path, exclude_dir: Path | None = None) -> list[Path]:
    """Return supported images recursively in stable path order."""
    root = images_dir.expanduser().resolve()
    excluded = exclude_dir.expanduser().resolve() if exclude_dir is not None else None
    if excluded is not None and not excluded.is_relative_to(root):
        excluded = None
    images = []
    for path in root.rglob("*"):
        if not caption_script.is_image_file(path):
            continue
        resolved = path.resolve()
        if excluded is not None and resolved.is_relative_to(excluded):
            continue
        images.append(resolved)
    return sorted(images, key=lambda path: str(path).casefold())


def normalize_caption(caption: str) -> str:
    caption = " ".join(str(caption).split()).strip(" ,")
    if not caption:
        raise ValueError("caption is empty")
    return caption


def sidecar_caption(image_path: Path) -> str | None:
    sidecar = image_path.with_suffix(".txt")
    if not sidecar.is_file():
        return None
    for line in sidecar.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return normalize_caption(line)
    return None


def _read_caption_cache(path: Path) -> dict:
    if not path.is_file():
        return {"version": CAPTION_CACHE_VERSION, "images": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"version": CAPTION_CACHE_VERSION, "images": {}}
    if data.get("version") != CAPTION_CACHE_VERSION or not isinstance(
        data.get("images"), dict
    ):
        return {"version": CAPTION_CACHE_VERSION, "images": {}}
    return data


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def file_fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class DeepInfraCaptioner:
    """Use the repository's character-safe DeepInfra captioning policy."""

    def __init__(self, model_name: str):
        self.script = caption_script
        self.key = caption_script.api_key()
        self.model_name = model_name

    def __call__(self, image_path: Path) -> str:
        return self.script.request_caption(
            image_path,
            key=self.key,
            model=self.model_name,
            endpoint=self.script.DEFAULT_ENDPOINT,
            system_prompt=self.script.DEFAULT_SYSTEM_PROMPT,
            prompt=self.script.DEFAULT_USER_PROMPT,
            reasoning_effort="none",
            max_tokens=512,
            temperature=0.2,
            timeout=120.0,
            retries=2,
        )


def prepare_captions(
    images: list[Path],
    cache_path: Path,
    *,
    model_name: str,
    recaption: bool = False,
    captioner_factory: Callable[[str], Callable[[Path], str]] = DeepInfraCaptioner,
) -> list[str]:
    """Resolve sidecar/cache captions and generate only the missing entries."""
    cache = _read_caption_cache(cache_path)
    records = cache["images"]
    active_images = {str(image) for image in images}
    for stale_path in records.keys() - active_images:
        del records[stale_path]
    captions: dict[Path, str] = {}
    missing = []
    caption_policy = file_fingerprint(CAPTION_SCRIPT)

    for image in images:
        key = str(image)
        fingerprint = file_fingerprint(image)
        sidecar = None if recaption else sidecar_caption(image)
        if sidecar is not None:
            captions[image] = sidecar
            records[key] = {
                **fingerprint,
                "caption": sidecar,
                "source": "sidecar",
                "model": None,
            }
            continue
        record = records.get(key, {})
        reusable = (
            not recaption
            and record.get("size") == fingerprint["size"]
            and record.get("mtime_ns") == fingerprint["mtime_ns"]
            and record.get("model") == model_name
            and record.get("caption_policy") == caption_policy
            and record.get("caption")
        )
        if reusable:
            captions[image] = normalize_caption(record["caption"])
        else:
            missing.append(image)

    owned_captioner = None
    if missing:
        click.echo(f"captioning {len(missing)} images with {model_name!r}")
        try:
            owned_captioner = captioner_factory(model_name)
        except RuntimeError as exc:
            raise click.ClickException(str(exc)) from exc
        try:
            with click.progressbar(missing, label="captioning") as progress:
                for image in progress:
                    try:
                        caption = normalize_caption(owned_captioner(image))
                    except RuntimeError as exc:
                        raise click.ClickException(str(exc)) from exc
                    captions[image] = caption
                    records[str(image)] = {
                        **file_fingerprint(image),
                        "caption": caption,
                        "source": "generated",
                        "model": model_name,
                        "caption_policy": caption_policy,
                    }
                    _write_json(cache_path, cache)
        finally:
            close = getattr(owned_captioner, "close", None)
            if close is not None:
                close()
    else:
        _write_json(cache_path, cache)

    return [captions[image] for image in images]


def write_dataset_csv(path: Path, images: list[Path], captions: list[str]) -> None:
    if len(images) != len(captions):
        raise ValueError("images and captions must have the same length")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "prompt"])
        writer.writeheader()
        for image, caption in zip(images, captions):
            writer.writerow(
                {
                    "image_path": str(image.resolve()),
                    "prompt": normalize_caption(caption),
                }
            )
    if path.is_file() and path.read_bytes() == temporary.read_bytes():
        temporary.unlink()
    else:
        temporary.replace(path)


def prepare_draft_prompts(
    dataset_csv: Path,
    output_path: Path,
    *,
    count: int,
    seed: int,
    model_name: str,
    regenerate: bool,
    prompt_generator: Callable[..., list[str]] = prompt_script.generate_prompts,
) -> list[str]:
    """Generate prompts once per dataset/config signature."""
    marker = output_path.with_name(output_path.name + ".stage.json")
    signature = {
        "count": count,
        "seed": seed,
        "model": model_name,
        "dependencies": _dependency_fingerprints([dataset_csv, PROMPT_SCRIPT]),
    }
    if not regenerate and output_path.is_file() and marker.is_file():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        prompts = [
            line.strip()
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if previous == signature and len(prompts) == count:
            click.echo(f"prompt generation: reusing {output_path}")
            return prompts

    click.echo(f"prompt generation: creating {count} prompts with {model_name!r}")
    try:
        source_prompts = prompt_script.read_source_prompts(dataset_csv)
        prompts = prompt_generator(
            source_prompts,
            count=count,
            seed=seed,
            model=model_name,
        )
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(f"prompt generation failed: {exc}") from exc
    if len(prompts) != count:
        raise click.ClickException(
            f"prompt generator wrote {len(prompts)} prompts, expected {count}"
        )
    prompt_script.write_prompts(output_path, prompts)
    _write_json(marker, signature)
    return prompts


def _append_trigger(command: list[str], trigger_word: str | None) -> None:
    trigger_word = normalize_trigger_word(trigger_word)
    if trigger_word is not None:
        command.extend(["--trigger-word", trigger_word])


def _save_interval(train_steps: int) -> int:
    return min(100, max(1, int(train_steps)))


def _common_training_args(
    config: TrainingConfig, *, train_steps: int, lr: float, seed: int
) -> list[str]:
    args = [
        "--rank",
        str(config.rank),
        "--train-steps",
        str(train_steps),
        "--batch-size",
        str(config.batch_size),
        "--lr",
        str(lr),
        "--checkpoint",
        config.checkpoint,
        "--seed",
        str(seed),
        "--save-every",
        str(_save_interval(train_steps)),
        "--validation-step",
        str(config.validation_step),
        "--validation-size",
        str(config.validation_size),
        "--validation-steps",
        str(config.validation_steps),
        "--quantization-type",
        "rowwise",
        "--compile-mode",
        "default",
    ]
    if config.validation_step > 0:
        args.append("--validation-at-end")
    return args


def build_training_commands(config: TrainingConfig) -> tuple[list[str], list[str]]:
    trainer = str(TRAINER_PATH)
    sft = [
        config.python,
        trainer,
        "--objective",
        "sft",
        "--csv",
        str(config.dataset_csv),
        "--validation-csv",
        str(config.dataset_csv),
        "--cache-latents",
        "--skip-final-sample",
        "--output-dir",
        str(config.sft_output_dir),
        *_common_training_args(
            config,
            train_steps=config.sft_steps,
            lr=config.sft_lr,
            seed=config.seed,
        ),
    ]
    _append_trigger(sft, config.trigger_word)

    reward_init = json.dumps(config.reward.init_kwargs, separators=(",", ":"))
    draft = [
        config.python,
        trainer,
        "--objective",
        "draft",
        "--prompts",
        str(config.draft_prompts),
        "--validation-prompts",
        str(config.draft_prompts),
        "--resume-lora",
        str(config.sft_output_dir / "lora_latest.safetensors"),
        "--reward",
        config.reward.target,
        "--reward-init-kwargs",
        reward_init,
        "--draft-k",
        str(config.draft_k),
        "--draft-lv-samples",
        str(config.draft_lv_samples),
        "--draft-diversity-every",
        str(config.draft_diversity_every),
        "--lora-target",
        "qkvo",
        "--steps",
        str(config.denoising_steps),
        "--cfg",
        str(config.cfg),
        "--output-dir",
        str(config.draft_output_dir),
        *_common_training_args(
            config,
            train_steps=config.draft_steps,
            lr=config.draft_lr,
            seed=config.seed + 1,
        ),
    ]
    if config.reward.call_kwargs:
        draft.extend(
            [
                "--reward-kwargs",
                json.dumps(config.reward.call_kwargs, separators=(",", ":")),
            ]
        )
    _append_trigger(draft, config.trigger_word)
    return sft, draft


def _dependency_fingerprints(paths: Iterable[Path]) -> dict[str, dict[str, int]]:
    fingerprints = {}
    for path in paths:
        path = path.resolve()
        fingerprints[str(path)] = file_fingerprint(path)
    return fingerprints


def run_training_stage(
    name: str,
    command: list[str],
    *,
    output_dir: Path,
    expected_output: Path,
    dependencies: Iterable[Path],
    force: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / ".pipeline-stage.json"
    signature = {
        "command_sha256": hashlib.sha256("\0".join(command).encode()).hexdigest(),
        "dependencies": _dependency_fingerprints(dependencies),
    }
    if not force and expected_output.is_file() and marker.is_file():
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None
        if previous == signature:
            click.echo(f"{name}: already complete, reusing {expected_output}")
            return expected_output

    click.echo(f"{name}: {shlex.join(_redacted_command(command))}")
    try:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as exc:
        raise click.ClickException(
            f"{name} failed with exit code {exc.returncode}"
        ) from exc
    if not expected_output.is_file():
        raise click.ClickException(
            f"{name} completed without producing {expected_output}"
        )
    _write_json(marker, signature)
    return expected_output


def validate_base_checkpoint(checkpoint: str) -> Path:
    env_name = {"oss_raw": "OSS_RAW", "oss_turbo": "OSS_TURBO"}[checkpoint]
    value = os.environ.get(env_name)
    if not value:
        raise click.ClickException(
            f"{env_name} must point to the Krea 2 checkpoint before training"
        )
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise click.ClickException(f"{env_name} does not point to a file: {path}")
    return path


_SENSITIVE_FRAGMENTS = ("key", "token", "secret", "password", "credential")


def _redact(value):
    if isinstance(value, dict):
        return {
            key: (
                "<redacted>"
                if any(fragment in key.lower() for fragment in _SENSITIVE_FRAGMENTS)
                else _redact(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _redacted_command(command: list[str]) -> list[str]:
    result = list(command)
    for option in ("--reward-init-kwargs", "--reward-kwargs"):
        if option not in result:
            continue
        index = result.index(option) + 1
        try:
            result[index] = json.dumps(_redact(json.loads(result[index])))
        except (IndexError, json.JSONDecodeError):
            result[index] = "<redacted>"
    return result


def reward_source_dependencies(target: str) -> tuple[Path, ...]:
    module_name = target.partition(":")[0]
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return ()
    if spec is None or spec.origin is None:
        return ()
    path = Path(spec.origin)
    return (path.resolve(),) if path.is_file() else ()


def run_pipeline(config: PipelineConfig, reward: RewardSpec) -> Path:
    """Run captioning, SFT, then DRaFT-K with the supplied reward."""
    base_checkpoint = validate_base_checkpoint(config.checkpoint)
    images_dir = config.images_dir.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    if output_dir == images_dir:
        raise click.ClickException(
            "--output-dir must be separate from the image folder"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    trigger_word = normalize_trigger_word(config.trigger_word)

    images = discover_images(images_dir, exclude_dir=output_dir)
    if not images:
        raise click.ClickException(f"no supported images found under {images_dir}")
    if config.batch_size > len(images):
        raise click.ClickException(
            f"--batch-size {config.batch_size} exceeds the "
            f"{len(images)} discovered images"
        )
    if config.draft_k > config.denoising_steps:
        raise click.ClickException("--draft-k cannot exceed --denoising-steps")
    if config.draft_lv_samples and config.draft_k != 1:
        raise click.ClickException("--draft-lv-samples requires --draft-k 1")
    click.echo(f"found {len(images)} reference images")

    data_dir = output_dir / "data"
    caption_cache = data_dir / "captions.json"
    dataset_csv = data_dir / "dataset.csv"
    draft_prompts_path = data_dir / "draft_prompts.txt"
    captions = prepare_captions(
        images,
        caption_cache,
        model_name=config.caption_model,
        recaption=config.recaption,
    )
    write_dataset_csv(dataset_csv, images, captions)
    prompts = prepare_draft_prompts(
        dataset_csv,
        draft_prompts_path,
        count=config.prompt_count,
        seed=config.seed + 10_000,
        model_name=config.prompt_model,
        regenerate=config.regenerate_prompts,
    )
    click.echo(f"wrote SFT dataset to {dataset_csv}")
    click.echo(f"wrote {len(prompts)} DRaFT prompts to {draft_prompts_path}")

    source_dependencies = reward_source_dependencies(reward.target)
    package_dependencies = tuple(sorted((PROJECT_ROOT / "src" / "krea2").rglob("*.py")))
    resolved_reward = RewardSpec(
        target=reward.target,
        init_kwargs=reward.init_kwargs,
        call_kwargs=reward.call_kwargs,
        dependencies=tuple(dict.fromkeys((*reward.dependencies, *source_dependencies))),
    )
    training = TrainingConfig(
        python=sys.executable,
        dataset_csv=dataset_csv,
        draft_prompts=draft_prompts_path,
        sft_output_dir=output_dir / "sft",
        draft_output_dir=output_dir / "draft",
        reward=resolved_reward,
        trigger_word=trigger_word,
        checkpoint=config.checkpoint,
        rank=config.rank,
        batch_size=config.batch_size,
        sft_steps=config.sft_steps,
        draft_steps=config.draft_steps,
        sft_lr=config.sft_lr,
        draft_lr=config.draft_lr,
        draft_k=config.draft_k,
        draft_lv_samples=config.draft_lv_samples,
        draft_diversity_every=config.draft_diversity_every,
        denoising_steps=config.denoising_steps,
        validation_steps=config.validation_steps,
        cfg=config.cfg,
        validation_step=config.validation_step,
        validation_size=config.validation_size,
        seed=config.seed,
    )
    sft_command, draft_command = build_training_commands(training)
    _write_json(
        output_dir / "pipeline_plan.json",
        {
            "images_dir": str(images_dir),
            "trigger_word": trigger_word,
            "reward": {
                "target": reward.target,
                "init_kwargs": _redact(reward.init_kwargs),
                "call_kwargs": _redact(reward.call_kwargs),
            },
            "sft_command": _redacted_command(sft_command),
            "draft_command": _redacted_command(draft_command),
        },
    )

    sft_adapter = run_training_stage(
        "SFT",
        sft_command,
        output_dir=training.sft_output_dir,
        expected_output=training.sft_output_dir / "lora_latest.safetensors",
        dependencies=[
            dataset_csv,
            base_checkpoint,
            TRAINER_PATH,
            *package_dependencies,
        ],
        force=config.force,
    )
    draft_adapter = run_training_stage(
        "DRaFT-K",
        draft_command,
        output_dir=training.draft_output_dir,
        expected_output=training.draft_output_dir / "lora_latest.safetensors",
        dependencies=[
            draft_prompts_path,
            sft_adapter,
            TRAINER_PATH,
            *package_dependencies,
            *resolved_reward.dependencies,
        ],
        force=config.force,
    )
    click.echo(f"pipeline complete: {draft_adapter}")
    return draft_adapter


def parse_json_object(value: str | None, option: str) -> dict:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"invalid JSON: {exc}", param_hint=option) from exc
    if not isinstance(parsed, dict):
        raise click.BadParameter("must be a JSON object", param_hint=option)
    return parsed


def pipeline_options(command):
    """Attach the options shared by generic and face-specialized entry points."""
    options = [
        click.argument(
            "images_dir",
            type=click.Path(exists=True, file_okay=False, path_type=Path),
        ),
        click.option(
            "--output-dir",
            required=True,
            type=click.Path(file_okay=False, path_type=Path),
        ),
        click.option("--trigger-word", default=None),
        click.option(
            "--sft-steps", default=500, show_default=True, type=click.IntRange(1)
        ),
        click.option(
            "--draft-steps", default=60, show_default=True, type=click.IntRange(1)
        ),
        click.option(
            "--prompt-count", default=64, show_default=True, type=click.IntRange(1)
        ),
        click.option(
            "--rank", default=32, show_default=True, type=click.Choice([32, 64])
        ),
        click.option(
            "--batch-size", default=1, show_default=True, type=click.IntRange(1)
        ),
        click.option(
            "--sft-lr",
            default=1e-4,
            show_default=True,
            type=click.FloatRange(min=0, min_open=True),
        ),
        click.option(
            "--draft-lr",
            default=1e-4,
            show_default=True,
            type=click.FloatRange(min=0, min_open=True),
        ),
        click.option("--draft-k", default=1, show_default=True, type=click.IntRange(1)),
        click.option(
            "--draft-lv-samples",
            default=1,
            show_default=True,
            type=click.IntRange(0),
        ),
        click.option(
            "--draft-diversity-every",
            default=4,
            show_default=True,
            type=click.IntRange(0),
        ),
        click.option(
            "--denoising-steps", default=12, show_default=True, type=click.IntRange(1)
        ),
        click.option(
            "--validation-steps", default=20, show_default=True, type=click.IntRange(1)
        ),
        click.option("--cfg", default=4.5, show_default=True, type=float),
        click.option(
            "--validation-step", default=100, show_default=True, type=click.IntRange(0)
        ),
        click.option(
            "--validation-size", default=10, show_default=True, type=click.IntRange(1)
        ),
        click.option("--seed", default=42, show_default=True, type=int),
        click.option(
            "--checkpoint",
            envvar="K2_CHECKPOINT",
            default="oss_raw",
            show_default=True,
            type=click.Choice(["oss_raw", "oss_turbo"]),
        ),
        click.option(
            "--caption-model", default=DEFAULT_CAPTION_MODEL, show_default=True
        ),
        click.option("--prompt-model", default=DEFAULT_PROMPT_MODEL, show_default=True),
        click.option("--recaption", is_flag=True),
        click.option("--regenerate-prompts", is_flag=True),
        click.option("--force", is_flag=True),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def pipeline_config_from_options(**options) -> PipelineConfig:
    return PipelineConfig(**options)


@click.command(help="Run SFT followed by DRaFT-K with a custom reward.")
@pipeline_options
@click.option("--reward", required=True, help="differentiable reward as module:object")
@click.option("--reward-init-kwargs", default=None, help="reward constructor JSON")
@click.option("--reward-kwargs", default=None, help="reward call JSON")
def main(
    reward: str,
    reward_init_kwargs: str | None,
    reward_kwargs: str | None,
    **options,
) -> None:
    spec = RewardSpec(
        target=reward,
        init_kwargs=parse_json_object(reward_init_kwargs, "--reward-init-kwargs"),
        call_kwargs=parse_json_object(reward_kwargs, "--reward-kwargs"),
    )
    run_pipeline(pipeline_config_from_options(**options), spec)


if __name__ == "__main__":
    main()
