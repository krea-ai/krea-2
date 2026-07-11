"""CPU tests for the full training pipeline and preprocessing."""

import csv
import json
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

from click.testing import CliRunner
from PIL import Image

from krea2.preprocessing import captioning as caption_script
from krea2.preprocessing import prompting as prompt_script
from krea2.rewards import face_models
from krea2.training import face_pipeline, pipeline
from krea2.training.config import RewardSpec, TrainingConfig


def make_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_discovery_and_caption_cache():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        images_dir = root / "images"
        output_dir = images_dir / "pipeline-output"
        first = images_dir / "a.jpg"
        second = images_dir / "nested" / "b.png"
        excluded = output_dir / "validation.png"
        make_image(first, (1, 2, 3))
        make_image(second, (4, 5, 6))
        make_image(excluded, (7, 8, 9))
        first.with_suffix(".txt").write_text("  a generic man by a window  \n")

        images = pipeline.discover_images(images_dir, exclude_dir=output_dir)
        assert images == [first.resolve(), second.resolve()]
        calls = []

        class FakeCaptioner:
            def __init__(self, model_name):
                assert model_name == "fake-caption-model"

            def __call__(self, path):
                calls.append(path)
                return "a generic man on a street"

        cache_path = root / "run" / "data" / "captions.json"
        captions = pipeline.prepare_captions(
            images,
            cache_path,
            model_name="fake-caption-model",
            captioner_factory=FakeCaptioner,
        )
        assert captions == [
            "a generic man by a window",
            "a generic man on a street",
        ]
        assert calls == [second.resolve()]

        class MustNotLoad:
            def __init__(self, _model_name):
                raise AssertionError("caption cache should avoid loading the model")

        repeated = pipeline.prepare_captions(
            images,
            cache_path,
            model_name="fake-caption-model",
            captioner_factory=MustNotLoad,
        )
        assert repeated == captions
        cache = json.loads(cache_path.read_text())
        assert cache["version"] == 1

        pipeline.prepare_captions(
            [first.resolve()],
            cache_path,
            model_name="fake-caption-model",
            captioner_factory=MustNotLoad,
        )
        cache = json.loads(cache_path.read_text())
        assert set(cache["images"]) == {str(first.resolve())}
    print("ok  recursive discovery, output exclusion, sidecars, and caption cache")


def test_repo_local_deepinfra_utilities():
    caption = caption_script.sanitize_caption(
        "A young man with dark hair, natural skin texture, beside a window"
    )
    lowered = caption.lower()
    assert "young" not in lowered
    assert "hair" not in lowered
    assert "skin" not in lowered
    assert caption.endswith(".")

    extracted = prompt_script.extract_prompts(
        '{"prompts":["A man in a studio.","A man outdoors."]}'
    )
    assert extracted == ["A man in a studio.", "A man outdoors."]
    calls = []
    old_request = prompt_script.request_prompt_batch

    def fake_request(**kwargs):
        calls.append(kwargs)
        return [
            f"A man in setting {index}, wearing a tailored jacket, framed as an "
            "editorial portrait with soft directional light and a calm expression."
            for index in range(5)
        ]

    prompt_script.request_prompt_batch = fake_request
    try:
        prompts = prompt_script.generate_prompts(
            ["A man beside a window in soft light."],
            count=3,
            seed=7,
            key="test-key",
        )
    finally:
        prompt_script.request_prompt_batch = old_request
    assert len(prompts) == len(set(prompts)) == 3
    assert calls and calls[0]["seed"] == 7
    assert all(prompt.startswith("A man ") for prompt in prompts)
    print("ok  repo-local caption and prompt policies run without parent imports")


def test_dataset_and_prompt_generation_cache():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        images = [root / "one.jpg", root / "two.jpg"]
        for index, image in enumerate(images):
            make_image(image, (index, index, index))
        dataset = root / "data" / "dataset.csv"
        pipeline.write_dataset_csv(dataset, images, ["first prompt", "second prompt"])
        with dataset.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert [row["image_path"] for row in rows] == [
            str(image.resolve()) for image in images
        ]
        assert [row["prompt"] for row in rows] == ["first prompt", "second prompt"]

        prompts_path = root / "data" / "draft_prompts.txt"
        calls = []

        def fake_generate(source_prompts, *, count, seed, model):
            assert source_prompts == ["first prompt", "second prompt"]
            assert (count, seed, model) == (3, 12, "fake-prompt-model")
            calls.append((count, seed, model))
            return [f"generated prompt {index}" for index in range(count)]

        prompts = pipeline.prepare_draft_prompts(
            dataset,
            prompts_path,
            count=3,
            seed=12,
            model_name="fake-prompt-model",
            regenerate=False,
            prompt_generator=fake_generate,
        )
        assert prompts == [
            "generated prompt 0",
            "generated prompt 1",
            "generated prompt 2",
        ]
        assert len(calls) == 1

        def must_not_generate(*args, **kwargs):
            raise AssertionError("prompt cache should avoid the API")

        repeated = pipeline.prepare_draft_prompts(
            dataset,
            prompts_path,
            count=3,
            seed=12,
            model_name="fake-prompt-model",
            regenerate=False,
            prompt_generator=must_not_generate,
        )
        assert repeated == prompts
        assert len(calls) == 1
    print("ok  absolute SFT CSV and cached repository prompt-generator stage")


def write_sparse(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)


def test_face_model_resolution_and_download():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        calls = []

        def fake_download(url, destination, minimum_bytes):
            calls.append((url, destination, minimum_bytes))
            write_sparse(destination, minimum_bytes)

        selected = face_models.locate_face_model_dir(root / "explicit-models")
        assert selected == (root / "explicit-models").resolve()
        ensured = face_models.ensure_face_models(selected, downloader=fake_download)
        assert ensured == selected
        assert len(calls) == len(face_models.FACE_MODEL_FILES)
        assert face_models.face_models_complete(ensured)
        face_models.ensure_face_models(selected, downloader=fake_download)
        assert len(calls) == len(face_models.FACE_MODEL_FILES)
        for spec in face_models.FACE_MODEL_FILES:
            assert (ensured / spec.relative_path).stat().st_size >= spec.minimum_bytes
    print("ok  explicit antelopev2 layout downloads missing models once")


def base_training_config(root: Path, trigger_word=None):
    images = [root / "one.jpg", root / "two.jpg"]
    reward = RewardSpec(
        target="krea2.rewards.face:FaceSimilarityReward",
        init_kwargs={
            "reference_images": [str(path) for path in images],
            "model_dir": str(root / "models" / "antelopev2"),
        },
    )
    return TrainingConfig(
        python="/venv/bin/python",
        dataset_csv=root / "data" / "dataset.csv",
        draft_prompts=root / "data" / "draft_prompts.txt",
        sft_output_dir=root / "sft",
        draft_output_dir=root / "draft",
        reward=reward,
        trigger_word=trigger_word,
        sft_steps=20,
        draft_steps=10,
        validation_step=5,
        validation_size=3,
    )


def test_training_commands_and_optional_trigger():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        config = base_training_config(root)
        config.dataset_csv.parent.mkdir(parents=True)
        config.dataset_csv.write_text("image_path,prompt\n/tmp/image.jpg,a person\n")
        config.draft_prompts.write_text("A portrait of a person.\n")
        config.sft_output_dir.mkdir(parents=True)
        (config.sft_output_dir / "lora_latest.safetensors").write_bytes(b"adapter")
        sft, draft = pipeline.build_training_commands(config)
        assert "--trigger-word" not in sft
        assert "--trigger-word" not in draft
        assert sft[sft.index("--objective") + 1] == "sft"
        assert "--cache-latents" in sft
        assert "--skip-final-sample" in sft
        assert draft[draft.index("--objective") + 1] == "draft"
        assert draft[draft.index("--lr") + 1] == "0.0004"
        assert draft[draft.index("--steps") + 1] == "12"
        assert draft[draft.index("--draft-lv-samples") + 1] == "1"
        assert draft[draft.index("--validation-steps") + 1] == "20"
        assert draft[draft.index("--resume-lora") + 1] == str(
            root / "sft" / "lora_latest.safetensors"
        )
        reward_kwargs = json.loads(draft[draft.index("--reward-init-kwargs") + 1])
        assert reward_kwargs["reference_images"] == [
            str(root / "one.jpg"),
            str(root / "two.jpg"),
        ]
        assert reward_kwargs["model_dir"] == str(root / "models" / "antelopev2")

        triggered = replace(config, trigger_word="char_tok")
        triggered_sft, triggered_draft = pipeline.build_training_commands(triggered)
        for command in (triggered_sft, triggered_draft):
            index = command.index("--trigger-word")
            assert command[index + 1] == "char_tok"

        from krea2.training import trainer

        for command in (sft, draft, triggered_sft, triggered_draft):
            context = trainer.main.make_context("train_draft_int8_lora", command[2:])
            context.close()
    print("ok  SFT-to-DRaFT commands omit None trigger and pass configured trigger")


def test_resumable_training_stage():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dependency = root / "dataset.csv"
        dependency.write_text("version one")
        output_dir = root / "sft"
        expected = output_dir / "lora_latest.safetensors"
        calls = []
        old_run = pipeline.subprocess.run

        def fake_run(command, cwd, check):
            assert cwd == pipeline.PROJECT_ROOT
            assert check
            calls.append(command)
            expected.write_bytes(b"adapter")

        pipeline.subprocess.run = fake_run
        try:
            command = ["python", "trainer.py", "--objective", "sft"]
            pipeline.run_training_stage(
                "SFT",
                command,
                output_dir=output_dir,
                expected_output=expected,
                dependencies=[dependency],
                force=False,
            )
            pipeline.run_training_stage(
                "SFT",
                command,
                output_dir=output_dir,
                expected_output=expected,
                dependencies=[dependency],
                force=False,
            )
            assert len(calls) == 1
            dependency.write_text("version two")
            pipeline.run_training_stage(
                "SFT",
                command,
                output_dir=output_dir,
                expected_output=expected,
                dependencies=[dependency],
                force=False,
            )
            assert len(calls) == 2
        finally:
            pipeline.subprocess.run = old_run
    print("ok  completed stages resume and dependency changes invalidate them")


def test_cli_help():
    result = CliRunner().invoke(pipeline.main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--trigger-word" in result.output
    assert "--reward" in result.output
    assert "--face-model-dir" not in result.output
    assert "--regenerate-prompts" in result.output
    face_result = CliRunner().invoke(face_pipeline.main, ["--help"])
    assert face_result.exit_code == 0, face_result.output
    assert "--face-model-dir" in face_result.output
    assert "--reward" not in face_result.output
    source = Path(pipeline.__file__).read_text(encoding="utf-8")
    assert "sys.path.insert" not in source
    assert pipeline.CAPTION_SCRIPT.is_relative_to(pipeline.PROJECT_ROOT)
    assert pipeline.PROMPT_SCRIPT.is_relative_to(pipeline.PROJECT_ROOT)
    print("ok  pipeline CLI help")


def test_generic_pipeline_does_not_import_face_dependencies():
    code = r"""
import importlib.abc
import sys

class RejectFaceDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.partition(".")[0] in {"cv2", "insightface", "onnx", "onnxruntime"}:
            raise ImportError(f"unexpected face dependency import: {fullname}")
        return None

sys.meta_path.insert(0, RejectFaceDependencies())
import krea2.training.pipeline
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_reward_source_and_command_redaction():
    dependencies = pipeline.reward_source_dependencies(
        "krea2.training.objectives:reward_loss"
    )
    assert dependencies == (
        Path(pipeline.__file__).parent.joinpath("objectives.py").resolve(),
    )
    command = [
        "python",
        "train.py",
        "--reward-init-kwargs",
        '{"api_key":"private","weight":0.5}',
    ]
    redacted = pipeline._redacted_command(command)
    assert json.loads(redacted[-1]) == {
        "api_key": "<redacted>",
        "weight": 0.5,
    }
