"""CPU tests for controlled comparison and exact SFT continuation."""

import json
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
from click.testing import CliRunner
from PIL import Image

from krea2.experiments import comparison as compare
from krea2.preprocessing import captioning as caption_script
from krea2.quantization.int8 import (
    LinearLoraINT8,
    load_lora_state_tensors,
    lora_parameters,
    lora_state_tensors,
    set_lora_trainable,
)
from krea2.training import pipeline, trainer
from krea2.training.objectives import reward_loss


class StateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = LinearLoraINT8(
            128,
            128,
            rank=32,
            bias=False,
            device="cpu",
            quantization_type="rowwise",
        )


class TargetAttention(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("wq", "wk", "wv", "gate", "wo"):
            setattr(
                self,
                name,
                LinearLoraINT8(
                    128,
                    128,
                    rank=32,
                    bias=False,
                    device="cpu",
                    quantization_type="rowwise",
                ),
            )


class TargetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TargetAttention()
        self.mlp = nn.Module()
        self.mlp.up = LinearLoraINT8(
            128,
            128,
            rank=32,
            bias=False,
            device="cpu",
            quantization_type="rowwise",
        )


def test_qkvo_target_keeps_full_adapter_state():
    model = nn.Module()
    model.blocks = nn.ModuleList([TargetBlock()])
    selected = set_lora_trainable(model, "qkvo")
    assert selected == [
        "blocks.0.attn.wq",
        "blocks.0.attn.wk",
        "blocks.0.attn.wv",
        "blocks.0.attn.wo",
    ]
    assert len(list(lora_parameters(model))) == 8
    assert not model.blocks[0].attn.gate.lora_A.requires_grad
    assert not model.blocks[0].mlp.up.lora_B.requires_grad
    assert len(lora_state_tensors(model)) == 12


class PairReward:
    def __call__(self, image, prompt, **kwargs):
        del prompt, kwargs
        return image.mean()

    def pairwise_reward(self, first, second, prompt, **kwargs):
        del prompt, kwargs
        return -(first - second).square().mean()


def test_reward_loss_pairwise_protocol():
    images = torch.randn(2, 3, 4, 4, requires_grad=True)
    loss, rewards = reward_loss(
        PairReward(),
        images,
        ["unspecified", "unspecified"],
        {},
        pair_indices=[(0, 1)],
    )
    assert rewards.shape == (2,)
    loss.backward()
    assert images.grad is not None and images.grad.abs().sum() > 0


def test_comparison_cli_baseline_defaults():
    result = CliRunner().invoke(compare.main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "SFT 500 + DRaFT-LV 60" in result.output
    assert "--draft-steps INTEGER" in result.output
    assert "--draft-lv-samples INTEGER" in result.output
    assert "--denoising-steps INTEGER" in result.output
    assert result.output.count("[default: 60;") >= 1
    assert result.output.count("[default: 12;") >= 1
    assert result.output.count("[default: 20;") >= 1


def test_training_state_roundtrip():
    torch.manual_seed(123)
    model = StateModel()
    optimizer = torch.optim.AdamW(list(lora_parameters(model)), lr=1e-4)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        for parameter in lora_parameters(model):
            parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    dataset = trainer.CachedSFTDataset(
        torch.randn(5, 2, 4, 4),
        torch.randn(5, 3, 1, 8),
        torch.ones(5, 3, dtype=torch.bool),
    )
    sampler = trainer.StatefulShuffleBatchSampler(5, 1, seed=99)
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(7)]
    assert len(consumed) == 7
    compatibility = {"objective": "sft", "dataset_signature": "abc"}

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "training_state.pt"
        trainer.write_training_state(
            path,
            model=model,
            optimizer=optimizer,
            dataset=dataset,
            sampler=sampler,
            global_step=7,
            compatibility=compatibility,
        )

        def continue_updates(current_model, current_optimizer, current_sampler):
            batches = []
            current_iterator = iter(current_sampler)
            for _ in range(8):
                batch = next(current_iterator)
                batches.append(batch)
                current_optimizer.zero_grad(set_to_none=True)
                for parameter in lora_parameters(current_model):
                    parameter.grad = torch.randn_like(parameter) + batch[0] * 1e-3
                current_optimizer.step()
            return batches

        expected_batches = continue_updates(model, optimizer, sampler)

        state = trainer.load_training_state(path)
        trainer.validate_training_state(state, compatibility)
        try:
            trainer.validate_training_state(
                state, {"objective": "sft", "dataset_signature": "changed"}
            )
        except ValueError as exc:
            assert "dataset_signature" in str(exc)
        else:
            raise AssertionError("incompatible training state must be rejected")
        restored_model = StateModel()
        load_lora_state_tensors(restored_model, state["lora"])
        restored_optimizer = torch.optim.AdamW(
            list(lora_parameters(restored_model)), lr=1e-4
        )
        restored_optimizer.load_state_dict(state["optimizer"])
        restored_sampler = trainer.StatefulShuffleBatchSampler(5, 1, seed=0)
        restored_sampler.load_state_dict(state["sampler"])
        trainer.restore_training_rng(state)
        assert (
            continue_updates(restored_model, restored_optimizer, restored_sampler)
            == expected_batches
        )
        for original, restored in zip(
            lora_parameters(model), lora_parameters(restored_model)
        ):
            torch.testing.assert_close(original, restored, atol=0, rtol=0)
        assert state["global_step"] == 7
        assert torch.equal(state["cached_sft"]["latents"], dataset.latents)
        assert not path.with_name(path.name + ".tmp").exists()
    print("ok  exact training state restores LoRA, AdamW, RNG, cache, and sampler")


def test_commands_and_heldout_prompts():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dataset = root / "dataset.csv"
        train_prompts = root / "draft.txt"
        eval_prompts = root / "eval.txt"
        dataset.write_text("image_path,prompt\n/tmp/a.png,a person\n")
        train_prompts.write_text("train prompt\n")
        eval_prompts.write_text("eval prompt\n")
        shared = root / "shared"
        draft = root / "draft"
        sft = root / "sft"
        shared.mkdir()
        (shared / "lora_latest.safetensors").write_bytes(b"adapter")
        (shared / "training_state_step_000500.pt").write_bytes(b"state")
        commands = compare.build_experiment_commands(
            python=sys.executable,
            dataset_csv=dataset,
            draft_prompts=train_prompts,
            evaluation_prompts=eval_prompts,
            reference_images=[root / "reference.png"],
            face_model_dir=root / "models",
            shared_dir=shared,
            draft_dir=draft,
            sft_dir=sft,
            trigger_word="subject_tok",
            checkpoint="oss_raw",
            rank=32,
            batch_size=1,
            shared_sft_steps=500,
            draft_steps=50,
            total_sft_steps=1000,
            sft_lr=1e-4,
            draft_lr=5e-5,
            draft_k=1,
            draft_lv_samples=1,
            draft_diversity_every=4,
            denoising_steps=20,
            validation_steps=20,
            cfg=4.5,
            validation_size=8,
            seed=7,
        )
        shared_command, draft_command, continued_command = commands
        assert shared_command[shared_command.index("--train-steps") + 1] == "500"
        assert "--validation-at-start" in shared_command
        assert "--validation-at-end" in shared_command
        assert draft_command[draft_command.index("--train-steps") + 1] == "50"
        assert draft_command[draft_command.index("--draft-lv-samples") + 1] == "1"
        assert draft_command[draft_command.index("--draft-diversity-every") + 1] == "4"
        assert draft_command[draft_command.index("--lora-target") + 1] == "qkvo"
        assert "--validation-at-start" not in draft_command
        assert draft_command[draft_command.index("--draft-image-every") + 1] == "0"
        assert continued_command[continued_command.index("--train-steps") + 1] == "500"
        assert "--resume-training-state" in continued_command
        assert "--validation-at-start" not in continued_command
        for command in commands:
            assert command[command.index("--validation-size") + 1] == "8"
            assert command[command.index("--validation-step") + 1] == "0"
            assert command[command.index("--validation-steps") + 1] == "20"
            assert command[command.index("--timing-warmup-steps") + 1] == "1"
            assert command[command.index("--validation-prompts") + 1] == str(
                eval_prompts
            )
            context = trainer.main.make_context("trainer", command[2:])
            context.close()

    training, evaluation = compare.split_prompt_pool(
        [f"unique prompt {index}" for index in range(72)], 64, 8
    )
    assert len(training) == 64 and len(evaluation) == 8
    assert set(training).isdisjoint(evaluation)
    print("ok  experiment commands and held-out prompt split")


def test_image_discovery_and_grid():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        jpeg = root / "extensionless"
        avif = root / "reference.avif"
        Image.new("RGB", (16, 16), "red").save(jpeg, format="JPEG")
        Image.new("RGB", (16, 16), "blue").save(avif, format="AVIF")
        assert pipeline.discover_images(root) == [jpeg.resolve(), avif.resolve()]
        assert caption_script.data_url(jpeg).startswith("data:image/jpeg;base64,")
        assert caption_script.data_url(avif).startswith("data:image/avif;base64,")

        draft_paths = []
        sft_paths = []
        for index in range(8):
            draft_path = root / f"draft_{index}.png"
            sft_path = root / f"sft_{index}.png"
            Image.new("RGB", (512, 512), (255, index, 0)).save(draft_path)
            Image.new("RGB", (512, 512), (0, index, 255)).save(sft_path)
            draft_paths.append(draft_path)
            sft_paths.append(sft_path)
        output = root / "grid.png"
        compare.create_grid(draft_paths, sft_paths, output)
        with Image.open(output) as grid:
            assert grid.size == (260 + 8 * 512, 48 + 2 * 512)
            assert grid.getpixel((260 + 256, 48 + 256))[0] == 255
            assert grid.getpixel((260 + 256, 48 + 512 + 256))[2] == 255
    print("ok  AVIF/extensionless discovery, MIME detection, and annotated grid")


def test_resumable_stage():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dependency = root / "dependency.txt"
        dependency.write_text("one")
        output_dir = root / "stage"
        expected = output_dir / "result.txt"
        counter = output_dir / "counter.txt"
        code = (
            "from pathlib import Path; "
            f"p=Path({str(counter)!r}); p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text(str(int(p.read_text()) + 1) if p.exists() else '1'); "
            f"Path({str(expected)!r}).write_text('done')"
        )
        command = [sys.executable, "-c", code]
        first = compare.run_stage(
            "test",
            command,
            output_dir=output_dir,
            expected=[expected],
            dependencies=[dependency],
            force=False,
        )
        second = compare.run_stage(
            "test",
            command,
            output_dir=output_dir,
            expected=[expected],
            dependencies=[dependency],
            force=False,
        )
        assert counter.read_text() == "1"
        assert first == second
        dependency.write_text("two")
        compare.run_stage(
            "test",
            command,
            output_dir=output_dir,
            expected=[expected],
            dependencies=[dependency],
            force=False,
        )
        assert counter.read_text() == "2"
        marker = json.loads((output_dir / ".compare-stage.json").read_text())
        assert "signature" in marker and "wall_seconds" in marker
    print("ok  comparison stages resume and invalidate on dependency changes")
