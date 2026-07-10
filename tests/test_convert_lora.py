"""CPU tests for the ComfyUI LoRA converter."""

import json
import tempfile
from pathlib import Path

import torch
from click.testing import CliRunner
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scripts import convert_lora_to_comfyui as converter


def test_conversion_and_scale_fold():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "training.safetensors"
        destination = root / "comfy.safetensors"
        tensors = {
            "blocks.0.attn.wq.lora_A": torch.arange(8).reshape(2, 4).float(),
            "blocks.0.attn.wq.lora_B": torch.arange(6).reshape(3, 2).float(),
            "blocks.1.mlp.down.lora_A": torch.ones(2, 3),
            "blocks.1.mlp.down.lora_B": torch.full((5, 2), 4.0),
        }
        save_file(
            tensors,
            source,
            metadata={
                "rank": "2",
                "lora_alpha": "1",
                "lora_scale": "0.5",
                "final_step": "50",
                "prompts": "/private/prompts.txt",
            },
        )

        result = CliRunner().invoke(
            converter.main,
            [str(source), str(destination), "--name", "test-character"],
        )
        assert result.exit_code == 0, result.output
        output = load_file(destination)
        assert set(output) == {
            "diffusion_model.blocks.0.attn.wq.lora_A.weight",
            "diffusion_model.blocks.0.attn.wq.lora_B.weight",
            "diffusion_model.blocks.1.mlp.down.lora_A.weight",
            "diffusion_model.blocks.1.mlp.down.lora_B.weight",
        }
        output_a = output["diffusion_model.blocks.0.attn.wq.lora_A.weight"]
        output_b = output["diffusion_model.blocks.0.attn.wq.lora_B.weight"]
        assert output_a.dtype == torch.bfloat16
        assert torch.equal(output_a, tensors["blocks.0.attn.wq.lora_A"].bfloat16())
        assert torch.equal(
            output_b,
            (tensors["blocks.0.attn.wq.lora_B"] * 0.25).bfloat16(),
        )
        with safe_open(destination, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        assert metadata["format"] == "pt"
        assert metadata["name"] == "test-character"
        assert metadata["ss_base_model_version"] == "krea2"
        assert json.loads(metadata["training_info"]) == {"step": 50}
        assert json.loads(metadata["conversion_info"])["folded_scale"] == 0.25
        assert "lora_alpha" not in metadata
        assert "lora_scale" not in metadata
        assert "prompts" not in metadata
    print("ok  ComfyUI keys, BF16 conversion, metadata, and scale folding")


def test_default_path_float32_and_dry_run():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source = root / "character.safetensors"
        save_file(
            {
                "blocks.0.attn.wk.lora_A": torch.ones(2, 4),
                "blocks.0.attn.wk.lora_B": torch.ones(3, 2),
            },
            source,
            metadata={"rank": "2"},
        )

        dry = CliRunner().invoke(converter.main, [str(source), "--dry-run"])
        assert dry.exit_code == 0, dry.output
        destination = root / "character_comfyui.safetensors"
        assert not destination.exists()

        result = CliRunner().invoke(converter.main, [str(source), "--dtype", "float32"])
        assert result.exit_code == 0, result.output
        output = load_file(destination)
        assert all(tensor.dtype == torch.float32 for tensor in output.values())

        blocked = CliRunner().invoke(converter.main, [str(source)])
        assert blocked.exit_code != 0
        assert "--overwrite" in blocked.output
    print("ok  default path, dry run, dtype selection, and overwrite guard")


def test_invalid_sources():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        incomplete = root / "incomplete.safetensors"
        save_file(
            {"blocks.0.attn.wq.lora_A": torch.ones(2, 4)},
            incomplete,
            metadata={"rank": "2"},
        )
        result = CliRunner().invoke(converter.main, [str(incomplete), "--dry-run"])
        assert result.exit_code != 0
        assert "incomplete LoRA pair" in result.output

        foreign = root / "foreign.safetensors"
        save_file(
            {
                "transformer.blocks.0.attn.lora_A": torch.ones(2, 4),
                "transformer.blocks.0.attn.lora_B": torch.ones(3, 2),
            },
            foreign,
        )
        result = CliRunner().invoke(converter.main, [str(foreign), "--dry-run"])
        assert result.exit_code != 0
        assert "unsupported source tensor key" in result.output
    print("ok  incomplete and foreign adapters are rejected")
