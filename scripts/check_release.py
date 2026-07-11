#!/usr/bin/env python3
"""Run the local Krea 2 release gates."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_FILES = (
    ROOT / "main.py",
    ROOT / "train_face.py",
    ROOT / "scripts" / "train.py",
    ROOT / "scripts" / "compare.py",
    ROOT / "scripts" / "inference.py",
    ROOT / "scripts" / "inference_fp8.py",
    ROOT / "scripts" / "inference_int8.py",
    ROOT / "scripts" / "caption_images_deepinfra.py",
    ROOT / "scripts" / "generate_character_prompts_deepinfra.py",
    ROOT / "scripts" / "convert_images_to_jpg.py",
    ROOT / "scripts" / "convert_lora_to_comfyui.py",
    ROOT / "scripts" / "autoresearch.py",
    ROOT / "scripts" / "evaluate_autoresearch.py",
)
PACKAGE_IMPORTS = (
    "krea2.models.transformer",
    "krea2.inference.bf16",
    "krea2.inference.fp8",
    "krea2.inference.int8",
    "krea2.quantization.int8",
    "krea2.training.trainer",
    "krea2.training.pipeline",
    "krea2.experiments.comparison",
    "krea2.experiments.autoresearch",
)


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="also run CUDA kernels and antelopev2 face-reward tests",
    )
    args = parser.parse_args(argv)

    run(["uv", "lock", "--check"])
    run([sys.executable, "-m", "ruff", "check", "."])
    run([sys.executable, "-m", "ruff", "format", "--check", "."])
    run([sys.executable, "-m", "pytest", "-q", "-m", "not cuda"])
    for module in PACKAGE_IMPORTS:
        run([sys.executable, "-c", f"import {module}"])
    for script in CLI_FILES:
        run([sys.executable, str(script), "--help"])
    if args.cuda:
        run([sys.executable, "-m", "pytest", "-q", "-m", "cuda"])
    print("release checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
