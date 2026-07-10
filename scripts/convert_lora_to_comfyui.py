#!/usr/bin/env python3
"""Convert a krea-2 training LoRA to Krea 2's ComfyUI LoRA layout."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import click
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SOURCE_KEY = re.compile(
    r"^(?P<module>blocks\.(?:0|[1-9]\d*)\..+)\.(?P<side>lora_[AB])$"
)
DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
SAFETENSORS_DTYPES = {
    torch.bfloat16: "BF16",
    torch.float16: "F16",
    torch.float32: "F32",
}
SAFE_SOURCE_METADATA = (
    "objective",
    "rank",
    "checkpoint",
    "trigger_word",
    "quantization_type",
    "initial_step",
    "final_step",
    "flow_convention",
)


@dataclass(frozen=True)
class LoraModule:
    name: str
    lora_a_key: str
    lora_b_key: str
    rank: int
    in_features: int
    out_features: int


@dataclass(frozen=True)
class ConversionPlan:
    modules: tuple[LoraModule, ...]
    source_metadata: dict[str, str]
    source_dtypes: tuple[str, ...]
    rank: int
    alpha: float
    source_scale: float

    @property
    def folded_scale(self) -> float:
        return self.alpha / self.rank * self.source_scale

    @property
    def tensor_count(self) -> int:
        return len(self.modules) * 2


def comfyui_key(source_key: str) -> str:
    match = SOURCE_KEY.fullmatch(source_key)
    if match is None:
        raise ValueError(
            f"unsupported source tensor key {source_key!r}; expected "
            "blocks.<index>.<module>.lora_A or .lora_B"
        )
    return f"diffusion_model.{match.group('module')}.{match.group('side')}.weight"


def _metadata_float(metadata: dict[str, str], key: str, default: float) -> float:
    value = metadata.get(key)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"metadata {key!r} is not a number: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"metadata {key!r} must be finite, got {value!r}")
    return parsed


def inspect_source(path: Path) -> ConversionPlan:
    pairs: dict[str, dict[str, tuple[str, tuple[int, ...]]]] = {}
    dtypes = set()
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        keys = list(handle.keys())
        if not keys:
            raise ValueError("the source file contains no tensors")
        for key in keys:
            match = SOURCE_KEY.fullmatch(key)
            if match is None:
                raise ValueError(
                    f"unsupported source tensor key {key!r}; this converter "
                    "accepts only krea-2 training LoRA tensors"
                )
            shape = tuple(handle.get_slice(key).get_shape())
            if len(shape) != 2:
                raise ValueError(f"{key!r} must be a matrix, got shape {shape}")
            dtype = str(handle.get_slice(key).get_dtype())
            dtypes.add(dtype)
            module = match.group("module")
            side = match.group("side")[-1]
            module_pair = pairs.setdefault(module, {})
            if side in module_pair:
                raise ValueError(f"duplicate LoRA {side} tensor for {module!r}")
            module_pair[side] = (key, shape)

    modules = []
    for name, pair in sorted(pairs.items()):
        missing = {"A", "B"} - pair.keys()
        if missing:
            raise ValueError(
                f"incomplete LoRA pair for {name!r}: missing lora_{missing.pop()}"
            )
        a_key, a_shape = pair["A"]
        b_key, b_shape = pair["B"]
        rank, in_features = a_shape
        out_features, b_rank = b_shape
        if rank <= 0 or in_features <= 0 or out_features <= 0:
            raise ValueError(f"invalid LoRA shapes for {name!r}: {a_shape}, {b_shape}")
        if rank != b_rank:
            raise ValueError(
                f"LoRA rank mismatch for {name!r}: A={a_shape}, B={b_shape}"
            )
        modules.append(
            LoraModule(
                name=name,
                lora_a_key=a_key,
                lora_b_key=b_key,
                rank=rank,
                in_features=in_features,
                out_features=out_features,
            )
        )

    ranks = {module.rank for module in modules}
    if len(ranks) != 1:
        raise ValueError(f"mixed LoRA ranks are unsupported: {sorted(ranks)}")
    rank = ranks.pop()
    metadata_rank = _metadata_float(metadata, "rank", float(rank))
    if not metadata_rank.is_integer() or int(metadata_rank) != rank:
        raise ValueError(
            f"metadata rank {metadata_rank:g} does not match tensor rank {rank}"
        )
    alpha = _metadata_float(metadata, "lora_alpha", float(rank))
    source_scale = _metadata_float(metadata, "lora_scale", 1.0)
    return ConversionPlan(
        modules=tuple(modules),
        source_metadata=metadata,
        source_dtypes=tuple(sorted(dtypes)),
        rank=rank,
        alpha=alpha,
        source_scale=source_scale,
    )


def output_metadata(plan: ConversionPlan, *, name: str, source: Path) -> dict[str, str]:
    metadata = {
        key: plan.source_metadata[key]
        for key in SAFE_SOURCE_METADATA
        if key in plan.source_metadata
    }
    metadata.update(
        {
            "format": "pt",
            "version": "1.0",
            "name": name,
            "ss_output_name": name,
            "ss_base_model_version": "krea2",
            "software": json.dumps(
                {
                    "name": "krea-2-oss",
                    "script": "scripts/convert_lora_to_comfyui.py",
                },
                separators=(",", ":"),
            ),
            "conversion_info": json.dumps(
                {
                    "source_file": source.name,
                    "source_format": "krea-2-training-lora",
                    "source_lora_alpha": plan.alpha,
                    "source_lora_scale": plan.source_scale,
                    "folded_scale": plan.folded_scale,
                },
                separators=(",", ":"),
            ),
        }
    )
    if "final_step" in plan.source_metadata:
        try:
            step = int(plan.source_metadata["final_step"])
        except ValueError:
            pass
        else:
            metadata["training_info"] = json.dumps(
                {"step": step}, separators=(",", ":")
            )
    return metadata


def convert(
    source: Path,
    destination: Path,
    *,
    dtype: torch.dtype,
    name: str,
    plan: ConversionPlan,
) -> None:
    tensors = {}
    with safe_open(source, framework="pt", device="cpu") as handle:
        for module in plan.modules:
            a = handle.get_tensor(module.lora_a_key).to(dtype=dtype).contiguous()
            b = handle.get_tensor(module.lora_b_key)
            if plan.folded_scale != 1.0:
                b = b.float().mul(plan.folded_scale)
            b = b.to(dtype=dtype).contiguous()
            tensors[comfyui_key(module.lora_a_key)] = a
            tensors[comfyui_key(module.lora_b_key)] = b

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        save_file(
            tensors,
            temporary,
            metadata=output_metadata(plan, name=name, source=source),
        )
        validate_output(temporary, plan, dtype=dtype)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_output(
    destination: Path, plan: ConversionPlan, *, dtype: torch.dtype
) -> None:
    expected = {
        comfyui_key(module.lora_a_key): (module.rank, module.in_features)
        for module in plan.modules
    }
    expected.update(
        {
            comfyui_key(module.lora_b_key): (module.out_features, module.rank)
            for module in plan.modules
        }
    )
    with safe_open(destination, framework="pt", device="cpu") as handle:
        found = set(handle.keys())
        if found != expected.keys():
            missing = sorted(expected.keys() - found)
            extra = sorted(found - expected.keys())
            raise RuntimeError(
                f"output validation failed: missing={missing}, extra={extra}"
            )
        for key in found:
            tensor = handle.get_slice(key)
            shape = tuple(tensor.get_shape())
            if shape != expected[key]:
                raise RuntimeError(
                    f"output validation failed: {key!r} has shape {shape}, "
                    f"expected {expected[key]}"
                )
            found_dtype = str(tensor.get_dtype())
            if found_dtype != SAFETENSORS_DTYPES[dtype]:
                raise RuntimeError(
                    f"output validation failed: {key!r} has dtype {found_dtype}"
                )


@click.command(help=__doc__)
@click.argument("source", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument(
    "destination",
    required=False,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--dtype",
    type=click.Choice(tuple(DTYPES), case_sensitive=False),
    default="bfloat16",
    show_default=True,
    help="output tensor dtype; BF16 matches typical Krea 2 ComfyUI LoRAs",
)
@click.option("--name", default=None, help="adapter name stored in metadata")
@click.option("--overwrite", is_flag=True, help="replace an existing output file")
@click.option("--dry-run", is_flag=True, help="validate and print the plan only")
def main(
    source: Path,
    destination: Path | None,
    dtype: str,
    name: str | None,
    overwrite: bool,
    dry_run: bool,
) -> None:
    source = source.expanduser().resolve()
    if destination is None:
        destination = source.with_name(f"{source.stem}_comfyui.safetensors")
    destination = destination.expanduser().resolve()
    if source == destination:
        raise click.ClickException("source and destination must be different files")
    if destination.exists() and not overwrite and not dry_run:
        raise click.ClickException(
            f"destination exists: {destination}; pass --overwrite to replace it"
        )

    try:
        plan = inspect_source(source)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    output_name = destination.stem if name is None else name
    output_dtype = DTYPES[dtype.lower()]
    click.echo(
        f"{len(plan.modules)} modules / {plan.tensor_count} tensors, "
        f"rank {plan.rank}, source dtype {', '.join(plan.source_dtypes)}"
    )
    click.echo(
        f"effective LoRA scale: ({plan.alpha:g} / {plan.rank}) * "
        f"{plan.source_scale:g} = {plan.folded_scale:g} (folded into lora_B)"
    )
    click.echo(f"{source} -> {destination} ({dtype.lower()})")
    if dry_run:
        click.echo("dry run: source is valid; no file written")
        return

    try:
        convert(
            source,
            destination,
            dtype=output_dtype,
            name=output_name,
            plan=plan,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    size_mib = destination.stat().st_size / (1024 * 1024)
    click.echo(f"wrote {destination} ({size_mib:.1f} MiB); validation passed")


if __name__ == "__main__":
    main()
