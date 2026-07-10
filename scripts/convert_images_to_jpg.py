#!/usr/bin/env python3
"""Recursively convert every Pillow-readable image in a folder to JPEG."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import click
from PIL import Image, ImageOps, UnidentifiedImageError


def is_image(path: Path) -> bool:
    if not path.is_file() or path.name.endswith(".tmp"):
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def discover_images(
    root: Path, *, recursive: bool, exclude_dir: Path | None = None
) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    images = []
    for path in root.glob(pattern):
        resolved = path.resolve()
        if exclude_dir is not None and resolved.is_relative_to(exclude_dir):
            continue
        if is_image(resolved):
            images.append(resolved)
    return sorted(images, key=lambda path: str(path).casefold())


def target_path(source: Path, input_dir: Path, output_dir: Path) -> Path:
    return (output_dir / source.relative_to(input_dir)).with_suffix(".jpg")


def conversion_plan(
    sources: list[Path], input_dir: Path, output_dir: Path
) -> list[tuple[Path, Path]]:
    planned = [
        (source, target_path(source, input_dir, output_dir)) for source in sources
    ]
    by_target: dict[Path, list[Path]] = defaultdict(list)
    for source, target in planned:
        by_target[target.resolve()].append(source)
    collisions = {}
    for target, values in by_target.items():
        converted_sources = [source for source in values if source != target]
        if len(converted_sources) > 1:
            collisions[target] = converted_sources
    if collisions:
        details = "; ".join(
            f"{target} <- {', '.join(map(str, values))}"
            for target, values in sorted(
                collisions.items(), key=lambda item: str(item[0])
            )
        )
        raise click.ClickException(f"multiple images map to the same JPEG: {details}")
    return planned


def _rgb_on_white(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        return Image.alpha_composite(background, rgba).convert("RGB")
    return image.convert("RGB")


def convert_image(source: Path, target: Path, *, quality: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        with Image.open(source) as image:
            icc_profile = image.info.get("icc_profile")
            rgb = _rgb_on_white(image)
            save_kwargs = {
                "format": "JPEG",
                "quality": quality,
                "subsampling": 0,
                "optimize": True,
            }
            exif = rgb.getexif()
            if exif:
                save_kwargs["exif"] = exif.tobytes()
            if icc_profile:
                save_kwargs["icc_profile"] = icc_profile
            rgb.save(temporary, **save_kwargs)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


@click.command(help=__doc__)
@click.argument(
    "input_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False, path_type=Path),
    help="destination root; by default JPEGs are written beside their sources",
)
@click.option("--recursive/--no-recursive", default=True, show_default=True)
@click.option("--quality", default=95, show_default=True, type=click.IntRange(1, 100))
@click.option(
    "--overwrite",
    is_flag=True,
    help="replace existing destination JPEGs",
)
@click.option(
    "--delete-originals",
    is_flag=True,
    help="delete non-destination source files only after every conversion succeeds",
)
@click.option("--dry-run", is_flag=True, help="show the plan without writing files")
def main(
    input_dir: Path,
    output_dir: Path | None,
    recursive: bool,
    quality: int,
    overwrite: bool,
    delete_originals: bool,
    dry_run: bool,
) -> None:
    input_dir = input_dir.expanduser().resolve()
    output_dir = input_dir if output_dir is None else output_dir.expanduser().resolve()
    excluded = (
        output_dir
        if output_dir != input_dir and output_dir.is_relative_to(input_dir)
        else None
    )
    sources = discover_images(input_dir, recursive=recursive, exclude_dir=excluded)
    if not sources:
        raise click.ClickException(f"no readable images found under {input_dir}")
    plan = conversion_plan(sources, input_dir, output_dir)

    pending = []
    already_jpeg = []
    for source, target in plan:
        same_file = source == target
        if same_file and source.suffix.lower() == ".jpg":
            already_jpeg.append(source)
            continue
        if target.exists() and not overwrite:
            raise click.ClickException(
                f"destination exists: {target}; pass --overwrite to replace it"
            )
        pending.append((source, target))

    for source, target in pending:
        click.echo(f"{source} -> {target}")
    if dry_run:
        click.echo(
            f"dry run: {len(pending)} conversions, {len(already_jpeg)} existing JPGs"
        )
        return

    converted = []
    with click.progressbar(pending, label="converting") as progress:
        for source, target in progress:
            try:
                convert_image(source, target, quality=quality)
            except (OSError, UnidentifiedImageError) as exc:
                raise click.ClickException(
                    f"failed to convert {source}: {exc}"
                ) from exc
            converted.append((source, target))

    if delete_originals:
        for source, target in converted:
            if source != target:
                source.unlink()

    click.echo(
        f"converted {len(converted)} images; {len(already_jpeg)} were already .jpg"
    )


if __name__ == "__main__":
    main()
