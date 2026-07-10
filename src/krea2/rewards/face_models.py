"""Locate and atomically download antelopev2 face models."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import click


@dataclass(frozen=True)
class FaceModelFile:
    relative_path: Path
    url: str
    minimum_bytes: int


FACE_MODEL_FILES = (
    FaceModelFile(
        Path("recognition/model.onnx"),
        "https://huggingface.co/immich-app/antelopev2/resolve/main/recognition/model.onnx?download=true",
        200_000_000,
    ),
    FaceModelFile(
        Path("detection/model.onnx"),
        "https://huggingface.co/immich-app/antelopev2/resolve/main/detection/model.onnx?download=true",
        10_000_000,
    ),
)


def _valid_face_model(path: Path, minimum_bytes: int) -> bool:
    return path.is_file() and path.stat().st_size >= minimum_bytes


def face_models_complete(model_dir: Path) -> bool:
    return all(
        _valid_face_model(model_dir / spec.relative_path, spec.minimum_bytes)
        for spec in FACE_MODEL_FILES
    )


def locate_face_model_dir(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env = os.environ.get("ANTELOPEV2_DIR")
    if env:
        return Path(env).expanduser().resolve()
    project_root = Path(__file__).resolve().parents[3]
    candidates = (
        project_root.parent / "antelopev2",
        project_root / "antelopev2",
        Path.home() / ".cache" / "face-krea" / "antelopev2",
    )
    return next(
        (path for path in candidates if face_models_complete(path)), candidates[-1]
    )


def download_file(url: str, destination: Path, minimum_bytes: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "krea-2-oss/0.1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=120) as response:
            append = offset > 0 and getattr(response, "status", None) == 206
            if not append:
                offset = 0
            mode = "ab" if append else "wb"
            total = response.headers.get("Content-Length")
            length = None if total is None else int(total) + offset
            with (
                partial.open(mode) as handle,
                click.progressbar(
                    length=length,
                    label=f"downloading {destination.name}",
                    show_pos=True,
                ) as progress,
            ):
                if offset:
                    progress.update(offset)
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
                    progress.update(len(chunk))
    except (HTTPError, URLError, OSError) as exc:
        raise click.ClickException(f"failed to download {url}: {exc}") from exc
    if partial.stat().st_size < minimum_bytes:
        raise click.ClickException(
            f"downloaded face model is unexpectedly small: {partial}"
        )
    partial.replace(destination)


def ensure_face_models(
    model_dir: Path,
    *,
    downloader: Callable[[str, Path, int], None] = download_file,
) -> Path:
    model_dir = model_dir.expanduser().resolve()
    for spec in FACE_MODEL_FILES:
        destination = model_dir / spec.relative_path
        if _valid_face_model(destination, spec.minimum_bytes):
            continue
        click.echo(f"face model missing: {destination}")
        downloader(spec.url, destination, spec.minimum_bytes)
        if not _valid_face_model(destination, spec.minimum_bytes):
            raise click.ClickException(
                f"invalid face model after download: {destination}"
            )
    return model_dir
