#!/usr/bin/env python3
"""Caption character reference images with DeepInfra's vision API."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, UnidentifiedImageError

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".avif"}
DEFAULT_SYSTEM_PROMPT = """You write Krea 2 image-generation training prompts.
Use natural language, not analysis. Write one polished prompt that describes
the visual target: subject category, setting, composition, camera/framing,
lighting, colors, medium/style, texture, and notable objects.

Do not teach unwanted identity or dataset artifacts. For people, keep identity
generic: use only words like man, woman, person, group, figure, or child when
visibly appropriate. Do not describe age, hair color, skin, ethnicity,
nationality, exact face shape, facial likeness, celebrity identity, or names.
For a human subject, "man", "woman", or "person" is enough; focus the rest of
the prompt on pose, clothing, objects, setting, lighting, composition, and
visual style. Do not write phrases like "in this image", "in the photo", or
"the image shows". Do not describe watermarks, UI chrome, compression artifacts,
filenames, broadcast overlays, or source-document context."""
DEFAULT_USER_PROMPT = (
    "Create one Krea-style caption for this training image. Output only the "
    "caption as a single line."
)
BANNED_CLEANUPS = (
    (
        re.compile(r"\b(elderly|older|young|middle-aged)\s+(man|woman|person)\b", re.I),
        r"\2",
    ),
    (re.compile(r"\b(man|woman|person) with [^,.;]*\bhair\b", re.I), r"\1"),
    (re.compile(r",?\s*(natural|smooth|detailed)?\s*skin texture\b", re.I), ""),
    (re.compile(r",?\s*lower third graphic overlay[^,.]*", re.I), ""),
    (re.compile(r",?\s*broadcast television aesthetic(?: with [^,.]*)?", re.I), ""),
)


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_env() -> None:
    candidates = [Path.cwd() / ".env", Path(__file__).resolve().parents[3] / ".env"]
    for parent in Path.cwd().parents:
        candidates.append(parent / ".env")
    seen = set()
    for path in candidates:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        load_dotenv(path)


def api_key() -> str:
    load_env()
    key = os.environ.get("DEEPINFRA_KEY") or os.environ.get("DEEPINFRA_TOKEN")
    if not key:
        raise RuntimeError("DEEPINFRA_KEY is not set in the environment or .env")
    return key


def is_image_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.suffix.lower() in IMAGE_EXTS:
        return True
    if path.suffix:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def image_files(path: Path, recursive: bool = True) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        if not is_image_file(path):
            raise ValueError(f"unsupported or invalid image: {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"path does not exist: {path}")
    pattern = "**/*" if recursive else "*"
    return sorted(item.resolve() for item in path.glob(pattern) if is_image_file(item))


def data_url(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None or not mime.startswith("image/"):
        try:
            with Image.open(path) as image:
                mime = Image.MIME.get(image.format)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"cannot determine image type for {path}") from exc
    if not mime:
        raise ValueError(f"cannot determine image MIME type for {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def sanitize_caption(caption: str) -> str:
    caption = " ".join(caption.strip().split())
    for pattern, replacement in BANNED_CLEANUPS:
        caption = pattern.sub(replacement, caption)
    caption = re.sub(r"\s+,", ",", caption)
    caption = re.sub(r",\s*,+", ",", caption)
    caption = re.sub(r"\s{2,}", " ", caption)
    caption = re.sub(r",\s*\.", ".", caption)
    caption = re.sub(r"\s+with\.$", ".", caption, flags=re.I)
    caption = caption.strip(" ,")
    if caption and caption[-1] not in ".!?":
        caption += "."
    return caption


def request_caption(
    path: Path,
    *,
    key: str,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt: str = DEFAULT_USER_PROMPT,
    reasoning_effort: str = "none",
    max_tokens: int = 512,
    temperature: float = 0.2,
    timeout: float = 120.0,
    retries: int = 2,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url(path)}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if reasoning_effort:
        payload["reasoning_effort"] = reasoning_effort
        if reasoning_effort == "none":
            payload["reasoning"] = {"enabled": False}
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        request = urllib.request.Request(
            endpoint, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            message = result["choices"][0]["message"]
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in content
                )
            caption = sanitize_caption(str(content))
            if not caption:
                raise RuntimeError(f"empty caption response: {message}")
            return caption
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {details}")
        except Exception as exc:  # noqa: BLE001 - retry transient API failures.
            last_error = exc
        if attempt < retries:
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"caption request failed for {path}: {last_error}")


def write_labels(path: Path, rows: list[tuple[Path, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "prompt"])
        writer.writeheader()
        for image, caption in rows:
            writer.writerow(
                {
                    "image_path": str(image.resolve()),
                    "prompt": sanitize_caption(caption),
                }
            )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path)
    parser.add_argument("--output", type=Path, default=Path("labels.csv"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-recursive", action="store_true")
    args = parser.parse_args(argv)
    images = image_files(args.images, recursive=not args.no_recursive)
    if not images:
        parser.error(f"no supported images found under {args.images}")
    key = api_key()
    rows = []
    for index, image in enumerate(images, 1):
        print(f"[{index}/{len(images)}] caption {image}", flush=True)
        rows.append((image, request_caption(image, key=key, model=args.model)))
    write_labels(args.output.resolve(), rows)
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 - CLI should show a concise failure.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
