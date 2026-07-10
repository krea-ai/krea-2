#!/usr/bin/env python3
"""Generate diverse identity-safe Krea character prompts with DeepInfra."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import click

from krea2.preprocessing.captioning import DEFAULT_ENDPOINT, api_key, sanitize_caption

DEFAULT_MODEL = "Qwen/Qwen3.6-35B-A3B"
DEFAULT_SYSTEM_PROMPT = """You generate Krea 2 image-generation training prompts.
Write polished, single-paragraph visual prompts in natural language.

The prompts are related to the same generic character concept as the examples,
but they must not teach identity or physical-person details. Keep the person
generic: use "a man", "a woman", "a person", "a group", or "a figure" only
when visibly appropriate. Do not describe age, hair, skin, ethnicity,
nationality, exact facial features, likeness, celebrity identity, names,
political identity, source context, or biography. Do not mention the source
image, dataset, screenshot, photo, caption, watermark, UI, compression, or
broadcast artifacts.

Describe what should be learned instead: clothing, pose, action, objects,
setting, composition, camera/framing, lighting, color palette, material
textures, medium, and visual style. Make prompts diverse across locations,
wardrobe, poses, camera distance, lighting, color mood, and professional or
documentary contexts. Avoid duplicates and near-duplicates."""

IDENTITY_CLEANUPS = (
    (
        re.compile(
            r"\b(older|old|elderly|young|middle-aged)\s+"
            r"(man|woman|person|figure)\b",
            re.I,
        ),
        r"\2",
    ),
    (
        re.compile(
            r"\b(gray|grey|dark|black|brown|blond|blonde|white|silver)"
            r"[ -]?haired\s+(man|woman|person|figure)\b",
            re.I,
        ),
        r"\2",
    ),
    (
        re.compile(
            r"\b(bearded|mustached|moustached|clean-shaven|bald|balding)\s+"
            r"(man|woman|person|figure)\b",
            re.I,
        ),
        r"\2",
    ),
    (
        re.compile(
            r"\b(man|woman|person|figure) with [^,.;]*"
            r"(?:hair|beard|mustache|moustache|skin|face|facial feature|glasses)"
            r"[^,.;]*",
            re.I,
        ),
        r"\1",
    ),
    (
        re.compile(
            r",?\s*(?:wearing|with)\s+"
            r"(?:thin|thick|round|rectangular|dark|clear|wire-frame|rimless)?\s*"
            r"glasses\b",
            re.I,
        ),
        "",
    ),
    (
        re.compile(
            r",?\s*(?:natural|smooth|detailed|textured)?\s*"
            r"skin(?:\s+texture|\s+tone)?\b",
            re.I,
        ),
        "",
    ),
)
BANNED_IDENTITY_TERMS = {
    "elderly",
    "older",
    "middle-aged",
    "young",
    "hair",
    "haired",
    "beard",
    "bearded",
    "mustache",
    "moustache",
    "bald",
    "skin",
    "ethnicity",
    "nationality",
    "kazakh",
    "asian",
    "caucasian",
    "celebrity",
    "politician",
    "famous",
    "portrait of the same",
}
GENERIC_PERSON_RE = re.compile(
    r"\b(?:a\s+)?(?:man|woman|person|child|group|figure)\b", re.I
)
SUBJECT_PATTERNS = {
    "a man": re.compile(r"\ba man\b|\bman\b", re.I),
    "a woman": re.compile(r"\ba woman\b|\bwoman\b", re.I),
    "a child": re.compile(r"\ba child\b|\bchild\b", re.I),
    "a group": re.compile(r"\ba group\b|\bgroup\b", re.I),
    "a person": re.compile(r"\ba person\b|\bperson\b", re.I),
    "a figure": re.compile(r"\ba figure\b|\bfigure\b", re.I),
}


def read_source_prompts(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "prompt" not in (reader.fieldnames or []):
            raise ValueError(f"{path} must contain a prompt column")
        prompts = [
            row["prompt"].strip() for row in reader if row.get("prompt", "").strip()
        ]
    if not prompts:
        raise ValueError(f"{path} does not contain any prompts")
    return prompts


def sanitize_generated_prompt(prompt: str) -> str:
    prompt = re.sub(r"^[-*\d.)\s]+", "", prompt.strip()).strip("`'\" ")
    prompt = re.sub(
        r"\b(?:in this image|in the photo|the image shows|the photo shows)\b"
        r"[:,]?\s*",
        "",
        prompt,
        flags=re.I,
    )
    prompt = sanitize_caption(prompt)
    for pattern, replacement in IDENTITY_CLEANUPS:
        prompt = pattern.sub(replacement, prompt)
    prompt = re.sub(r"\s+,", ",", prompt)
    prompt = re.sub(r",\s*,+", ",", prompt)
    prompt = re.sub(r"\s{2,}", " ", prompt)
    prompt = re.sub(r",\s*\.", ".", prompt).strip(" ,")
    for index, character in enumerate(prompt):
        if character.isalpha():
            prompt = prompt[:index] + character.upper() + prompt[index + 1 :]
            break
    if prompt and prompt[-1] not in ".!?":
        prompt += "."
    return prompt


def detect_subject(prompts: list[str]) -> str | None:
    counts = {
        subject: sum(1 for prompt in prompts if pattern.search(prompt))
        for subject, pattern in SUBJECT_PATTERNS.items()
    }
    subject, count = max(counts.items(), key=lambda item: item[1])
    return subject if count > 0 else None


def subject_regex(subject: str | None) -> re.Pattern[str] | None:
    if subject is None:
        return None
    return re.compile(rf"\b{re.escape(subject.strip())}\b", re.I)


def is_allowed_prompt(prompt: str, *, required_subject: re.Pattern[str] | None) -> bool:
    if not prompt or not GENERIC_PERSON_RE.search(prompt):
        return False
    if required_subject is not None and not required_subject.search(prompt):
        return False
    lowered = prompt.lower()
    return not any(term in lowered for term in BANNED_IDENTITY_TERMS)


def prompt_key(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", prompt.lower()).strip()


def extract_prompts(text: str) -> list[str]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.I | re.S).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    object_match = re.search(r"\{.*\}", text, flags=re.S)
    array_match = re.search(r"\[.*\]", text, flags=re.S)
    if object_match:
        candidates.append(object_match.group(0))
    if array_match:
        candidates.append(array_match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            value = value.get("prompts", [])
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
    return [
        re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
        for line in text.splitlines()
        if re.sub(r"^\s*[-*\d.)]+\s*", "", line).strip()
    ]


def post_chat(
    *, key: str, endpoint: str, payload: dict, timeout: float, retries: int
) -> str:
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = json.dumps(payload).encode("utf-8")
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
            content = str(content).strip()
            if not content:
                raise RuntimeError(f"empty prompt response: {message}")
            return content
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {details}")
        except Exception as exc:  # noqa: BLE001 - retry transient API failures.
            last_error = exc
        if attempt < retries:
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"prompt request failed: {last_error}")


def build_user_prompt(
    *,
    source_prompts: list[str],
    count: int,
    seed: int,
    subject: str | None,
    already_generated: list[str],
) -> str:
    source_lines = "\n".join(
        f"{index + 1}. {prompt}" for index, prompt in enumerate(source_prompts)
    )
    existing = ""
    if already_generated:
        existing = "\nAvoid repeating these already generated prompts:\n" + "\n".join(
            f"- {prompt}" for prompt in already_generated[-20:]
        )
    return f"""Use these source prompts as generic character and visual-style anchors:
{source_lines}

Generate exactly {count} new diverse prompts related to this character concept.
Randomization seed: {seed}

Requirements:
- Use this exact main subject phrase in every prompt: "{subject or "a person"}".
- Keep the subject generic; do not add physical identity details.
- Vary environment, action, framing, lighting, wardrobe, props, and mood.
- Keep each prompt one sentence, 35 to 80 words.
- Do not copy source wording except unavoidable generic nouns.
- Return only valid JSON in this format: {{"prompts": ["...", "..."]}}.
{existing}"""


def request_prompt_batch(
    *,
    key: str,
    endpoint: str,
    model: str,
    source_prompts: list[str],
    count: int,
    seed: int,
    subject: str | None,
    already_generated: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: float,
    retries: int,
) -> list[str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    source_prompts=source_prompts,
                    count=count,
                    seed=seed,
                    subject=subject,
                    already_generated=already_generated,
                ),
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        text = post_chat(
            key=key,
            endpoint=endpoint,
            payload=payload,
            timeout=timeout,
            retries=retries,
        )
    except RuntimeError as exc:
        if "chat_template_kwargs" not in str(exc):
            raise
        payload.pop("chat_template_kwargs", None)
        text = post_chat(
            key=key,
            endpoint=endpoint,
            payload=payload,
            timeout=timeout,
            retries=retries,
        )
    return extract_prompts(text)


def generate_prompts(
    source_prompts: list[str],
    *,
    count: int,
    seed: int,
    model: str = DEFAULT_MODEL,
    key: str | None = None,
    endpoint: str = DEFAULT_ENDPOINT,
    batch_size: int = 20,
    max_source_prompts: int = 12,
    max_tokens: int = 4096,
    temperature: float = 0.95,
    top_p: float = 0.95,
    timeout: float = 120.0,
    retries: int = 2,
) -> list[str]:
    if count <= 0 or batch_size <= 0 or max_source_prompts <= 0:
        raise ValueError("count, batch_size, and max_source_prompts must be positive")
    rng = random.Random(seed)
    sources = [sanitize_generated_prompt(prompt) for prompt in source_prompts]
    sources = [prompt for prompt in sources if prompt]
    if len(sources) > max_source_prompts:
        sources = rng.sample(sources, max_source_prompts)
    if not sources:
        raise ValueError("source prompts are empty")
    subject = detect_subject(sources)
    required_subject = subject_regex(subject)
    key = key or api_key()
    generated: list[str] = []
    seen: set[str] = set()
    rounds = max(4, (count + batch_size - 1) // batch_size + 4)
    with click.progressbar(
        length=count, label="generating prompts", show_pos=True
    ) as progress:
        for round_index in range(rounds):
            if len(generated) >= count:
                break
            before = len(generated)
            remaining = count - before
            request_count = min(
                batch_size, max(remaining, min(batch_size, remaining * 2))
            )
            candidates = request_prompt_batch(
                key=key,
                endpoint=endpoint,
                model=model,
                source_prompts=sources,
                count=request_count,
                seed=seed + round_index,
                subject=subject,
                already_generated=generated,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
                retries=retries,
            )
            for candidate in candidates:
                prompt = sanitize_generated_prompt(candidate)
                normalized = prompt_key(prompt)
                if (
                    not is_allowed_prompt(prompt, required_subject=required_subject)
                    or normalized in seen
                ):
                    continue
                generated.append(prompt)
                seen.add(normalized)
                if len(generated) == count:
                    break
            progress.update(len(generated) - before)
    if len(generated) != count:
        raise RuntimeError(
            f"only generated {len(generated)} accepted prompts out of {count}"
        )
    return generated


def write_prompts(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("\n".join(prompts) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("-n", "--num-prompts", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("generated_prompts.txt"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    prompts = generate_prompts(
        read_source_prompts(args.labels),
        count=args.num_prompts,
        seed=args.seed,
        model=args.model,
    )
    write_prompts(args.output.resolve(), prompts)
    print(f"wrote {len(prompts)} prompts to {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001 - CLI should show a concise failure.
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
