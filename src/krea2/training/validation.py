"""Fixed prompt selection and validation image generation."""

from pathlib import Path

import torch

from krea2.inference.sampling import sample as sample_images
from krea2.training.data import PromptDataset, read_csv_prompts
from krea2.training.objectives import high_noise_schedule_mu


def apply_trigger_word(prompts, trigger_word: str | None) -> list[str]:
    prompts = list(prompts)
    trigger = "" if trigger_word is None else trigger_word.strip()
    if not trigger:
        return prompts
    return [f"{trigger} {prompt}".strip() for prompt in prompts]


def read_sample_prompts(
    *,
    objective: str,
    csv_path: str | Path | None,
    prompts_path: str | Path | None,
    validation_csv: str | Path | None,
    validation_prompts: str | Path | None,
    trigger_word: str | None,
) -> tuple[list[str], Path]:
    """Load the prompt source shared by validation and final sampling."""
    if validation_prompts is not None:
        source = Path(validation_prompts)
        prompts = PromptDataset(source).prompts
    elif objective == "draft":
        source = Path(prompts_path)
        prompts = PromptDataset(source).prompts
    else:
        source = Path(validation_csv or csv_path)
        prompts = read_csv_prompts(source)
    if objective in {"draft", "sft"}:
        prompts = apply_trigger_word(prompts, trigger_word)
    return prompts, source


def choose_final_sample_prompt(
    *,
    objective: str,
    csv_path: str | Path | None,
    prompts_path: str | Path | None,
    validation_csv: str | Path | None,
    validation_prompts: str | Path | None,
    trigger_word: str | None,
    seed: int,
) -> tuple[str, int, Path]:
    prompts, source = read_sample_prompts(
        objective=objective,
        csv_path=csv_path,
        prompts_path=prompts_path,
        validation_csv=validation_csv,
        validation_prompts=validation_prompts,
        trigger_word=trigger_word,
    )
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    index = int(torch.randint(len(prompts), (1,), generator=gen).item())
    return prompts[index], index, source


def choose_validation_prompts(
    *,
    objective: str,
    csv_path: str | Path | None,
    prompts_path: str | Path | None,
    validation_csv: str | Path | None,
    validation_prompts: str | Path | None,
    trigger_word: str | None,
    size: int,
    seed: int,
) -> tuple[list[str], list[int], Path]:
    """Choose a deterministic random validation subset once per training run."""
    if size <= 0:
        raise ValueError("validation size must be positive")
    available, source = read_sample_prompts(
        objective=objective,
        csv_path=csv_path,
        prompts_path=prompts_path,
        validation_csv=validation_csv,
        validation_prompts=validation_prompts,
        trigger_word=trigger_word,
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    indices = []
    while len(indices) < size:
        order = torch.randperm(len(available), generator=generator).tolist()
        indices.extend(order[: size - len(indices)])
    return [available[index] for index in indices], indices, source


class CachedValidationEncoder:
    """Serve fixed validation conditioning after the real encoder is offloaded."""

    def __init__(self, prompts, text, text_mask, negative_text, negative_mask, device):
        self.prompts = list(prompts)
        self.text = text.detach().cpu().contiguous()
        self.text_mask = text_mask.detach().cpu().bool().contiguous()
        self.negative_text = negative_text.detach().cpu().contiguous()
        self.negative_mask = negative_mask.detach().cpu().bool().contiguous()
        self.device = torch.device(device)
        self.prompt_indices = {
            prompt: index for index, prompt in enumerate(self.prompts)
        }

    def __call__(self, prompts):
        prompts = list(prompts)
        if prompts and all(prompt in self.prompt_indices for prompt in prompts):
            indices = torch.tensor(
                [self.prompt_indices[prompt] for prompt in prompts], dtype=torch.long
            )
            return (
                self.text.index_select(0, indices).to(self.device, non_blocking=True),
                self.text_mask.index_select(0, indices).to(
                    self.device, non_blocking=True
                ),
            )
        if prompts and all(prompt == "" for prompt in prompts):
            count = len(prompts)
            text = self.negative_text.expand(count, *self.negative_text.shape[1:])
            mask = self.negative_mask.expand(count, *self.negative_mask.shape[1:])
            return (
                text.contiguous().to(self.device, non_blocking=True),
                mask.contiguous().to(self.device, non_blocking=True),
            )
        raise ValueError("validation encoder received prompts outside its fixed cache")


@torch.no_grad()
def build_validation_encoder(encoder, prompts: list[str], device):
    text, text_mask = encoder(prompts)
    negative_text, negative_mask = encoder([""])
    return CachedValidationEncoder(
        prompts,
        text,
        text_mask,
        negative_text,
        negative_mask,
        device,
    )


@torch.no_grad()
def save_validation_images(
    model,
    ae,
    encoder,
    prompts: list[str],
    *,
    output_dir: Path,
    step: int,
    steps: int,
    cfg: float,
    seed: int,
    y1: float,
    y2: float,
    mu: float | None,
    high_noise_shift: float,
) -> list[Path]:
    """Generate fixed-prompt/fixed-seed validation images sequentially."""
    validation_dir = output_dir / "validation" / f"step_{step:06d}"
    validation_dir.mkdir(parents=True, exist_ok=True)
    sample_mu = high_noise_schedule_mu(
        1024,
        minres=256,
        maxres=1280,
        y1=y1,
        y2=y2,
        mu=mu,
        compression=8,
        patch=2,
        high_noise_shift=high_noise_shift,
    )
    paths = []
    for index, prompt in enumerate(prompts):
        image = sample_images(
            model,
            ae,
            encoder,
            [prompt],
            negative_prompts=[""],
            device=next(model.parameters()).device,
            dtype=torch.bfloat16,
            width=512,
            height=512,
            steps=steps,
            guidance=cfg,
            seed=int(seed) + index,
            y1=y1,
            y2=y2,
            mu=sample_mu,
            progress=False,
            report_latency=False,
        )[0]
        image_path = validation_dir / f"image_{index:03d}.png"
        image.save(image_path)
        image_path.with_suffix(".txt").write_text(prompt + "\n")
        paths.append(image_path)
    return paths
