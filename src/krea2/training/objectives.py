"""SFT flow matching and differentiable DRaFT-K objectives."""

from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import functional as TF

from krea2.inference.sampling import prepare, timesteps


def shifted_random_times(
    batch: int,
    seq_len: int,
    *,
    device,
    dtype,
    minres: int = 256,
    maxres: int = 1280,
    y1: float = 0.5,
    y2: float = 1.15,
    mu: float | None = None,
    sigma: float = 1.0,
    compression: int = 8,
    patch: int = 2,
    high_noise_shift: float = 0.5,
):
    mu = high_noise_schedule_mu(
        seq_len,
        minres=minres,
        maxres=maxres,
        y1=y1,
        y2=y2,
        mu=mu,
        compression=compression,
        patch=patch,
        high_noise_shift=high_noise_shift,
    )
    u = torch.rand(batch, device=device, dtype=torch.float32).clamp_(1e-5, 1.0 - 1e-5)
    emu = torch.exp(torch.tensor(float(mu), device=device))
    t = emu / (emu + (1.0 / u - 1.0) ** sigma)
    return t.to(dtype=dtype)


def high_noise_schedule_mu(
    seq_len: int,
    *,
    minres: int = 256,
    maxres: int = 1280,
    y1: float = 0.5,
    y2: float = 1.15,
    mu: float | None = None,
    compression: int = 8,
    patch: int = 2,
    high_noise_shift: float = 0.5,
) -> float:
    """Resolve the Krea resolution shift and bias it toward noisy states.

    Krea uses t=1 for noise and t=0 for data. A positive additive log-SNR
    shift therefore moves both stochastic SFT samples and deterministic DRaFT
    integration intervals toward higher noise while preserving both endpoints.
    """
    if mu is None:
        x1 = (minres // (compression * patch)) ** 2
        x2 = (maxres // (compression * patch)) ** 2
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    return float(mu) + float(high_noise_shift)


def unpatchify_latents(tokens, *, height: int, width: int, patch: int):
    return rearrange(
        tokens,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=height // patch,
        w=width // patch,
    )


def encode_latents(ae, images: torch.Tensor) -> torch.Tensor:
    param = next(ae.parameters())
    images = images.to(device=param.device, dtype=param.dtype, non_blocking=True)
    video = rearrange(images, "b c h w -> b c 1 h w")
    enc = ae.ae.encode(video)
    if hasattr(enc, "latent_dist"):
        dist = enc.latent_dist
        z = dist.mode() if hasattr(dist, "mode") else dist.sample()
    elif hasattr(enc, "latents"):
        z = enc.latents
    else:
        z = enc[0]
    z = (z - ae.latents_mean.to(z)) / ae.latents_std.to(z)
    return rearrange(z, "b c 1 h w -> b c h w")


def decode_latents_checkpointed(ae, latents: torch.Tensor) -> torch.Tensor:
    def run(z):
        return ae.decode(z.to(torch.bfloat16))

    return checkpoint(run, latents, use_reentrant=False, preserve_rng_state=False)


def cfg_velocity(model, img, txt, txt_mask, untxt, un_mask, t, pos, unpos, guidance):
    cond = model(img=img, context=txt, t=t, pos=pos, mask=txt_mask)
    if guidance <= 0:
        return cond
    with torch.no_grad():
        uncond = model(img=img, context=untxt, t=t, pos=unpos, mask=un_mask)
    return cond + guidance * (cond - uncond.detach())


def draft_sample_images(
    model,
    ae,
    prompts: list[str],
    *,
    text_embeddings: torch.Tensor,
    text_masks: torch.Tensor,
    negative_text_embeddings: torch.Tensor,
    negative_text_masks: torch.Tensor,
    steps: int,
    draft_k: int,
    guidance: float,
    seed: int,
    y1: float,
    y2: float,
    mu: float | None,
    high_noise_shift: float = 0.5,
):
    device = next(model.parameters()).device
    dtype = torch.bfloat16
    width = height = 512
    patch = model.config.patch
    latent_h = height // ae.compression
    latent_w = width // ae.compression
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        len(prompts),
        ae.channels,
        latent_h,
        latent_w,
        device=device,
        dtype=dtype,
        generator=gen,
    )
    with torch.no_grad():
        txt = text_embeddings.to(device=device, dtype=dtype, non_blocking=True)
        txtmask = text_masks.to(device=device, dtype=torch.bool, non_blocking=True)
        x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
        if guidance > 0:
            untxt = negative_text_embeddings.to(
                device=device, dtype=dtype, non_blocking=True
            )
            untxtmask = negative_text_masks.to(
                device=device, dtype=torch.bool, non_blocking=True
            )
            _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)
        else:
            untxt = unpos = unmask = None

    x1 = (256 // (ae.compression * patch)) ** 2
    x2 = (1280 // (ae.compression * patch)) ** 2
    schedule_mu = high_noise_schedule_mu(
        x.shape[1],
        minres=256,
        maxres=1280,
        y1=y1,
        y2=y2,
        mu=mu,
        compression=ae.compression,
        patch=patch,
        high_noise_shift=high_noise_shift,
    )
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=schedule_mu)
    grad_start = max(0, steps - draft_k)
    img = x
    for i, (tcurr, tprev) in enumerate(zip(ts[:-1], ts[1:])):
        t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
        delta = tprev - tcurr
        if i < grad_start:
            with torch.no_grad():
                v = cfg_velocity(
                    model, img, txt, mask, untxt, unmask, t, pos, unpos, guidance
                )
                img = (img + delta * v).detach()
        else:
            v = cfg_velocity(
                model, img, txt, mask, untxt, unmask, t, pos, unpos, guidance
            )
            img = img + delta * v

    latents = unpatchify_latents(img, height=latent_h, width=latent_w, patch=patch)
    return decode_latents_checkpointed(ae, latents).clamp(-1, 1)


def reward_loss(reward, images, prompts: list[str], reward_kwargs: dict):
    values = []
    for image, prompt in zip(images, prompts):
        value = reward(image.unsqueeze(0), prompt, **reward_kwargs)
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                "a DRaFT-K reward must return a differentiable torch.Tensor, "
                f"got {type(value).__name__}"
            )
        value = value.to(device=image.device, dtype=torch.float32).mean()
        if not torch.isfinite(value):
            raise ValueError(
                f"reward returned a non-finite value for prompt {prompt!r}"
            )
        if not value.requires_grad:
            raise ValueError(
                "reward output is not connected to the generated image; "
                "DRaFT-K requires a differentiable reward"
            )
        values.append(value)
    rewards = torch.stack(values)
    return -rewards.mean(), rewards.detach()


def save_draft_step_images(
    images: torch.Tensor,
    prompts: list[str],
    output_dir: Path,
    step: int,
) -> list[Path]:
    sample_dir = output_dir / "draft_step_images"
    sample_dir.mkdir(parents=True, exist_ok=True)
    images_cpu = ((images.detach().float().clamp(-1, 1).cpu() + 1.0) * 0.5).clamp(0, 1)
    paths = []
    for idx, image in enumerate(images_cpu):
        stem = f"step_{step:06d}_{idx:02d}"
        image_path = sample_dir / f"{stem}.png"
        prompt_path = sample_dir / f"{stem}.txt"
        TF.to_pil_image(image).save(image_path)
        prompt = prompts[idx] if idx < len(prompts) else ""
        prompt_path.write_text(prompt + "\n")
        paths.append(image_path)
    return paths


def flow_loss(model, ae, encoder, images, prompts, *, y1, y2, mu, high_noise_shift=0.5):
    device = next(model.parameters()).device
    images = images.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    with torch.no_grad():
        x1 = encode_latents(ae, images)
        txt, txtmask = encoder(list(prompts))
    return flow_matching_loss(
        model,
        x1,
        txt,
        txtmask,
        y1=y1,
        y2=y2,
        mu=mu,
        compression=ae.compression,
        high_noise_shift=high_noise_shift,
    )


def flow_matching_loss(
    model,
    x1,
    txt,
    txtmask,
    *,
    y1,
    y2,
    mu,
    compression,
    high_noise_shift=0.5,
):
    x0 = torch.randn_like(x1)
    patch = model.config.patch
    image_tokens = (x1.shape[-2] // patch) * (x1.shape[-1] // patch)
    t = shifted_random_times(
        x1.shape[0],
        image_tokens,
        device=x1.device,
        dtype=x1.dtype,
        y1=y1,
        y2=y2,
        mu=mu,
        compression=compression,
        patch=patch,
        high_noise_shift=high_noise_shift,
    )
    t_view = t.view(-1, 1, 1, 1)
    xt = t_view * x0 + (1.0 - t_view) * x1
    target = x0 - x1
    x_tokens, pos, mask = prepare(xt, txt.shape[1], patch, txtmask)
    target_tokens = rearrange(
        target,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        ph=patch,
        pw=patch,
    )
    pred = model(img=x_tokens, context=txt, t=t, pos=pos, mask=mask)
    return F.mse_loss(pred.float(), target_tokens.float())


def cached_flow_loss(
    model,
    latents,
    text_embeddings,
    text_masks,
    *,
    y1,
    y2,
    mu,
    high_noise_shift=0.5,
):
    device = next(model.parameters()).device
    x1 = latents.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    txt = text_embeddings.to(device=device, dtype=torch.bfloat16, non_blocking=True)
    txtmask = text_masks.to(device=device, dtype=torch.bool, non_blocking=True)
    return flow_matching_loss(
        model,
        x1,
        txt,
        txtmask,
        y1=y1,
        y2=y2,
        mu=mu,
        compression=8,
        high_noise_shift=high_noise_shift,
    )
