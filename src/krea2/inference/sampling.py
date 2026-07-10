"""Functional flow-matching sampler for the K2 MMDiT (no Scheduler class)."""

import contextlib
import math
import time

import click
import torch
from einops import rearrange, repeat
from PIL import Image


def roundup(value, multiple, name):
    """Round `value` up to the nearest multiple, logging when padding is applied."""
    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        print(
            f"[sample] {name}={value} is not a multiple of {multiple}; "
            f"padding to {aligned}"
        )
    return aligned


def prepare(img, txtlen, patch, txtmask):
    """Patchify the latent and build the combined text+image position / mask tensors.

    Returns (img_tokens, pos, mask).
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def timesteps(seq_len, steps, x1, x2, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    """Resolution-aware flow-matching timestep schedule (t: 1 -> 0).

    `mu` is interpolated linearly in image-sequence length between (x1,y1) and
    (x2,y2), then used to time-shift a uniform 1->0 grid. Pass an explicit `mu`
    to pin a constant shift regardless of resolution (used by the distilled
    checkpoint, which was trained at a fixed mu=1.15).
    """
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


def _sync_if_cuda(device):
    device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _mark_cudagraph_step():
    mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    if mark is not None:
        mark()


def _compiled_model_call(model, **kwargs):
    _mark_cudagraph_step()
    return model(**kwargs)


@torch.no_grad()
def sample(
    model,
    ae,
    encoder,
    prompts,
    *,
    negative_prompts=None,
    device="cuda",
    dtype=torch.bfloat16,
    width=1024,
    height=1024,
    steps=28,
    guidance=4.5,
    seed=0,
    minres=256,
    maxres=1280,
    y1=0.5,
    y2=1.15,
    mu=None,
    progress=True,
    report_latency=True,
):
    """End-to-end text-to-image sampling: encode -> euler+CFG denoise -> decode."""
    _sync_if_cuda(device)
    init_start = time.perf_counter()
    patch = model.config.patch

    # The latent grid (dim // ae.compression) is patchified in `patch`-sized blocks,
    # so width/height must be multiples of ae.compression * patch. Pad up otherwise.
    align = ae.compression * patch
    width, height = roundup(width, align, "width"), roundup(height, align, "height")

    n = len(prompts)
    cfg = guidance > 0
    if negative_prompts is None:
        negative_prompts = [""] * n

    # Per-prompt seeded gaussian latent noise.
    noise = torch.cat(
        [
            torch.randn(
                1,
                ae.channels,
                height // ae.compression,
                width // ae.compression,
                device=device,
                dtype=dtype,
                generator=torch.Generator(device=device).manual_seed(seed + i),
            )
            for i in range(n)
        ],
        dim=0,
    )

    # Positive (conditional) text conditioning.
    txt, txtmask = encoder(prompts)
    x, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)

    # The unconditional branch is only used for CFG; skip encoding/prep entirely
    # when guidance is disabled.
    if cfg:
        untxt, untxtmask = encoder(negative_prompts)
        _, unpos, unmask = prepare(noise, untxt.shape[1], patch, untxtmask)

    # min_res/max_res define the (x1,y1)-(x2,y2) interpolation endpoints for `mu`.
    x1 = (minres // (ae.compression * patch)) ** 2
    x2 = (maxres // (ae.compression * patch)) ** 2
    ts = timesteps(x.shape[1], steps, x1, x2, y1=y1, y2=y2, mu=mu)
    _sync_if_cuda(x.device)
    init_latency = time.perf_counter() - init_start

    # Euler integration of the flow ODE with CFG.
    img = x
    _sync_if_cuda(img.device)
    warmup_start = time.perf_counter()
    if steps > 0:
        t = torch.full((len(img),), ts[0], dtype=img.dtype, device=img.device)
        warm_cond = _compiled_model_call(
            model, img=img, context=txt, t=t, pos=pos, mask=mask
        )
        if cfg:
            warm_cond = warm_cond.clone()
            warm_uncond = _compiled_model_call(
                model, img=img, context=untxt, t=t, pos=unpos, mask=unmask
            )
            warm_v = warm_cond + guidance * (warm_cond - warm_uncond)
            del warm_uncond, warm_v
        del warm_cond, t
    _sync_if_cuda(img.device)
    warmup_latency = time.perf_counter() - warmup_start

    step_pairs = zip(ts[:-1], ts[1:])
    progress_ctx = (
        click.progressbar(step_pairs, length=steps, label="sampling")
        if progress
        else contextlib.nullcontext(step_pairs)
    )
    _sync_if_cuda(img.device)
    denoise_start = time.perf_counter()
    with progress_ctx as step_iter:
        for tcurr, tprev in step_iter:
            t = torch.full((len(img),), tcurr, dtype=img.dtype, device=img.device)
            cond = _compiled_model_call(
                model, img=img, context=txt, t=t, pos=pos, mask=mask
            )
            if cfg:
                cond = cond.clone()
                uncond = _compiled_model_call(
                    model, img=img, context=untxt, t=t, pos=unpos, mask=unmask
                )
                v = cond + guidance * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v
    _sync_if_cuda(img.device)
    denoise_latency = time.perf_counter() - denoise_start

    # Unpatchify back to a latent and decode to pixels.
    img = rearrange(
        img,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        ph=patch,
        pw=patch,
        h=height // (ae.compression * patch),
        w=width // (ae.compression * patch),
    )
    _sync_if_cuda(img.device)
    decode_start = time.perf_counter()
    img = ae.decode(img.to(torch.bfloat16))
    _sync_if_cuda(img.device)
    decode_latency = time.perf_counter() - decode_start
    img = img.clamp(-1, 1) * 0.5 + 0.5
    img = rearrange(img * 255.0, "b c h w -> b h w c").cpu().byte().numpy()
    if report_latency:
        click.echo(
            f"latency: initialization={init_latency:.3f}s, "
            f"warmup={warmup_latency:.3f}s, "
            f"denoising={denoise_latency:.3f}s, "
            f"vae={decode_latency:.3f}s"
        )
    return [Image.fromarray(img[i]) for i in range(len(img))]
