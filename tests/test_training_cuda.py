"""CUDA correctness tests for INT8 LoRA training kernels.

Run:
    uv run --extra train --extra face-reward --extra dev \
        pytest -m cuda tests/test_training_cuda.py
"""

import csv
import os
import tempfile
from pathlib import Path

import cv2
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from krea2.experiments.comparison import score_image_set
from krea2.inference.int8 import load_lora_into_int8_model
from krea2.inference.int8 import main as inference_int8_main
from krea2.quantization.int8 import (
    INT8Linear,
    LinearLoraINT8,
    add_lora_to_int8_blocks,
    int8_lora_linear,
    kernels,
    load_int8_state_dict,
    load_lora_adapters,
    quantize_int8_activation_blocks,
    quantize_int8_weight_blocks,
    quantize_int8_weight_tensorwise,
    save_lora_adapters,
    swap_linears_meta,
)
from krea2.rewards.face import FaceSimilarityReward, bgr_to_rgb_tensor
from krea2.training import cache as training_cache
from krea2.training import objectives as training_objectives
from krea2.training import trainer
from krea2.training import validation as training_validation
from krea2.training.cache import (
    build_draft_text_cache,
    build_sft_cache,
    offload_vae_encoder_to_cpu,
)
from krea2.training.data import (
    CachedDraftPromptDataset,
    CachedSFTDataset,
    ImagePromptDataset,
    PromptDataset,
    read_csv_prompts,
)
from krea2.training.objectives import (
    cached_flow_loss,
    cfg_velocity,
    save_draft_step_images,
)
from krea2.training.trainer import (
    load_reward,
    save_final_sample,
)
from krea2.training.validation import (
    apply_trigger_word,
    choose_final_sample_prompt,
)

torch.manual_seed(0)
DEV = "cuda"
pytestmark = pytest.mark.cuda


def dequant_activation(q, scale):
    m, k = q.shape
    s = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return q.float() * s[:m, :k]


def dequant_weight(q, scale):
    n, k = q.shape
    s = scale.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    return q.float() * s[:n, :k]


def rel_err(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).abs().mean() / b.abs().mean().clamp_min(1e-8)).item()


def check_fused_lora(rank, has_bias):
    m, k, n = 96, 256, 384
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    wq, ws = quantize_int8_weight_blocks(w)
    bias = torch.randn(n, device=DEV, dtype=torch.bfloat16) if has_bias else None
    lora_a = torch.randn(rank, k, device=DEV, dtype=torch.float32) * 0.01
    lora_b = torch.randn(n, rank, device=DEV, dtype=torch.float32) * 0.01
    lora_a.requires_grad_()
    lora_b.requires_grad_()

    scale = 0.75
    y = int8_lora_linear(x, wq, ws, bias, lora_a, lora_b, scale)

    xq, xs = quantize_int8_activation_blocks(x.detach())
    x_deq = dequant_activation(xq.reshape(m, k), xs)
    w_deq = dequant_weight(wq, ws)
    o = x.to(torch.bfloat16) @ lora_a.to(torch.bfloat16).t()
    ref = x_deq @ w_deq.t()
    if bias is not None:
        ref = ref + bias.float()
    ref = ref + scale * (o @ lora_b.to(torch.bfloat16).t()).float()
    fwd_rel = rel_err(y, ref.to(torch.bfloat16))
    assert fwd_rel < 0.02, f"rank={rank} bias={has_bias} fwd rel={fwd_rel:.4f}"

    grad = torch.randn_like(y)
    (y.float() * grad.float()).sum().backward()
    tmp = grad.to(torch.bfloat16) @ lora_b.to(torch.bfloat16)
    dx_ref = grad.float() @ w_deq + scale * (
        tmp.float() @ lora_a.to(torch.bfloat16).float()
    )
    da_ref = scale * (tmp.float().t() @ x.to(torch.bfloat16).float())
    db_ref = scale * (grad.to(torch.bfloat16).float().t() @ o.float())
    dx_rel = rel_err(x.grad, dx_ref)
    da_rel = rel_err(lora_a.grad, da_ref)
    db_rel = rel_err(lora_b.grad, db_ref)
    assert dx_rel < 0.01, f"rank={rank} dx rel={dx_rel:.4f}"
    assert da_rel < 0.01, f"rank={rank} dA rel={da_rel:.4f}"
    assert db_rel < 0.01, f"rank={rank} dB rel={db_rel:.4f}"
    print(
        f"ok  fused LoRA rank={rank:2d} bias={has_bias} "
        f"fwd={fwd_rel:.4f} dx={dx_rel:.4f} dA={da_rel:.4f} dB={db_rel:.4f}"
    )


def check_fused_lora_rowwise(rank, has_bias):
    """Check the exact quantized semantics of rowwise forward and base dX."""
    m, k, n = 97, 256, 384
    x = torch.randn(m, k, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16)
    wq, ws = quantize_int8_weight_tensorwise(w)
    assert ws.shape == () and ws.dtype == torch.float32
    bias = torch.randn(n, device=DEV, dtype=torch.bfloat16) if has_bias else None
    lora_a = torch.randn(rank, k, device=DEV, dtype=torch.float32) * 0.01
    lora_b = torch.randn(n, rank, device=DEV, dtype=torch.float32) * 0.01
    lora_a.requires_grad_()
    lora_b.requires_grad_()

    scale = 0.75
    y = int8_lora_linear(
        x,
        wq,
        ws,
        bias,
        lora_a,
        lora_b,
        scale,
        quantization_type="rowwise",
    )
    xq, xs = kernels.rowwise_quant(x.detach())
    x_deq = xq.float() * xs[:, None]
    w_deq = wq.float() * ws
    o = x.to(torch.bfloat16) @ lora_a.to(torch.bfloat16).t()
    ref = x_deq @ w_deq.t()
    if bias is not None:
        ref += bias.float()
    ref += scale * (o @ lora_b.to(torch.bfloat16).t()).float()
    fwd_rel = rel_err(y, ref.to(torch.bfloat16))
    assert fwd_rel < 0.01, f"rowwise rank={rank} fwd rel={fwd_rel:.4f}"

    grad = torch.randn_like(y)
    (y.float() * grad.float()).sum().backward()
    gq, gs = kernels.rowwise_quant(grad)
    tmp = grad.to(torch.bfloat16) @ lora_b.to(torch.bfloat16)
    dx_ref = (gq.float() * gs[:, None]) @ w_deq
    dx_ref += scale * (tmp.float() @ lora_a.to(torch.bfloat16).float())
    da_ref = scale * (tmp.float().t() @ x.to(torch.bfloat16).float())
    db_ref = scale * (grad.to(torch.bfloat16).float().t() @ o.float())
    errors = (
        rel_err(x.grad, dx_ref),
        rel_err(lora_a.grad, da_ref),
        rel_err(lora_b.grad, db_ref),
    )
    assert max(errors) < 0.01, f"rowwise rank={rank} backward rel={errors}"
    print(
        f"ok  rowwise LoRA rank={rank:2d} bias={has_bias} "
        f"fwd={fwd_rel:.4f} dx={errors[0]:.4f} "
        f"dA={errors[1]:.4f} dB={errors[2]:.4f}"
    )


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = INT8Linear(256, 256, bias=True, device=DEV)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([TinyBlock()])


@pytest.mark.parametrize("rank", [32, 64])
@pytest.mark.parametrize("has_bias", [False, True])
def test_fused_lora(rank, has_bias):
    check_fused_lora(rank, has_bias)


@pytest.mark.parametrize("rank", [32, 64])
@pytest.mark.parametrize("has_bias", [False, True])
def test_fused_lora_rowwise(rank, has_bias):
    check_fused_lora_rowwise(rank, has_bias)


def test_module_wrapping():
    model = TinyModel().to(DEV)
    names = add_lora_to_int8_blocks(model, rank=32, alpha=32)
    assert names == ["blocks.0.proj"]
    assert isinstance(model.blocks[0].proj, LinearLoraINT8)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"blocks.0.proj.lora_A", "blocks.0.proj.lora_B"}
    print("ok  module wrapping leaves only LoRA trainable")

    rowwise = TinyModel().to(DEV)
    rowwise.blocks[0].proj = INT8Linear(
        256, 256, bias=True, device=DEV, quantization_type="rowwise"
    )
    names = add_lora_to_int8_blocks(rowwise, rank=32, alpha=32)
    assert names == ["blocks.0.proj"]
    assert rowwise.blocks[0].proj.quantization_type == "rowwise"
    assert rowwise.blocks[0].proj.weight_scale.shape == ()
    print("ok  module wrapping preserves rowwise/tensorwise quantization")


def test_rowwise_state_loading():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "tiny.safetensors"
        src = nn.Sequential(nn.Linear(256, 384, bias=True))
        save_file(src.state_dict(), str(path))
        with torch.device("meta"):
            dst = nn.Sequential(nn.Linear(256, 384, bias=True))
        names = swap_linears_meta(dst, quantization_type="rowwise")
        assert names == ["0"]
        load_int8_state_dict(dst, str(path), device=DEV)
        assert isinstance(dst[0], INT8Linear)
        assert dst[0].weight.dtype == torch.int8
        assert dst[0].weight_scale.shape == ()
        assert dst[0].weight_scale.dtype == torch.float32
        assert dst[0].bias.dtype == torch.bfloat16
    print("ok  rowwise checkpoint loading creates scalar weight scales")


def test_compiled_checkpointed_rowwise_block():
    class TrainBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = INT8Linear(
                256,
                256,
                bias=False,
                device=DEV,
                quantization_type="rowwise",
            )
            self.proj.weight.zero_()
            self.proj.weight_scale.fill_(0.01)

        def forward(self, x, vec, freqs, mask=None):
            return self.proj(x)

    class TrainModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([TrainBlock()])

    model = TrainModel()
    add_lora_to_int8_blocks(model, rank=32, alpha=32)
    trainer.compile_training_blocks(model)
    trainer.apply_block_checkpointing(model)
    params = list(model.parameters())
    x = torch.randn(1, 17, 256, device=DEV, dtype=torch.bfloat16)
    for _ in range(2):
        for param in params:
            param.grad = None
        model.blocks[0](x, None, None, None).float().sum().backward()
        assert all(
            param.grad is not None and torch.isfinite(param.grad).all()
            for param in params
        )
    with torch.no_grad():
        output = model.blocks[0](x, None, None, None)
    assert torch.isfinite(output).all()
    print("ok  compiled checkpointed rowwise block trains and runs no-grad")


def test_lora_save_load():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lora.safetensors"
        src = TinyModel().to(DEV)
        dst = TinyModel().to(DEV)
        add_lora_to_int8_blocks(src, rank=32, alpha=32)
        add_lora_to_int8_blocks(dst, rank=32, alpha=32)
        with torch.no_grad():
            for param in src.parameters():
                if param.requires_grad:
                    param.normal_()
        save_lora_adapters(src, path, metadata={"test": "1"})
        load_lora_adapters(dst, path, strict=True)
        for (name_src, p_src), (name_dst, p_dst) in zip(
            src.named_parameters(), dst.named_parameters()
        ):
            if p_src.requires_grad:
                assert name_src == name_dst
                torch.testing.assert_close(p_src, p_dst)
    print("ok  LoRA save/load restores adapter tensors exactly")


class WrappedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, *args, **kwargs):
        return self.block(*args, **kwargs)


def test_lora_wrapped_key_compat():
    with tempfile.TemporaryDirectory() as td:
        old_path = Path(td) / "old_wrapped_lora.safetensors"
        new_path = Path(td) / "new_wrapped_lora.safetensors"
        src = TinyModel().to(DEV)
        dst_old = TinyModel().to(DEV)
        dst_new = TinyModel().to(DEV)
        add_lora_to_int8_blocks(src, rank=32, alpha=32)
        add_lora_to_int8_blocks(dst_old, rank=32, alpha=32)
        add_lora_to_int8_blocks(dst_new, rank=32, alpha=32)
        with torch.no_grad():
            src.blocks[0].proj.lora_A.normal_()
            src.blocks[0].proj.lora_B.normal_()

        save_file(
            {
                "blocks.0.block.proj.lora_A": src.blocks[0].proj.lora_A.detach().cpu(),
                "blocks.0.block.proj.lora_B": src.blocks[0].proj.lora_B.detach().cpu(),
            },
            str(old_path),
            metadata={"rank": "32", "lora_alpha": "32", "lora_scale": "1.0"},
        )
        load_lora_adapters(dst_old, old_path, strict=True)
        torch.testing.assert_close(
            dst_old.blocks[0].proj.lora_A, src.blocks[0].proj.lora_A
        )
        torch.testing.assert_close(
            dst_old.blocks[0].proj.lora_B, src.blocks[0].proj.lora_B
        )

        src.blocks[0] = WrappedBlock(src.blocks[0])
        save_lora_adapters(src, new_path)
        with safe_open(str(new_path), framework="pt", device="cpu") as f:
            assert set(f.keys()) == {
                "blocks.0.proj.lora_A",
                "blocks.0.proj.lora_B",
            }
        load_lora_adapters(dst_new, new_path, strict=True)
        torch.testing.assert_close(
            dst_new.blocks[0].proj.lora_A, src.blocks[0].block.proj.lora_A
        )
        torch.testing.assert_close(
            dst_new.blocks[0].proj.lora_B, src.blocks[0].block.proj.lora_B
        )
    print("ok  LoRA loader accepts checkpoint-wrapper keys and saves canonical keys")


def test_inference_lora_loader():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "lora.safetensors"
        src = TinyModel().to(DEV)
        dst = TinyModel().to(DEV)
        add_lora_to_int8_blocks(src, rank=32, alpha=16)
        with torch.no_grad():
            src.blocks[0].proj.lora_A.normal_()
            src.blocks[0].proj.lora_B.normal_()
        save_lora_adapters(
            src,
            path,
            metadata={
                "rank": "32",
                "lora_alpha": "16",
                "lora_scale": "0.25",
            },
        )
        config = load_lora_into_int8_model(dst, path, extra_scale=2.0)
        mod = dst.blocks[0].proj
        assert isinstance(mod, LinearLoraINT8)
        assert config["rank"] == 32
        assert config["alpha"] == 16
        assert config["effective_scale"] == 0.25
        assert mod.lora_scale == 0.25
        torch.testing.assert_close(mod.lora_A, src.blocks[0].proj.lora_A)
        torch.testing.assert_close(mod.lora_B, src.blocks[0].proj.lora_B)
    print("ok  inference loader inserts LoRA and applies adapter scale metadata")


def test_inference_cli_contract():
    from click.testing import CliRunner

    result = CliRunner().invoke(inference_int8_main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--quantization-type" in result.output
    assert "--compile-mode" not in result.output
    print("ok  INT8 inference CLI exposes only effective options")


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.ones((), device=DEV))
        self.grad_flags = []

    def forward(self, img, context, t, pos, mask):
        self.grad_flags.append(torch.is_grad_enabled())
        return img * self.p


def test_cfg_grad_branch():
    model = ToyModel()
    img = torch.ones(1, 4, 8, device=DEV)
    txt = torch.zeros(1, 2, 8, device=DEV)
    mask = torch.ones(1, 6, device=DEV, dtype=torch.bool)
    t = torch.ones(1, device=DEV)
    pos = torch.zeros(1, 6, 3, device=DEV)
    out = cfg_velocity(model, img, txt, mask, txt, mask, t, pos, pos, 3.5)
    assert model.grad_flags == [True, False]
    assert out.requires_grad
    out.float().sum().backward()
    assert model.p.grad is not None
    print("ok  CFG keeps gradients on conditional branch only")


def test_dataset_smoke():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        image_path = root / "img.png"
        Image.new("RGB", (20, 20), color=(128, 64, 32)).save(image_path)
        csv_path = root / "data.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "prompt"])
            writer.writeheader()
            writer.writerow({"image_path": "img.png", "prompt": "test prompt"})
        dataset = ImagePromptDataset(csv_path, size=512)
        image, prompt = dataset[0]
        assert image.shape == (3, 512, 512)
        assert image.min() >= -1 and image.max() <= 1
        assert prompt == "test prompt"
        assert read_csv_prompts(csv_path) == ["test prompt"]
        txt_path = root / "prompts.txt"
        txt_path.write_text("\nfirst prompt\n\nsecond prompt\n")
        prompts = PromptDataset(txt_path)
        assert len(prompts) == 2
        assert prompts[0] == "first prompt"
        assert apply_trigger_word([prompts[1]], "trg") == ["trg second prompt"]
        assert apply_trigger_word([prompts[1]], "") == ["second prompt"]

        sft_prompt, sft_index, sft_source = choose_final_sample_prompt(
            objective="sft",
            csv_path=csv_path,
            prompts_path=None,
            validation_csv=None,
            validation_prompts=None,
            trigger_word="trg",
            seed=0,
        )
        assert sft_prompt == "trg test prompt"
        assert sft_index == 0
        assert sft_source == csv_path

        val_csv_path = root / "val.csv"
        with val_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "prompt"])
            writer.writeheader()
            writer.writerow({"image_path": "img.png", "prompt": "validation prompt"})
        val_prompt, _, val_source = choose_final_sample_prompt(
            objective="sft",
            csv_path=csv_path,
            prompts_path=None,
            validation_csv=val_csv_path,
            validation_prompts=None,
            trigger_word="trg",
            seed=0,
        )
        assert val_prompt == "trg validation prompt"
        assert val_source == val_csv_path

        draft_prompt, _, draft_source = choose_final_sample_prompt(
            objective="draft",
            csv_path=None,
            prompts_path=txt_path,
            validation_csv=None,
            validation_prompts=None,
            trigger_word="trg",
            seed=0,
        )
        assert draft_prompt.startswith("trg ")
        assert draft_prompt in {"trg first prompt", "trg second prompt"}
        assert draft_source == txt_path
    print("ok  CSV image/prompt dataset smoke")


def test_validation_prompt_selection():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        csv_path = root / "validation.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "prompt"])
            writer.writeheader()
            for index in range(3):
                writer.writerow(
                    {"image_path": f"unused-{index}.png", "prompt": f"prompt {index}"}
                )
        prompts_path = root / "validation.txt"
        prompts_path.write_text("draft zero\ndraft one\ndraft two\n")

        args = dict(
            objective="sft",
            csv_path=csv_path,
            prompts_path=None,
            validation_csv=None,
            validation_prompts=None,
            trigger_word="trg",
            size=5,
            seed=123,
        )
        prompts, indices, source = trainer.choose_validation_prompts(**args)
        repeat = trainer.choose_validation_prompts(**args)
        assert (prompts, indices, source) == repeat
        assert len(prompts) == len(indices) == 5
        assert len(set(indices[:3])) == 3
        assert all(prompt.startswith("trg ") for prompt in prompts)
        assert source == csv_path

        flow_prompts, _, _ = trainer.choose_validation_prompts(
            **{**args, "objective": "flow"}
        )
        assert all(not prompt.startswith("trg ") for prompt in flow_prompts)

        draft_prompts, draft_indices, draft_source = trainer.choose_validation_prompts(
            objective="draft",
            csv_path=None,
            prompts_path=prompts_path,
            validation_csv=None,
            validation_prompts=None,
            trigger_word="trg",
            size=3,
            seed=123,
        )
        assert len(set(draft_indices)) == 3
        assert all(prompt.startswith("trg draft ") for prompt in draft_prompts)
        assert draft_source == prompts_path
    print("ok  validation prompts are selected once with deterministic random order")


class FakeEncoder:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompts):
        self.prompts.extend(prompts)
        batch = len(prompts)
        return (
            torch.ones(batch, 4, 1, 8, dtype=torch.bfloat16),
            torch.ones(batch, 4, dtype=torch.bool),
        )


class ValidationFakeEncoder:
    def __init__(self):
        self.calls = []

    def __call__(self, prompts):
        prompts = list(prompts)
        self.calls.append(prompts)
        values = torch.tensor([len(prompt) for prompt in prompts], dtype=torch.bfloat16)
        text = values[:, None, None, None].expand(-1, 2, 1, 3).contiguous()
        mask = torch.ones(len(prompts), 2, dtype=torch.bool)
        return text, mask


def test_cached_validation_encoder():
    source = ValidationFakeEncoder()
    cache = trainer.build_validation_encoder(source, ["one", "longer"], DEV)
    assert source.calls == [["one", "longer"], [""]]

    text, mask = cache(["longer"])
    assert text.device.type == "cuda"
    assert mask.device.type == "cuda"
    assert text.shape == (1, 2, 1, 3)
    assert text.unique().item() == len("longer")

    negative, negative_mask = cache(["", ""])
    assert negative.shape == (2, 2, 1, 3)
    assert negative.unique().item() == 0
    assert negative_mask.all()
    try:
        cache(["not cached"])
    except ValueError as exc:
        assert "outside its fixed cache" in str(exc)
    else:
        raise AssertionError("uncached validation prompts must be rejected")
    print("ok  fixed validation conditioning works after text-encoder offload")


def test_cached_sft_build():
    old_encode = training_cache.encode_latents

    def fake_encode(_ae, images):
        return images[:, :1, ::64, ::64].to(torch.bfloat16)

    training_cache.encode_latents = fake_encode
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for idx in range(2):
                Image.new("RGB", (32, 32), color=(idx * 30, 64, 128)).save(
                    root / f"img{idx}.png"
                )
            csv_path = root / "data.csv"
            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["image_path", "prompt"])
                writer.writeheader()
                writer.writerow({"image_path": "img0.png", "prompt": "one"})
                writer.writerow({"image_path": "img1.png", "prompt": "two"})
            encoder = FakeEncoder()
            dataset = ImagePromptDataset(csv_path, size=512)
            cache = build_sft_cache(
                dataset,
                ae=object(),
                encoder=encoder,
                trigger_word="trg",
                batch_size=1,
                num_workers=0,
            )
            assert isinstance(cache, CachedSFTDataset)
            assert len(cache) == 2
            assert cache.latents.shape == (2, 1, 8, 8)
            assert cache.text_embeddings.shape == (2, 4, 1, 8)
            assert cache.text_masks.shape == (2, 4)
            assert encoder.prompts == ["trg one", "trg two"]
    finally:
        training_cache.encode_latents = old_encode
    print("ok  cached SFT builder stores latents and text tensors")


def test_cached_draft_text_build():
    with tempfile.TemporaryDirectory() as td:
        prompt_path = Path(td) / "prompts.txt"
        prompt_path.write_text("first\nsecond\n")
        encoder = FakeEncoder()
        cache = build_draft_text_cache(
            PromptDataset(prompt_path),
            encoder=encoder,
            trigger_word="trg",
            batch_size=2,
            num_workers=0,
        )
        assert isinstance(cache, CachedDraftPromptDataset)
        assert len(cache) == 2
        assert cache.prompts == ["trg first", "trg second"]
        assert cache.text_embeddings.shape == (2, 4, 1, 8)
        assert cache.text_masks.shape == (2, 4)
        assert cache.negative_text_embeddings.shape == (2, 4, 1, 8)
        assert cache.negative_text_masks.shape == (2, 4)
        assert encoder.prompts == ["trg first", "trg second", "", ""]
        item = cache[0]
        assert item["prompt"] == "trg first"
        assert item["text_embeddings"].shape == (4, 1, 8)
    print("ok  cached DRaFT text builder stores conditional and negative tensors")


def test_vae_encoder_offload_selection():
    class FakeInner(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(1, 1)
            self.quant_conv = nn.Linear(1, 1)
            self.decoder = nn.Linear(1, 1)

    class FakeAEWrap:
        def __init__(self):
            self.ae = FakeInner()

    fake = FakeAEWrap()
    moved = offload_vae_encoder_to_cpu(fake)
    assert moved == ["encoder", "quant_conv"]
    assert isinstance(fake.ae.decoder, nn.Linear)
    print("ok  VAE offload selects encode-side modules only")


class FlowToyModel(nn.Module):
    class Config:
        patch = 2

    def __init__(self):
        super().__init__()
        self.config = self.Config()
        self.p = nn.Parameter(torch.ones((), device=DEV))

    def forward(self, img, context, t, pos, mask):
        del context, t, pos, mask
        return img * self.p


class KreaFlowConventionModel(nn.Module):
    class Config:
        patch = 2

    def __init__(self, x1_tokens):
        super().__init__()
        self.config = self.Config()
        self.p = nn.Parameter(torch.zeros((), device=DEV))
        self.register_buffer("x1_tokens", x1_tokens.to(DEV))

    def forward(self, img, context, t, pos, mask):
        del context, pos, mask
        target = (img.float() - self.x1_tokens.float()) / t.float().view(-1, 1, 1)
        return target.to(img.dtype) + self.p.to(img.dtype) * 0.0


def test_cached_flow_loss():
    model = FlowToyModel()
    latents = torch.randn(2, 4, 8, 8, dtype=torch.bfloat16)
    txt = torch.randn(2, 5, 1, 8, dtype=torch.bfloat16)
    mask = torch.ones(2, 5, dtype=torch.bool)
    loss = cached_flow_loss(model, latents, txt, mask, y1=0.5, y2=1.15, mu=None)
    assert loss.ndim == 0
    loss.backward()
    assert model.p.grad is not None

    x1_tokens, _, _ = training_objectives.prepare(
        latents.to(DEV), txt.shape[1], 2, mask.to(DEV)
    )
    exact_model = KreaFlowConventionModel(x1_tokens)
    old_shifted_random_times = training_objectives.shifted_random_times

    def fixed_times(batch, seq_len, *, device, dtype, **kwargs):
        del seq_len, kwargs
        return torch.full((batch,), 0.5, device=device, dtype=dtype)

    training_objectives.shifted_random_times = fixed_times
    try:
        exact_loss = cached_flow_loss(
            exact_model,
            latents,
            txt,
            mask,
            y1=0.5,
            y2=1.15,
            mu=None,
        )
        assert exact_loss.item() < 1e-3, exact_loss.item()
    finally:
        training_objectives.shifted_random_times = old_shifted_random_times
    print("ok  cached SFT flow loss backpropagates and matches Krea t convention")


def test_high_noise_schedule():
    base_mu = training_objectives.high_noise_schedule_mu(1024, high_noise_shift=0.0)
    high_mu = training_objectives.high_noise_schedule_mu(1024, high_noise_shift=0.5)
    assert abs(base_mu - 0.58125) < 1e-6
    assert abs(high_mu - 1.08125) < 1e-6

    torch.manual_seed(123)
    base = training_objectives.shifted_random_times(
        64,
        1024,
        device=DEV,
        dtype=torch.float32,
        high_noise_shift=0.0,
    )
    torch.manual_seed(123)
    high = training_objectives.shifted_random_times(
        64,
        1024,
        device=DEV,
        dtype=torch.float32,
        high_noise_shift=0.5,
    )
    assert torch.all(high > base)

    standard_steps = torch.tensor(
        training_objectives.timesteps(1024, 20, 256, 6400, mu=base_mu)
    )
    noisy_steps = torch.tensor(
        training_objectives.timesteps(1024, 20, 256, 6400, mu=high_mu)
    )
    torch.testing.assert_close(standard_steps[[0, -1]], noisy_steps[[0, -1]])
    assert torch.all(noisy_steps[1:-1] > standard_steps[1:-1])
    print("ok  shared high-noise schedule shifts SFT samples and DRaFT intervals")


class DummyReward:
    def __init__(self, value=1):
        self.value = value

    def __call__(self, image, prompt, **kwargs):
        return image.sum() * 0.0 + float(self.value)


def test_reward_loader_init_kwargs():
    reward = load_reward(f"{__name__}:DummyReward", {"value": 7})
    assert isinstance(reward, DummyReward)
    assert reward.value == 7
    print("ok  reward constructor kwargs are applied")


class SampleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros((), device=DEV))


def test_final_sample_save():
    calls = []

    def fake_sample(model, ae, encoder, prompts, **kwargs):
        del model, ae, encoder
        calls.append((prompts, kwargs))
        return [Image.new("RGB", (8, 8), color=(12, 34, 56))]

    old_sample = trainer.sample_images
    trainer.sample_images = fake_sample
    try:
        with tempfile.TemporaryDirectory() as td:
            out = save_final_sample(
                SampleModel(),
                ae=object(),
                encoder=object(),
                prompt="trg prompt",
                output_dir=Path(td),
                objective="sft",
                train_steps=12,
                steps=3,
                cfg=4.5,
                seed=99,
                y1=0.5,
                y2=1.15,
                mu=None,
            )
            assert out.name == "sft_final_sample_step_000012.png"
            assert out.exists()
            assert out.with_suffix(".txt").read_text() == "trg prompt\n"
            assert calls[0][0] == ["trg prompt"]
            assert calls[0][1]["width"] == 512
            assert calls[0][1]["height"] == 512
            assert calls[0][1]["seed"] == 99
    finally:
        trainer.sample_images = old_sample
    print("ok  final sample helper writes image and prompt sidecar")


def test_validation_image_save():
    calls = []

    def fake_sample(model, ae, encoder, prompts, **kwargs):
        del model, ae, encoder
        calls.append((list(prompts), kwargs))
        return [Image.new("RGB", (8, 8), color=(12, 34, 56))]

    old_sample = training_validation.sample_images
    training_validation.sample_images = fake_sample
    try:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            prompts = [
                "first fixed prompt",
                "second fixed prompt",
                "third fixed prompt",
            ]
            paths = training_validation.save_validation_images(
                SampleModel(),
                ae=object(),
                encoder=object(),
                prompts=prompts,
                output_dir=root,
                step=0,
                steps=3,
                cfg=4.5,
                seed=200_007,
                y1=0.5,
                y2=1.15,
                mu=None,
                high_noise_shift=0.5,
            )
            assert [path.name for path in paths] == [
                "image_000.png",
                "image_001.png",
                "image_002.png",
            ]
            assert all(
                path.parent == root / "validation" / "step_000000" for path in paths
            )
            assert all(path.exists() for path in paths)
            for path, prompt in zip(paths, prompts):
                assert path.with_suffix(".txt").read_text() == prompt + "\n"

            assert [call[0] for call in calls] == [[prompt] for prompt in prompts]
            assert [call[1]["seed"] for call in calls] == [200_007, 200_008, 200_009]
            expected_mu = training_objectives.high_noise_schedule_mu(
                1024,
                minres=256,
                maxres=1280,
                y1=0.5,
                y2=1.15,
                mu=None,
                compression=8,
                patch=2,
                high_noise_shift=0.5,
            )
            assert all(call[1]["mu"] == expected_mu for call in calls)
    finally:
        training_validation.sample_images = old_sample
    print("ok  step-0 validation writes fixed-prompt images and sidecars sequentially")


def test_draft_step_image_save():
    with tempfile.TemporaryDirectory() as td:
        images = torch.tensor(
            [
                [
                    [[-1.0, 1.0], [0.0, 0.5]],
                    [[1.0, -1.0], [0.0, -0.5]],
                    [[0.0, 0.0], [1.0, -1.0]],
                ],
            ],
            dtype=torch.float32,
        )
        paths = save_draft_step_images(images, ["trg prompt"], Path(td), step=7)
        assert len(paths) == 1
        assert paths[0].name == "step_000007_00.png"
        assert paths[0].exists()
        assert paths[0].with_suffix(".txt").read_text() == "trg prompt\n"
        image = Image.open(paths[0]).convert("RGB")
        assert image.size == (2, 2)
    print("ok  draft step image helper writes PNG and prompt sidecar")


@pytest.mark.face_models
def test_face_similarity_reward():
    root = Path(__file__).resolve().parents[2]
    model_dir = Path(os.environ.get("ANTELOPEV2_DIR", root / "antelopev2"))
    image_dir = Path(os.environ.get("KREA2_TEST_IMAGES", root / "test_images"))
    reward = FaceSimilarityReward(
        reference_images=image_dir,
        model_dir=model_dir,
        providers=["CPUExecutionProvider"],
        device=DEV,
    )
    assert reward.reference_embeddings.ndim == 2
    assert reward.reference_embeddings.shape[1] == 512
    assert reward.reference_embeddings.shape[0] == 8
    assert reward.skipped_reference_images == []
    norms = reward.reference_embeddings.norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        reward.reference_prototype.norm(),
        torch.ones((), device=DEV),
        atol=2e-5,
        rtol=2e-5,
    )

    x = torch.randn(1, 3, 112, 112, device=DEV, dtype=torch.float32)
    with torch.no_grad():
        torch_out = reward.recognition(x).detach().cpu().numpy()
    session = ort.InferenceSession(
        str(model_dir / "recognition" / "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    ort_out = session.run(None, {"input.1": x.detach().cpu().numpy()})[0]
    max_abs = abs(torch_out - ort_out).max()
    assert max_abs < 6e-3, f"recognition ONNX/PyTorch max abs diff {max_abs}"

    blank = torch.zeros(1, 3, 512, 512, device=DEV, requires_grad=True)
    no_face = reward(blank, "prompt")
    assert torch.isfinite(no_face).all()
    assert no_face.item() < 0.0
    (-no_face.mean()).backward()
    assert blank.grad is not None
    assert blank.grad.abs().sum().item() > 0

    bgr = cv2.imread(reward.valid_reference_images[0], cv2.IMREAD_COLOR)
    image = bgr_to_rgb_tensor(bgr).unsqueeze(0).to(DEV).requires_grad_()
    value = reward(image, "prompt")
    assert value.shape == (1,)
    (-value.mean()).backward()
    assert image.grad is not None
    assert image.grad.abs().sum().item() > 0

    metric_summary, metric_rows = score_image_set(
        "reference", [Path(reward.valid_reference_images[0])], reward
    )
    assert metric_summary["face_detection_rate"] == 1.0
    assert metric_summary["identity_similarity"]["mean"] > 0.0
    assert metric_rows[0]["detected"]

    with tempfile.TemporaryDirectory() as td:
        blank_path = Path(td) / "blank.png"
        Image.new("RGB", (512, 512), color=(0, 0, 0)).save(blank_path)
        try:
            FaceSimilarityReward(
                reference_images=td,
                model_dir=model_dir,
                providers=["CPUExecutionProvider"],
                det_thresh=0.99,
                device=DEV,
            )
        except RuntimeError as exc:
            assert "no valid reference faces" in str(exc)
        else:
            raise AssertionError("all-skipped reference images should fail")
    print(
        "ok  face reward uses all references and backpropagates through "
        "aligned and no-detection fallback crops"
    )
