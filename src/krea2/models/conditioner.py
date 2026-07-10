import itertools
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor
from transformers import (
    AutoTokenizer,
    Qwen2TokenizerFast,
    Qwen3VLForConditionalGeneration,
)

DEFAULT_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)


@dataclass
class TextEncoderConfig:
    model_id: str
    max_length: int = 512
    select_layers: tuple[int, ...] = DEFAULT_SELECT_LAYERS


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(
        self,
        version: str,
        max_length: int = 512,
        select_layers: tuple[int, ...] = DEFAULT_SELECT_LAYERS,
        *,
        dtype: torch.dtype | None = None,
        attn_implementation: str | None = None,
    ):
        super().__init__()
        load_kwargs = {}
        if dtype is not None:
            load_kwargs["dtype"] = dtype
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = attn_implementation
        self.qwen = Qwen3VLForConditionalGeneration.from_pretrained(
            version, **load_kwargs
        )
        self.tokenizer = AutoTokenizer.from_pretrained(version, max_length=max_length)
        self.processor = Qwen2TokenizerFast.from_pretrained(
            version, max_length=max_length
        )
        self.qwen = self.qwen.eval().requires_grad_(False)
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = (
            "<|im_start|>system\nDescribe the image by detailing the color, shape, "
            "size, texture, quantity, text, spatial relationships of the objects and "
            "background:<|im_end|>\n<|im_start|>user\n"
        )
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        return self._encode(text, self.qwen.device)

    @torch.no_grad()
    def _encode(
        self,
        text: list[str],
        device: torch.device,
        *,
        logits_to_keep: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        prefix_idx = self.prompt_template_encode_start_idx
        prefixed_text = [self.prompt_template_encode_prefix + item for item in text]
        suffix_inputs = self.processor(
            text=[self.prompt_template_encode_suffix] * len(text),
            return_tensors="pt",
        ).to(device, non_blocking=True)

        inputs = self.tokenizer(
            prefixed_text,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            padding="max_length",
            max_length=(
                self.max_length
                + prefix_idx
                - self.prompt_template_encode_suffix_start_idx
            ),
            return_tensors="pt",
        ).to(device, non_blocking=True)
        input_ids = torch.cat([inputs["input_ids"], suffix_inputs["input_ids"]], dim=1)
        mask = torch.cat(
            [
                inputs["attention_mask"].bool(),
                suffix_inputs["attention_mask"].bool(),
            ],
            dim=1,
        )
        model_kwargs = {
            "input_ids": input_ids,
            "attention_mask": mask,
            "output_hidden_states": True,
        }
        if logits_to_keep is not None:
            model_kwargs["logits_to_keep"] = logits_to_keep
        states = self.qwen(**model_kwargs)

        hiddens = torch.stack(
            [states.hidden_states[i] for i in self.select_layers], dim=2
        )
        return hiddens[:, prefix_idx:], mask[:, prefix_idx:]


class Qwen3VLConditionerLowMem(Qwen3VLConditioner):
    """Qwen text conditioner with quantized or host-offloaded weights.

    A quantizer callback keeps the memory-management implementation independent
    of the FP8 and INT8 linear implementations.
    """

    def __init__(
        self,
        version: str,
        max_length: int = 512,
        select_layers: tuple[int, ...] = DEFAULT_SELECT_LAYERS,
        *,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        bf16_weights: bool = False,
        offload: bool = True,
        quantize_linears: Callable[[torch.nn.Module], object] | None = None,
    ):
        super().__init__(
            version,
            max_length,
            select_layers,
            dtype=dtype,
            attn_implementation="sdpa",
        )
        self._device = torch.device(device)
        self._offload_pairs: list[tuple[Tensor, Tensor]] | None = None

        if not bf16_weights:
            if quantize_linears is None:
                raise ValueError("quantize_linears is required for quantized weights")
            quantize_linears(self.qwen)
            self.qwen.to(self._device)
        elif offload:
            self._prepare_host_offload()
        else:
            self.qwen.to(self._device)

    def _prepare_host_offload(self) -> None:
        # Text-only conditioning never executes the vision tower, so leave its
        # tensors on the CPU rather than copying them for every prompt batch.
        visual_tensors = {
            id(t)
            for t in itertools.chain(
                self.qwen.model.visual.parameters(),
                self.qwen.model.visual.buffers(),
            )
        }
        pairs = []
        for tensor in itertools.chain(self.qwen.parameters(), self.qwen.buffers()):
            if id(tensor) in visual_tensors:
                continue
            tensor.data = tensor.data.pin_memory()
            pairs.append((tensor, tensor.data))
        self._offload_pairs = pairs

    def forward(self, text: list[str]) -> tuple[Tensor, Tensor]:
        if self._offload_pairs is None:
            return self._encode(text, self._device, logits_to_keep=1)

        try:
            for tensor, cpu_data in self._offload_pairs:
                tensor.data = cpu_data.to(self._device, non_blocking=True)
            return self._encode(text, self._device, logits_to_keep=1)
        finally:
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)
            for tensor, cpu_data in self._offload_pairs:
                tensor.data = cpu_data
            if self._device.type == "cuda":
                torch.cuda.empty_cache()
