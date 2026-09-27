"""Gemma 4 text-only Decision adapter admission, separate from Qwen training.

The official multimodal checkpoint is loaded and verified by the caller. Only
its ``Gemma4TextModel`` may be passed here: vision, expert tensors and the MoE
router remain frozen. This module does not run an optimizer or select a model.
"""

from __future__ import annotations

import hashlib
from typing import Any

from torch import nn

from .data import canonical
from .decision_model import encode

GEMMA_PROMPT_VERSION = "decision2-gemma4-bos-segmented-options-v1"
ATTENTION_TARGETS = ("self_attn.q_proj", "self_attn.o_proj")


def lora_plan(text: nn.Module, *, rank: int = 8, head_dim: int = 256) -> dict[str, Any]:
    """Enumerate every q/o projection in the pinned Gemma text decoder.

    The five full-attention layers have wider q/o projections and no linear
    v projection. Exact per-module dimensions therefore determine the count;
    suffix-only PEFT matching would silently include an unintended module.
    """
    config = getattr(text, "config", None)
    if (
        getattr(config, "model_type", None) != "gemma4_text"
        or getattr(config, "num_hidden_layers", None) != 30
        or getattr(config, "hidden_size", None) != 2816
        or rank < 1
        or head_dim < 1
    ):
        raise ValueError("Expected pinned Gemma 4 26B text decoder and positive ranks")
    modules = dict(text.named_modules())
    if any(name.startswith(("vision_tower", "embed_vision")) for name in modules):
        raise ValueError(
            "Pass only the Gemma text decoder, never the multimodal wrapper"
        )
    layers = getattr(text, "layers", None)
    if layers is None or len(layers) != 30:
        raise ValueError("Gemma text decoder layer count differs from configuration")
    targets: list[str] = []
    dimensions: dict[str, list[int]] = {}
    wide_layers: list[int] = []
    for index in range(30):
        q_name = f"layers.{index}.self_attn.q_proj"
        o_name = f"layers.{index}.self_attn.o_proj"
        q, o = modules.get(q_name), modules.get(o_name)
        if not isinstance(q, nn.Linear) or not isinstance(o, nn.Linear):
            raise ValueError(f"Gemma layer {index} lacks linear q/o attention paths")
        if (
            q.in_features != 2816
            or o.out_features != 2816
            or q.out_features != o.in_features
            or q.out_features not in (4096, 8192)
        ):
            raise ValueError(f"Gemma layer {index} q/o dimensions changed")
        if q.out_features == 8192:
            wide_layers.append(index)
        targets.extend((q_name, o_name))
        dimensions[q_name] = [q.in_features, q.out_features]
        dimensions[o_name] = [o.in_features, o.out_features]
    if wide_layers != [5, 11, 17, 23, 29]:
        raise ValueError("Gemma full-attention layer topology changed")
    adapter_count = rank * sum(sum(shape) for shape in dimensions.values())
    # CandidateHead: two LayerNorms, four hidden-to-head projections, one
    # candidate MLP bias, and the final head-to-scalar projection.
    head_count = 4 * 2816 + 4 * 2816 * head_dim + 2 * head_dim
    return {
        "target_modules": targets,
        "target_dimensions": dimensions,
        "wide_attention_layers": wide_layers,
        "adapter_parameters": adapter_count,
        "decision_head_parameters": head_count,
        "combined_trainable_parameters": adapter_count + head_count,
        "rank": rank,
        "head_dim": head_dim,
    }


def attach_text_lora(
    text: nn.Module, *, rank: int = 8, alpha: int = 16, dropout: float = 0.05
) -> tuple[nn.Module, dict[str, Any]]:
    """Attach PEFT to the text decoder and reject any unexpected trainables."""
    from peft import LoraConfig, get_peft_model

    if alpha < 1 or not 0 <= dropout < 1:
        raise ValueError("Invalid Gemma LoRA alpha or dropout")
    plan = lora_plan(text, rank=rank)
    text.requires_grad_(False)
    adapter = get_peft_model(
        text,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=plan["target_modules"],
            bias="none",
            task_type=None,
        ),
    )
    trainable = {
        name: parameter
        for name, parameter in adapter.named_parameters()
        if parameter.requires_grad
    }
    expected = {
        f"base_model.model.{target}.lora_{matrix}.default.weight"
        for target in plan["target_modules"]
        for matrix in ("A", "B")
    }
    if (
        set(trainable) != expected
        or sum(p.numel() for p in trainable.values()) != plan["adapter_parameters"]
    ):
        raise RuntimeError("Gemma PEFT trainables differ from exact q/o LoRA plan")
    plan.update({"alpha": alpha, "dropout": dropout})
    return adapter, plan


def assert_zero_lora_b(adapter: nn.Module, target_count: int) -> None:
    """Fresh PEFT B matrices must be all zero before source identity checks."""
    from torch import count_nonzero

    named = dict(adapter.named_parameters())
    b = {
        name: parameter
        for name, parameter in named.items()
        if name.endswith(".lora_B.default.weight")
    }
    if len(b) != target_count or any(
        parameter.is_meta or count_nonzero(parameter).item() != 0
        for parameter in b.values()
    ):
        raise RuntimeError("Fresh Gemma LoRA B matrices are not zero-initialized")


def encode_gemma(
    row: dict[str, Any], tokenizer: Any, max_length: int
) -> dict[str, Any]:
    """Preserve the shared segmented prompt and explicitly prepend Gemma BOS."""
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is None or max_length < 2:
        raise ValueError("Gemma tokenizer needs BOS and a positive context budget")
    encoded = encode(row, tokenizer, max_length - 1)
    ids = [bos, *encoded["ids"]]
    if len(ids) > max_length:
        raise ValueError("Gemma input would be truncated")
    return {
        **encoded,
        "ids": ids,
        "candidate_positions": [p + 1 for p in encoded["candidate_positions"]],
        "query_position": encoded["query_position"] + 1,
        "token_ids_sha256": hashlib.sha256(canonical(ids).encode()).hexdigest(),
        "prompt_version": GEMMA_PROMPT_VERSION,
    }
