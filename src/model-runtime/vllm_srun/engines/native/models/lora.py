"""Unmerged PEFT LoRA over native Linear layers.

``result = base(x) + B(A(x)) * scaling`` in the adapter's dtype, cast back
to the base result's dtype, exactly as PEFT's LoRA Linear computes it in
evaluation mode (dropout is the identity). Merging into BF16 weights changes
answers, so the adapter always stays separate on the exact path.
"""

from __future__ import annotations

import torch
from torch import nn

PEFT_PREFIX = "base_model.model."


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, scaling: float):
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        self.scaling = scaling

    @property
    def in_features(self) -> int:
        return self.base_layer.in_features

    @property
    def out_features(self) -> int:
        return self.base_layer.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.base_layer(x)
        result_dtype = result.dtype
        adapter_dtype = self.lora_A.weight.dtype
        if x.dtype != adapter_dtype and x.is_floating_point():
            x = x.to(adapter_dtype)
        result = result + self.lora_B(self.lora_A(x)) * self.scaling
        return result.to(result_dtype)


def attach(
    backbone: nn.Module, targets: list[str], rank: int, scaling: float
) -> list[str]:
    """Wrap each target Linear in place; returns the wrapped module names."""
    modules = dict(backbone.named_modules())
    wrapped = []
    for name in targets:
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(
                f"LoRA target {name!r} is not a Linear layer of the backbone"
            )
        parent_name, _, child = name.rpartition(".")
        parent = modules[parent_name] if parent_name else backbone
        setattr(parent, child, LoRALinear(module, rank, scaling))
        wrapped.append(name)
    return wrapped


def adapter_key(name: str) -> tuple[str, str]:
    """``base_model.model.<module>.lora_A.weight`` -> (module, ``lora_A``)."""
    if not name.startswith(PEFT_PREFIX) or not name.endswith(".weight"):
        raise ValueError(f"unexpected adapter tensor {name!r}")
    stem = name[len(PEFT_PREFIX) : -len(".weight")]
    module, _, factor = stem.rpartition(".")
    if factor not in ("lora_A", "lora_B"):
        raise ValueError(f"unexpected adapter tensor {name!r}")
    return module, factor
