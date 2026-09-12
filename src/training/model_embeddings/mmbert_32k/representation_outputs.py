"""Execute the explicit representation contract for native ModernBERT."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional

from .representation_contract import NORMALIZATION_FINAL, read_representation_contract


def select_hidden_state(encoder, outputs, layer: int, contract: dict, *, task: str):
    """Select raw or once-normalized states without assuming HF tuple semantics."""
    if encoder.config.model_type != "modernbert":
        raise ValueError("Representation execution requires native ModernBERT")
    read_representation_contract({"representation_contract": contract}, task)
    total = len(encoder.layers)
    if not 1 <= layer <= total:
        raise ValueError(f"Layer must be in 1..{total}, got {layer}")
    if outputs.hidden_states is None or len(outputs.hidden_states) != total + 1:
        raise ValueError("Expected native ModernBERT hidden states for every layer")
    mode = contract[
        "final_normalization" if layer == total else "intermediate_normalization"
    ]
    # HF 4.57.6 stores even hidden_states[-1] BEFORE final_norm.
    raw = outputs.hidden_states[layer]
    return encoder.final_norm(raw) if mode == NORMALIZATION_FINAL else raw


def masked_mean(hidden, attention_mask):
    """Pool attended tokens in FP32, rejecting empty rows before division."""
    mask = attention_mask.unsqueeze(-1).to(torch.float32)
    counts = mask.sum(dim=1)
    if not torch.compiler.is_compiling() and torch.any(counts <= 0):
        raise ValueError("Cannot pool an all-padding input")
    return (hidden.to(torch.float32) * mask).sum(dim=1) / counts


def truncate_and_normalize(pooled, dimension: int):
    """Truncate vectors first; recompute the norm in the retained subspace."""
    if not 1 <= dimension <= pooled.shape[-1]:
        raise ValueError("Embedding dimension exceeds hidden size")
    return functional.normalize(pooled[..., :dimension].float(), dim=-1)


class PhysicalPrefixEncoder(nn.Module):
    """Export-only prefix of an owned encoder with the correct terminal norm.

    Construct with an independent encoder instance: this intentionally replaces
    its layer container and final normalization module for a static ONNX graph.
    """

    def __init__(self, encoder, layer: int, task: str):
        super().__init__()
        if encoder.config.model_type != "modernbert":
            raise ValueError("Prefix export requires native ModernBERT")
        contract = read_representation_contract(encoder.config, task)
        if contract is None:
            raise ValueError("Export requires an explicit representation_contract")
        total = len(encoder.layers)
        if not 1 <= layer <= total:
            raise ValueError(f"Layer must be in 1..{total}, got {layer}")
        mode = contract[
            "final_normalization" if layer == total else "intermediate_normalization"
        ]
        encoder.layers = nn.ModuleList(list(encoder.layers)[:layer])
        if mode != NORMALIZATION_FINAL:
            encoder.final_norm = nn.Identity()
        encoder.config.num_hidden_layers = layer
        self.encoder = encoder

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
