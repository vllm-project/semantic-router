"""Typed marker heads: per question type, transformer head layers and a scorer over candidate markers.

The Decision 1.0 encoders (Kai, Lex, Route) read each question type with its
own head: the type embedding is added to the encoder's hidden states, two
pre-norm ``nn.TransformerEncoderLayer`` blocks run over the row, and an MLP
scores the hidden state of every candidate marker. Runs in FP32.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch import nn

from ..errors import PackageError

KINDS = ("choice", "noul", "score")


class TypeHeadLayer(nn.Module):
    """``nn.TransformerEncoderLayer`` (pre-norm, ReLU, batch-first) through its reference, non-fused path.

    The released runtime disables PyTorch's fused encoder fast path; calling
    the attention function directly runs the same operations without touching
    that process-wide switch. Parameter names match the module's.
    """

    def __init__(self, hidden: int, heads: int, feedforward: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.linear1 = nn.Linear(hidden, feedforward)
        self.linear2 = nn.Linear(feedforward, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, hidden: torch.Tensor, key_padding: torch.Tensor) -> torch.Tensor:
        """``key_padding`` is the additive key mask (0 for tokens, -inf for padding), ``[rows, length]``."""
        attention = self.self_attn
        normed = self.norm1(hidden).transpose(1, 0)
        output, _ = F.multi_head_attention_forward(
            normed, normed, normed, attention.embed_dim, attention.num_heads,
            attention.in_proj_weight, attention.in_proj_bias, None, None, False, 0.0,
            attention.out_proj.weight, attention.out_proj.bias,
            training=False, key_padding_mask=key_padding, need_weights=False,
        )  # fmt: skip
        hidden = hidden + output.transpose(1, 0)
        out: torch.Tensor = hidden + self.linear2(
            F.relu(self.linear1(self.norm2(hidden)))
        )
        return out


class TypeReadout(nn.Module):
    """Per question type: a type embedding, transformer head layers and a scorer over the candidate markers."""

    def __init__(self, hidden: int, heads: int, layers: int):
        super().__init__()
        self.type_embedding = nn.Embedding(len(KINDS), hidden)
        self.heads = nn.ModuleDict(
            {
                kind: nn.ModuleList(
                    TypeHeadLayer(hidden, heads, 4 * hidden) for _ in range(layers)
                )
                for kind in KINDS
            }
        )
        self.scorers = nn.ModuleDict(
            {
                kind: nn.Sequential(
                    nn.LayerNorm(hidden),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, 1),
                )
                for kind in KINDS
            }
        )

    def forward(
        self,
        kind: str,
        hidden: torch.Tensor,
        padding: torch.Tensor,
        markers: torch.Tensor,
    ) -> torch.Tensor:
        """Marker logits ``[rows, width]`` of rows of one type; ``padding`` is True on padded tokens.

        The heads run in FP32 whatever precision the encoder ran in.
        """
        offset = self.type_embedding.weight[KINDS.index(kind)]
        hidden = hidden.float() + offset
        key_padding = torch.zeros_like(padding, dtype=hidden.dtype).masked_fill_(
            padding, float("-inf")
        )
        for layer in cast(nn.ModuleList, self.heads[kind]):
            hidden = layer(hidden, key_padding)
        gathered = torch.gather(
            hidden, 1, markers[:, :, None].expand(-1, -1, hidden.shape[-1])
        )
        scores: torch.Tensor = self.scorers[kind](gathered).squeeze(-1).float()
        return scores


def load_readout(path: Path, hidden: int, head: dict[str, Any]) -> TypeReadout:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    readout = TypeReadout(hidden, head["head_heads"], head["head_layers"])
    if set(tensors) != set(readout.state_dict()):
        raise PackageError(
            "decision_heads.safetensors does not hold the typed heads the config declares"
        )
    if any(tensor.dtype != torch.float32 for tensor in tensors.values()):
        raise PackageError("Decision 1.0 encoder weights must be FP32")
    readout.load_state_dict(tensors, strict=True)
    return readout.eval()
