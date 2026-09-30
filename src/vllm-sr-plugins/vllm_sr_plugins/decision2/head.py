"""The Decision 2.0 candidate head, always computed in FP32.

Parameter names, shapes and arithmetic follow ``CandidateHead`` in
``src/training/decision2/training/model/decision_model.py``, so a package's
``decision_head.safetensors`` loads unchanged. ``score_flat`` scores the
candidates of several requests in one pass; each candidate still sees only its
own request's global query.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

HEAD_PARAMETERS = (
    "candidate_norm.weight",
    "candidate_norm.bias",
    "query_norm.weight",
    "query_norm.bias",
    "key.weight",
    "query.weight",
    "candidate_mlp.weight",
    "candidate_mlp.bias",
    "query_mlp.weight",
    "scalar.weight",
)


class CandidateHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_dim: int = 256,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        factory = {"dtype": dtype, "device": device}
        self.head_dim = head_dim
        self.candidate_norm = nn.LayerNorm(hidden_size, **factory)
        self.query_norm = nn.LayerNorm(hidden_size, **factory)
        self.key = nn.Linear(hidden_size, head_dim, bias=False, **factory)
        self.query = nn.Linear(hidden_size, head_dim, bias=False, **factory)
        self.candidate_mlp = nn.Linear(hidden_size, head_dim, **factory)
        self.query_mlp = nn.Linear(hidden_size, head_dim, bias=False, **factory)
        self.scalar = nn.Linear(head_dim, 1, bias=False, **factory)

    def forward(self, candidates: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """``[batch, candidates, hidden]`` and ``[batch, hidden]`` -> ``[batch, candidates]``."""
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            candidate = self.candidate_norm(candidates.float())
            global_query = self.query_norm(query.float())
            bilinear = (self.key(candidate) * self.query(global_query)[:, None, :]).sum(
                -1
            ) / math.sqrt(self.head_dim)
            nonlinear = self.scalar(
                F.gelu(
                    self.candidate_mlp(candidate)
                    + self.query_mlp(global_query)[:, None, :]
                )
            ).squeeze(-1)
            return bilinear + nonlinear

    def score_flat(
        self, candidates: torch.Tensor, queries: torch.Tensor, owner: torch.Tensor
    ) -> torch.Tensor:
        """``[n, hidden]`` candidates of ``[r, hidden]`` queries; ``owner[i]`` is candidate i's query row."""
        with torch.autocast(device_type=candidates.device.type, enabled=False):
            candidate = self.candidate_norm(candidates.float())
            global_query = self.query_norm(queries.float())
            bilinear = (self.key(candidate) * self.query(global_query)[owner]).sum(
                -1
            ) / math.sqrt(self.head_dim)
            nonlinear = self.scalar(
                F.gelu(
                    self.candidate_mlp(candidate) + self.query_mlp(global_query)[owner]
                )
            ).squeeze(-1)
            return bilinear + nonlinear
