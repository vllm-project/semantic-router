"""The candidate head (Decision 2.0's readout), in FP32 outside autocast.

Option endpoint rows keep each option's local context; the query row sees all
options. ``logit = <K c, Q q> / sqrt(d) + w . GELU(Mc c + Mq q)`` with
LayerNorm-ed inputs.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from ..plugins.base import EngineModel, ForwardBatch

HEAD_TENSORS = {
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
}


class CandidateHead(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int = 256):
        super().__init__()
        self.head_dim = head_dim
        self.candidate_norm = nn.LayerNorm(hidden_size)
        self.query_norm = nn.LayerNorm(hidden_size)
        self.key = nn.Linear(hidden_size, head_dim, bias=False)
        self.query = nn.Linear(hidden_size, head_dim, bias=False)
        self.candidate_mlp = nn.Linear(hidden_size, head_dim)
        self.query_mlp = nn.Linear(hidden_size, head_dim, bias=False)
        self.scalar = nn.Linear(head_dim, 1, bias=False)

    def forward(self, candidates: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
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
            scores: torch.Tensor = bilinear + nonlinear
            return scores


def load_head(path: Path, hidden_size: int, head_dim: int) -> CandidateHead:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    if set(tensors) != HEAD_TENSORS:
        raise ValueError(
            "decision_head.safetensors does not hold the shared candidate head"
        )
    head = CandidateHead(hidden_size, head_dim)
    head.load_state_dict(
        {name: tensor.float() for name, tensor in tensors.items()}, strict=True
    )
    return head.float().eval()


def logits(
    head: CandidateHead, gathered: torch.Tensor, query: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Candidate logits with padded option slots set to -inf (the scored model's output)."""
    scores: torch.Tensor = head(gathered, query)
    return scores.masked_fill(~mask, -float("inf"))


def forward_logits(
    engine_model: EngineModel,
    head: CandidateHead,
    batch: dict[str, torch.Tensor],
    lengths: list[int],
    shared_prefix: int,
) -> torch.Tensor:
    """A collated decision batch through the backbone, then the candidate logits."""
    output = engine_model.forward(
        ForwardBatch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            gather=batch["candidate_positions"],
            query=batch["query_positions"],
            lengths=lengths,
            shared_prefix=shared_prefix,
        )
    )
    mask = batch["candidate_mask"].to(output.gathered.device)
    with torch.inference_mode():
        return logits(head, output.gathered, output.query, mask)
