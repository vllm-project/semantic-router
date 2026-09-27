"""Experimental typed readout for a matched 0.6B architecture ablation.

This module is deliberately separate from the released DecisionModel path.
An ablation runner must explicitly opt in, version its checkpoint format, and
prove native reload parity before using the head for model selection.
"""

from __future__ import annotations

import torch
from torch import nn

from .decision_model import TASK_TYPES, CandidateHead

TYPE_ORDER = TASK_TYPES


class TypeSeparatedCandidateHead(nn.Module):
    """Use one dynamic-candidate readout per System One question type.

    Candidate cardinality is unchanged: Choice can have 2..255 options, Noul
    uses two, and Score can have 2..10 ordered levels. This module only emits
    logits. It does not decode text or assume a fixed set of candidate IDs.
    """

    def __init__(self, hidden_size: int, head_dim: int = 256):
        super().__init__()
        self.heads = nn.ModuleList(
            CandidateHead(hidden_size, head_dim) for _ in TYPE_ORDER
        )

    def forward(
        self,
        candidates: torch.Tensor,
        query: torch.Tensor,
        task_type_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        if candidates.ndim != 3 or query.shape != (
            candidates.shape[0],
            candidates.shape[2],
        ):
            raise ValueError(
                "Candidates and query need [batch, options, hidden] shapes"
            )
        if (
            task_type_ids.shape != (candidates.shape[0],)
            or task_type_ids.dtype != torch.long
        ):
            raise ValueError("task_type_ids must be one integer per question")
        if (
            candidate_mask.shape != candidates.shape[:2]
            or candidate_mask.dtype != torch.bool
        ):
            raise ValueError("candidate_mask must identify the offered options")
        counts = candidate_mask.sum(-1)
        if not 2 <= candidates.shape[1] <= 255 or (counts < 2).any():
            raise ValueError("Every question needs 2..255 offered options")
        if ((task_type_ids < 0) | (task_type_ids >= len(TYPE_ORDER))).any():
            raise ValueError("Unsupported System One question type")
        if ((task_type_ids == 1) & (counts != 2)).any() or (
            (task_type_ids == 2) & (counts > 10)
        ).any():
            raise ValueError("Noul needs two options; Score needs 2..10 levels")
        if not (
            candidates.device
            == query.device
            == task_type_ids.device
            == candidate_mask.device
        ):
            raise ValueError("Inputs must share a device")

        logits = torch.empty(
            candidates.shape[:2], device=candidates.device, dtype=torch.float32
        )
        for type_id, head in enumerate(self.heads):
            rows = torch.nonzero(task_type_ids == type_id, as_tuple=True)[0]
            if rows.numel():
                logits[rows] = head(candidates[rows], query[rows])
        return logits.masked_fill(~candidate_mask, -float("inf"))
