"""Zero-initialized Score-cardinality residual over the shared candidate head.

The treatment can change an ordinal prior for a given number of levels without
changing Choice, Noul, or the pretrained shared readout at initialization.
"""

from __future__ import annotations

import torch
from torch import nn

from .decision_model import CandidateHead


class ScoreCardinalityHead(nn.Module):
    def __init__(self, hidden_size: int, head_dim: int = 256):
        super().__init__()
        self.shared = CandidateHead(hidden_size, head_dim)
        self.score_bias = nn.Parameter(torch.zeros(9, 10, dtype=torch.float32))

    def forward(
        self,
        candidates: torch.Tensor,
        query: torch.Tensor,
        task_type_ids: torch.Tensor,
        candidate_mask: torch.Tensor,
        score_level_indices: torch.Tensor,
    ) -> torch.Tensor:
        if (
            task_type_ids.shape != (candidates.shape[0],)
            or task_type_ids.dtype != torch.long
            or candidate_mask.shape != candidates.shape[:2]
            or candidate_mask.dtype != torch.bool
            or score_level_indices.shape != candidate_mask.shape
            or score_level_indices.dtype != torch.long
        ):
            raise ValueError("Score-cardinality head input contract differs")
        counts = candidate_mask.sum(-1)
        score_rows = task_type_ids == 2
        if torch.any(score_rows & ((counts < 2) | (counts > 10))):
            raise ValueError("Score needs 2..10 offered levels")
        for row in torch.nonzero(score_rows).flatten().tolist():
            levels = score_level_indices[row, candidate_mask[row]]
            expected = torch.arange(len(levels), device=levels.device)
            if not torch.equal(levels.sort().values, expected):
                raise ValueError("Score levels must enumerate 0..K-1")
        logits = self.shared(candidates, query)
        if not torch.any(score_rows):
            return logits
        cardinality = (counts[score_rows] - 2)[:, None].expand(-1, candidates.shape[1])
        levels = score_level_indices[score_rows]
        # Clamp padding indices only; the mask below makes them inert.
        bias = self.score_bias[cardinality, levels.clamp(0, 9)]
        logits = logits.clone()
        logits[score_rows] += bias * candidate_mask[score_rows]
        return logits
