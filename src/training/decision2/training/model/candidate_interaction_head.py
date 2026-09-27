"""Experimental set-equivariant residual over native decision candidates.

The causal backbone still encodes options in sequence order. This head is
equivariant given fixed candidate vectors; it does not make the whole model
permutation invariant. The baseline readout is inherited unchanged, and the
zero-initialized residual preserves exact zero-step logits.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .decision_model import CandidateHead


class CandidateInteractionHead(CandidateHead):
    """Score Choice and Score with a shared leave-self-out set interaction.

    Noul uses the inherited CandidateHead path exactly. No candidate IDs or
    positions are embedded, so candidate permutation permutes the logits when
    the candidate and query hidden states themselves are held fixed.
    """

    def __init__(
        self, hidden_size: int, head_dim: int = 256, interaction_dim: int = 64
    ) -> None:
        super().__init__(hidden_size, head_dim)
        if interaction_dim < 1:
            raise ValueError("interaction_dim must be positive")
        self.interaction_dim = interaction_dim
        self.interaction_candidate = nn.Linear(hidden_size, interaction_dim, bias=False)
        self.interaction_query = nn.Linear(hidden_size, interaction_dim, bias=False)
        self.interaction_key = nn.Linear(interaction_dim, interaction_dim, bias=False)
        self.interaction_value = nn.Linear(interaction_dim, interaction_dim, bias=False)
        self.interaction_fusion = nn.Linear(3 * interaction_dim, interaction_dim)
        self.interaction_out = nn.Linear(interaction_dim, 1, bias=False)
        nn.init.zeros_(self.interaction_out.weight)

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
        if ((task_type_ids < 0) | (task_type_ids > 2)).any():
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

        baseline = super().forward(candidates, query)
        active = torch.nonzero(task_type_ids != 1, as_tuple=True)[0]
        if active.numel() == 0:
            return baseline.masked_fill(~candidate_mask, -float("inf"))

        with torch.autocast(device_type=candidates.device.type, enabled=False):
            projected = self.interaction_candidate(
                self.candidate_norm(candidates[active].float())
            )
            global_query = self.interaction_query(
                self.query_norm(query[active].float())
            )
            keys = self.interaction_key(projected)
            values = self.interaction_value(projected)
            affinity = torch.bmm(
                projected + global_query[:, None, :], keys.transpose(1, 2)
            ) / math.sqrt(self.interaction_dim)
            width = candidates.shape[1]
            other = ~torch.eye(width, dtype=torch.bool, device=candidates.device)
            affinity = affinity.masked_fill(
                ~(candidate_mask[active, None, :] & other[None, :, :]),
                -float("inf"),
            )
            neighbors = torch.bmm(affinity.softmax(-1), values)
            fused = F.gelu(
                self.interaction_fusion(
                    torch.cat(
                        (
                            projected,
                            neighbors,
                            global_query[:, None, :].expand_as(projected),
                        ),
                        dim=-1,
                    )
                )
            )
            correction = self.interaction_out(fused).squeeze(-1)
            residual = torch.zeros_like(baseline)
            residual[active] = correction
            return (baseline + residual).masked_fill(~candidate_mask, -float("inf"))
