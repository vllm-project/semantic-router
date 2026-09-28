"""Frozen arm registry and the models that read cached features."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .features import sorted_levels
from .heads import DualProjection, OrdinalScoreHead, RawCosine

HIDDEN = 4096

ARMS: dict[str, dict[str, Any]] = {
    "A0J": {
        "label": "ordinary Decision head, joint native encoding",
        "representation": "joint",
        "relative": "candidate_head",
        "objective": "ce_brier",
    },
    "A0D": {
        "label": "ordinary Decision head, disaggregated encoding",
        "representation": "disaggregated",
        "relative": "candidate_head",
        "objective": "ce_brier",
    },
    "A1": {
        "label": "raw embedding cosine, no projection",
        "representation": "disaggregated",
        "relative": "raw_cosine",
        "objective": "none",
    },
    "A2": {
        "label": "dual projection + bidirectional in-batch InfoNCE",
        "representation": "disaggregated",
        "relative": "dual_projection",
        "objective": "infonce",
    },
    "A3": {
        "label": "A2 + explicit own-question hard negatives",
        "representation": "disaggregated",
        "relative": "dual_projection",
        "objective": "infonce_hard",
    },
    "A4": {
        "label": "A3 + own-Lux soft-distribution replay",
        "representation": "disaggregated",
        "relative": "dual_projection",
        "objective": "infonce_hard_replay",
    },
}


class ArmModel(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(arm)
        from training.model.decision_model import CandidateHead

        self.arm, self.spec = arm, ARMS[arm]
        relative = self.spec["relative"]
        if relative == "candidate_head":
            self.head: nn.Module = CandidateHead(HIDDEN, 256)
        elif relative == "raw_cosine":
            self.head = RawCosine()
        else:
            self.head = DualProjection(HIDDEN)
        self.score = OrdinalScoreHead(512 if relative == "dual_projection" else HIDDEN)

    @property
    def projected(self) -> bool:
        return self.spec["relative"] == "dual_projection"

    def relative_logits(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        mask = batch["candidate_mask"]
        if self.spec["relative"] == "candidate_head":
            logits = self.head(batch["candidates"], batch["query"])
        else:
            state = self.head.encode_state(batch["query"])
            action = self.head.encode_action(batch["candidates"])
            logits = self.head.scale() * torch.einsum("bd,bkd->bk", state, action)
        return logits.float().masked_fill(~mask, -float("inf"))

    def score_cumulative(
        self, batch: dict[str, torch.Tensor], score_rows: torch.Tensor
    ):
        vectors, mask, counts, gold_level, order = sorted_levels(batch, score_rows)
        state = batch["query"][score_rows]
        if self.projected:
            state = self.head.encode_state(state).detach()
            vectors = self.head.encode_action(vectors).detach() * mask[..., None]
        return self.score(state, vectors, mask), counts, gold_level, order
