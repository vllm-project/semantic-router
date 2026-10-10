"""Soft-target cross-entropy over the offered answer codes (the Vega/Perplexity objective)."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

MASK_VALUE = -1e9


class RowLosses(NamedTuple):
    cross_entropy: torch.Tensor
    brier: torch.Tensor


def mask_logits(logits: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Codes beyond each row's option count get ``-1e9`` (finite, so ``0 * logp`` stays 0)."""
    beyond = (
        torch.arange(logits.shape[-1], device=logits.device)[None]
        >= counts.to(logits.device)[:, None]
    )
    return logits.float().masked_fill(beyond, MASK_VALUE)


def row_losses(
    logits: torch.Tensor, targets: torch.Tensor, counts: torch.Tensor
) -> RowLosses:
    """Per-row ``-sum(target * log softmax)`` and Brier score, both in FP32."""
    log_probabilities = F.log_softmax(mask_logits(logits, counts), dim=-1)
    targets = targets.to(log_probabilities.device, torch.float32)
    cross_entropy = -(targets * log_probabilities).sum(-1)
    brier = (log_probabilities.exp() - targets).square().sum(-1)
    return RowLosses(cross_entropy, brier)


def step_loss(
    losses: RowLosses,
    weights: torch.Tensor,
    total_weight: float,
    world: int,
    brier_weight: float = 0.0,
) -> torch.Tensor:
    """This microbatch's share of the step loss ``sum(w * l) / sum(w)`` over all ranks.

    FSDP averages gradients over ranks, hence the ``world`` factor.
    """
    per_row = (
        losses.cross_entropy
        if brier_weight == 0
        else losses.cross_entropy + brier_weight * losses.brier
    )
    return (per_row * weights.to(per_row.device)).sum() * (world / total_weight)
