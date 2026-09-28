"""Training objectives for the frozen-feature ablation grid."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .heads import ordinal_probabilities


def bidirectional_infonce(
    state_z: torch.Tensor,
    pool_z: torch.Tensor,
    scale: torch.Tensor,
    gold_pool: torch.Tensor,
    own_distractor_pool: torch.Tensor,
    hard_negative_z: torch.Tensor | None = None,
    hard_negative_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """State-to-action and action-to-state InfoNCE over a batch.

    ``pool_z`` holds the distinct gold candidate texts of the batch;
    ``gold_pool[i]`` indexes state i's gold text there. Pool texts that are one
    of state i's *own* non-gold options are masked in both directions, so the
    in-batch control never sees a within-question distractor. Explicit hard
    negatives (state i's own distractors) enter only the state-to-action
    denominator, following the hard-negative InfoNCE form.
    """
    logits = scale * state_z @ pool_z.t()
    masked = logits.masked_fill(own_distractor_pool, -float("inf"))
    if hard_negative_z is not None:
        extra = scale * torch.einsum("bd,bkd->bk", state_z, hard_negative_z)
        extra = extra.masked_fill(~hard_negative_mask, -float("inf"))
        forward_logits = torch.cat([masked, extra], dim=-1)
    else:
        forward_logits = masked
    forward = F.cross_entropy(forward_logits, gold_pool)
    positives = F.one_hot(gold_pool, num_classes=pool_z.shape[0]).t().float()
    keep = positives.sum(-1) > 0
    backward_logits = masked.t()[keep]
    target = positives[keep] / positives[keep].sum(-1, keepdim=True)
    log_probs = F.log_softmax(backward_logits, dim=-1)
    backward = -(target * log_probs.masked_fill(target == 0, 0.0)).sum(-1).mean()
    return {"forward": forward, "backward": backward, "total": (forward + backward) / 2}


def ordinal_loss(
    cumulative: torch.Tensor,
    counts: torch.Tensor,
    gold_level: torch.Tensor,
    brier_weight: float = 0.5,
    teacher: torch.Tensor | None = None,
    teacher_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    probabilities = ordinal_probabilities(cumulative, counts)
    picked = probabilities.gather(1, gold_level[:, None]).squeeze(1).clamp_min(1e-12)
    nll = -picked.log()
    target = F.one_hot(gold_level, num_classes=probabilities.shape[1]).float()
    brier = (probabilities - target).square().sum(-1)
    total = nll + brier_weight * brier
    kl = torch.zeros_like(nll)
    if teacher is not None and teacher_weight:
        safe = teacher.clamp_min(1e-30)
        kl = (teacher * (safe.log() - probabilities.clamp_min(1e-12).log())).sum(-1)
        total = total + teacher_weight * kl
    return {
        "total": total.mean(),
        "nll": nll.mean(),
        "brier": brier.mean(),
        "replay_kl": kl.mean(),
    }


def replay_kl(
    logits: torch.Tensor, mask: torch.Tensor, teacher: torch.Tensor
) -> torch.Tensor:
    """Mean KL(teacher || student) over each row's own offered options."""
    log_student = F.log_softmax(logits.masked_fill(~mask, -float("inf")), dim=-1)
    safe = teacher.clamp_min(1e-30)
    per = teacher * (safe.log() - log_student.masked_fill(~mask, 0.0))
    return (per * mask).sum(-1).mean()
