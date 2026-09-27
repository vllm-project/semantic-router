"""FP32 categorical proper losses over only the offered candidates."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOSS_VERSION = "decision2-valid-k-ce-plus-optional-brier-and-replay-kl-v1"
ORDINAL_LOSS_VERSION = "decision2-valid-k-ce-brier-plus-score-rps-v1"


def per_example_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    objective: str = "ce",
    brier_weight: float = 0.5,
    teacher_probs: torch.Tensor | None = None,
    replay_mask: torch.Tensor | None = None,
    replay_kl_weight: float = 0.0,
    task_type_ids: torch.Tensor | None = None,
    score_level_indices: torch.Tensor | None = None,
    ordinal_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Return per-example terms; caller controls accumulation sample weighting.

    Brier is the sum of K squared errors, following the Decision 1.0 pilot.
    Benchmark reports divide this by two; the training coefficient is explicit.
    KL uses teacher || student and contributes only on designated replay rows.
    """
    if objective not in {"ce", "ce_brier"}:
        raise ValueError("objective must be ce or ce_brier")
    if not math.isfinite(brier_weight) or brier_weight < 0:
        raise ValueError("brier_weight must be finite and nonnegative")
    if not math.isfinite(replay_kl_weight) or replay_kl_weight < 0:
        raise ValueError("replay_kl_weight must be finite and nonnegative")
    if not math.isfinite(ordinal_weight) or ordinal_weight < 0:
        raise ValueError("ordinal_weight must be finite and nonnegative")
    if ordinal_weight and objective != "ce_brier":
        raise ValueError("Ordinal Score loss needs ce_brier")
    if (
        logits.ndim != 2
        or candidate_mask.shape != logits.shape
        or candidate_mask.dtype != torch.bool
    ):
        raise ValueError(
            "logits and candidate_mask need matching rank-2 shapes and a boolean mask"
        )
    if labels.ndim != 1 or len(labels) != len(logits) or labels.dtype != torch.long:
        raise ValueError("labels must be a rank-1 long tensor with one value per row")
    if not (logits.device == labels.device == candidate_mask.device):
        raise ValueError("logits, labels, and mask must share a device")
    if not 2 <= logits.shape[1] <= 255 or torch.any(candidate_mask.sum(-1) < 2):
        raise ValueError("Each row needs 2..255 valid candidates")
    if torch.any(labels < 0) or torch.any(labels >= logits.shape[1]):
        raise ValueError("A gold label is out of bounds")
    if not candidate_mask.gather(1, labels[:, None]).all():
        raise ValueError("A gold label points to padding")
    if not torch.isfinite(logits[candidate_mask]).all():
        raise ValueError("Valid logits must be finite")
    if (teacher_probs is None) != (replay_mask is None):
        raise ValueError("teacher_probs and replay_mask must be supplied together")
    if teacher_probs is not None:
        if (
            teacher_probs.shape != logits.shape
            or replay_mask.shape != labels.shape
            or replay_mask.dtype != torch.bool
        ):
            raise ValueError("Invalid replay tensor shapes or mask dtype")
        if teacher_probs.device != logits.device or replay_mask.device != logits.device:
            raise ValueError("Replay tensors must share the logits device")
        if not torch.isfinite(teacher_probs).all() or torch.any(teacher_probs < 0):
            raise ValueError("Teacher probabilities must be finite and nonnegative")
        if torch.any(teacher_probs[~candidate_mask] != 0):
            raise ValueError("Teacher probability assigned to padding")
        if torch.any((teacher_probs[replay_mask].sum(-1) - 1).abs() > 1e-5):
            raise ValueError("Replay teacher probabilities must sum to one")
        if torch.any(teacher_probs[~replay_mask] != 0):
            raise ValueError("Non-replay rows must have zero teacher probabilities")
    elif replay_kl_weight:
        raise ValueError("Positive replay_kl_weight needs replay tensors")
    if ordinal_weight:
        if (
            task_type_ids is None
            or task_type_ids.shape != labels.shape
            or task_type_ids.dtype != torch.long
            or task_type_ids.device != logits.device
            or torch.any((task_type_ids < 0) | (task_type_ids > 2))
        ):
            raise ValueError("Ordinal loss needs valid task_type_ids")
        if (
            score_level_indices is None
            or score_level_indices.shape != logits.shape
            or score_level_indices.dtype != torch.long
            or score_level_indices.device != logits.device
        ):
            raise ValueError("Ordinal loss needs score_level_indices")
        for row in torch.nonzero(task_type_ids == 2).flatten().tolist():
            offered = score_level_indices[row, candidate_mask[row]]
            if len(offered) > 10:
                raise ValueError("Score has more than the 10 System One levels")
            expected = torch.arange(len(offered), device=logits.device)
            if not torch.equal(offered.sort().values, expected):
                raise ValueError("Score levels must enumerate 0..K-1")

    with torch.autocast(device_type=logits.device.type, enabled=False):
        masked = logits.float().masked_fill(~candidate_mask, -float("inf"))
        log_probabilities = F.log_softmax(masked, dim=-1)
        ce = F.nll_loss(log_probabilities, labels, reduction="none")
        brier = torch.zeros_like(ce)
        if objective == "ce_brier":
            probabilities = log_probabilities.exp()
            target = F.one_hot(labels, num_classes=logits.shape[1]).float()
            brier = ((probabilities - target).square() * candidate_mask).sum(-1)
        kl = torch.zeros_like(ce)
        if teacher_probs is not None and replay_kl_weight:
            safe_log_student = log_probabilities.masked_fill(~candidate_mask, 0.0)
            safe_teacher = teacher_probs.float().clamp_min(1e-30)
            per_candidate = teacher_probs.float() * (
                safe_teacher.log() - safe_log_student
            )
            kl = (per_candidate * candidate_mask).sum(-1) * replay_mask
        ordinal = torch.zeros_like(ce)
        if ordinal_weight:
            probabilities = log_probabilities.exp().masked_fill(~candidate_mask, 0.0)
            ordered = torch.zeros_like(probabilities).scatter_add(
                1, score_level_indices, probabilities
            )
            thresholds = torch.arange(logits.shape[1], device=logits.device)[None, :]
            level_count = candidate_mask.sum(-1)
            gold_level = score_level_indices.gather(1, labels[:, None])
            threshold_mask = thresholds < (level_count - 1)[:, None]
            target_cdf = (thresholds >= gold_level).float()
            ordinal = (
                ((ordered.cumsum(-1) - target_cdf).square() * threshold_mask).sum(-1)
                / (level_count - 1)
            ) * (task_type_ids == 2)
        total = (
            ce
            + (brier_weight * brier if objective == "ce_brier" else 0.0)
            + replay_kl_weight * kl
            + ordinal_weight * ordinal
        )
        return {
            "total": total,
            "ce": ce,
            "brier": brier,
            "replay_kl": kl,
            "ordinal": ordinal,
        }
