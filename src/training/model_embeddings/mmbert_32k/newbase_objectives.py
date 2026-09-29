"""Masked retrieval objectives for the prospective new-Base experiments.

Functions consume scores, not tokenizers or models. Chunking a forward therefore
cannot silently change candidate lists, relevance masks, or query weighting.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional

_MATRIX_DIMENSIONS = 2
_MINIMUM_CANDIDATES = 2


def _validate_scores(scores: torch.Tensor, valid: torch.Tensor) -> None:
    if scores.ndim != _MATRIX_DIMENSIONS or valid.shape != scores.shape:
        raise ValueError(
            "Scores and validity must be matching query/candidate matrices"
        )
    if valid.dtype != torch.bool or not torch.isfinite(scores).all():
        raise ValueError("Finite scores and a boolean validity mask are required")
    if not valid.any(dim=1).all():
        raise ValueError("A query has no valid candidate")


def multi_positive_loss(
    scores: torch.Tensor, positives: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Mean query log loss, marginalizing all known relevant candidates."""
    _validate_scores(scores, valid)
    if positives.dtype != torch.bool or positives.shape != scores.shape:
        raise ValueError("Positive mask must match the score matrix")
    if (positives & ~valid).any() or not positives.any(dim=1).all():
        raise ValueError("Every query needs at least one valid positive")
    values = scores.float()
    denominator = torch.logsumexp(values.masked_fill(~valid, -torch.inf), dim=1)
    numerator = torch.logsumexp(values.masked_fill(~positives, -torch.inf), dim=1)
    return (denominator - numerator).mean()


def order_distillation(
    student: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    """KL on candidate ordering; either an internal or external teacher is detached."""
    _validate_scores(student, valid)
    _validate_scores(teacher, valid)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Distillation temperature must be positive and finite")
    selected = valid.sum(dim=1) >= _MINIMUM_CANDIDATES
    if not selected.any():
        return student.sum() * 0.0
    mask = valid[selected]
    # A finite sentinel yields exact zero softmax mass without 0*(-inf) in KL.
    left = (student[selected].float() / temperature).masked_fill(~mask, -1e9)
    right = (teacher[selected].detach().float() / temperature).masked_fill(~mask, -1e9)
    terms = functional.kl_div(
        functional.log_softmax(left, dim=-1),
        functional.softmax(right, dim=-1),
        reduction="none",
    )
    return terms.masked_fill(~mask, 0).sum(dim=1).mean() * temperature**2


def relational_cosine_loss(
    student_left: torch.Tensor,
    student_right: torch.Tensor,
    teacher_left: torch.Tensor,
    teacher_right: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Mean squared cosine discrepancy, with equal weight per eligible anchor.

    Only relations are matched, so the two models may use different dimensions
    and representation bases. The teacher supplies geometry, not relevance gold.
    """
    if (
        student_left.ndim != _MATRIX_DIMENSIONS
        or student_right.ndim != _MATRIX_DIMENSIONS
        or teacher_left.ndim != _MATRIX_DIMENSIONS
        or teacher_right.ndim != _MATRIX_DIMENSIONS
        or len(teacher_left) != len(student_left)
        or len(teacher_right) != len(student_right)
        or student_left.shape[1] != student_right.shape[1]
        or teacher_left.shape[1] != teacher_right.shape[1]
        or student_left.shape[1] == 0
        or teacher_left.shape[1] == 0
        or valid.shape != (len(student_left), len(student_right))
        or valid.dtype != torch.bool
    ):
        raise ValueError("Cosine relation geometry and boolean mask must agree")
    vectors = (student_left, student_right, teacher_left, teacher_right)
    if any(not torch.isfinite(value).all() for value in vectors):
        raise ValueError("Cosine relation inputs must be finite")
    left, right = [functional.normalize(value.float(), dim=-1) for value in vectors[:2]]
    reference_left, reference_right = [
        functional.normalize(value.detach().float(), dim=-1) for value in vectors[2:]
    ]
    error = (left @ right.T - reference_left @ reference_right.T).square()
    counts = valid.sum(-1)
    selected = counts > 0
    if not selected.any():
        return error.sum() * 0.0
    return (error.masked_fill(~valid, 0).sum(-1)[selected] / counts[selected]).mean()


def cosent_loss(
    cosines: torch.Tensor, labels: torch.Tensor, scale: float = 20.0
) -> torch.Tensor:
    """CoSENT scale20, matching Sentence Transformers 5.1.2 score arithmetic."""
    if cosines.ndim != 1 or labels.shape != cosines.shape:
        raise ValueError("CoSENT expects paired score and label vectors")
    if not torch.isfinite(cosines).all() or not torch.isfinite(labels).all():
        raise ValueError("CoSENT inputs must be finite")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("CoSENT scale must be positive and finite")
    scores = cosines.float() * scale
    differences = scores[:, None] - scores[None, :]
    ordered = labels[:, None] < labels[None, :]
    terms = differences - (~ordered).float() * 1e12
    return torch.logsumexp(torch.cat((scores.new_zeros(1), terms.flatten())), dim=0)


def paraphrase_loss(
    cosines: torch.Tensor, labels: torch.Tensor, margin: float = 0.5
) -> torch.Tensor:
    """Keep binary paraphrase supervision separate from retrieval relevance."""
    if cosines.ndim != 1 or labels.shape != cosines.shape:
        raise ValueError("Paraphrase scores and labels must be matching vectors")
    if not torch.isfinite(cosines).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError("Finite cosines and binary paraphrase labels are required")
    if not math.isfinite(margin) or not -1 <= margin <= 1:
        raise ValueError("Paraphrase cosine margin must be finite and in [-1,1]")
    return (
        labels * (1.0 - cosines.float())
        + (1.0 - labels) * functional.relu(cosines.float() - margin)
    ).mean()


def _query_mean(terms: torch.Tensor, included: torch.Tensor) -> torch.Tensor:
    counts = included.sum(dim=(-2, -1))
    eligible = counts > 0
    if not eligible.any():
        return terms.sum() * 0.0
    totals = terms.masked_fill(~included, 0).sum(dim=(-2, -1))
    return (totals[eligible] / counts[eligible]).mean()


def ranking_terms(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    judged: torch.Tensor,
    *,
    preference_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Per-query RankNet/BCE and explicitly separate unjudged preferences.

    labels=1 marks a source-supported relevant candidate, labels=0 a judged
    negative, and labels=-1 an unjudged alternative. ``judged`` controls BCE:
    weak source-paper pairs may opt out without inventing negative judgments.
    An optional preference mask selects unjudged alternatives for weak ranking;
    excluded alternatives remain available to separately computed soft targets.
    """
    _validate_scores(scores, valid)
    if labels.shape != scores.shape or judged.shape != scores.shape:
        raise ValueError("Relevance and judgment masks must match scores")
    if judged.dtype != torch.bool or (judged & ~valid).any():
        raise ValueError("Judged entries must be valid")
    if not ((labels == -1) | (labels == 0) | (labels == 1)).all():
        raise ValueError("Relevance labels must be -1, 0 or 1")
    if (judged & (labels < 0)).any():
        raise ValueError("Unjudged alternatives cannot enter supervised BCE")
    positive = (labels == 1) & valid
    negative = (labels == 0) & valid & judged
    unknown = (labels == -1) & valid
    if preference_mask is not None:
        if (
            preference_mask.shape != scores.shape
            or preference_mask.dtype != torch.bool
            or (preference_mask & ~unknown).any()
        ):
            raise ValueError("Weak preferences must select valid unjudged candidates")
        unknown = preference_mask
    if not positive.any(dim=1).all():
        raise ValueError("Every training query requires a positive")
    values = scores.float()
    pair_loss = functional.softplus(-(values[:, :, None] - values[:, None, :]))
    known_pairs = positive[:, :, None] & negative[:, None, :]
    weak_pairs = positive[:, :, None] & unknown[:, None, :]
    bce = functional.binary_cross_entropy_with_logits(
        values, labels.clamp(min=0).float(), reduction="none"
    )
    counts = judged.sum(dim=1)
    supported = counts > 0
    bce_mean = (
        (bce.masked_fill(~judged, 0).sum(dim=1)[supported] / counts[supported]).mean()
        if supported.any()
        else values.sum() * 0.0
    )
    return {
        "pairwise": _query_mean(pair_loss, known_pairs),
        "bce": bce_mean,
        "unjudged_preference": _query_mean(pair_loss, weak_pairs),
    }


def _lambda_one(scores: torch.Tensor, relevance: torch.Tensor, k: int) -> torch.Tensor:
    """ST5.1.2 NDCGLoss2++ arithmetic for one compact, judged query list."""
    order = torch.argsort(scores, descending=True, stable=True)
    ranked_scores = scores.float()[order]
    ranked_labels = relevance.float()[order]
    ideal = torch.sort(relevance.float(), descending=True).values
    count = scores.numel()
    positions = torch.arange(1, count + 1, device=scores.device)
    discount = torch.log2(1.0 + positions.float())
    max_dcg = (((2.0**ideal - 1.0) / discount)[:k]).sum().clamp(min=1e-10)
    gain = (2.0**ranked_labels - 1.0) / max_dcg
    gap = torch.abs(positions[:, None] - positions[None, :])
    delta = (discount[gap - 1].reciprocal() - discount[gap].reciprocal()).abs()
    delta = delta.masked_fill(gap == 0, 0)
    lambda_rank = (
        discount[:, None].reciprocal() - discount[None, :].reciprocal()
    ).abs()
    weights = (10.0 * delta + lambda_rank) * (gain[:, None] - gain[None, :]).abs()
    differences = (ranked_scores[:, None] - ranked_scores[None, :]).clamp(-1e8, 1e8)
    probabilities = (torch.sigmoid(differences).clamp(min=1e-10) ** weights).clamp(
        min=1e-10
    )
    comparable = ranked_labels[:, None] > ranked_labels[None, :]
    topk = positions <= k
    comparable = comparable & topk[:, None] & topk[None, :]
    if not comparable.any():
        return scores.sum() * 0.0
    return -torch.log2(probabilities[comparable]).mean()


def lambda_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    judged: torch.Tensor,
    k: int = 10,
) -> torch.Tensor:
    """NDCGLoss2++ over judged candidates, equally weighting supported queries.

    Unlike ST's pooled-pair reduction, this plan normalizes within each query
    first. Single-query arithmetic matches the pinned official implementation.
    Unknowns are removed before sorting and cannot change discounts or support.
    """
    _validate_scores(scores, valid)
    if (
        type(k) is not int
        or k <= 0
        or judged.dtype != torch.bool
        or judged.shape != scores.shape
    ):
        raise ValueError("LambdaLoss requires positive k and a matching judgment mask")
    if labels.shape != scores.shape or (judged & ~valid).any():
        raise ValueError("LambdaLoss relevance/mask mismatch")
    values = []
    for query_scores, query_labels, mask in zip(scores, labels, judged, strict=True):
        selected_scores, selected_labels = query_scores[mask], query_labels[mask]
        if (selected_labels < 0).any() or not torch.isfinite(selected_labels).all():
            raise ValueError("Unknown/nonfinite relevance cannot enter LambdaLoss")
        if selected_labels.numel() and selected_labels.max() > selected_labels.min():
            values.append(_lambda_one(selected_scores, selected_labels, k))
    return torch.stack(values).mean() if values else scores.sum() * 0.0
