"""Statistics for the grounded-fusion eval: rank correlation + paired bootstrap.

Level-1 (intrinsic): does the grounding score rank panel responses by quality?
    -> Spearman correlation between grounding score and panel-response rubric score.
Level-2 (extrinsic): does grounding-aware synthesis beat plain fusion?
    -> paired bootstrap CI on the per-item score delta (same items in both arms).
"""

from __future__ import annotations

import random
from collections.abc import Sequence

_MIN_N_FOR_SPEARMAN = 3


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Spearman rank correlation; None if undefined (n<3 or zero variance)."""
    if len(xs) != len(ys) or len(xs) < _MIN_N_FOR_SPEARMAN:
        return None
    rx, ry = _rank(xs), _rank(ys)
    return _pearson(rx, ry)


def discard_quality(
    grounding_scores: Sequence[float],
    rubric_scores: Sequence[float],
    dropped: Sequence[bool],
) -> dict:
    """How good were the drop decisions?

    discard_precision: of dropped responses, fraction that were below-median quality.
    discard_recall:    of below-median responses, fraction that were dropped.
    """
    n = len(rubric_scores)
    if n == 0:
        return {"discard_precision": None, "discard_recall": None, "n_dropped": 0}
    median = sorted(rubric_scores)[n // 2]
    bad = [s <= median for s in rubric_scores]
    n_dropped = sum(1 for d in dropped if d)
    tp = sum(1 for d, b in zip(dropped, bad, strict=True) if d and b)
    n_bad = sum(1 for b in bad if b)
    return {
        "discard_precision": (tp / n_dropped) if n_dropped else None,
        "discard_recall": (tp / n_bad) if n_bad else None,
        "n_dropped": n_dropped,
    }


def paired_bootstrap_ci(
    deltas: Sequence[float],
    iters: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return (mean_delta, lo, hi) percentile bootstrap CI over per-item deltas."""
    deltas = [float(d) for d in deltas]
    n = len(deltas)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(deltas) / n
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(iters):
        s = sum(deltas[rng.randrange(n)] for _ in range(n))
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters) - 1]
    return (mean, lo, hi)


# ---- internals -------------------------------------------------------------


def _rank(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average rank for ties (1-based)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)
