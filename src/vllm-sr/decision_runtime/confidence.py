"""Versioned, Decision-owned confidence statistics.

These statistics describe an answer distribution; they are not calibrated
probabilities that an answer is correct and do not claim TypeSafe equivalence.
Callers pass the normalized, unrounded probability distribution.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

MIN_PROBABILITY_COUNT = 1


def _validate_distribution(probabilities: Sequence[float]) -> tuple[float, ...]:
    """Reject malformed distributions before computing a statistic."""

    values = tuple(probabilities)
    if len(values) < MIN_PROBABILITY_COUNT:
        raise ValueError("confidence requires at least one probability")
    total = math.fsum(values)
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in values
    ) or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=2e-5):
        raise ValueError("confidence requires a probability distribution")
    return values


def choice_confidence(probabilities: Sequence[float]) -> float:
    """Top-two margin; a sole available choice has concentration one."""

    values = _validate_distribution(probabilities)
    if len(values) == 1:
        return 1.0
    first, second = sorted(values, reverse=True)[:2]
    return min(1.0, max(0.0, first - second))


def score_confidence(probabilities: Sequence[float]) -> float:
    """Concentration around the expected ordered Score, relative to uniform."""

    values = _validate_distribution(probabilities)
    if len(values) == 1:
        return 1.0
    mean = math.fsum(index * value for index, value in enumerate(values))
    variance = math.fsum(
        value * (index - mean) ** 2 for index, value in enumerate(values)
    )
    uniform_variance = (len(values) ** 2 - 1) / 12
    return min(1.0, max(0.0, 1.0 - variance / uniform_variance))
