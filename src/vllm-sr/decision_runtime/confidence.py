"""Decision-owned confidence statistics.

This module does not implement or claim equivalence to TypeSafe/Jev confidence.
Their public documentation does not disclose the formula. Decision uses a
normalized top-probability concentration statistic so a uniform distribution is
zero and a one-hot distribution is one, independent of option count.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

MIN_PROBABILITY_COUNT = 2


def normalized_top_confidence(probabilities: Sequence[float]) -> float:
    """Return Decision normalized-top concentration in the closed interval [0, 1]."""

    count = len(probabilities)
    if count < MIN_PROBABILITY_COUNT:
        raise ValueError("confidence requires at least two probabilities")
    total = math.fsum(probabilities)
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0 for value in probabilities
    ) or not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=2e-5):
        raise ValueError("confidence requires a probability distribution")
    uniform = 1.0 / count
    normalized = (max(probabilities) - uniform) / (1.0 - uniform)
    return min(1.0, max(0.0, normalized))
