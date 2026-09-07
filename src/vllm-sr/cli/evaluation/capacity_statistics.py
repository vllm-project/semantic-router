"""Deterministic statistics used by the live capacity reducer."""

from __future__ import annotations

from math import sqrt

_ONE_SIDED_95_Z = 1.6448536269514722
_MIN_STABILITY_REPETITIONS = 2


def arithmetic_mean(values: tuple[float, ...]) -> float:
    if not values:
        raise ValueError("mean requires at least one observation")
    return sum(values) / len(values)


def sample_coefficient_of_variation(values: tuple[float, ...]) -> float:
    """Return sample standard deviation divided by the arithmetic mean."""

    if len(values) < _MIN_STABILITY_REPETITIONS:
        raise ValueError("stability requires at least two independent repetitions")
    mean = arithmetic_mean(values)
    if mean == 0:
        if any(value != 0 for value in values):
            raise ValueError("zero-mean stability observations are invalid")
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return sqrt(max(variance, 0.0)) / mean


def one_sided_wilson_upper(events: int, total: int) -> float:
    """Return the one-sided 95% Wilson upper bound for a Bernoulli rate."""

    if total < 1 or events < 0 or events > total:
        raise ValueError("Wilson observations are invalid")
    estimate = events / total
    z2 = _ONE_SIDED_95_Z**2
    denominator = 1 + z2 / total
    center = estimate + z2 / (2 * total)
    margin = _ONE_SIDED_95_Z * sqrt(
        estimate * (1 - estimate) / total + z2 / (4 * total**2)
    )
    return min(1.0, (center + margin) / denominator)
