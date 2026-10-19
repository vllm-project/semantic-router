"""Candidate probabilities to System One answers, exactly as Decision 1.0's bundled runtime builds them.

Choice confidence is the top-two margin and Score confidence the
concentration around the expected level relative to a uniform distribution
(the ``decision_type_aware_v1`` statistics); ties pick the first candidate.
A Score legend holds text, as the API contract types it: structured level
descriptions appear as their canonical JSON (the bundled runtime returns them
as objects).
"""

from __future__ import annotations

import math
from typing import Any

from ...errors import INVALID_MODEL_OUTPUT
from ...systemone import canonical

PROBABILITY_SUM_TOLERANCE = 2e-5


def choice_confidence(probabilities: list[float]) -> float:
    """Top-two margin; a sole candidate has concentration one."""
    if len(probabilities) == 1:
        return 1.0
    first, second = sorted(probabilities, reverse=True)[:2]
    return min(1.0, max(0.0, first - second))


def score_confidence(probabilities: list[float]) -> float:
    """One minus the variance around the expected level, relative to the uniform variance."""
    if len(probabilities) == 1:
        return 1.0
    mean = math.fsum(index * value for index, value in enumerate(probabilities))
    variance = math.fsum(
        value * (index - mean) ** 2 for index, value in enumerate(probabilities)
    )
    uniform = (len(probabilities) ** 2 - 1) / 12
    return min(1.0, max(0.0, 1.0 - variance / uniform))


def answer(
    kind: str,
    keys: list[str],
    descriptions: list[Any],
    probabilities: list[float] | None,
) -> dict[str, Any]:
    """The typed answer for one question's candidate distribution; malformed output is ``invalid_model_output``."""
    values = [float(value) for value in probabilities or ()]
    if (
        len(values) != len(keys)
        or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values)
        or not math.isclose(
            math.fsum(values), 1.0, rel_tol=0.0, abs_tol=PROBABILITY_SUM_TOLERANCE
        )
    ):
        return {"type": kind, "error": INVALID_MODEL_OUTPUT}
    if kind == "noul":
        return {"type": "noul", "noul": values[1]}
    distribution = dict(zip(keys, values, strict=True))
    if kind == "choice":
        winner = max(range(len(keys)), key=values.__getitem__)
        return {
            "type": "choice",
            "choice": keys[winner],
            "confidence": choice_confidence(values),
            "probabilities": distribution,
        }
    return {
        "type": "score",
        "score": math.fsum(index * value for index, value in enumerate(values)),
        "confidence": score_confidence(values),
        "legend": {
            key: description if isinstance(description, str) else canonical(description)
            for key, description in zip(keys, descriptions, strict=True)
        },
        "probabilities": distribution,
    }
