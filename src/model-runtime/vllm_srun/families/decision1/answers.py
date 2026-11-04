"""Candidate probabilities to System One answers, as Decision 1.0's bundled runtime builds them.

Choice and Score confidence are System One's published statistics
(``systemone.choice_confidence`` and ``score_confidence``), as every family
reports them; ties pick the first candidate.
A Score legend holds text, as the API contract types it: structured level
descriptions appear as their canonical JSON (the bundled runtime returns them
as objects).
"""

from __future__ import annotations

import math
from typing import Any

from ...errors import INVALID_MODEL_OUTPUT
from ...systemone import canonical, choice_confidence, score_confidence

PROBABILITY_SUM_TOLERANCE = 2e-5


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
