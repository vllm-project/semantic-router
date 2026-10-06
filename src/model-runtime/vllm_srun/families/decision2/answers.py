"""Candidate logits to System One answers, exactly as the released runtime computes them."""

from __future__ import annotations

import math
from typing import Any

from ...systemone import MAX_OPTIONS, MIN_OPTIONS

# Probabilities this close to the maximum count as tied (the scored adapter's rule).
TIE_TOLERANCE = 1e-8


def apply_score_bias(
    offsets: dict[int, list[float]], logits: list[float], level_count: int
) -> list[float]:
    """Logits plus the offsets of this level count; malformed output is returned unchanged."""
    row = offsets.get(level_count)
    if (
        row is None
        or len(logits) != level_count
        or any(type(value) not in (int, float) for value in logits)
    ):
        return list(logits)
    return [float(value) + bias for value, bias in zip(logits, row, strict=True)]


def normalized_answer(
    kind: str, keys: list[str], logits: list[float], temperature: float
) -> dict[str, Any]:
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if len(keys) != len(logits) or not MIN_OPTIONS <= len(keys) <= MAX_OPTIONS:
        raise ValueError("model returned the wrong number of candidate logits")
    if any(
        not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in logits
    ):
        raise ValueError("model returned a nonfinite valid-candidate logit")
    scaled = [float(value) / temperature for value in logits]
    top = max(scaled)
    exponentials = [math.exp(value - top) for value in scaled]
    total = sum(exponentials)
    probabilities = [value / total for value in exponentials]
    probability_map = dict(zip(keys, probabilities, strict=True))
    maximum = max(probabilities)
    winners = [
        index
        for index, value in enumerate(probabilities)
        if abs(value - maximum) <= TIE_TOLERANCE
    ]
    winner = winners[0] if len(winners) == 1 else None
    if kind == "noul":
        return {"type": "noul", "noul": probability_map["true"]}
    if kind == "score":
        return {
            "type": "score",
            "score": sum(int(key) * probability_map[key] for key in keys),
            "probabilities": probability_map,
        }
    return {
        "type": "choice",
        "choice": keys[winner] if winner is not None else None,
        "probabilities": probability_map,
    }


def product_answer(
    kind: str,
    keys: list[str],
    logits: list[float],
    temperature: float,
    descriptions: list[Any],
) -> dict[str, Any]:
    """System One fields: confidence is 1 - normalized entropy; Score adds a legend."""
    answer = normalized_answer(kind, keys, logits, temperature)
    if kind == "noul":
        return answer
    if len(descriptions) != len(keys):
        raise ValueError("candidate descriptions do not match the model answer")
    probabilities = list(answer["probabilities"].values())
    entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
    answer["confidence"] = max(0.0, min(1.0, 1.0 - entropy / math.log(len(keys))))
    if kind == "score":
        from ...systemone import canonical

        answer["legend"] = {
            key: description if isinstance(description, str) else canonical(description)
            for key, description in zip(keys, descriptions, strict=True)
        }
    elif answer["choice"] is None:
        # Exact ties resolve to the first option in request order; the distribution is unchanged.
        maximum = max(probabilities)
        answer["choice"] = keys[probabilities.index(maximum)]
    return answer
