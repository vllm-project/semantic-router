"""Audited, benchmark-only projection of the old Decision preview envelope.

Projection runs after the complete HTTP response has been timed and hashed. It
does not change the old service, the request bytes, or the response digest in a
receipt. The old max-probability statistic is checked, then the deliberately
different Decision v1 confidence is calculated solely to validate the common
answer distribution with the new contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from decision_runtime.confidence import choice_confidence, score_confidence
from decision_runtime.contracts import SystemOneRequest

_TOP_LEVEL = {"model", "answers", "usage", "profile", "timing", "source"}
_ANSWER_FIELDS = {
    "noul": {"type", "input_tokens", "noul"},
    "choice": {
        "type",
        "input_tokens",
        "probabilities",
        "top_probability",
        "choice",
        "confidence",
    },
    "score": {
        "type",
        "input_tokens",
        "probabilities",
        "top_probability",
        "score",
        "legend",
        "confidence",
    },
}


def project_legacy_preview(
    payload: object, request: SystemOneRequest
) -> dict[str, object]:
    """Reject unexpected old shapes and return only common semantic fields."""

    if not isinstance(payload, dict) or set(payload) != _TOP_LEVEL:
        raise ValueError("old response is not the known preview envelope")
    if payload["model"] != request.model or not isinstance(payload["source"], str):
        raise ValueError("old response model or source identity differs")
    profile = payload["profile"]
    timing = payload["timing"]
    if (
        not isinstance(profile, dict)
        or not isinstance(profile.get("confidence_definition"), str)
        or not profile["confidence_definition"].startswith("max(p)")
        or not isinstance(timing, dict)
        or not isinstance(payload["answers"], dict)
        or set(payload["answers"]) != set(request.questions)
    ):
        raise ValueError("old response preview metadata or answer IDs differ")

    projected: dict[str, object] = {}
    token_sum = 0
    for question_id, question in request.questions.items():
        answer = payload["answers"][question_id]
        if not isinstance(answer, dict) or answer.get("type") != question.type:
            raise ValueError("old response answer type differs")
        if set(answer) != _ANSWER_FIELDS[question.type]:
            raise ValueError("old response answer fields differ")
        input_tokens = answer["input_tokens"]
        if type(input_tokens) is not int or input_tokens < 1:
            raise ValueError("old response input token count is invalid")
        token_sum += input_tokens
        if question.type == "noul":
            projected[question_id] = {"type": "noul", "noul": answer["noul"]}
            continue

        probabilities = answer["probabilities"]
        if not isinstance(probabilities, Mapping) or not probabilities:
            raise ValueError("old response probabilities are invalid")
        values = tuple(probabilities.values())
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in values
        ):
            raise ValueError("old response probabilities are not finite")
        peak = max(values)
        if not math.isclose(
            answer["top_probability"], peak, rel_tol=0, abs_tol=2e-5
        ) or not math.isclose(answer["confidence"], peak, rel_tol=0, abs_tol=2e-5):
            raise ValueError("old response max-probability statistic differs")
        if question.type == "choice":
            projected[question_id] = {
                "type": "choice",
                "choice": answer["choice"],
                "confidence": choice_confidence(values),
                "probabilities": probabilities,
            }
        else:
            projected[question_id] = {
                "type": "score",
                "score": answer["score"],
                "confidence": score_confidence(values),
                "legend": answer["legend"],
                "probabilities": probabilities,
            }
    usage = payload["usage"]
    if not isinstance(usage, dict) or usage.get("input_tokens") != token_sum:
        raise ValueError("old response usage differs from answer tokens")
    return {
        "model": payload["model"],
        "answers": projected,
        "usage": usage,
    }
