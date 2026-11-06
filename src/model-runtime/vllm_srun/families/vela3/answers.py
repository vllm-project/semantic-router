"""Raw logits to Vela 3.0 answers, in the System One response of the decisions surface.

Choice probabilities are a softmax over the real options with no temperature
(router questions report their abstain option apart); Noul is P(yes) of the
two-option choice. A router Set question is a sigmoid per label against the
request's threshold, else the package's (0.5): it answers ``sets.<id>`` and a
Noul per label (``<id>.<label>``). A Span question decodes its word
probabilities, a sigmoid at its route's temperature, against its route's
threshold (``routing.Router.span_threshold``, by the target's token count),
or the request's: it answers ``spans.<id>`` (code-point offsets into the field
it reads) and a Noul that is its highest word probability over that field.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...errors import INVALID_MODEL_OUTPUT
from ...systemone import choice_confidence
from ..vela2.calibration import sigmoid, softmax
from ..vela2.raw import RawRow
from ..vela2.request import State
from ..vela2.words import decode_spans, trim
from .request import Asked
from .routing import Router

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Answerer:
    """Answers of one package (its router set threshold and span routes)."""

    def __init__(self, router: Router):
        self.router = router

    def answer(
        self,
        asked: Asked,
        logits: NDArray[np.float64] | None,
        raw: RawRow | None,
        state: State | None,
        response: dict[str, Any],
    ) -> None:
        """Write one question's answer (and its span / set entries) into ``response``."""
        question = asked.question
        try:
            if question.type == "span":
                assert raw is not None and state is not None
                self._span(asked, raw, state, response)
            elif question.type == "set":
                assert logits is not None
                self._set(asked, logits, response)
            else:
                assert logits is not None
                response["answers"][question.id] = self._choice(asked, logits)
        except (KeyError, ValueError, FloatingPointError):
            response["answers"][question.id] = {
                "type": question.kind,
                "error": INVALID_MODEL_OUTPUT,
            }

    def _choice(self, asked: Asked, values: NDArray[np.float64]) -> dict[str, Any]:
        question = asked.question
        names = question.names
        logits = np.asarray(values, np.float64)
        if not np.all(np.isfinite(logits)):
            raise ValueError("nonfinite option logits")
        probabilities = softmax(logits[: len(names)])
        probability = dict(zip(names, (float(p) for p in probabilities), strict=True))
        if question.kind == "noul":
            return {"type": "noul", "noul": probability["yes"]}
        answer: dict[str, Any] = {
            "type": "choice",
            "choice": max(probability, key=probability.__getitem__),
            "confidence": choice_confidence(list(probability.values())),
            "probabilities": probability,
        }
        if question.abstain:
            answer["abstain_probability"] = float(softmax(logits)[-1])
        return answer

    def _set(
        self, asked: Asked, values: NDArray[np.float64], response: dict[str, Any]
    ) -> None:
        question = asked.question
        names = question.names
        logits = np.asarray(values, np.float64)[: len(names)]
        if not np.all(np.isfinite(logits)):
            raise ValueError("nonfinite set logits")
        threshold = (
            question.threshold
            if question.threshold is not None
            else self.router.set_threshold
        )
        probability = dict(zip(names, (float(p) for p in sigmoid(logits)), strict=True))
        response.setdefault("sets", {})[question.id] = {
            "selected": [name for name, p in probability.items() if p > threshold],
            "probabilities": probability,
        }
        response.setdefault("thresholds", {})[question.id] = float(threshold)
        for name, p in probability.items():
            response["answers"][f"{question.id}.{name}"] = {"type": "noul", "noul": p}

    def _span(
        self, asked: Asked, raw: RawRow, state: State, response: dict[str, Any]
    ) -> None:
        question, route = asked.question, asked.route
        assert route is not None
        span = raw.span
        assert span is not None
        role = cast(str, question.over)
        text = state.text(role)
        threshold, temperature = self.router.span_threshold(route, raw.tokens[role])
        if question.threshold is not None:
            threshold = question.threshold
        probabilities = sigmoid(span.logits / temperature)
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("nonfinite span logits")
        decoded = (
            decode_spans(probabilities, span.offsets, span.labels, text, threshold)
            if len(span.offsets)
            else []
        )
        low, high = question.span_range or (0, len(text))
        spans = []
        for entry in decoded:
            start, end = entry["start"], entry["end"]
            if question.span_range is not None:
                start, end = trim(text, max(start, low), min(end, high))
                if end <= start:
                    continue
            spans.append(
                {
                    "label": entry["label"],
                    "start": start - low,
                    "end": end - low,
                    "text": text[start:end],
                    "probability": float(entry["probability"]),
                }
            )
        best: NDArray[np.float64] = (
            probabilities.max(1) if len(span.offsets) else np.zeros(0)
        )
        if question.span_range is not None and len(best):
            inside = (span.offsets[:, 0] >= low) & (span.offsets[:, 1] <= high)
            best = best[inside]
        response.setdefault("spans", {})[question.id] = spans
        response.setdefault("thresholds", {})[question.id] = float(threshold)
        response["answers"][question.id] = {
            "type": "noul",
            "noul": float(np.max(best)) if len(best) else 0.0,
        }
