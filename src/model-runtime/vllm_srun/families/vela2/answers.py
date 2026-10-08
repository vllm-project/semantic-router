"""Raw logits to the Vela 2.0 response: System One answers plus ``spans``, ``sets`` and ``thresholds``.

Choice and Score probabilities are a softmax over the real options at the
type's temperature (the abstain option is reported apart); ``confidence`` is
the top-two margin for Choice and one minus the normalised variance for
Score; Noul is P(yes). A Set question answers ``sets.<id>`` and one Noul per
label (``<id>.<label>``); a Span question answers ``spans.<id>`` (code-point
offsets into the field it reads, clipped to that field when it shares a
part) and a Noul that is its highest word probability over that field.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...errors import INVALID_MODEL_OUTPUT
from ...systemone import canonical
from .calibration import Calibration, sigmoid, softmax
from .raw import RawRow
from .request import Question, State
from .words import decode_spans, trim

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True)
class Answerer:
    """Calibrated answers of one package (``noul_calibration`` applies its optional Noul fit)."""

    calibration: Calibration
    noul_calibration: bool = False
    report_heads: bool = False

    def answer(
        self, question: Question, raw: RawRow, state: State, response: dict[str, Any]
    ) -> None:
        """Write one question's answer (and its span / set entries) into ``response``."""
        try:
            if question.type == "span":
                self._span(question, raw, state, response)
            elif question.type == "set":
                self._set(question, raw.logits[question.id], response)
            else:
                response["answers"][question.id] = self._single(
                    question, raw.logits[question.id]
                )
        except (KeyError, ValueError, FloatingPointError):
            response["answers"][question.id] = {
                "type": question.kind,
                "error": INVALID_MODEL_OUTPUT,
            }

    def _single(
        self, question: Question, values: NDArray[np.float32]
    ) -> dict[str, Any]:
        logits = np.asarray(values, np.float64)
        names = question.names
        count = len(names)
        if not np.all(np.isfinite(logits[:count])):
            raise ValueError("nonfinite option logits")
        temperature = self.calibration.temperature(question.type)
        probabilities = softmax(logits[:count] / temperature)
        probability = dict(zip(names, (float(p) for p in probabilities), strict=True))
        if question.kind == "noul":
            return {
                "type": "noul",
                "noul": self.calibration.noul(
                    probability["yes"], self.noul_calibration
                ),
            }
        abstain = (
            {"abstain_probability": float(softmax(logits / temperature)[-1])}
            if question.abstain
            else {}
        )
        if question.kind == "choice":
            ordered = sorted(probability.values(), reverse=True)
            margin = 1.0 if count == 1 else min(1.0, max(0.0, ordered[0] - ordered[1]))
            return {
                "type": "choice",
                "choice": names[int(np.argmax(logits[:count]))],
                "confidence": margin,
                "probabilities": probability,
                **abstain,
            }
        levels = [float(p) for p in probabilities]
        mean = math.fsum(index * p for index, p in enumerate(levels))
        variance = math.fsum(p * (index - mean) ** 2 for index, p in enumerate(levels))
        return {
            "type": "score",
            "score": mean,
            "confidence": min(
                1.0, max(0.0, 1.0 - variance / ((count * count - 1) / 12))
            ),
            "legend": {
                str(index): level if isinstance(level, str) else canonical(level)
                for index, level in enumerate(question.criteria)
            },
            "probabilities": {str(index): p for index, p in enumerate(levels)},
        }

    def _set(
        self, question: Question, values: NDArray[np.float32], response: dict[str, Any]
    ) -> None:
        logits = np.asarray(values, np.float64)
        names = question.names
        if not np.all(np.isfinite(logits[: len(names)])):
            raise ValueError("nonfinite set logits")
        threshold = (
            question.threshold
            if question.threshold is not None
            else self.calibration.set_threshold(question.key)
        )
        scores = sigmoid(logits[: len(names)] / self.calibration.temperature("set"))
        probability = dict(zip(names, (float(p) for p in scores), strict=True))
        response.setdefault("sets", {})[question.id] = {
            "selected": [name for name, p in probability.items() if p > threshold],
            "probabilities": probability,
        }
        response.setdefault("thresholds", {})[question.id] = float(threshold)
        for name, p in probability.items():
            response["answers"][f"{question.id}.{name}"] = {"type": "noul", "noul": p}

    def _span(
        self, question: Question, raw: RawRow, state: State, response: dict[str, Any]
    ) -> None:
        span = raw.span
        assert span is not None
        role = cast(str, question.over)
        text = state.text(role)
        calibration = self.calibration
        if question.threshold is not None:
            threshold, rule = question.threshold, "given"
        else:
            threshold, rule = calibration.span_threshold(
                question.key, span.labels, raw.tokens[role], span.head
            )
        probabilities = sigmoid(span.logits / calibration.span_temperature(span.head))
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("nonfinite span logits")
        if rule.startswith("pii:"):
            threshold, rule = calibration.sparse_gate(
                probabilities, span.offsets, span.labels, text, threshold, rule
            )
        decoded = (
            decode_spans(probabilities, span.offsets, span.labels, text, threshold)
            if len(span.offsets)
            else []
        )
        alias = span.alias or {}
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
                    "label": alias.get(entry["label"], entry["label"]),
                    "start": start - low if question.span_range else start,
                    "end": end - low if question.span_range else end,
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
        if self.report_heads:
            response.setdefault("span_heads", {})[question.id] = span.head
        response["answers"][question.id] = {
            "type": "noul",
            "noul": float(np.max(best)) if len(best) else 0.0,
        }
