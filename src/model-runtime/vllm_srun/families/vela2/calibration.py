"""A Vela 2.0 package's ``calibration.json``: temperatures and decision thresholds.

Choice and Score probabilities are a softmax over the real options at the
question type's temperature; Set labels and span words are sigmoids, decided
against a threshold: the request's own, else the package's per-question or
default value. PII span questions (question ID ``pii``, or labels drawn from
the 17 trained PII types) use the length rule (log threshold piecewise linear
in the log token count of the span part) and then the sparse-document gate,
a second decode of the same probabilities. The broad span head of the 4B and
9B releases has one temperature and one threshold.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .words import decode_spans

if TYPE_CHECKING:
    from numpy.typing import NDArray

ROUTER_HEAD, BROAD_HEAD = "router", "broad"
SPAN_HEADS = (ROUTER_HEAD, BROAD_HEAD)


def sigmoid(values: Any) -> NDArray[np.float64]:
    """Elementwise logistic in float64, as the packages compute probabilities."""
    return 1 / (1 + np.exp(-np.asarray(values, dtype=np.float64)))


def softmax(values: Any) -> NDArray[np.float64]:
    """Softmax over a vector in float64."""
    values = np.asarray(values, dtype=np.float64)
    exponentials = np.exp(values - values.max())
    normalized: NDArray[np.float64] = exponentials / exponentials.sum()
    return normalized


@dataclass(frozen=True)
class Calibration:
    """The parsed ``calibration.json`` of one package."""

    raw: dict[str, Any]

    @property
    def pii_types(self) -> frozenset[str]:
        return frozenset(self.raw.get("pii_types", ()))

    @property
    def broad(self) -> dict[str, Any] | None:
        return self.raw.get("broad_head") or None

    def schema(self, name: str) -> dict[str, Any] | None:
        """A trained question schema (``pii_schema``, ``halu_schema``, ``relevance_schema``)."""
        value = self.raw.get(f"{name}_schema")
        return value if isinstance(value, dict) else None

    def temperature(self, question_type: str) -> float:
        return float(self.raw["temperature"].get(question_type, 1.0))

    def span_temperature(self, head: str = ROUTER_HEAD) -> float:
        if head == BROAD_HEAD:
            return float(self.raw["broad_head"]["temperature"])
        return self.temperature("span")

    def set_threshold(self, question_id: str) -> float:
        thresholds = self.raw["thresholds"]
        return float(thresholds.get(f"set:{question_id}", thresholds["set:*"]))

    def is_pii(self, question_id: str, labels: list[str]) -> bool:
        return question_id == "pii" or bool(labels and set(labels) <= self.pii_types)

    def pii_length_threshold(self, tokens: int) -> float:
        anchors = self.raw["pii_length_rule"]["anchors"]
        return float(
            np.exp(
                np.interp(
                    np.log(max(tokens, 1)),
                    [np.log(anchor["n_tokens"]) for anchor in anchors],
                    [np.log(anchor["threshold"]) for anchor in anchors],
                )
            )
        )

    def span_threshold(
        self, question_id: str, labels: list[str], tokens: int, head: str = ROUTER_HEAD
    ) -> tuple[float, str]:
        """The threshold before the PII sparse gate, and the rule that chose it.

        ``tokens`` is the token count of the whole span part (before any cut).
        """
        if head == BROAD_HEAD:
            return float(self.raw["broad_head"]["threshold"]), BROAD_HEAD
        thresholds = self.raw["thresholds"]
        if self.is_pii(question_id, labels):
            if "pii_length_rule" in self.raw:
                return self.pii_length_threshold(tokens), "pii:length"
            for bucket in self.raw["pii_buckets"]:
                if tokens <= bucket["max_tokens"]:
                    return float(bucket["threshold"]), f"pii:{bucket['name']}"
        if question_id in ("halu", "toxic"):
            return float(thresholds[f"span:{question_id}"]), question_id
        key = f"span:{question_id}"
        if key in thresholds:
            return float(thresholds[key]), key
        return float(thresholds["span:*"]), "span:*"

    def sparse_gate(
        self,
        probabilities: NDArray[np.float64],
        offsets: NDArray[np.int32],
        labels: list[str],
        text: str,
        threshold: float,
        rule: str,
    ) -> tuple[float, str]:
        """A document with at most K spans at the probe threshold decodes at ``max(threshold, t_sparse)``."""
        gate = self.raw.get("pii_sparse_gate")
        if (
            not gate
            or gate.get("K", -1) < 0
            or threshold >= gate["t_sparse"]
            or not len(offsets)
        ):
            return threshold, rule
        probe = float(gate.get("probe_threshold", 0.5))
        if len(decode_spans(probabilities, offsets, labels, text, probe)) <= gate["K"]:
            return float(gate["t_sparse"]), rule + "+sparse"
        return threshold, rule

    def noul(self, probability: float, enabled: bool) -> float:
        """The optional Noul calibration ``sigmoid(logit(p) / T + b)``; ``p`` itself when off or absent."""
        block = self.raw.get("noul_calibration")
        if not enabled or not block:
            return probability
        p = min(max(probability, 1e-12), 1.0 - 1e-12)
        logit = math.log(p) - math.log1p(-p)
        return 1.0 / (1.0 + math.exp(-(logit / float(block["T"]) + float(block["b"]))))

    def noul_default(self) -> bool:
        return bool((self.raw.get("noul_calibration") or {}).get("default", False))
