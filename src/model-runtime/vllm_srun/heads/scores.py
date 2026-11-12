"""Independent label scores: a pooled row, the classifier, a sigmoid per label.

A package's operating point (Vela Hazard's ``operating_point.json``) fixes the
per-label thresholds, the comparison and the window policy: every input is
read in its windows and each label keeps its maximum over them; ``selected``
lists the labels whose reduced score passes its threshold.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ..accel.kernels import rowwise
from .sequence import SequenceHead

COMPARISONS: dict[str, Callable[[float, float], bool]] = {
    "score >= threshold": lambda score, threshold: score >= threshold,
    "score > threshold": lambda score, threshold: score > threshold,
}


@dataclass(frozen=True)
class OperatingPoint:
    """Per-label thresholds and the input policy they were calibrated under."""

    labels: tuple[str, ...]
    thresholds: tuple[float, ...]
    comparison: str
    window: tuple[int, int] | None
    max_tokens: int | None

    @classmethod
    def parse(cls, document: dict[str, Any], labels: Sequence[str]) -> OperatingPoint:
        """An independent-sigmoid operating point whose labels match the head's, in order."""
        if document.get("score_type") != "independent_sigmoid":
            raise ValueError("the operating point is not an independent-sigmoid policy")
        comparison = document.get("comparison")
        if comparison not in COMPARISONS:
            raise ValueError(f"unsupported operating point comparison {comparison!r}")
        if tuple(document.get("labels") or ()) != tuple(labels):
            raise ValueError(
                "operating point labels differ from the classifier's labels"
            )
        thresholds = tuple(float(value) for value in document.get("thresholds") or ())
        if len(thresholds) != len(labels):
            raise ValueError("operating point needs one threshold per label")
        policy = document.get("input_policy") or {}
        window = None
        if policy.get("strategy") == "overlapping_content_windows":
            window = (
                int(policy["window_tokens_including_special_tokens"]),
                int(policy["overlap_content_tokens"]),
            )
        limit = policy.get("max_document_tokens_including_special_tokens")
        return cls(
            tuple(labels), thresholds, comparison, window, int(limit) if limit else None
        )

    def select(self, scores: Sequence[float]) -> list[str]:
        passes = COMPARISONS[self.comparison]
        return [
            label
            for label, score, threshold in zip(
                self.labels, scores, self.thresholds, strict=True
            )
            if passes(score, threshold)
        ]


class ScoresHead(SequenceHead):
    """Independent sigmoid scores, with ``selected`` labels when an operating point is packaged."""

    kind = "scores"
    value_key = "scores"

    def __init__(
        self, *args: Any, operating_point: OperatingPoint | None = None, **kwargs: Any
    ):
        if operating_point is not None and operating_point.window is not None:
            kwargs.update(overflow="window", window=operating_point.window)
        super().__init__(*args, **kwargs)
        self.operating_point = operating_point

    def describe(self) -> dict[str, Any]:
        card = super().describe()
        if self.operating_point is not None:
            card["thresholds"] = self.operating_point.thresholds
        return card

    def activate(self, logits: torch.Tensor) -> torch.Tensor:
        return rowwise(torch.sigmoid, logits)

    def annotate(self, out: dict[str, Any], reduced: Sequence[float]) -> None:
        if self.operating_point is not None:
            out["selected"] = self.operating_point.select(reduced)
