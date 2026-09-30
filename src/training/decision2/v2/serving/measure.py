"""Pure helpers for the serving sessions: answer comparison and latency statistics.

Answer comparison uses the release parity's own definitions
(``v2.release.examples``): an answer change is a different category (Choice
key, Noul side of 0.5, Score argmax level) or a different set of numbers, and
drift is the largest absolute difference of any reported number. This module
also splits drift into probabilities (Choice/Score probabilities, Noul) and
expected Score values, and keeps the drift distribution.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from v2.release.examples import category, compare_answers, numbers


def payload_sha256(prompt: dict[str, Any]) -> str:
    """The frozen panels' input digest: state and questions in insertion order."""
    payload = {"state": prompt["state"], "questions": prompt["questions"]}
    return hashlib.sha256(
        json.dumps(
            payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def slot_drifts(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    """Per question present on both sides: category change, probability and score drift."""
    rows = []
    for qid in sorted(set(left) & set(right)):
        a, b = numbers(left[qid]), numbers(right[qid])
        shared = set(a) & set(b)
        prob = [abs(a[k] - b[k]) for k in shared if k != "score"]
        rows.append(
            {
                "qid": qid,
                "changed": category(left[qid]) != category(right[qid])
                or set(a) != set(b),
                "prob_drift": max(prob, default=0.0),
                "score_drift": (
                    abs(a["score"] - b["score"]) if "score" in shared else None
                ),
                "type": (
                    (right[qid] or {}).get("type")
                    if isinstance(right[qid], dict)
                    else None
                ),
            }
        )
    return rows


def quantile(values: list[float], q: float) -> float | None:
    """Nearest-rank quantile (q in [0, 1]); None for no values."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def latency_summary(seconds: list[float]) -> dict[str, Any]:
    ms = [s * 1000 for s in seconds]
    return {
        "n": len(ms),
        "mean_ms": sum(ms) / len(ms) if ms else None,
        "p50_ms": quantile(ms, 0.50),
        "p95_ms": quantile(ms, 0.95),
        "p99_ms": quantile(ms, 0.99),
        "max_ms": max(ms, default=None),
    }


class PanelTally:
    """Running totals for one panel: release-parity counts plus drift distributions."""

    def __init__(self, name: str):
        self.name = name
        self.totals = {
            "prompts": 0,
            "slots": 0,
            "category_changes": 0,
            "missing": 0,
            "max_abs_drift": 0.0,
            "input_mismatch": 0,
            "errors": 0,
        }
        self.prob_drifts: list[float] = []
        self.score_drifts: list[float] = []
        self.changed_by_type: dict[str, int] = {}
        self.slots_by_type: dict[str, int] = {}

    def add(
        self,
        prompt: dict[str, Any],
        response: dict[str, Any] | None,
        stored: dict[str, Any] | None,
    ) -> None:
        self.totals["prompts"] += 1
        stored = stored or {}
        if stored.get("source_input_sha256") != payload_sha256(prompt):
            self.totals["input_mismatch"] += 1
        if response is None:
            self.totals["errors"] += 1
            response = {}
        left = response.get("answers") or {}
        right = stored.get("answers") or {}
        result = compare_answers(left, right)
        for key in ("slots", "category_changes", "missing"):
            self.totals[key] += result[key]
        self.totals["max_abs_drift"] = max(
            self.totals["max_abs_drift"], result["max_abs_drift"]
        )
        for row in slot_drifts(left, right):
            kind = row["type"] or "unknown"
            self.slots_by_type[kind] = self.slots_by_type.get(kind, 0) + 1
            if row["changed"]:
                self.changed_by_type[kind] = self.changed_by_type.get(kind, 0) + 1
            self.prob_drifts.append(row["prob_drift"])
            if row["score_drift"] is not None:
                self.score_drifts.append(row["score_drift"])

    def summary(self) -> dict[str, Any]:
        return {
            **self.totals,
            "changed_by_type": dict(sorted(self.changed_by_type.items())),
            "slots_by_type": dict(sorted(self.slots_by_type.items())),
            "prob_drift": {
                "max": max(self.prob_drifts, default=0.0),
                "p50": quantile(self.prob_drifts, 0.5),
                "p99": quantile(self.prob_drifts, 0.99),
                "mean": (
                    sum(self.prob_drifts) / len(self.prob_drifts)
                    if self.prob_drifts
                    else None
                ),
            },
            "score_value_drift_max": max(self.score_drifts, default=None),
        }
