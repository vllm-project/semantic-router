"""Runtime state and the golden check that gates readiness."""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any

STATES = ("starting", "loading", "warming", "ready", "degraded", "failed")
GPU_TOLERANCE = 0.02
SUM_TOLERANCE = 1e-6


@dataclass
class GoldenResult:
    status: str = "pending"
    checked: int = 0
    matched: int = 0
    reference: str | None = None
    detail: str | None = None

    def describe(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checked": self.checked,
            "matched": self.matched,
            "reference": self.reference,
        }


@dataclass
class Health:
    state: str = "starting"
    reason: str | None = None
    golden: GoldenResult = field(default_factory=GoldenResult)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set(self, state: str, reason: str | None = None) -> None:
        if state not in STATES:
            raise ValueError(f"unknown runtime state {state!r}")
        with self._lock:
            self.state = state
            self.reason = reason

    @property
    def ready(self) -> bool:
        return self.state == "ready"


def well_formed(answer: dict[str, Any]) -> bool:
    """A finite, normalized answer of the declared type."""
    if "error" in answer:
        return False
    kind = answer.get("type")
    if kind == "noul":
        value = answer.get("noul")
        return isinstance(value, float) and 0.0 <= value <= 1.0
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, dict) or not probabilities:
        return False
    values = list(probabilities.values())
    if any(not isinstance(v, float) or not math.isfinite(v) or v < 0 for v in values):
        return False
    return abs(sum(values) - 1.0) < SUM_TOLERANCE


def compare(
    answers: dict[str, Any], expected: dict[str, Any], exact: bool
) -> tuple[int, int]:
    """(checked, matched) of answers against reference answers for the same questions."""
    checked = matched = 0
    for question_id, reference in expected.items():
        answer = answers.get(question_id)
        if answer is None:
            continue
        checked += 1
        if exact:
            matched += answer == reference
            continue
        if answer.get("type") != reference.get("type"):
            continue
        if answer.get("type") == "noul":
            matched += abs(answer["noul"] - reference["noul"]) <= GPU_TOLERANCE
            continue
        left, right = answer.get("probabilities", {}), reference.get(
            "probabilities", {}
        )
        if set(left) == set(right) and all(
            abs(left[k] - right[k]) <= GPU_TOLERANCE for k in left
        ):
            matched += 1
    return checked, matched


def golden_check(
    run: Any,
    goldens: list[dict[str, Any]],
    device_class: str,
) -> GoldenResult:
    """Run each golden request twice; require determinism, well-formed answers and the reference when known.

    ``run(state, questions)`` returns the answers dict. References are keyed
    by device class (``cpu``, ``rocm``, ``cuda``); CPU references must match
    exactly, GPU references within ``GPU_TOLERANCE``.
    """
    result = GoldenResult(status="unverified")
    for golden in goldens:
        first = run(golden["state"], golden["questions"])
        second = run(golden["state"], golden["questions"])
        if first != second:
            return GoldenResult(
                status="failed", detail="golden answers are not deterministic"
            )
        if not all(well_formed(answer) for answer in first.values()):
            return GoldenResult(status="failed", detail="golden answers are malformed")
        reference = (golden.get("expected") or {}).get(device_class)
        if reference:
            checked, matched = compare(first, reference, exact=device_class == "cpu")
            result.checked += checked
            result.matched += matched
            result.reference = device_class
            if matched != checked:
                return GoldenResult(
                    status="failed",
                    checked=result.checked,
                    matched=result.matched,
                    reference=device_class,
                    detail="golden answers differ from the reference",
                )
    if result.reference is not None:
        result.status = "matched"
    return result
