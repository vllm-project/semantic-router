"""Runtime state and the golden check that gates readiness."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

STATES = ("starting", "loading", "warming", "ready", "degraded", "failed")
GPU_TOLERANCE = 0.02
# CPU kernels round differently across instruction sets (AVX2, AVX-512, NEON).
CPU_TOLERANCE = 1e-3


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


def flatten(surface: str, response: dict[str, Any]) -> dict[str, float]:
    """Comparable numbers of a classify, embeddings or rerank response, by stable key."""
    values: dict[str, float] = {}
    if surface == "classify":
        for result in response.get("results", []):
            index = result.get("index")
            for name in ("probabilities", "scores"):
                for position, value in enumerate(result.get(name) or []):
                    values[f"{index}.{name}.{position}"] = float(value)
            for position, span in enumerate(result.get("spans") or []):
                key = f"{index}.span.{position}.{span['label']}.{span['start']}.{span['end']}"
                values[key] = float(span["probability"])
    elif surface == "embeddings":
        for item in response.get("data", []):
            for position, value in enumerate(item.get("embedding") or []):
                values[f"{item['index']}.{position}"] = float(value)
    elif surface == "rerank":
        for result in response.get("results", []):
            values[f"{result['index']}.logit"] = float(result["logit"])
    return values


def compare_numbers(
    values: dict[str, Any], reference: dict[str, Any], tolerance: float
) -> tuple[int, int] | None:
    """(checked, matched) of golden numbers against a reference; None when there are none or one is not finite.

    Every reference value is checked; a key set that differs from the
    reference's fails at least one.
    """
    if not values or not all(math.isfinite(value) for value in values.values()):
        return None
    if not reference:
        return 0, 0
    checked = len(reference)
    matched = sum(
        key in values and abs(values[key] - float(value)) <= tolerance
        for key, value in reference.items()
    )
    if set(values) != set(reference):
        matched = min(matched, checked - 1)
    return checked, matched


def golden_check(
    run: Callable[[str, dict[str, Any]], dict[str, Any]],
    compare: Callable[
        [str, dict[str, Any], dict[str, Any], float], tuple[int, int] | None
    ],
    goldens: list[dict[str, Any]],
    device_class: str,
) -> GoldenResult:
    """Run each golden request twice; require determinism, well-formed answers and the reference when known.

    A golden is ``{surface, body, expected}``. ``run(surface, body)`` returns
    the response's comparable values (``LoadedModel.golden_values``) and
    ``compare(surface, values, reference, tolerance)`` checks them against the
    reference recorded for this device class (``LoadedModel.golden_compare``).
    References are keyed by device class (``cpu``, ``rocm``, ``cuda``, ``mps``); answers
    must match within ``CPU_TOLERANCE`` on CPUs and ``GPU_TOLERANCE`` on GPUs.
    MPS requires a per-model ``tolerances.mps`` alongside its recorded reference.
    """
    result = GoldenResult(status="unverified")
    tolerance = CPU_TOLERANCE if device_class == "cpu" else GPU_TOLERANCE
    for golden in goldens:
        surface, body = golden["surface"], golden["body"]
        first = run(surface, body)
        second = run(surface, body)
        if first != second:
            return GoldenResult(
                status="failed", detail="golden answers are not deterministic"
            )
        reference = (golden.get("expected") or {}).get(device_class)
        model_tolerance = tolerance
        if device_class == "mps" and reference:
            recorded = (golden.get("tolerances") or {}).get("mps")
            if (
                isinstance(recorded, bool)
                or not isinstance(recorded, (int, float))
                or not math.isfinite(recorded)
                or recorded < 0
            ):
                return GoldenResult(
                    status="failed",
                    reference="mps",
                    detail="MPS reference requires a recorded per-model tolerance",
                )
            model_tolerance = float(recorded)
        counts = compare(surface, first, reference or {}, model_tolerance)
        if counts is None:
            return GoldenResult(status="failed", detail="golden answers are malformed")
        if not reference:
            continue
        checked, matched = counts
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
