"""Gold-blind source SELECT parity check before a new optimizer step."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from training.model.data import file_sha256


def check_baseline_parity(
    reference: Path,
    current: Path,
    *,
    reference_sha256: str,
    max_probability_drift: float = 1e-6,
) -> dict[str, Any]:
    if file_sha256(reference) != reference_sha256:
        raise ValueError("Frozen source SELECT reference SHA-256 differs")
    if not math.isfinite(max_probability_drift) or max_probability_drift < 0:
        raise ValueError("Invalid parity tolerance")
    original = [json.loads(line) for line in reference.read_text().splitlines()]
    candidate = [json.loads(line) for line in current.read_text().splitlines()]
    if len(original) != len(candidate) or not original:
        raise ValueError("Source SELECT row count differs")
    maximum = 0.0
    for expected, actual in zip(original, candidate):
        for field in ("id", "source_input_sha256", "keys", "prediction_key"):
            if expected[field] != actual[field]:
                raise ValueError(f"Source SELECT {field} differs")
        if set(expected["probabilities"]) != set(actual["probabilities"]):
            raise ValueError("Source SELECT option domain differs")
        for key in expected["probabilities"]:
            a = expected["probabilities"][key]
            b = actual["probabilities"][key]
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                raise ValueError("Source SELECT probability is nonnumeric")
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Source SELECT probability is nonfinite")
            maximum = max(maximum, abs(a - b))
    if maximum > max_probability_drift:
        raise ValueError("Source SELECT numeric parity failed")
    return {
        "reference_sha256": reference_sha256,
        "current_sha256": file_sha256(current),
        "rows": len(original),
        "categorical_mismatches": 0,
        "max_probability_drift": maximum,
        "tolerance": max_probability_drift,
        "gold_opened_for_parity": False,
    }
