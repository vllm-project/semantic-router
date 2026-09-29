"""Per-level Score logit offsets bound to one model's weight fingerprint.

A `score_bias.json` holds, for some Score level counts L, offsets b_L,0..b_L,L-1
that are added to the model's Score logits (keys "0".."L-1" in level order)
before temperature and softmax. Every other L, Choice and Noul are unchanged.

    {"format": "dev2-score-bias-v1", "model_sha256": "<64 hex>",
     "offsets": {"3": [...], "4": [...], "5": [...]}, "fit": {...provenance...}}

The file binds the `model_sha256` that `training.model.infer` reports for the
checkpoint (see `checkpoint_fingerprint`), exactly like a calibration report.
Standard library only.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .calibration import _sha
from .data import MAX_OPTIONS

SCORE_BIAS_FORMAT = "dev2-score-bias-v1"
SCORE_BIAS_KEYS = {"format", "model_sha256", "offsets", "fit"}


def validate_offsets(value: Any) -> dict[int, list[float]]:
    if not isinstance(value, dict) or not value:
        raise ValueError("score bias offsets must be a nonempty object")
    offsets: dict[int, list[float]] = {}
    for key, row in value.items():
        if (
            not isinstance(key, str)
            or not key.isdigit()
            or str(int(key)) != key
            or not 2 <= int(key) <= MAX_OPTIONS
        ):
            raise ValueError(
                f"score bias level count {key!r} is not in 2..{MAX_OPTIONS}"
            )
        level_count = int(key)
        if not isinstance(row, list) or len(row) != level_count:
            raise ValueError(
                f"score bias offsets for L={key} need exactly {key} values"
            )
        if any(
            type(item) not in (int, float) or not math.isfinite(item) for item in row
        ):
            raise ValueError(f"score bias offsets for L={key} must be finite numbers")
        offsets[level_count] = [float(item) for item in row]
    return offsets


def validate_score_bias(
    report: Any, expected_model_sha256: str
) -> dict[int, list[float]]:
    if not isinstance(report, dict) or report.get("format") != SCORE_BIAS_FORMAT:
        raise ValueError("Unknown Decision 2.0 score bias format")
    if set(report) != SCORE_BIAS_KEYS:
        raise ValueError(
            f"score bias keys differ: {sorted(set(report) ^ SCORE_BIAS_KEYS)}"
        )
    _sha(expected_model_sha256, "inference model_sha256")
    if _sha(report["model_sha256"], "score bias model_sha256") != expected_model_sha256:
        raise ValueError("Score bias model hash differs from the inference checkpoint")
    if not isinstance(report["fit"], dict):
        raise ValueError("score bias fit provenance must be an object")
    return validate_offsets(report["offsets"])


def load_score_bias(
    path: Path, expected_model_sha256: str
) -> tuple[dict[int, list[float]], dict[str, Any]]:
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_score_bias(report, expected_model_sha256), report


def apply(
    offsets: dict[int, list[float]], logits: list[float], level_count: int
) -> list[float]:
    """Logits plus the offsets of this level count, or unchanged without one.

    Malformed model output (wrong length, non-numbers) is returned unchanged so
    that `normalized_answer` rejects it exactly as it would without offsets.
    """
    row = offsets.get(level_count)
    if (
        row is None
        or len(logits) != level_count
        or any(type(value) not in (int, float) for value in logits)
    ):
        return list(logits)
    return [float(value) + bias for value, bias in zip(logits, row)]
