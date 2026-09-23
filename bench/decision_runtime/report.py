"""Per-model latency, throughput, errors, and six-model receipt collation."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .cases import MODELS
from .transport import Sample

SCHEMA_VERSION = "decision-http-pair-v1"


def percentile(values: list[float], fraction: float) -> float | None:
    """Use the nearest-rank definition, including for small cohorts."""

    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _latency_summary(samples: list[Sample]) -> dict[str, Any]:
    successful = [sample.latency_ms for sample in samples if sample.success]
    failures = Counter(
        sample.error_code for sample in samples if sample.error_code is not None
    )
    return {
        "attempts": len(samples),
        "successes": len(successful),
        "errors": dict(sorted(failures.items())),
        "p50_ms": percentile(successful, 0.50),
        "p95_ms": percentile(successful, 0.95),
        "p99_ms": percentile(successful, 0.99),
    }


def _throughput_summary(samples: list[Sample]) -> dict[str, Any]:
    base = _latency_summary(samples)
    windows = []
    for round_number in sorted({sample.round for sample in samples}):
        round_samples = [sample for sample in samples if sample.round == round_number]
        if not round_samples:
            continue
        duration_seconds = (
            max(sample.ended_ns for sample in round_samples)
            - min(sample.started_ns for sample in round_samples)
        ) / 1_000_000_000
        windows.append(
            {
                "round": round_number,
                "attempts": len(round_samples),
                "successes": sum(sample.success for sample in round_samples),
                "window_seconds": duration_seconds,
            }
        )
    total_seconds = sum(window["window_seconds"] for window in windows)
    base.update(
        {
            "rounds": windows,
            "total_window_seconds": total_seconds,
            "successful_requests_per_second": (
                base["successes"] / total_seconds if total_seconds > 0 else None
            ),
            "attempted_requests_per_second": (
                base["attempts"] / total_seconds if total_seconds > 0 else None
            ),
        }
    )
    return base


def summarize(
    samples: list[Sample], old: dict[str, str], new: dict[str, str]
) -> dict[str, Any]:
    arms: dict[str, dict[str, Any]] = {}
    for arm in ("old", "new"):
        arms[arm] = {
            "warmup": _latency_summary(
                [
                    sample
                    for sample in samples
                    if sample.arm == arm and sample.phase == "warmup"
                ]
            ),
            "latency": _latency_summary(
                [
                    sample
                    for sample in samples
                    if sample.arm == arm and sample.phase == "latency"
                ]
            ),
            "throughput": _throughput_summary(
                [
                    sample
                    for sample in samples
                    if sample.arm == arm and sample.phase == "throughput"
                ]
            ),
        }

    reasons = []
    for key in ("model_revision", "hardware", "network_scope"):
        if old[key] != new[key]:
            reasons.append(f"different_{key}")
    for arm in ("old", "new"):
        for phase in ("warmup", "latency", "throughput"):
            if arms[arm][phase]["errors"]:
                reasons.append(f"{arm}_{phase}_errors")
        if not arms[arm]["latency"]["successes"]:
            reasons.append(f"{arm}_no_latency_samples")
        if not arms[arm]["throughput"]["successes"]:
            reasons.append(f"{arm}_no_throughput_samples")
    comparable = not reasons
    old_p50 = arms["old"]["latency"]["p50_ms"]
    new_p50 = arms["new"]["latency"]["p50_ms"]
    old_rps = arms["old"]["throughput"]["successful_requests_per_second"]
    new_rps = arms["new"]["throughput"]["successful_requests_per_second"]
    return {
        "arms": arms,
        "comparison": {
            "comparable": comparable,
            "reasons": reasons,
            "old_p50_over_new_p50": (
                old_p50 / new_p50 if comparable and new_p50 else None
            ),
            "new_throughput_over_old_throughput": (
                new_rps / old_rps if comparable and old_rps else None
            ),
        },
    }


def build_matrix(receipt_paths: list[Path]) -> dict[str, Any]:
    """Require one independently measured receipt for each catalog Decision model."""

    if len(receipt_paths) != len(MODELS):
        raise ValueError("matrix requires exactly six per-model receipts")
    receipts = [json.loads(path.read_text(encoding="utf-8")) for path in receipt_paths]
    indexed = {}
    for receipt in receipts:
        if receipt.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported or missing receipt schema")
        model = receipt.get("model")
        if model in indexed:
            raise ValueError(f"duplicate model receipt: {model}")
        indexed[model] = receipt
    if set(indexed) != set(MODELS):
        raise ValueError("matrix receipts must cover the six Decision models")
    fixtures = {receipt["fixture_sha256"] for receipt in receipts}
    settings = {json.dumps(receipt["settings"], sort_keys=True) for receipt in receipts}
    harnesses = {receipt["harness_sha256"] for receipt in receipts}
    if len(fixtures) != 1 or len(settings) != 1 or len(harnesses) != 1:
        raise ValueError("matrix receipts use different fixtures, harness, or settings")
    rows = []
    for model in MODELS:
        receipt = indexed[model]
        summary = receipt["summary"]
        rows.append(
            {
                "model": model,
                "old_provenance": receipt["old"],
                "new_provenance": receipt["new"],
                "old": summary["arms"]["old"],
                "new": summary["arms"]["new"],
                "comparison": summary["comparison"],
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "fixture_sha256": fixtures.pop(),
        "harness_sha256": harnesses.pop(),
        "settings": receipts[0]["settings"],
        "models": rows,
    }
