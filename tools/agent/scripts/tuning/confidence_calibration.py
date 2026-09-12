"""Versioned offline confidence calibration for model escalation policies.

The module deliberately operates on recorded small/large-model results.  It
does not call a router or mutate active configuration.  A manifest binds the
input files, model identities, score transform, data splits, and selection
constraints so that a candidate threshold can be reviewed and reproduced.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analyzer import OfflineAnalyzer
from .scenarios.confidence import normalize_logprob, question_severity

ARTIFACT_SCHEMA_VERSION = "confidence-calibration-artifact/v1"
MANIFEST_SCHEMA_VERSION = "confidence-calibration/v1"
SPLITS = ("train", "calibration", "held_out")
DEFAULT_BIN_COUNT = 10


class ConfidenceCalibrationError(ValueError):
    """Raised when confidence calibration inputs are unsafe or incomplete."""


@dataclass(frozen=True)
class ConfidenceRecord:
    """Paired outcomes for one question and one routing decision."""

    question_id: str
    category: str
    confidence: float
    correct_small: bool
    correct_large: bool
    split: str

    @property
    def quadrant(self) -> str:
        if not self.correct_small and self.correct_large:
            return "uplift"
        if self.correct_small and not self.correct_large:
            return "regression"
        if self.correct_small and self.correct_large:
            return "both_correct"
        return "both_wrong"


def load_confidence_inputs(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, list[ConfidenceRecord]], dict[str, Any]]:
    """Load and validate the manifest, paired results, and source digests."""
    manifest = _load_json_object(manifest_path, "manifest")
    _validate_manifest(manifest, manifest_path)

    split_records: dict[str, list[ConfidenceRecord]] = {}
    source_digests: dict[str, Any] = {}
    seen: dict[str, str] = {}
    for split in SPLITS:
        split_spec = manifest["splits"][split]
        small_path = _resolve_input_path(manifest_path, split_spec["small_results"])
        large_path = _resolve_input_path(manifest_path, split_spec["large_results"])
        small_results = _load_result_list(small_path, "small")
        large_results = _load_result_list(large_path, "large")
        records = _pair_results(small_results, large_results, split)
        if not records:
            raise ConfidenceCalibrationError(
                f"{split} split must contain at least one paired result"
            )
        for record in records:
            previous_split = seen.get(record.question_id)
            if previous_split is not None:
                raise ConfidenceCalibrationError(
                    f"question_id {record.question_id!r} appears in both "
                    f"{previous_split!r} and {split!r}"
                )
            seen[record.question_id] = split
        split_records[split] = records
        source_digests[split] = {
            "small_results": {
                "path": str(split_spec["small_results"]),
                "sha256": _sha256_file(small_path),
            },
            "large_results": {
                "path": str(split_spec["large_results"]),
                "sha256": _sha256_file(large_path),
            },
        }

    return manifest, split_records, source_digests


def select_threshold(
    records: Sequence[ConfidenceRecord],
    objective: Mapping[str, Any],
) -> dict[str, Any]:
    """Select the best calibration candidate that satisfies hard constraints."""
    if not records:
        return {"status": "no_safe_threshold", "reason": "empty_calibration_set"}

    analyzer = OfflineAnalyzer(severity_fn=question_severity)
    result = analyzer.find_optimal_threshold(
        items=list(records),
        confidence_fn=lambda item: item.confidence,
        quadrant_fn=lambda item: item.quadrant,
        correct_small_fn=lambda item: item.correct_small,
        correct_large_fn=lambda item: item.correct_large,
        id_fn=lambda item: item.question_id,
    )
    max_escalation = float(objective["max_escalation_rate"])
    min_net_uplift = float(objective["min_net_uplift"])
    max_regression = objective.get("max_regression_rate")
    safe = []
    for candidate in result["candidates"]:
        threshold = float(candidate["threshold"])
        escalated = [record for record in records if record.confidence < threshold]
        escalation_rate = len(escalated) / len(records)
        regressions = sum(record.quadrant == "regression" for record in escalated)
        regression_rate = regressions / len(records)
        if threshold < 0.0 or threshold > 1.0:
            continue
        if escalation_rate > max_escalation:
            continue
        if candidate["net"] < min_net_uplift:
            continue
        if max_regression is not None and regression_rate > float(max_regression):
            continue
        safe.append(
            {
                **candidate,
                "_escalation_rate_exact": escalation_rate,
                "_regression_rate_exact": regression_rate,
            }
        )

    if not safe:
        return {
            "status": "no_safe_threshold",
            "reason": "no_candidate_satisfies_constraints",
            "candidate_count": len(result["candidates"]),
        }

    selected = max(
        safe,
        key=lambda candidate: (
            candidate["correct"],
            -candidate["_escalation_rate_exact"],
            -candidate["threshold"],
        ),
    )
    return {
        "status": "candidate",
        "threshold": float(selected["threshold"]),
        "accuracy": selected["correct"] / len(records),
        "escalation_rate": selected["_escalation_rate_exact"],
        "net_uplift": selected["net"],
        "regression_rate": selected["_regression_rate_exact"],
        "candidate_count": len(result["candidates"]),
        "safe_candidate_count": len(safe),
    }


def evaluate_threshold(
    records: Sequence[ConfidenceRecord],
    threshold: float,
    resources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a fixed threshold without selecting a new one."""
    if not 0.0 <= threshold <= 1.0:
        raise ConfidenceCalibrationError("threshold must be within [0, 1]")
    if not records:
        raise ConfidenceCalibrationError("cannot evaluate an empty split")

    escalated = [record for record in records if record.confidence < threshold]
    correct = sum(
        record.correct_large if record.confidence < threshold else record.correct_small
        for record in records
    )
    quadrants = Counter(record.quadrant for record in escalated)
    total = len(records)
    escalation_rate = len(escalated) / total
    metrics = {
        "n_items": total,
        "correct": correct,
        "accuracy": _rounded(correct / total),
        "accuracy_pct": round(100.0 * correct / total, 2),
        "escalated": len(escalated),
        "escalation_rate": _rounded(escalation_rate),
        "escalation_rate_pct": round(100.0 * escalation_rate, 2),
        "uplifts": quadrants["uplift"],
        "regressions": quadrants["regression"],
        "net_uplift": quadrants["uplift"] - quadrants["regression"],
        "coverage": 1.0,
        "abstention_rate": 0.0,
        "uncertainty": {
            "accuracy_95_ci": _wilson_interval(correct, total),
            "escalation_rate_95_ci": _wilson_interval(len(escalated), total),
        },
        "calibration": _calibration_metrics(records),
        "resources": _resource_metrics(total, len(escalated), resources),
        "quadrants": dict(sorted(quadrants.items())),
    }
    return metrics


def build_failure_slices(
    records: Sequence[ConfidenceRecord], threshold: float
) -> dict[str, Any]:
    """Return compact category and quadrant slices for review."""
    by_category: dict[str, list[ConfidenceRecord]] = defaultdict(list)
    for record in records:
        by_category[record.category].append(record)

    category_slices = {}
    for category in sorted(by_category):
        subset = by_category[category]
        metrics = evaluate_threshold(subset, threshold)
        category_slices[category] = {
            key: metrics[key]
            for key in (
                "n_items",
                "accuracy",
                "escalation_rate",
                "uplifts",
                "regressions",
                "net_uplift",
            )
        }
    return {
        "by_category": category_slices,
        "by_quadrant": dict(
            sorted(Counter(record.quadrant for record in records).items())
        ),
    }


def build_artifact(manifest_path: Path) -> dict[str, Any]:
    """Build a reviewable artifact without changing active router policy."""
    manifest, splits, source_digests = load_confidence_inputs(manifest_path)
    train = splits["train"]
    calibration = splits["calibration"]
    held_out = splits["held_out"]
    selection = select_threshold(calibration, manifest["objective"])
    policy = manifest["policy"]
    fallback = manifest["fallback"]

    baseline = None
    if "current_threshold" in policy:
        current_threshold = float(policy["current_threshold"])
        baseline = {
            "threshold": _rounded(current_threshold),
            "metrics": {
                split: evaluate_threshold(
                    splits[split], current_threshold, manifest.get("resources")
                )
                for split in SPLITS
            },
        }

    threshold = selection.get("threshold")
    if threshold is None:
        threshold = _fallback_threshold(policy, fallback)
    train_metrics = evaluate_threshold(train, threshold, manifest.get("resources"))
    calibration_metrics = evaluate_threshold(
        calibration, threshold, manifest.get("resources")
    )
    held_out_metrics = evaluate_threshold(
        held_out, threshold, manifest.get("resources")
    )
    candidate_diff = None
    if selection["status"] == "candidate":
        candidate_diff = {
            "path": "algorithm.confidence.threshold",
            "from": policy.get("current_threshold"),
            "to": _rounded(threshold),
        }

    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": selection["status"],
        "approval_state": manifest.get("approval_state", "pending_review"),
        "name": manifest["name"],
        "method": manifest["method"],
        "score_domain": manifest["score_domain"],
        "normalization": manifest["normalization"],
        "dataset": manifest["dataset"],
        "population": manifest["population"],
        "outcome": manifest["outcome"],
        "expected_impact": manifest["expected_impact"],
        "models": manifest["models"],
        "source": {
            "manifest_sha256": _sha256_file(manifest_path),
            "splits": source_digests,
        },
        "split_counts": {split: len(splits[split]) for split in SPLITS},
        "selection": {
            "split": "calibration",
            **selection,
        },
        "baseline": baseline,
        "fallback": {
            "on_no_safe_threshold": fallback["on_no_safe_threshold"],
            "effective_threshold": _rounded(threshold),
        },
        "objective": manifest["objective"],
        "rollback_identity": policy["rollback_identity"],
        "candidate_config_diff": candidate_diff,
        "collection": manifest.get("collection", {}),
        "metrics": {
            "train": train_metrics,
            "calibration": calibration_metrics,
            "held_out": held_out_metrics,
        },
        "failure_slices": {
            "calibration": build_failure_slices(calibration, threshold),
            "held_out": build_failure_slices(held_out, threshold),
        },
        "unsupported_regions": manifest.get(
            "unsupported_regions",
            ["confidence methods other than avg_logprob"],
        ),
    }
    artifact["artifact_id"] = _artifact_id(artifact)
    return artifact


def write_artifact(artifact: Mapping[str, Any], output_path: Path) -> None:
    """Write a deterministic JSON artifact; this is the only output mutation."""
    output_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _validate_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    required = {
        "schema_version",
        "name",
        "method",
        "score_domain",
        "normalization",
        "dataset",
        "population",
        "outcome",
        "expected_impact",
        "models",
        "splits",
        "objective",
        "fallback",
        "policy",
    }
    missing = sorted(field for field in required if field not in manifest)
    if missing:
        raise ConfidenceCalibrationError(
            f"{path} is missing required fields: {', '.join(missing)}"
        )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ConfidenceCalibrationError(
            f"{path} schema_version must be {MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest["method"] != "avg_logprob":
        raise ConfidenceCalibrationError("T9 only supports method='avg_logprob'")
    _validate_score_domain(manifest["score_domain"])
    _validate_normalization(manifest["normalization"])
    _validate_dataset(manifest["dataset"])
    _validate_description(manifest["population"], "population")
    _validate_description(manifest["outcome"], "outcome")
    _validate_description(manifest["expected_impact"], "expected_impact")
    _validate_models(manifest["models"])
    _validate_splits(manifest["splits"], path)
    _validate_objective(manifest["objective"])
    _validate_fallback(manifest["fallback"])
    _validate_policy(manifest["policy"], manifest["fallback"])
    approval_state = manifest.get("approval_state", "pending_review")
    if approval_state not in {"pending_review", "approved"}:
        raise ConfidenceCalibrationError(
            "approval_state must be 'pending_review' or 'approved'"
        )


def _validate_score_domain(domain: Any) -> None:
    if not isinstance(domain, Mapping):
        raise ConfidenceCalibrationError("score_domain must be an object")
    if domain.get("min") != 0.0 or domain.get("max") != 1.0:
        raise ConfidenceCalibrationError("avg_logprob score_domain must be [0.0, 1.0]")


def _validate_normalization(normalization: Any) -> None:
    if not isinstance(normalization, Mapping):
        raise ConfidenceCalibrationError("normalization must be an object")
    expected = {"type": "linear_clamped", "min_logprob": -3.0, "max_logprob": 0.0}
    if dict(normalization) != expected:
        raise ConfidenceCalibrationError(
            "normalization must match the runtime avg_logprob transform "
            "(linear_clamped, -3.0 to 0.0)"
        )


def _validate_dataset(dataset: Any) -> None:
    if not isinstance(dataset, Mapping):
        raise ConfidenceCalibrationError("dataset must be an object")
    for field in ("name", "version", "digest"):
        if not _non_empty_text(dataset.get(field)):
            raise ConfidenceCalibrationError(f"dataset.{field} is required")


def _validate_description(value: Any, field_name: str) -> None:
    if not _non_empty_text(value):
        raise ConfidenceCalibrationError(f"{field_name} must be a non-empty string")


def _validate_models(models: Any) -> None:
    if not isinstance(models, Mapping):
        raise ConfidenceCalibrationError("models must be an object")
    for role in ("small", "large"):
        model = models.get(role)
        if not isinstance(model, Mapping):
            raise ConfidenceCalibrationError(f"models.{role} must be an object")
        for field in ("id", "version"):
            if not _non_empty_text(model.get(field)):
                raise ConfidenceCalibrationError(f"models.{role}.{field} is required")


def _validate_splits(splits: Any, path: Path) -> None:
    if not isinstance(splits, Mapping):
        raise ConfidenceCalibrationError("splits must be an object")
    for split in SPLITS:
        spec = splits.get(split)
        if not isinstance(spec, Mapping):
            raise ConfidenceCalibrationError(f"splits.{split} must be an object")
        for role in ("small_results", "large_results"):
            value = spec.get(role)
            if not _non_empty_text(value):
                raise ConfidenceCalibrationError(f"splits.{split}.{role} is required")
            _resolve_input_path(path, value)


def _validate_objective(objective: Any) -> None:
    if not isinstance(objective, Mapping):
        raise ConfidenceCalibrationError("objective must be an object")
    if objective.get("primary_metric") != "accuracy":
        raise ConfidenceCalibrationError("objective.primary_metric must be 'accuracy'")
    max_escalation = _bounded_number(
        objective.get("max_escalation_rate"), "max_escalation_rate"
    )
    if max_escalation < 0.0 or max_escalation > 1.0:
        raise ConfidenceCalibrationError("max_escalation_rate must be within [0, 1]")
    _number(objective.get("min_net_uplift"), "min_net_uplift")
    if "max_regression_rate" in objective:
        max_regression = _bounded_number(
            objective["max_regression_rate"], "max_regression_rate"
        )
        if max_regression < 0.0 or max_regression > 1.0:
            raise ConfidenceCalibrationError(
                "max_regression_rate must be within [0, 1]"
            )


def _validate_fallback(fallback: Any) -> None:
    if not isinstance(fallback, Mapping):
        raise ConfidenceCalibrationError("fallback must be an object")
    allowed = {"retain_current", "avoid_escalation"}
    if fallback.get("on_no_safe_threshold") not in allowed:
        raise ConfidenceCalibrationError(
            "fallback.on_no_safe_threshold must be retain_current or avoid_escalation"
        )


def _validate_policy(policy: Any, fallback: Mapping[str, Any]) -> None:
    if not isinstance(policy, Mapping):
        raise ConfidenceCalibrationError("policy must be an object")
    if not _non_empty_text(policy.get("rollback_identity")):
        raise ConfidenceCalibrationError("policy.rollback_identity is required")
    if "current_threshold" in policy:
        threshold = _bounded_number(policy["current_threshold"], "current_threshold")
        if threshold < 0.0 or threshold > 1.0:
            raise ConfidenceCalibrationError("current_threshold must be within [0, 1]")
    if (
        fallback["on_no_safe_threshold"] == "retain_current"
        and "current_threshold" not in policy
    ):
        raise ConfidenceCalibrationError(
            "current_threshold is required when fallback retains the current policy"
        )


def _resolve_input_path(manifest_path: Path, raw_path: Any) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ConfidenceCalibrationError("input path must be a non-empty string")
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    if not path.is_file():
        raise ConfidenceCalibrationError(f"input file does not exist: {path}")
    return path


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfidenceCalibrationError(f"unable to read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ConfidenceCalibrationError(f"{label} must be a JSON object: {path}")
    return value


def _load_result_list(path: Path, role: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConfidenceCalibrationError(
            f"unable to read {role} results: {path}"
        ) from error
    if isinstance(value, dict):
        value = value.get("results")
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ConfidenceCalibrationError(f"{role} results must be a JSON array: {path}")
    return value


def _pair_results(
    small_results: Sequence[Mapping[str, Any]],
    large_results: Sequence[Mapping[str, Any]],
    split: str,
) -> list[ConfidenceRecord]:
    small = _index_results(small_results, "small")
    large = _index_results(large_results, "large")
    if set(small) != set(large):
        missing_large = sorted(set(small) - set(large))
        missing_small = sorted(set(large) - set(small))
        raise ConfidenceCalibrationError(
            f"{split} results are not paired; missing_large={missing_large[:5]}, "
            f"missing_small={missing_small[:5]}"
        )

    records = []
    for question_id in sorted(small):
        small_item = small[question_id]
        large_item = large[question_id]
        small_category = str(small_item.get("category") or "").strip()
        large_category = str(large_item.get("category") or "").strip()
        if small_category and large_category and small_category != large_category:
            raise ConfidenceCalibrationError(
                f"{split} item {question_id} has inconsistent categories"
            )
        category = small_category or large_category
        if not category:
            raise ConfidenceCalibrationError(
                f"{split} item {question_id} has no category"
            )
        confidence = normalize_logprob(
            _number(small_item.get("avg_logprob"), f"{question_id}.avg_logprob")
        )
        records.append(
            ConfidenceRecord(
                question_id=question_id,
                category=category,
                confidence=confidence,
                correct_small=_correctness(small_item, question_id, "small"),
                correct_large=_correctness(large_item, question_id, "large"),
                split=split,
            )
        )
    return records


def _index_results(
    results: Sequence[Mapping[str, Any]], role: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for item in results:
        question_id = str(item.get("question_id") or "").strip()
        if not question_id:
            raise ConfidenceCalibrationError(f"{role} result has no question_id")
        if question_id in indexed:
            raise ConfidenceCalibrationError(
                f"duplicate {role} question_id: {question_id}"
            )
        indexed[question_id] = item
    return indexed


def _correctness(item: Mapping[str, Any], question_id: str, role: str) -> bool:
    explicit = item.get("correct")
    if isinstance(explicit, bool):
        return explicit
    if "predicted" not in item or "correct_answer" not in item:
        raise ConfidenceCalibrationError(
            f"{role} result {question_id} needs correct or predicted/correct_answer"
        )
    return item["predicted"] == item["correct_answer"]


def _calibration_metrics(records: Sequence[ConfidenceRecord]) -> dict[str, Any]:
    predictions = [record.confidence for record in records]
    outcomes = [1.0 if record.correct_small else 0.0 for record in records]
    bins = _reliability_bins(predictions, outcomes, DEFAULT_BIN_COUNT)
    ece = sum(
        item["count"]
        / len(records)
        * abs(item["mean_prediction"] - item["observed_frequency"])
        for item in bins
        if item["count"]
    )
    brier = sum(
        (prediction - outcome) ** 2
        for prediction, outcome in zip(predictions, outcomes, strict=True)
    ) / len(records)
    return {
        "brier": _rounded(brier),
        "ece_10": _rounded(ece),
        "reliability_bins": bins,
    }


def _reliability_bins(
    predictions: Sequence[float], outcomes: Sequence[float], bin_count: int
) -> list[dict[str, Any]]:
    counts = [0] * bin_count
    prediction_totals = [0.0] * bin_count
    outcome_totals = [0.0] * bin_count
    for prediction, outcome in zip(predictions, outcomes, strict=True):
        index = min(int(prediction * bin_count), bin_count - 1)
        counts[index] += 1
        prediction_totals[index] += prediction
        outcome_totals[index] += outcome
    return [
        {
            "index": index,
            "lower": _rounded(index / bin_count),
            "upper": _rounded((index + 1) / bin_count),
            "count": counts[index],
            "mean_prediction": (
                _rounded(prediction_totals[index] / counts[index])
                if counts[index]
                else 0.0
            ),
            "observed_frequency": (
                _rounded(outcome_totals[index] / counts[index])
                if counts[index]
                else 0.0
            ),
        }
        for index in range(bin_count)
    ]


def _resource_metrics(
    total: int, escalated: int, resources: Mapping[str, Any] | None
) -> dict[str, Any]:
    if not resources:
        return {"available": False, "reason": "resource_costs_not_provided"}
    small = resources.get("small")
    large = resources.get("large")
    if not isinstance(small, Mapping) or not isinstance(large, Mapping):
        raise ConfidenceCalibrationError(
            "resources.small and resources.large are required"
        )
    small_cost = _number(small.get("cost"), "resources.small.cost")
    large_cost = _number(large.get("cost"), "resources.large.cost")
    small_latency = _number(small.get("latency_ms"), "resources.small.latency_ms")
    large_latency = _number(large.get("latency_ms"), "resources.large.latency_ms")
    return {
        "available": True,
        "total_cost": _rounded(total * small_cost + escalated * large_cost),
        "average_cost": _rounded(small_cost + escalated / total * large_cost),
        "expected_latency_ms": _rounded(
            small_latency + escalated / total * large_latency
        ),
    }


def _fallback_threshold(
    policy: Mapping[str, Any], fallback: Mapping[str, Any]
) -> float:
    if fallback["on_no_safe_threshold"] == "avoid_escalation":
        return 0.0
    return float(policy["current_threshold"])


def _wilson_interval(successes: int, total: int) -> list[float]:
    z = 1.96
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return [_rounded(max(0.0, center - margin)), _rounded(min(1.0, center + margin))]


def _artifact_id(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "artifact_id"}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfidenceCalibrationError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ConfidenceCalibrationError(f"{label} must be finite")
    return number


def _bounded_number(value: Any, label: str) -> float:
    number = _number(value, label)
    if number < 0.0 or number > 1.0:
        raise ConfidenceCalibrationError(f"{label} must be within [0, 1]")
    return number


def _rounded(value: float) -> float:
    return round(float(value), 8)
