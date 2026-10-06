"""Calibrate one signal family onto the declared label_correctness/v1 scale.

A calibrated score is the probability that the label a signal matched is the
request's true label. The builder fits an isotonic map on a calibration split of
recorded predictions and reports it on a disjoint held-out split. It reuses the
manifest checks, reliability bins and artifact identity of the confidence
calibration and never touches a router.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from .confidence_calibration import (
    DEFAULT_BIN_COUNT,
    ConfidenceCalibrationError,
    _artifact_id,
    _load_json_object,
    _load_result_list,
    _reliability_bins,
    _resolve_input_path,
    _rounded,
    _sha256_file,
    _validate_dataset,
    _validate_description,
    _wilson_interval,
    write_artifact,
)

MANIFEST_SCHEMA_VERSION = "signal-calibration/v1"
ARTIFACT_SCHEMA_VERSION = "signal-calibration-artifact/v1"
SCALE = "label_correctness/v1"
SPLITS = ("calibration", "held_out")
MIN_LABELS = 2
SHA256_HEX_LENGTH = 64
# At or above this threshold only the top label can match.
MIN_OPERATING_THRESHOLD = 0.5


def build_signal_artifact(manifest_path: Path) -> dict[str, Any]:
    """Fit on calibration, report on held_out, and return a reviewable artifact."""
    manifest = _load_json_object(manifest_path, "manifest")
    _validate_manifest(manifest, manifest_path)
    labels = manifest["model"]["labels"]
    splits: dict[str, list[tuple[str, str, float, bool]]] = {}
    digests: dict[str, Any] = {}
    seen: set[str] = set()
    for split in SPLITS:
        path = _resolve_input_path(manifest_path, manifest["splits"][split])
        rows = _load_rows(_load_result_list(path, split), labels, split, seen)
        splits[split] = rows
        digests[split] = {
            "path": manifest["splits"][split],
            "sha256": _sha256_file(path),
        }

    knots = fit_isotonic([(row[2], row[3]) for row in splits["calibration"]])
    threshold = float(manifest["operating_threshold"])
    held_out = splits["held_out"]
    metrics = {
        split: {
            "raw": _metrics(splits[split], lambda score: score),
            "calibrated": _metrics(splits[split], lambda score: apply(knots, score)),
        }
        for split in SPLITS
    }
    improved = (
        metrics["held_out"]["calibrated"]["brier"]
        <= metrics["held_out"]["raw"]["brier"]
    )
    present = {row[1] for row in held_out}
    artifact = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "calibrated" if improved else "no_improvement",
        "name": manifest["name"],
        "family": manifest["family"],
        "scale": SCALE,
        "dataset": manifest["dataset"],
        "population": manifest["population"],
        "outcome": manifest["outcome"],
        "model": manifest["model"],
        "source": {
            "manifest_sha256": _sha256_file(manifest_path),
            "splits": digests,
        },
        "split_counts": {split: len(splits[split]) for split in SPLITS},
        "mapping": {"method": "isotonic", "fitted_on": "calibration", "knots": knots},
        "metrics": metrics,
        "operating_point": _operating_point(held_out, knots, threshold),
        "failure_slices": _category_slices(held_out, knots, threshold),
        "unsupported_regions": [
            f"{label}: no held-out rows" for label in labels if label not in present
        ],
        "rollback_identity": manifest["policy"]["rollback_identity"],
    }
    artifact["artifact_id"] = _artifact_id(artifact)
    return artifact


def fit_isotonic(pairs: Sequence[tuple[float, bool]]) -> list[list[float]]:
    """Pool adjacent violators over scores, returning strictly rising knots.

    Each knot is a pooled block's mean score and observed accuracy. Equal values
    are pooled too, so linear interpolation between knots never flattens two
    different scores into a tie.
    """
    totals: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for score, correct in pairs:
        totals[score][0] += 1.0
        totals[score][1] += 1.0 if correct else 0.0
    blocks: list[list[float]] = []  # weight, score sum, outcome sum
    for score in sorted(totals):
        weight, hits = totals[score]
        blocks.append([weight, score * weight, hits])
        while len(blocks) > 1 and (
            blocks[-2][2] / blocks[-2][0] >= blocks[-1][2] / blocks[-1][0]
        ):
            last = blocks.pop()
            for index in range(3):
                blocks[-1][index] += last[index]
    knots = [[block[1] / block[0], block[2] / block[0]] for block in blocks]
    if len(knots) == 1:
        low, high = min(totals), max(totals)
        if low == high:
            raise ConfidenceCalibrationError(
                "calibration split needs at least two distinct scores"
            )
        knots = [[low, knots[0][1]], [high, knots[0][1]]]
    return knots


def apply(knots: Sequence[Sequence[float]], score: float) -> float:
    """Interpolate between knots and clamp outside them, as the router does."""
    if score <= knots[0][0]:
        return knots[0][1]
    if score >= knots[-1][0]:
        return knots[-1][1]
    for low, high in pairwise(knots):
        if score <= high[0]:
            return low[1] + (score - low[0]) * (high[1] - low[1]) / (high[0] - low[0])
    return knots[-1][1]


def _metrics(rows: Sequence[tuple[str, str, float, bool]], transform) -> dict[str, Any]:
    predictions = [transform(row[2]) for row in rows]
    outcomes = [1.0 if row[3] else 0.0 for row in rows]
    bins = _reliability_bins(predictions, outcomes, DEFAULT_BIN_COUNT)
    ece = sum(
        item["count"]
        / len(rows)
        * abs(item["mean_prediction"] - item["observed_frequency"])
        for item in bins
        if item["count"]
    )
    brier = sum(
        (prediction - outcome) ** 2
        for prediction, outcome in zip(predictions, outcomes, strict=True)
    ) / len(rows)
    return {
        "n_items": len(rows),
        "accuracy": _rounded(sum(outcomes) / len(rows)),
        "mean_prediction": _rounded(sum(predictions) / len(rows)),
        "brier": _rounded(brier),
        "ece_10": _rounded(ece),
        "reliability_bins": bins,
    }


def _operating_point(
    rows: Sequence[tuple[str, str, float, bool]],
    knots: Sequence[Sequence[float]],
    threshold: float,
) -> dict[str, Any]:
    matched = [row for row in rows if row[2] >= threshold]
    correct = sum(row[3] for row in matched)
    point = {
        "threshold": threshold,
        "calibrated_threshold": _rounded(apply(knots, threshold)),
        "coverage": _rounded(len(matched) / len(rows)),
        "coverage_95_ci": _wilson_interval(len(matched), len(rows)),
        "matched": len(matched),
    }
    if matched:
        point["accuracy_when_matched"] = _rounded(correct / len(matched))
        point["accuracy_95_ci"] = _wilson_interval(correct, len(matched))
    return point


def _category_slices(
    rows: Sequence[tuple[str, str, float, bool]],
    knots: Sequence[Sequence[float]],
    threshold: float,
) -> dict[str, Any]:
    by_category: dict[str, list[tuple[str, str, float, bool]]] = defaultdict(list)
    for row in rows:
        by_category[row[1]].append(row)
    slices = {}
    for category in sorted(by_category):
        subset = by_category[category]
        calibrated = _metrics(subset, lambda score: apply(knots, score))
        slices[category] = {
            "n_items": len(subset),
            "accuracy": calibrated["accuracy"],
            "mean_calibrated": calibrated["mean_prediction"],
            "brier_calibrated": calibrated["brier"],
            "coverage": _rounded(
                sum(row[2] >= threshold for row in subset) / len(subset)
            ),
        }
    return {"by_category": slices}


def _load_rows(
    items: Sequence[Mapping[str, Any]],
    labels: Sequence[str],
    split: str,
    seen: set[str],
) -> list[tuple[str, str, float, bool]]:
    rows = []
    for item in items:
        row_id = str(item.get("id") or "").strip()
        label = item.get("label")
        category = str(item.get("category") or "").strip()
        score = item.get("score")
        if not row_id or row_id in seen:
            raise ConfidenceCalibrationError(
                f"{split} row id {row_id!r} is empty or repeated"
            )
        if label not in labels or not category:
            raise ConfidenceCalibrationError(
                f"{split} row {row_id} needs a known label and a category"
            )
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ConfidenceCalibrationError(
                f"{split} row {row_id} score must be numeric"
            )
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ConfidenceCalibrationError(
                f"{split} row {row_id} score must lie in [0, 1]"
            )
        seen.add(row_id)
        rows.append((row_id, category, float(score), label == category))
    if not rows:
        raise ConfidenceCalibrationError(f"{split} split must contain at least one row")
    return rows


def _validate_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    required = (
        "schema_version",
        "name",
        "family",
        "scale",
        "method",
        "dataset",
        "population",
        "outcome",
        "model",
        "operating_threshold",
        "splits",
        "policy",
    )
    missing = [field for field in required if field not in manifest]
    if missing:
        raise ConfidenceCalibrationError(
            f"{path} is missing required fields: {', '.join(missing)}"
        )
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ConfidenceCalibrationError(
            f"schema_version must be {MANIFEST_SCHEMA_VERSION!r}"
        )
    if manifest["scale"] != SCALE or manifest["method"] != "isotonic":
        raise ConfidenceCalibrationError(
            f"only isotonic calibration onto {SCALE} is supported"
        )
    if manifest["family"] != "domain":
        raise ConfidenceCalibrationError("only the domain family is calibrated so far")
    _validate_dataset(manifest["dataset"])
    _validate_description(manifest["name"], "name")
    _validate_description(manifest["population"], "population")
    _validate_description(manifest["outcome"], "outcome")
    model = manifest["model"]
    if not isinstance(model, Mapping) or not all(
        isinstance(model.get(field), str) and model[field].strip()
        for field in ("id", "revision")
    ):
        raise ConfidenceCalibrationError("model.id and model.revision are required")
    labels = model.get("labels")
    if (
        not isinstance(labels, list)
        or len(labels) < MIN_LABELS
        or len(set(labels)) != len(labels)
    ):
        raise ConfidenceCalibrationError(
            "model.labels must list at least two distinct labels"
        )
    identity = model.get("model_sha256")
    if (
        not isinstance(identity, str)
        or len(identity) != SHA256_HEX_LENGTH
        or identity.strip("0123456789abcdef")
    ):
        raise ConfidenceCalibrationError(
            "model.model_sha256 must be the runtime's model identity, 64 lowercase hex"
        )
    threshold = manifest["operating_threshold"]
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not MIN_OPERATING_THRESHOLD <= threshold <= 1.0
    ):
        raise ConfidenceCalibrationError("operating_threshold must lie in [0.5, 1]")
    splits = manifest["splits"]
    if not isinstance(splits, Mapping) or set(splits) != set(SPLITS):
        raise ConfidenceCalibrationError(
            "splits must name exactly calibration and held_out"
        )
    _validate_description(
        manifest["policy"].get("rollback_identity"), "policy.rollback_identity"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = build_signal_artifact(args.manifest)
    write_artifact(artifact, args.output)
    print(
        f"{artifact['status']}: held-out brier "
        f"{artifact['metrics']['held_out']['raw']['brier']} -> "
        f"{artifact['metrics']['held_out']['calibrated']['brier']}, "
        f"ece_10 {artifact['metrics']['held_out']['raw']['ece_10']} -> "
        f"{artifact['metrics']['held_out']['calibrated']['ece_10']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
