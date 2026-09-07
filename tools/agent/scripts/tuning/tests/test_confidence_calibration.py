"""Tests for the manifest-backed offline confidence calibration workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuning.confidence_calibration import (
    ConfidenceCalibrationError,
    ConfidenceRecord,
    build_artifact,
    evaluate_threshold,
    load_confidence_inputs,
)

FIXTURE_MANIFEST = (
    Path(__file__).parent / "fixtures" / "confidence_calibration_v1" / "manifest.json"
)


def _avg_logprob(confidence: float) -> float:
    return 3.0 * confidence - 3.0


def _write_split(
    root: Path,
    split: str,
    rows: list[tuple[str, float, bool, bool]],
) -> None:
    small = []
    large = []
    for question_id, confidence, correct_small, correct_large in rows:
        small.append(
            {
                "question_id": question_id,
                "category": "math",
                "avg_logprob": _avg_logprob(confidence),
                "correct": correct_small,
            }
        )
        large.append(
            {
                "question_id": question_id,
                "category": "math",
                "predicted": "right" if correct_large else "wrong",
                "correct_answer": "right",
            }
        )
    (root / f"small-{split}.json").write_text(json.dumps(small), encoding="utf-8")
    (root / f"large-{split}.json").write_text(json.dumps(large), encoding="utf-8")


def _write_manifest(
    root: Path,
    *,
    calibration: list[tuple[str, float, bool, bool]],
    held_out: list[tuple[str, float, bool, bool]],
    train: list[tuple[str, float, bool, bool]] | None = None,
    max_escalation_rate: float = 0.5,
    min_net_uplift: int = 0,
) -> Path:
    _write_split(root, "train", train or [("train-1", 0.5, True, True)])
    _write_split(root, "calibration", calibration)
    _write_split(root, "held-out", held_out)
    manifest = {
        "schema_version": "confidence-calibration/v1",
        "name": "fixture-confidence-calibration",
        "method": "avg_logprob",
        "score_domain": {"min": 0.0, "max": 1.0},
        "normalization": {
            "type": "linear_clamped",
            "min_logprob": -3.0,
            "max_logprob": 0.0,
        },
        "dataset": {
            "name": "fixture",
            "version": "v1",
            "digest": "sha256:fixture",
        },
        "population": "Fixed paired fixture questions",
        "outcome": "Whether the model answer matches the answer key",
        "expected_impact": "Improve escalation quality within the declared budget",
        "models": {
            "small": {"id": "small-fixture", "version": "v1"},
            "large": {"id": "large-fixture", "version": "v1"},
        },
        "splits": {
            "train": {
                "small_results": "small-train.json",
                "large_results": "large-train.json",
            },
            "calibration": {
                "small_results": "small-calibration.json",
                "large_results": "large-calibration.json",
            },
            "held_out": {
                "small_results": "small-held-out.json",
                "large_results": "large-held-out.json",
            },
        },
        "objective": {
            "primary_metric": "accuracy",
            "max_escalation_rate": max_escalation_rate,
            "min_net_uplift": min_net_uplift,
        },
        "fallback": {"on_no_safe_threshold": "retain_current"},
        "policy": {
            "current_threshold": 0.72,
            "rollback_identity": "threshold-0.72",
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def test_evaluate_threshold_reports_calibration_and_policy_metrics():
    records = [
        ConfidenceRecord("q1", "math", 0.9, True, True, "held_out"),
        ConfidenceRecord("q2", "math", 0.8, False, True, "held_out"),
        ConfidenceRecord("q3", "math", 0.2, True, True, "held_out"),
    ]

    metrics = evaluate_threshold(records, 0.85)

    assert metrics["accuracy"] == 1.0
    assert metrics["escalation_rate"] == pytest.approx(2 / 3)
    assert metrics["uplifts"] == 1
    assert metrics["regressions"] == 0
    assert metrics["calibration"]["brier"] == pytest.approx(0.43)
    assert metrics["calibration"]["ece_10"] == pytest.approx(17 / 30)
    assert metrics["uncertainty"]["accuracy_95_ci"] == pytest.approx(
        [0.4385, 1.0], abs=1e-4
    )


def test_artifact_selects_on_calibration_and_reports_held_out(tmp_path: Path):
    manifest_path = _write_manifest(
        tmp_path,
        calibration=[
            ("c1", 0.2, False, True),
            ("c2", 0.3, False, True),
            ("c3", 0.8, True, True),
            ("c4", 0.9, True, True),
        ],
        held_out=[
            ("h1", 0.2, True, False),
            ("h2", 0.3, True, False),
            ("h3", 0.8, True, True),
            ("h4", 0.9, True, True),
        ],
    )

    artifact = build_artifact(manifest_path)

    assert artifact["status"] == "candidate"
    assert artifact["selection"]["split"] == "calibration"
    assert artifact["selection"]["threshold"] == pytest.approx(0.55)
    assert artifact["baseline"]["threshold"] == 0.72
    assert artifact["baseline"]["metrics"]["held_out"]["n_items"] == 4
    assert artifact["metrics"]["train"]["n_items"] == 1
    assert artifact["metrics"]["calibration"]["accuracy"] == 1.0
    assert artifact["metrics"]["held_out"]["accuracy"] == 0.5
    assert artifact["candidate_config_diff"]["path"] == (
        "algorithm.confidence.threshold"
    )
    assert artifact["artifact_id"].startswith("sha256:")


def test_no_safe_threshold_uses_declared_fallback(tmp_path: Path):
    manifest_path = _write_manifest(
        tmp_path,
        calibration=[("c1", 0.2, False, True), ("c2", 0.8, True, True)],
        held_out=[("h1", 0.2, True, False), ("h2", 0.8, True, True)],
        max_escalation_rate=0.0,
        min_net_uplift=1,
    )

    artifact = build_artifact(manifest_path)

    assert artifact["status"] == "no_safe_threshold"
    assert artifact["selection"]["reason"] == "no_candidate_satisfies_constraints"
    assert artifact["fallback"]["effective_threshold"] == 0.72
    assert artifact["candidate_config_diff"] is None


def test_selection_uses_exact_rates_for_hard_constraints(tmp_path: Path):
    manifest_path = _write_manifest(
        tmp_path,
        calibration=[
            ("c1", 0.2, False, True),
            ("c2", 0.8, True, True),
            ("c3", 0.9, True, True),
        ],
        held_out=[
            ("h1", 0.2, False, True),
            ("h2", 0.8, True, True),
            ("h3", 0.9, True, True),
        ],
        max_escalation_rate=0.333,
        min_net_uplift=1,
    )

    artifact = build_artifact(manifest_path)

    assert artifact["status"] == "no_safe_threshold"
    assert artifact["selection"]["reason"] == "no_candidate_satisfies_constraints"


def test_manifest_rejects_question_id_overlap(tmp_path: Path):
    manifest_path = _write_manifest(
        tmp_path,
        train=[("same", 0.2, True, True)],
        calibration=[("same", 0.3, False, True)],
        held_out=[("held", 0.8, True, True)],
    )

    with pytest.raises(ConfidenceCalibrationError, match="appears in both"):
        load_confidence_inputs(manifest_path)


def test_artifact_id_is_deterministic(tmp_path: Path):
    manifest_path = _write_manifest(
        tmp_path,
        calibration=[("c1", 0.2, False, True), ("c2", 0.8, True, True)],
        held_out=[("h1", 0.2, False, True), ("h2", 0.8, True, True)],
    )

    first = build_artifact(manifest_path)
    second = build_artifact(manifest_path)

    assert first == second


def test_checked_in_fixture_rebuilds_without_network():
    first = build_artifact(FIXTURE_MANIFEST)
    second = build_artifact(FIXTURE_MANIFEST)

    assert first == second
    assert first["artifact_schema_version"] == "confidence-calibration-artifact/v1"
    assert first["split_counts"] == {
        "train": 1,
        "calibration": 4,
        "held_out": 4,
    }
    assert first["selection"]["split"] == "calibration"
    assert first["selection"]["threshold"] == pytest.approx(0.55)
    assert first["collection"]["mode"] == "checked-in-fixture"
