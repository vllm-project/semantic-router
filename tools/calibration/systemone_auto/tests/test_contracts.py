"""Fail-closed typed losses, leakage guards and deployable feature parity."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from systemone_auto.artifacts import canonical, digest
from systemone_auto.dataset import synthetic_edges, validate_dataset
from systemone_auto.metrics import FEATURE_NAMES, evaluate_response, features, summarize
from systemone_auto.policy import (
    calibrated_quality,
    fit_calibrator,
    predict,
    ridge,
    terminal_result,
)


def response(row):
    answers = {}
    for key, question in row["request"]["questions"].items():
        label = row["labels"][key]["label"]
        kind = question["type"]
        value = {"type": kind, "input_coverage": "complete"}
        if kind == "noul":
            value["noul"] = float(label == "true")
        else:
            keys = (
                list(question["criteria"])
                if kind == "choice"
                else [str(i) for i in range(len(question["criteria"]))]
            )
            value["probabilities"] = {option: float(option == label) for option in keys}
            value["choice" if kind == "choice" else "score"] = (
                label if kind == "choice" else float(label)
            )
        answers[key] = value
    return {"answers": answers}


def test_complete_bundle_perfect_losses_and_native_features():
    row = synthetic_edges()[0]
    result = evaluate_response(row, response(row))
    assert result["correct"] and result["loss"] == 0
    vector = features(row, result)
    assert vector[:5] == [1, 1, 1, 0, 1]
    assert vector[5] == math.log1p(len(canonical(row["request"]["state"]).encode()))
    assert vector[7:10] == [1 / 3] * 3
    assert vector[10] == 0


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["answers"].pop("exceeds"),
        lambda value: value["answers"]["comparison"].pop("input_coverage"),
        lambda value: value["answers"]["exceeds"].update(noul=float("nan")),
    ],
)
def test_invalid_question_is_not_dropped_from_loss_or_feature_denominator(mutate):
    row = synthetic_edges()[0]
    raw = response(row)
    mutate(raw)
    result = evaluate_response(row, raw)
    assert not result["valid"] and result["loss"] == 1
    assert len(result["questions"]) == 3
    vector = features(row, result)
    assert vector[1] == 0 and vector[2] == pytest.approx(2 / 3)
    assert vector[3] == 1 and vector[4] == 0 and vector[10] == 1


@pytest.mark.parametrize(
    "distribution",
    [None, {"a": 0.2, "b": 0.8}, {"below": 0.9, "equal": 0.9, "above": 0.1}],
)
def test_missing_evidence_does_not_destroy_valid_point_answer(distribution):
    row = synthetic_edges()[0]
    raw = response(row)
    raw["answers"]["comparison"]["probabilities"] = distribution
    result = evaluate_response(row, raw)
    assert result["valid"] and result["correct"]
    assert result["questions"]["comparison"]["brier"] is None
    vector = features(row, result)
    assert vector[1:5] == pytest.approx([0, 2 / 3, 1, 0])
    assert vector[10] == 1


def test_failed_questions_remain_in_per_type_denominators():
    row = synthetic_edges()[0]
    good = evaluate_response(row, response(row))
    failed = evaluate_response(row, {})
    report = summarize([good, failed])
    assert report["bundle_error"] == 0.5
    for kind in ("choice", "noul", "score"):
        assert report["per_type"][kind]["questions"] == 2
        assert report["per_type"][kind]["valid_questions"] == 1
        assert report["per_type"][kind]["accuracy"] == 0.5


def test_shared_go_python_feature_golden():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/native-features.json").read_text()
    )
    assert fixture["feature_names"] == FEATURE_NAMES
    for case in fixture["cases"]:
        row = {"request": case["request"], "labels": case["labels"]}
        observed = evaluate_response(row, case["response"])
        assert observed["valid"] is case["valid"], case["name"]
        assert features(row, observed) == pytest.approx(
            case["features"], abs=1e-12
        ), case["name"]


def test_chat_points_have_no_probability_metrics_even_if_self_reported():
    row = synthetic_edges()[0]
    raw = response(row)
    raw["answers"]["exceeds"]["noul"] = False
    result = evaluate_response(row, raw, native=False)
    assert result["correct"]
    assert all(
        value["brier"] is None and value["top_probability"] is None
        for value in result["questions"].values()
    )


def test_noul_feature_confidence_uses_both_outcomes():
    row = synthetic_edges()[0]
    row["request"]["questions"] = {"exceeds": row["request"]["questions"]["exceeds"]}
    row["labels"] = {"exceeds": row["labels"]["exceeds"]}
    raw = response(row)
    raw["answers"]["exceeds"]["noul"] = 0.1
    values = features(row, evaluate_response(row, raw))
    assert values[1:3] == [0.9, 0.9]
    assert values[4] == pytest.approx(0.8)


def test_group_leak_and_changed_payload_digest_rejected():
    rows = synthetic_edges()[:2]
    data = {
        "schema_version": "systemone-pilot/v1",
        "records": rows,
        "records_sha256": digest(rows),
    }
    validate_dataset(data)
    rows[1]["split"] = "train"
    with pytest.raises(ValueError, match="digest"):
        validate_dataset(data)
    data["records_sha256"] = digest(rows)
    with pytest.raises(ValueError, match="leaks"):
        validate_dataset(data)


def test_quality_calibration_pools_ties_and_is_not_gain():
    calibration = fit_calibrator([(0.2, True), (0.2, False), (0.5, False), (0.9, True)])
    assert calibrated_quality(calibration, 0.1) == pytest.approx(1 / 3)
    assert calibrated_quality(calibration, 0.8) == 1
    x = [[1, float(i % 2), *([0] * 9)] for i in range(100)]
    weights = ridge(x, [1 if i % 2 else -1 for i in range(100)])
    assert predict(weights, x[1]) > 0.8
    assert predict(weights, x[0]) < -0.8


def test_failed_second_call_preserves_last_valid_bundle():
    row = synthetic_edges()[0]
    valid = evaluate_response(row, response(row))
    failed = evaluate_response(row, {})
    assert terminal_result(valid, failed) == valid
    assert not terminal_result(failed, failed)["valid"]
    wrong = {**valid, "correct": False, "loss": 1.0}
    assert terminal_result(valid, wrong)["loss"] == 1.0


def test_point_only_upgrade_is_not_deliverable_but_direct_answer_remains_valid():
    row = synthetic_edges()[0]
    valid = evaluate_response(row, response(row))
    raw = response(row)
    raw["answers"]["comparison"].pop("probabilities")
    point_only = evaluate_response(row, raw)
    assert point_only["valid"] and point_only["correct"]
    assert terminal_result(valid, point_only) == valid
    assert terminal_result(point_only, valid) == valid
    rejected = terminal_result(point_only, point_only)
    assert not rejected["valid"] and rejected["loss"] == 1 and not rejected["correct"]
