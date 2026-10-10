"""Primary quality objective and honest runtime cost evidence."""

from __future__ import annotations

import copy

import pytest
from systemone_auto.policy import fit_calibrator
from systemone_auto.replay import execute, select_setting
from systemone_auto.timing import native_timing


def test_bundle_correctness_has_priority_over_mean_typed_loss():
    rows = [{"id": str(i), "group_id": str(i)} for i in range(2)]

    def observed(correct, loss):
        question = {
            "valid": True,
            "correct": correct,
            "loss": loss,
            "type": "score",
            "top_probability": 0.5,
            "brier": 0.5,
            "nll": 0.5,
            "probability_correct": correct,
        }
        return {
            "result": {
                "valid": True,
                "correct": correct,
                "loss": loss,
                "questions": {"q": question},
            },
            "features": [1, 0.5],
            "policy_cost_ms": 1,
            "client_elapsed_ms": 1,
        }

    matrix = {
        row["id"]: {
            "kai": observed(False, 1),
            "risky": observed(index == 0, 0.1),
            "safe": observed(True, 0.2),
        }
        for index, row in enumerate(rows)
    }
    settings = [
        {"kind": "cascade", "action": action, "risk_threshold": -1}
        for action in ("risky", "safe")
    ]
    selected = select_setting(
        rows, matrix, "kai", settings, 2, {}, fit_calibrator([(0.5, True)])
    )
    assert selected["action"] == "safe"


def test_compute_time_requires_pinned_model_and_complete_profile():
    target = {"model_id": "vllm-sr/Decision-2.0-Kai-0.6B", "revision": "a" * 40}
    response = {
        "model": target["model_id"],
        "meta": {
            "revision": target["revision"],
            "model_sha256": "b" * 64,
            "engine": "native",
            "profile": "exact",
            "numerics": "exact",
            "accelerator": "rocm",
            "compute_ms": 18.379,
        },
    }
    cost, identity = native_timing(response, target)
    assert cost == 18.379 and identity["revision"] == target["revision"]
    for field, value in (
        ("compute_ms", None),
        ("compute_ms", 0),
        ("compute_ms", True),
        ("compute_ms", float("nan")),
        ("revision", "c" * 40),
        ("profile", ""),
        ("model_sha256", "not-a-digest"),
    ):
        changed = copy.deepcopy(response)
        changed["meta"][field] = value
        with pytest.raises(ValueError):
            native_timing(changed, target)
    with pytest.raises(ValueError, match="model/revision"):
        native_timing({**response, "model": "wrong-model"}, target)


def test_rejected_cascade_first_answer_is_not_upgrade_failure_fallback():
    question = {
        "valid": True,
        "correct": True,
        "loss": 0.0,
        "type": "choice",
        "top_probability": 0.6,
    }
    accepted = {
        "valid": True,
        "correct": True,
        "loss": 0.0,
        "questions": {"task": question},
    }
    failed = {
        "valid": False,
        "correct": False,
        "loss": 1.0,
        "questions": {"task": {**question, "valid": False, "top_probability": None}},
    }
    matrix = {
        "sample": {
            name: {
                "result": result,
                "features": [1.0] + [0.0] * 10,
                "policy_cost_ms": 1,
                "client_elapsed_ms": 1,
            }
            for name, result in (("kai", accepted), ("strong", failed))
        }
    }
    rows = [{"id": "sample", "group_id": "sample"}]
    policy = {
        "stop_value": 0.0,
        "heads": {
            "kai": {
                "strong": {"weights": [1.0] + [0.0] * 10, "training_mean_cost_ms": 1.0}
            }
        },
    }
    calibration = fit_calibrator([(0.5, True)])
    cascade = execute(
        rows,
        matrix,
        "kai",
        {"kind": "cascade", "action": "strong", "risk_threshold": -1.0},
        policy,
        calibration,
    )
    learned = execute(
        rows, matrix, "kai", {"kind": "policy", "cost_weight": 0}, policy, calibration
    )
    assert cascade[0]["calls"] == learned[0]["calls"] == ["kai", "strong"]
    assert not cascade[0]["result"]["valid"] and cascade[0]["result"]["loss"] == 1
    assert learned[0]["result"] == accepted
