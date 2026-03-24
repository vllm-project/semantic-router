from __future__ import annotations

from meta_routing_policy_support import (
    ARTIFACT_VERSION,
    FEATURE_SCHEMA_NAME,
    FEATURE_SCHEMA_VERSION,
    validate_policy_artifact,
)


def valid_artifact() -> dict:
    return {
        "version": ARTIFACT_VERSION,
        "artifact_id": "artifact-123",
        "provider": {
            "kind": "calibrated_policy",
            "name": "shadow-calibration",
            "version": "2026-03-24",
        },
        "feature_schema": {
            "name": FEATURE_SCHEMA_NAME,
            "version": FEATURE_SCHEMA_VERSION,
        },
        "rollout": {
            "min_replay_records": 100,
            "min_trigger_precision": 0.8,
            "min_action_precision": 0.75,
            "min_overturn_gain": 0.05,
            "max_p95_latency_delta_ms": 200.0,
        },
        "evaluation": {
            "replay_records": 120,
            "trigger_precision": 0.9,
            "action_precision": 0.82,
            "overturn_gain": 0.08,
            "p95_latency_delta_ms": 90.0,
            "accepted": True,
        },
        "policy": {
            "trigger_policy": {
                "decision_margin_below": 0.18,
            },
            "allowed_actions": [
                {
                    "type": "rerun_signal_families",
                    "signal_families": ["embedding"],
                }
            ],
        },
    }


def test_validate_policy_artifact_accepts_valid_artifact() -> None:
    assert validate_policy_artifact(valid_artifact()) == []


def test_validate_policy_artifact_rejects_threshold_failure() -> None:
    artifact = valid_artifact()
    artifact["evaluation"]["accepted"] = True
    artifact["evaluation"]["trigger_precision"] = 0.4
    errors = validate_policy_artifact(artifact)
    assert errors
    assert "trigger_precision" in errors[0]
