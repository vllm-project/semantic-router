from __future__ import annotations

import pytest
from cli.evaluation.evidence import ExecutionRecord
from pydantic import ValidationError


def _base(track_id: str, ceiling: str) -> dict[str, object]:
    return {
        "id": f"{track_id}-record",
        "track_id": track_id,
        "case_id": "case-1",
        "attempt_id": f"{track_id}-attempt",
        "status": "succeeded",
        "evidence_kind": f"normalized-suite-replay.v1;ceiling={ceiling}",
    }


def test_normalized_routing_requires_consistent_typed_decision_facts() -> None:
    record = ExecutionRecord.model_validate(
        {
            **_base("routing", "E4"),
            "selected_arm_id": "arm-a",
            "selection_status": "selected",
            "selection_method": "normalized-suite-replay.v1",
            "success": True,
            "fallback": False,
            "quality": 1.0,
        }
    )
    assert record.selected_arm_id == "arm-a"

    with pytest.raises(ValidationError, match="typed decision facts"):
        ExecutionRecord.model_validate(
            record.model_dump(exclude={"fallback"}, exclude_none=True)
        )
    with pytest.raises(ValidationError, match="agree with success"):
        ExecutionRecord.model_validate(
            record.model_dump(exclude_none=True) | {"status": "failed"}
        )


def test_normalized_agentic_requires_complete_trajectory_counters() -> None:
    payload = _base("agentic", "E5") | {
        "success": True,
        "quality": 1.0,
        "trajectory_steps": 3,
        "tool_calls": 2,
        "invalid_tool_calls": 1,
        "privacy_violations": 0,
    }
    assert ExecutionRecord.model_validate(payload).trajectory_steps == 3

    with pytest.raises(ValidationError, match="typed trajectory facts"):
        ExecutionRecord.model_validate(payload | {"privacy_violations": None})
    with pytest.raises(ValidationError, match="cannot exceed"):
        ExecutionRecord.model_validate(payload | {"invalid_tool_calls": 3})


def test_normalized_preference_distinguishes_offline_and_e5_propensity() -> None:
    payload = _base("preference", "E4") | {
        "selected_arm_id": "action-a",
        "success": True,
        "quality": 1.0,
        "preference_match": True,
    }
    assert ExecutionRecord.model_validate(payload).behavior_propensity is None

    with pytest.raises(ValidationError, match="requires propensity"):
        ExecutionRecord.model_validate(
            payload | {"evidence_kind": "normalized-suite-replay.v1;ceiling=E5"}
        )
    assert (
        ExecutionRecord.model_validate(
            payload
            | {
                "evidence_kind": "normalized-suite-replay.v1;ceiling=E5",
                "behavior_propensity": 0.25,
            }
        ).behavior_propensity
        == 0.25
    )


def test_unavailable_normalized_diagnostics_cannot_smuggle_values() -> None:
    payload = _base("routing", "E4") | {
        "status": "unavailable",
        "error": "decision evidence is missing",
    }
    assert ExecutionRecord.model_validate(payload).status == "unavailable"

    with pytest.raises(ValidationError, match="cannot carry diagnostic values"):
        ExecutionRecord.model_validate(payload | {"quality": 1.0})
    with pytest.raises(ValidationError, match="requires a reason"):
        ExecutionRecord.model_validate(payload | {"error": ""})
