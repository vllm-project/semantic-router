"""AG4: agentic diagnosis - per-phase distributions, sub-agent separate
retention, failed-selection disclosure."""

from __future__ import annotations

import json

from recipe_lab.adapters.router_replay import records_to_rows
from recipe_lab.diagnose import (
    agent_separation,
    build_diagnosis,
    failure_disclosure,
    phase_distributions,
    render_markdown,
)

CONFIG = {
    "decisions": [
        {
            "name": "domain_code",
            "rules": [
                {
                    "operator": "difficulty",
                    "type": "score",
                    "predicate": "gte",
                    "value": 2,
                }
            ],
        }
    ]
}


def _record(
    request_id: str,
    *,
    prompt: str = "fix the bug",
    model: str = "local/model-a",
    decision: str = "domain_code",
    difficulty: float = 3.0,
    phase_body: str = "user",
    conversation: str | None = "conv-1",
    turn: int = 0,
    agent_id: str | None = None,
    identity_status: str | None = None,
    memory_key: str | None = None,
    lifecycle_state: str | None = None,
    terminal_reason: str | None = None,
) -> dict:
    messages = (
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "tool_calls": [{"id": "t1"}]},
            {"role": "tool", "tool_call_id": "t1", "content": "out"},
        ]
        if phase_body == "tool"
        else [{"role": "user", "content": prompt}]
    )
    record = {
        "request_id": request_id,
        "prompt": prompt,
        "decision": decision,
        "selected_model": model,
        "signal_values": {"decision:difficulty": difficulty},
        "request_body": json.dumps({"messages": messages}),
        "timestamp": f"2026-11-03T00:00:{turn:02d}",
    }
    if conversation is not None:
        record["conversation_id"] = conversation
        record["session_id"] = conversation
        record["turn_index"] = turn
    if agent_id is not None:
        record["agent_id"] = agent_id
    if identity_status is not None or memory_key is not None:
        record["session_policy"] = {
            "method": "protection",
            "identity": {
                "scope": "conversation",
                "conversation": {"required": True, "status": identity_status},
                **({"memory_key_hash": memory_key} if memory_key else {}),
            },
        }
    if lifecycle_state is not None:
        record["lifecycle_state"] = lifecycle_state
    if terminal_reason is not None:
        record["terminal_reason"] = terminal_reason
    return record


# ---- adapter: the rows carry the agentic evidence ----


def test_records_to_rows_derives_phase_agent_and_policy():
    rows = records_to_rows(
        [
            _record("r1", phase_body="user", agent_id="agent-main"),
            _record("r2", phase_body="tool", identity_status="missing"),
        ]
    )
    assert rows[0]["phase"] == "user_turn"
    assert rows[0]["agent_id"] == "agent-main"
    assert rows[1]["phase"] == "tool_loop"
    assert rows[1]["session_policy"]["identity"]["scope"] == "conversation"


# ---- phase distributions: the BUG-3 lesson made visible ----


def test_phase_distributions_split_decisions_and_signals():
    rows = records_to_rows(
        [
            _record("r1", phase_body="user", difficulty=3.0),
            _record("r2", phase_body="user", difficulty=3.4),
            _record("r3", phase_body="tool", difficulty=0.2),
            _record("r4", phase_body="tool", difficulty=0.4),
        ]
    )
    distribution = phase_distributions(rows)
    by_phase = {entry["phase"]: entry for entry in distribution["phases"]}
    assert set(by_phase) == {"user_turn", "tool_loop"}
    assert by_phase["user_turn"]["requests"] == 2
    assert by_phase["user_turn"]["decisions"] == {"domain_code": 2}
    # the mixed-population p50 would hide this: tool-loop difficulty runs low
    assert (
        by_phase["tool_loop"]["signals"]["decision:difficulty"]["p50"]
        < by_phase["user_turn"]["signals"]["decision:difficulty"]["p50"]
    )


# ---- sub-agent separate retention ----


def test_agent_marker_sharing_main_identity_is_not_separate():
    rows = records_to_rows(
        [
            _record(
                "r1",
                conversation="conv-1",
                agent_id="agent-main",
                identity_status="present",
            ),
            _record(
                "r2",
                conversation="conv-1",
                turn=1,
                agent_id="agent-sub",
                identity_status="present",
            ),
        ]
    )
    result = agent_separation(rows)
    assert result["status"] == "not_separate"
    assert result["shared_identity_conversations"] == ["conv-1"]


def test_agent_marker_with_own_identity_is_separate():
    rows = records_to_rows(
        [
            _record(
                "r1",
                conversation="conv-main",
                agent_id="agent-main",
                identity_status="present",
            ),
            _record(
                "r2",
                conversation="conv-sub",
                turn=1,
                agent_id="agent-sub",
                identity_status="present",
            ),
        ]
    )
    assert agent_separation(rows)["status"] == "separate"


def test_missing_identity_reports_not_separate_without_agent_marker():
    # the AG8 fixture case: the client sends no identity headers at all, so
    # sub-agents cannot keep a separate identity (completion standard)
    rows = records_to_rows(
        [
            _record("r1", conversation=None, identity_status="missing"),
            _record("r2", conversation=None, turn=1, identity_status="missing"),
        ]
    )
    result = agent_separation(rows)
    assert result["status"] == "not_separate"
    assert result["requests_missing_identity"] == 2
    assert "cannot keep a separate identity" in result["reason"]


def test_no_agent_marker_with_identities_is_unverifiable():
    rows = records_to_rows([_record("r1", identity_status="present")])
    result = agent_separation(rows)
    assert result["status"] == "unverifiable"


# ---- the client manifest join (Z9) ----


def _hash_rows():
    return records_to_rows(
        [
            _record("req-main-1", identity_status="present", memory_key="main-hash"),
            _record(
                "req-main-2", turn=1, identity_status="present", memory_key="main-hash"
            ),
            _record(
                "req-sub-1", turn=2, identity_status="present", memory_key="sub-hash"
            ),
        ]
    )


def test_manifest_sub_agent_own_identity_is_separate():
    # Z9: the probe's manifest names the sub-agent request ids; the join
    # compares their protection identity hash against the main traffic's
    result = agent_separation(
        _hash_rows(), agent_requests=[{"request_id": "req-sub-1"}]
    )
    assert result["status"] == "separate"
    assert result["manifest_requests"] == 1


def test_manifest_sub_agent_sharing_the_main_hash_is_not_separate():
    rows = records_to_rows(
        [
            _record("req-main-1", memory_key="main-hash"),
            _record("req-sub-1", turn=1, memory_key="main-hash"),
        ]
    )
    result = agent_separation(rows, agent_requests=["req-sub-1"])
    assert result["status"] == "not_separate"
    assert result["shared_identity_request_ids"] == ["req-sub-1"]


def test_manifest_sub_agent_without_identity_is_not_separate():
    rows = records_to_rows(
        [
            _record("req-main-1", memory_key="main-hash"),
            _record("req-sub-1", turn=1, identity_status="present"),
        ]
    )
    result = agent_separation(rows, agent_requests=["req-sub-1"])
    assert result["status"] == "not_separate"
    assert result["requests_missing_identity"] == 1


def test_manifest_unjoinable_request_ids_stay_unverifiable():
    # a manifest id the replays cannot join is partial evidence: the check
    # is never asserted on incomplete evidence
    result = agent_separation(_hash_rows(), agent_requests=["req-sub-1", "req-lost"])
    assert result["status"] == "unverifiable"
    assert result["unjoinable_request_ids"] == ["req-lost"]


def test_manifest_without_sub_agent_traffic_is_unverifiable():
    result = agent_separation(_hash_rows(), agent_requests=[])
    assert result["status"] == "unverifiable"
    assert result["manifest_requests"] == 0


def test_build_diagnosis_passes_the_manifest_to_the_join():
    rows = _hash_rows()
    diagnosis = build_diagnosis(
        rows, CONFIG, agent_requests=[{"request_id": "req-sub-1"}]
    )
    agentic = diagnosis["agentic"]
    assert agentic["agent_separation"]["status"] == "separate"
    # without the manifest the same rows stay unverifiable (no record markers)
    diagnosis = build_diagnosis(rows, CONFIG)
    assert diagnosis["agentic"]["agent_separation"]["status"] == "unverifiable"


# ---- failed selections (sub-6 P1) ----


def test_failure_disclosure_counts_rejected_selections():
    rows = records_to_rows(
        [
            _record(
                "r1", lifecycle_state="failed", terminal_reason="selection_rejected"
            ),
            _record(
                "r2", lifecycle_state="failed", terminal_reason="selection_rejected"
            ),
            _record("r3", lifecycle_state="failed", terminal_reason="upstream_error"),
            _record("r4"),
        ]
    )
    failures = failure_disclosure(rows)
    assert failures["failed_requests"] == 3
    assert failures["selection_rejected"] == 2
    assert failures["terminal_reasons"] == {
        "selection_rejected": 2,
        "upstream_error": 1,
    }


# ---- end to end through build_diagnosis ----


def test_build_diagnosis_includes_agentic_section_for_agentic_rows():
    rows = records_to_rows(
        [
            _record("r1", phase_body="user", difficulty=3.0, identity_status="missing"),
            _record(
                "r2",
                phase_body="tool",
                difficulty=0.3,
                identity_status="missing",
                lifecycle_state="failed",
                terminal_reason="selection_rejected",
            ),
        ]
    )
    diagnosis = build_diagnosis(rows, CONFIG)
    agentic = diagnosis["agentic"]
    assert {p["phase"] for p in agentic["phase_distributions"]["phases"]} == {
        "user_turn",
        "tool_loop",
    }
    assert agentic["agent_separation"]["status"] == "not_separate"
    assert agentic["failures"]["selection_rejected"] == 1
    markdown = render_markdown(diagnosis)
    assert "## Agentic session evidence" in markdown
    assert "Sub-agent separate retention" in markdown
    assert "**not_separate**" in markdown
    assert "Failed selections" in markdown


def test_build_diagnosis_legacy_rows_keep_their_shape():
    rows = [
        {
            "matched_decision": "domain_code",
            "signal_values": {"decision:difficulty": 2.0},
        }
    ]
    diagnosis = build_diagnosis(rows, CONFIG)
    assert "agentic" not in diagnosis
    assert "Agentic session evidence" not in render_markdown(diagnosis)
