"""Sub-issue 6 P2 tests: session-aware agentic attribution.

Evidence classes are layered on the single-turn classes, keyed by
(conversation, turn) from the router_replay session fields. One agentic user
turn fans out into many requests that all share the same turn_index, so the
in-turn order comes from the record timestamp.
"""

from __future__ import annotations

import json

from recipe_lab.session_attribution import (
    SESSION_CLASS_LANE_FLAP,
    SESSION_CLASS_SELECTION_FAILED,
    SESSION_CLASS_TOOL_LOOP_INTERRUPTED,
    analyze_sessions,
    render_session_report,
    write_session_report,
)


def row(
    conversation: str,
    turn: int,
    routed: str,
    *,
    ts: str | None = None,
    request_id: str | None = None,
    lifecycle_state: str | None = None,
    response_status: int | None = None,
    terminal_reason: str | None = None,
    **session_extra,
) -> dict:
    session = {"conversation_id": conversation, "turn_index": turn, **session_extra}
    out = {
        "case_id": f"{conversation}-t{turn}",
        "routed": routed,
        "session": session,
    }
    if ts is not None:
        out["record_ts"] = ts
    if request_id is not None:
        out["request_id"] = request_id
    if lifecycle_state is not None:
        out["lifecycle_state"] = lifecycle_state
    if response_status is not None:
        out["response_status"] = response_status
    if terminal_reason is not None:
        out["terminal_reason"] = terminal_reason
    return out


def test_lane_flap_detected_with_router_reasoning():
    rows = [
        row(
            "conv-A",
            0,
            "local/private-qwen",
            ts="2026-10-28T10:00:00Z",
            session_action="establish_baseline",
        ),
        row(
            "conv-A",
            1,
            "cloud/frontier-reasoning",
            ts="2026-10-28T10:01:00Z",
            request_id="r2",
            session_action="switch",
            selection_reasoning="session_aware: previous_model_not_in_candidates",
        ),
    ]
    analysis = analyze_sessions(rows)
    assert analysis["event_counts"][SESSION_CLASS_LANE_FLAP] == 1
    event = analysis["events"][0]
    assert event["turn_index"] == 1
    assert event["detail"]["from"] == "local/private-qwen"
    assert event["detail"]["to"] == "cloud/frontier-reasoning"
    assert event["detail"]["session_action"] == "switch"


def test_same_turn_requests_ordered_by_timestamp():
    # one agentic user turn fans out into many tool-loop requests that all
    # share turn_index=0; the in-turn order must come from record_ts, not
    # from the (API-return-ordered) input order. The reversed input would
    # otherwise report the flap in the wrong direction.
    rows = [
        row(
            "conv-A",
            0,
            "cloud/frontier-reasoning",
            ts="2026-10-28T10:00:03Z",
            request_id="r3",
        ),
        row(
            "conv-A",
            0,
            "local/private-qwen",
            ts="2026-10-28T10:00:01Z",
            request_id="r1",
        ),
        row(
            "conv-A",
            0,
            "cloud/frontier-reasoning",
            ts="2026-10-28T10:00:02Z",
            request_id="r2",
        ),
    ]
    analysis = analyze_sessions(rows)
    assert analysis["event_counts"][SESSION_CLASS_LANE_FLAP] == 1
    flap_events = [
        e for e in analysis["events"] if e["class"] == SESSION_CLASS_LANE_FLAP
    ]
    assert [(e["detail"]["from"], e["detail"]["to"]) for e in flap_events] == [
        ("local/private-qwen", "cloud/frontier-reasoning"),
    ]
    # the reverse input order must yield the identical, timestamp-ordered flap
    reversed_analysis = analyze_sessions(list(reversed(rows)))
    assert reversed_analysis["events"] == analysis["events"]
    # summary reflects the fan-out shape: one turn, three requests
    assert analysis["sessions"][0]["requests_recorded"] == 3
    assert analysis["sessions"][0]["turn_span"] == [0, 0]


def test_failed_request_is_selection_failed_and_interrupted():
    # the router records the 503 rejection itself (lifecycle_state=failed +
    # terminal_reason=selection_rejected) and the same tool loop retries and
    # continues within the user turn: one record-level selection_failed plus
    # one conversation-level tool_loop_interrupted.
    rows = [
        row(
            "conv-A",
            0,
            "local/private-qwen",
            ts="2026-10-28T10:00:00Z",
            request_id="r1",
        ),
        row(
            "conv-A",
            0,
            "",
            ts="2026-10-28T10:00:01Z",
            request_id="r2",
            lifecycle_state="failed",
            response_status=503,
            terminal_reason="selection_rejected",
        ),
        row(
            "conv-A",
            0,
            "local/private-qwen",
            ts="2026-10-28T10:00:02Z",
            request_id="r3",
        ),
    ]
    analysis = analyze_sessions(rows)
    assert analysis["event_counts"][SESSION_CLASS_SELECTION_FAILED] == 1
    assert analysis["event_counts"][SESSION_CLASS_TOOL_LOOP_INTERRUPTED] == 1
    failed = next(
        e for e in analysis["events"] if e["class"] == SESSION_CLASS_SELECTION_FAILED
    )
    assert failed["turn_index"] == 0
    assert failed["detail"]["terminal_reason"] == "selection_rejected"
    assert failed["detail"]["response_status"] == 503
    assert failed["detail"]["request_id"] == "r2"
    interrupted = next(
        e
        for e in analysis["events"]
        if e["class"] == SESSION_CLASS_TOOL_LOOP_INTERRUPTED
    )
    assert interrupted["detail"]["interrupted_request"] == "r2"
    assert interrupted["detail"]["resumed_request"] == "r3"
    summary = analysis["sessions"][0]
    assert summary["selection_failures"] == 1
    assert summary["tool_loop_interruptions"] == 1


def test_failed_request_without_resume_is_not_interrupted():
    # the loop was interrupted but the turn ended there (user abort / next
    # turn): record-level evidence only, no in-turn resume.
    rows = [
        row(
            "conv-A",
            0,
            "local/private-qwen",
            ts="2026-10-28T10:00:00Z",
            request_id="r1",
        ),
        row(
            "conv-A",
            0,
            "",
            ts="2026-10-28T10:00:01Z",
            request_id="r2",
            lifecycle_state="failed",
        ),
        row(
            "conv-A",
            1,
            "local/private-qwen",
            ts="2026-10-28T10:05:00Z",
            request_id="r3",
        ),
    ]
    analysis = analyze_sessions(rows)
    assert analysis["event_counts"][SESSION_CLASS_SELECTION_FAILED] == 1
    assert analysis["event_counts"][SESSION_CLASS_TOOL_LOOP_INTERRUPTED] == 0


def test_turn_gap_alone_is_no_evidence():
    # a turn_index jump with no failed record proves nothing: one agentic
    # turn legitimately spans many same-index requests, so a missing index
    # may simply be a turn that sent no replayed request. The old turn-gap
    # heuristic is gone; without lifecycle_state=failed nothing fires.
    rows = [
        row("conv-A", 0, "local/private-qwen", ts="2026-10-28T10:00:00Z"),
        row("conv-A", 1, "local/private-qwen", ts="2026-10-28T10:01:00Z"),
        row("conv-A", 3, "local/private-qwen", ts="2026-10-28T10:03:00Z"),
    ]
    analysis = analyze_sessions(rows)
    assert analysis["events"] == []
    assert analysis["event_counts"][SESSION_CLASS_TOOL_LOOP_INTERRUPTED] == 0


def test_rows_without_timestamp_fall_back_to_request_id_order():
    # rows collected by an older adapter carry no record_ts: ordering then
    # falls back to (turn_index, request_id) and stays deterministic.
    rows = [
        row("conv-A", 0, "cloud/frontier-reasoning", request_id="r2"),
        row("conv-A", 0, "local/private-qwen", request_id="r1"),
    ]
    analysis = analyze_sessions(rows)
    flap = next(e for e in analysis["events"] if e["class"] == SESSION_CLASS_LANE_FLAP)
    assert flap["detail"]["from"] == "local/private-qwen"
    assert flap["detail"]["to"] == "cloud/frontier-reasoning"


def test_single_turn_rows_are_ignored_here():
    rows = [{"case_id": "solo", "routed": "local/private-qwen"}]
    analysis = analyze_sessions(rows)
    assert analysis["single_turn_rows"] == 1
    assert analysis["sessions"] == []
    assert analysis["events"] == []


def test_report_is_deterministic_and_honest():
    rows = [
        row("conv-B", 0, "local/private-qwen", ts="2026-10-28T10:00:00Z"),
        row("conv-A", 0, "local/private-qwen", ts="2026-10-28T10:00:00Z"),
        row(
            "conv-A",
            2,
            "cloud/frontier-reasoning",
            ts="2026-10-28T10:02:00Z",
            session_action="switch",
        ),
    ]
    first = render_session_report(analyze_sessions(rows))
    second = render_session_report(analyze_sessions(rows))
    assert first == second  # byte-identical across renders
    assert first.startswith("# Session-aware agentic attribution report")
    assert "statistical significance is not established" in first
    # conv-A sorts before conv-B; the flap detail carries the router reasoning
    assert "`conv-A`" in first and "`conv-B`" in first
    assert (
        "previous_model_not_in_candidates" not in first
    )  # no reasoning given on this flap


def test_write_session_report_artifacts(tmp_path):
    rows = [
        row(
            "conv-A",
            0,
            "local/private-qwen",
            ts="2026-10-28T10:00:00Z",
            request_id="r1",
        ),
        row(
            "conv-A",
            0,
            "",
            ts="2026-10-28T10:00:01Z",
            request_id="r2",
            lifecycle_state="failed",
            response_status=503,
            terminal_reason="selection_rejected",
        ),
        row(
            "conv-A",
            2,
            "cloud/frontier-reasoning",
            ts="2026-10-28T10:02:00Z",
            request_id="r3",
        ),
    ]
    summary = write_session_report(rows, str(tmp_path))
    assert summary["sessions"] == 1
    assert summary["events"][SESSION_CLASS_SELECTION_FAILED] == 1
    report = (tmp_path / "session-report.md").read_text(encoding="utf-8")
    events = json.loads((tmp_path / "session-events.json").read_text(encoding="utf-8"))
    assert "Session-aware agentic attribution report" in report
    assert len(events) == 2  # selection_failed plus its in-turn resume
