"""Session-aware attribution over replay records (Sub-issue 6, P2).

Single-turn attribution (attribution.py) keys evidence per request. Agentic
traffic lives per (conversation, turn): the replay record already carries the
session evidence (conversation_id / turn_index / session_action /
selection_reasoning), so this module aggregates rows into sessions and layers
three agentic evidence classes on top of the single-turn E1/E2/E3 classes -
it never redefines them:

- lane_flap            adjacent model-bearing requests of one conversation
                       selected different models (may be a legitimate switch
                       or a protection miss; reported with the router's own
                       session reasoning)
- tool_loop_interrupted a recorded failed selection whose turn continued
                       afterwards: an agentic tool loop was interrupted by the
                       rejected (503) selection and resumed within the same
                       user turn. Requires the rejection-recording router fix
                       (lifecycle_state=failed +
                       terminal_reason=selection_rejected); on unfixed routers
                       nothing emits those records, so this class stays empty
- selection_failed     every recorded failed selection (lifecycle_state=failed)
                       as record-level evidence; tool_loop_interrupted is the
                       conversation-level subset that resumed in-turn

Requests inside one conversation are ordered by (turn_index, record_ts,
request_id). The router counts turn_index per user turn, and one agentic user
turn fans out into many tool-loop requests that all share the same
turn_index - only the record timestamp orders them. Rows collected by an
older adapter without record_ts fall back to (turn_index, request_id) order.
Pure functions, no I/O, deterministic: same rows -> same report bytes.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

SESSION_CLASS_LANE_FLAP = "lane_flap"
SESSION_CLASS_TOOL_LOOP_INTERRUPTED = "tool_loop_interrupted"
SESSION_CLASS_SELECTION_FAILED = "selection_failed"
SESSION_CLASSES = (
    SESSION_CLASS_LANE_FLAP,
    SESSION_CLASS_TOOL_LOOP_INTERRUPTED,
    SESSION_CLASS_SELECTION_FAILED,
)

LIFECYCLE_FAILED = "failed"


def _session_of(row: dict[str, Any]) -> dict[str, Any] | None:
    session = row.get("session")
    return session if isinstance(session, dict) else None


def _conversation_key(row: dict[str, Any]) -> str | None:
    session = _session_of(row)
    if not session:
        return None
    for key in ("conversation_id", "session_id"):
        value = session.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _turn_index(row: dict[str, Any]) -> int | None:
    session = _session_of(row)
    if not session:
        return None
    value = session.get("turn_index")
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _record_ts(row: dict[str, Any]) -> str:
    value = row.get("record_ts")
    return value if isinstance(value, str) else ""


def _lifecycle_state(row: dict[str, Any]) -> str:
    value = row.get("lifecycle_state")
    return value if isinstance(value, str) else ""


def _request_order(row: dict[str, Any]) -> tuple[int | None, str, str]:
    return (_turn_index(row), _record_ts(row), str(row.get("request_id") or ""))


def _event(
    conversation: str,
    kind: str,
    *,
    turn_index: int | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {"conversation": conversation, "class": kind}
    if turn_index is not None:
        event["turn_index"] = turn_index
    if detail:
        event["detail"] = detail
    return event


def ordered_sessions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group attribution rows into conversations in replay order (AG3).

    The AG3 route-replay tier and `analyze_sessions` share one ordering:
    rows keyed by conversation identity (conversation_id, else session_id),
    requests sorted by (turn_index, record_ts, request_id). Rows without a
    conversation identity are dropped here (single-turn attribution owns
    them). Deterministic: same rows -> same mapping.
    """
    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = _conversation_key(row)
        if key is None:
            continue
        sessions.setdefault(key, []).append(row)
    for key, session_rows in sessions.items():
        sessions[key] = sorted(
            (row for row in session_rows if _turn_index(row) is not None),
            key=_request_order,
        )
    return sessions


def analyze_sessions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate attribution rows into per-conversation agentic evidence.

    Rows without session fields are counted as single-turn and ignored here
    (the single-turn attribution owns them). Deterministic output: sessions
    sorted by key, requests by (turn_index, record_ts, request_id).
    """
    sessions = ordered_sessions(rows)
    single_turn = sum(1 for row in rows if _conversation_key(row) is None)

    events: list[dict[str, Any]] = []
    session_summaries: list[dict[str, Any]] = []
    for conversation in sorted(sessions):
        requests = sessions[conversation]
        lane_flaps = 0
        selection_failures = 0
        interruptions = 0
        # lane_flap: adjacent model-bearing requests (failed selections carry
        # no routed model, so they never join a lane and never break adjacency)
        routed_requests = [row for row in requests if str(row.get("routed") or "")]
        for i in range(1, len(routed_requests)):
            prev, curr = routed_requests[i - 1], routed_requests[i]
            prev_model = str(prev.get("routed") or "")
            curr_model = str(curr.get("routed") or "")
            if prev_model != curr_model:
                lane_flaps += 1
                session = _session_of(curr) or {}
                events.append(
                    _event(
                        conversation,
                        SESSION_CLASS_LANE_FLAP,
                        turn_index=_turn_index(curr),
                        detail={
                            "from": prev_model,
                            "to": curr_model,
                            "session_action": session.get("session_action"),
                            "selection_reasoning": session.get("selection_reasoning"),
                        },
                    )
                )
        # selection_failed / tool_loop_interrupted: recorded failed selections
        for i, req in enumerate(requests):
            if _lifecycle_state(req) != LIFECYCLE_FAILED:
                continue
            selection_failures += 1
            events.append(
                _event(
                    conversation,
                    SESSION_CLASS_SELECTION_FAILED,
                    turn_index=_turn_index(req),
                    detail={
                        "terminal_reason": str(req.get("terminal_reason") or ""),
                        "response_status": req.get("response_status"),
                        "request_id": str(req.get("request_id") or ""),
                    },
                )
            )
            turn = _turn_index(req)
            resumed = next(
                (later for later in requests[i + 1 :] if _turn_index(later) == turn),
                None,
            )
            if resumed is not None:
                interruptions += 1
                events.append(
                    _event(
                        conversation,
                        SESSION_CLASS_TOOL_LOOP_INTERRUPTED,
                        turn_index=turn,
                        detail={
                            "interrupted_request": str(req.get("request_id") or ""),
                            "resumed_request": str(resumed.get("request_id") or ""),
                        },
                    )
                )
        session_summaries.append(
            {
                "conversation": conversation,
                "requests_recorded": len(requests),
                "turn_span": (
                    [_turn_index(requests[0]), _turn_index(requests[-1])]
                    if requests
                    else []
                ),
                "lane_flaps": lane_flaps,
                "selection_failures": selection_failures,
                "tool_loop_interruptions": interruptions,
            }
        )
    counts = Counter(event["class"] for event in events)
    return {
        "sessions": session_summaries,
        "events": events,
        "event_counts": {cls: counts.get(cls, 0) for cls in SESSION_CLASSES},
        "single_turn_rows": single_turn,
        "session_rows": sum(len(v) for v in sessions.values()),
    }


def render_session_report(analysis: dict[str, Any]) -> str:
    """Render the session-level agentic evidence report (markdown)."""
    lines = [
        "# Session-aware agentic attribution report",
        "",
        f"- conversations: {len(analysis['sessions'])} ({analysis['session_rows']} session rows, "
        f"{analysis['single_turn_rows']} single-turn rows ignored here)",
        "- evidence classes: lane_flap / tool_loop_interrupted / selection_failed "
        "(layered on the single-turn E1/E2/E3 classes, never replacing them)",
        "",
        "## Event counts",
        "",
        "| class | count |",
        "|---|---|",
    ]
    for cls in SESSION_CLASSES:
        lines.append(f"| {cls} | {analysis['event_counts'][cls]} |")
    lines += [
        "",
        "## Sessions",
        "",
        "| conversation | requests | span | lane_flaps | selection_failures | loop_interruptions |",
        "|---|---|---|---|---|---|",
    ]
    for session in analysis["sessions"]:
        span = session["turn_span"]
        span_text = f"{span[0]}..{span[1]}" if span else "-"
        lines.append(
            f"| `{session['conversation']}` | {session['requests_recorded']} | {span_text} "
            f"| {session['lane_flaps']} | {session['selection_failures']} "
            f"| {session['tool_loop_interruptions']} |"
        )
    if analysis["events"]:
        lines += ["", "## Events", ""]
        for event in analysis["events"]:
            detail = json.dumps(
                event.get("detail") or {}, ensure_ascii=False, sort_keys=True
            )
            lines.append(
                f"- `{event['class']}` conversation `{event['conversation']}`"
                + (
                    f" turn {event['turn_index']}"
                    if event.get("turn_index") is not None
                    else ""
                )
                + f" {detail}"
            )
    else:
        lines += ["", "## Events", "", "- none"]
    lines += [
        "",
        "---",
        "",
        "Replayed from recipe-lab records only; descriptive, statistical "
        "significance is not established.",
    ]
    return "\n".join(lines) + "\n"


def write_session_report(rows: list[dict[str, Any]], output_dir: str) -> dict[str, Any]:
    """Analyze + render + write session attribution artifacts; returns summary."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    analysis = analyze_sessions(rows)
    report_path = out / "session-report.md"
    report_path.write_text(render_session_report(analysis), encoding="utf-8")
    events_path = out / "session-events.json"
    events_path.write_text(
        json.dumps(analysis["events"], ensure_ascii=False, indent=1, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return {
        "sessions": len(analysis["sessions"]),
        "events": analysis["event_counts"],
        "session_report": str(report_path),
        "session_events": str(events_path),
    }
