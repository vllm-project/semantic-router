"""Replay the recorded coding-agent session fixture turn by turn.

The fixture (`bench/data/coding_agent_session.v1.json` in the repository)
records one coding-agent session: a tool catalog and turns labelled
`user_turn`, `tool_loop` and `follow_up`. Each turn appends recorded messages;
a replay resends the cumulative prefix, never feeding generated output back or
executing tools. One case is one replay, so a session-aware target gives every
replay exactly one session ID.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import SESSION_AWARE

BENCHMARK_ID = "agent-session-replay"
ADAPTER_VERSION = "agent-session-replay-v1"
SCHEMA_VERSION = "agent-session-fixture.v1"
FIXTURE_PATH_HINT = "bench/data/coding_agent_session.v1.json"
FIXTURE_KEYS: frozenset[str] = frozenset(
    {"schema_version", "id", "description", "tools", "turns"}
)
TURN_KEYS: frozenset[str] = frozenset({"id", "phase", "append_messages"})
CASE_TURN_KEYS: frozenset[str] = frozenset({"id", "phase", "message_count"})


class FixtureError(ValueError):
    pass


@dataclass(frozen=True)
class Turn:
    id: str
    phase: str
    message_count: int


@dataclass(frozen=True)
class AgentSession:
    id: str
    tools: tuple[dict[str, Any], ...]
    messages: tuple[dict[str, Any], ...]
    turns: tuple[Turn, ...]


def parse_session(raw: Any) -> AgentSession:
    fixture = _object(value=raw, where="fixture", allowed=FIXTURE_KEYS)
    if fixture.get("schema_version") != SCHEMA_VERSION:
        raise FixtureError(f"schema_version must be {SCHEMA_VERSION!r}")
    tools = _parse_tools(fixture.get("tools"))
    tool_names = {tool["function"]["name"] for tool in tools}
    messages, turns = _parse_turns(value=fixture.get("turns"), tool_names=tool_names)
    return AgentSession(
        id=_text(value=fixture.get("id"), where="id"),
        tools=tools,
        messages=messages,
        turns=turns,
    )


def build_case(session: AgentSession, *, split: str = "dev") -> dict[str, Any]:
    return {
        "id": session.id,
        "benchmark": BENCHMARK_ID,
        "messages": [dict(message) for message in session.messages],
        "tools": [dict(tool) for tool in session.tools],
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "split": split,
            "turns": [
                {
                    "id": turn.id,
                    "phase": turn.phase,
                    "message_count": turn.message_count,
                }
                for turn in session.turns
            ],
        },
    }


def case_turns(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate the recorded turn boundaries an inline case may hand-edit."""
    messages = case.get("messages")
    metadata = case.get("metadata")
    turns = metadata.get("turns") if isinstance(metadata, dict) else None
    if not isinstance(messages, list) or not isinstance(turns, list) or not turns:
        raise ValueError("Agent-session cases require recorded messages and turns")
    previous = 0
    for turn in turns:
        if not isinstance(turn, dict) or set(turn) != CASE_TURN_KEYS:
            raise ValueError("Agent-session turns require id, phase and message_count")
        for key in ("id", "phase"):
            if not isinstance(turn[key], str) or not turn[key].strip():
                raise ValueError(f"Agent-session turn {key} must be nonempty text")
        count = turn["message_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= previous:
            raise ValueError("Agent-session turn boundaries must strictly grow")
        if count > len(messages):
            raise ValueError("Agent-session turn exceeds the recorded transcript")
        last_role = messages[count - 1].get("role")
        if last_role not in {"user", "tool"}:
            raise ValueError(
                "Agent-session turns must end with a user message or tool result"
            )
        if (turn["phase"] == "tool_loop") != (last_role == "tool"):
            raise ValueError(
                "Agent-session tool_loop turns must end with a tool result"
            )
        previous = count
    if previous != len(messages):
        raise ValueError("Agent-session turns must cover the recorded transcript")
    return turns


def execute(case: dict[str, Any], context: Any) -> dict[str, Any]:
    turns = case_turns(case)
    rows: list[dict[str, Any]] = []
    answer = None
    for turn in turns:
        result = context.call(
            case["messages"][: turn["message_count"]],
            extra_body={"tools": case["tools"]},
            fixture_phase=turn["phase"],
        )
        answer = result.get("final") or answer
        rows.append(
            {
                "turn": turn["id"],
                "phase": turn["phase"],
                "selected_model": result.get("selected_model") or "",
                "finish_reason": result.get("finish_reason") or "",
            }
        )
    return {"answer": answer, "correct": True, "score": 1, "details": {"turns": rows}}


def preflight(case: dict[str, Any], manifest: dict[str, Any], cache: Any) -> None:
    del cache
    case_turns(case)
    tools = case.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("Agent-session cases require the recorded tool catalog")
    for target in manifest.get("targets", []):
        if target.get("session_mode") != SESSION_AWARE:
            raise ValueError(
                "agent-session-replay sends one session ID per replay; every "
                "target requires session_mode: session_aware"
            )


def prepare(
    *,
    benchmark: str,
    profile: str,
    store: Any,
    source_path: Any = None,
    revision: str | None = None,
    seed: int = 20260918,
    limit: int | None = None,
) -> dict[str, Any]:
    if benchmark != BENCHMARK_ID:
        raise ValueError("Unknown benchmark for the agent-session adapter")
    if profile not in {"smoke", "quick", "standard"}:
        raise ValueError("Unknown benchmark or profile")
    if source_path is None:
        raise ValueError(
            "agent-session-replay prepares from the repository fixture; pass "
            f"--local --source-path {FIXTURE_PATH_HINT}"
        )
    if limit is not None and limit < 1:
        raise ValueError("limit cannot drop the only recorded session")
    path = Path(source_path).expanduser()
    raw_bytes = path.read_bytes()
    try:
        raw = json.loads(raw_bytes.decode("utf-8"), object_pairs_hook=_unique_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FixtureError(f"{path}: {exc}") from exc
    session = parse_session(raw)
    case = build_case(session, split="holdout" if profile == "standard" else "dev")
    source = {
        "url": "local-import",
        "revision": revision or SCHEMA_VERSION,
        "files": [{"name": path.name, "sha256": hashlib.sha256(raw_bytes).hexdigest()}],
        "revision_verification": "declared-revision-with-content-digest",
        "access_note": (
            "Operator-supplied source; retain upstream access and license "
            "records alongside this manifest."
        ),
        "normalizer": ADAPTER_VERSION,
    }
    # Dataset publication imports adapter registration.
    from .sources import _write_dataset  # noqa: PLC0415

    return _write_dataset(store, [case], profile, seed, {BENCHMARK_ID: source})


def build_adapter():
    # Adapter registration imports this module.
    from .adapters import BenchmarkAdapter  # noqa: PLC0415

    return BenchmarkAdapter(
        id=BENCHMARK_ID,
        title="Agent session replay",
        kind="agent",
        source_url=(
            "https://github.com/vllm-project/semantic-router/blob/main/"
            "bench/data/coding_agent_session.v1.json"
        ),
        version=ADAPTER_VERSION,
        execute=execute,
        preflight=preflight,
        prepare=prepare,
    )


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FixtureError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _object(*, value: Any, where: str, allowed: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FixtureError(f"{where} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise FixtureError(f"{where} has unknown keys {unknown}")
    return value


def _text(*, value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FixtureError(f"{where} must be a non-empty string")
    return value


def _parse_tools(value: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise FixtureError("tools must be a non-empty list")
    names: set[str] = set()
    for index, tool in enumerate(value):
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or tool.get("type") != "function":
            raise FixtureError(f"tools[{index}] must be a function tool")
        name = _text(value=function.get("name"), where=f"tools[{index}] name")
        if name in names:
            raise FixtureError(f"duplicate tool name {name!r}")
        parameters = function.get("parameters")
        if not isinstance(parameters, dict) or parameters.get("type") != "object":
            raise FixtureError(f"tool {name!r} parameters must be an object schema")
        names.add(name)
    return tuple(value)


def _parse_turns(
    *, value: Any, tool_names: set[str]
) -> tuple[tuple[dict[str, Any], ...], tuple[Turn, ...]]:
    if not isinstance(value, list) or not value:
        raise FixtureError("turns must be a non-empty list")
    messages: list[dict[str, Any]] = []
    turns: list[Turn] = []
    pending: set[str] = set()
    seen_calls: set[str] = set()
    for index, raw_turn in enumerate(value):
        turn = _object(value=raw_turn, where=f"turns[{index}]", allowed=TURN_KEYS)
        turn_id = _text(value=turn.get("id"), where=f"turns[{index}] id")
        if any(existing.id == turn_id for existing in turns):
            raise FixtureError(f"duplicate turn id {turn_id!r}")
        appended = turn.get("append_messages")
        if not isinstance(appended, list) or not appended:
            raise FixtureError(f"turn {turn_id!r} must append at least one message")
        for message in appended:
            _check_message(
                message=message,
                position=len(messages),
                pending=pending,
                seen_calls=seen_calls,
                tool_names=tool_names,
            )
            messages.append(message)
        if pending:
            raise FixtureError(
                f"turn {turn_id!r} leaves tool calls unanswered: {sorted(pending)}"
            )
        last_role = messages[-1]["role"]
        if last_role not in {"user", "tool"}:
            raise FixtureError(
                f"turn {turn_id!r} must end with a user message or tool result"
            )
        phase = _text(value=turn.get("phase"), where=f"turn {turn_id!r} phase")
        if (phase == "tool_loop") != (last_role == "tool"):
            raise FixtureError(
                f"turn {turn_id!r} is a tool_loop turn only if it ends with a tool result"
            )
        turns.append(Turn(id=turn_id, phase=phase, message_count=len(messages)))
    return tuple(messages), tuple(turns)


def _check_message(
    *,
    message: Any,
    position: int,
    pending: set[str],
    seen_calls: set[str],
    tool_names: set[str],
) -> None:
    role = message.get("role") if isinstance(message, dict) else None
    where = f"message {position} ({role})"
    if (role == "system") != (position == 0):
        raise FixtureError(
            f"{where}: the session must open with its only system message"
        )
    if pending and role != "tool":
        raise FixtureError(f"{where} arrives before results for {sorted(pending)}")
    if role in {"system", "user"}:
        _text(value=message.get("content"), where=f"{where} content")
    elif role == "assistant":
        calls = message.get("tool_calls")
        if calls is None:
            _text(value=message.get("content"), where=f"{where} content")
            return
        if not isinstance(calls, list) or not calls:
            raise FixtureError(f"{where} tool_calls must be a non-empty list")
        for call in calls:
            _check_tool_call(
                call=call, where=where, seen_calls=seen_calls, tool_names=tool_names
            )
            pending.add(call["id"])
    elif role == "tool":
        call_id = message.get("tool_call_id")
        if call_id not in pending:
            raise FixtureError(f"{where} does not answer a pending call: {call_id!r}")
        _text(value=message.get("content"), where=f"{where} content")
        pending.discard(call_id)
    else:
        raise FixtureError(f"{where}: unsupported role")


def _check_tool_call(
    *, call: Any, where: str, seen_calls: set[str], tool_names: set[str]
) -> None:
    function = call.get("function") if isinstance(call, dict) else None
    if not isinstance(function, dict) or call.get("type") != "function":
        raise FixtureError(f"{where} has a tool call that is not a function call")
    call_id = _text(value=call.get("id"), where=f"{where} tool call id")
    if call_id in seen_calls:
        raise FixtureError(f"{where} reuses tool call id {call_id!r}")
    if function.get("name") not in tool_names:
        raise FixtureError(
            f"{where} calls {function.get('name')!r}, which is not in the tool catalog"
        )
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (TypeError, json.JSONDecodeError):
        arguments = None
    if not isinstance(arguments, dict):
        raise FixtureError(f"{where} call {call_id!r} arguments must be a JSON object")
    seen_calls.add(call_id)
