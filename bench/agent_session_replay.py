#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agentic_routing_live_benchmark import (
    HTTP_OK,
    HTTP_REDIRECT_START,
    cached_tokens,
    parse_extra_headers,
    post_json,
    selected_model_from,
    usage_value,
)

SCHEMA_VERSION: str = "agent-session-fixture.v1"
DEFAULT_FIXTURE: Path = (
    Path(__file__).with_name("data") / "coding_agent_session.v1.json"
)
FIXTURE_KEYS: frozenset[str] = frozenset(
    {"schema_version", "id", "description", "tools", "turns"}
)
TURN_KEYS: frozenset[str] = frozenset({"id", "phase", "append_messages"})
ROUTER_HEADERS: tuple[str, ...] = (
    "x-vsr-selected-decision",
    "x-vsr-session-phase",
    "x-vsr-replay-id",
)


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


def load_session(path: Path = DEFAULT_FIXTURE) -> AgentSession:
    try:
        raw = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_keys
        )
    except json.JSONDecodeError as exc:
        raise FixtureError(f"{path}: {exc}") from exc
    return parse_session(raw)


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


def build_request(
    *,
    session: AgentSession,
    turn: Turn,
    model: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "messages": list(session.messages[: turn.message_count]),
        "tools": list(session.tools),
        "max_tokens": max_tokens,
        "temperature": temperature,
    }


def replay(session: AgentSession, args: argparse.Namespace) -> list[dict[str, Any]]:
    url = args.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "vllm-sr-agent-session-replay",
        args.session_header: args.session_id,
    }
    if args.api_key_env:
        headers["Authorization"] = "Bearer " + os.environ[args.api_key_env]
    headers.update(parse_extra_headers(args.extra_header))
    rows: list[dict[str, Any]] = []
    previous_model = ""
    for turn in session.turns:
        body = build_request(
            session=session,
            turn=turn,
            model=args.model,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        row: dict[str, Any] = {
            "turn": turn.id,
            "phase": turn.phase,
            "messages": turn.message_count,
            "request_bytes": len(json.dumps(body)),
        }
        if not args.dry_run:
            result = post_json(
                url=url, body=body, headers=headers, timeout=args.timeout
            )
            row.update(
                result_row(
                    result=result, phase=turn.phase, previous_model=previous_model
                )
            )
            previous_model = row["selected_model"] or previous_model
        rows.append(row)
    return rows


def result_row(
    *, result: dict[str, Any], phase: str, previous_model: str
) -> dict[str, Any]:
    response = result.get("json") or {}
    headers = result.get("headers") or {}
    status = int(result.get("status") or 0)
    selected = selected_model_from(result)
    switched = bool(previous_model and selected and selected != previous_model)
    choices = response.get("choices") or [{}]
    return {
        "status": status,
        "success": HTTP_OK <= status < HTTP_REDIRECT_START,
        "latency_ms": round(float(result.get("latency_ms") or 0), 1),
        "selected_model": selected,
        "model_switched": switched,
        "tool_loop_switch_violation": phase == "tool_loop" and switched,
        "prompt_tokens": usage_value(response, "prompt_tokens"),
        "cached_tokens": cached_tokens(response),
        "finish_reason": choices[0].get("finish_reason") or "",
        "error": result.get("error", ""),
        **{name: headers.get(name, "") for name in ROUTER_HEADERS},
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    sent = [row for row in rows if "status" in row]
    return {
        "requests": len(rows),
        "sent": len(sent),
        "succeeded": sum(row["success"] for row in sent),
        "model_switches": sum(row["model_switched"] for row in sent),
        "tool_loop_switch_violations": sum(
            row["tool_loop_switch_violation"] for row in sent
        ),
        "selected_models": sorted(
            {row["selected_model"] for row in sent if row["selected_model"]}
        ),
    }


def stable_prefix_bytes(session: AgentSession) -> int:
    return len(json.dumps(session.messages[0])) + len(json.dumps(list(session.tools)))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a recorded coding-agent session through a router."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--base-url", default="http://127.0.0.1:8899/v1")
    parser.add_argument("--model", default="auto")
    parser.add_argument(
        "--api-key-env", default="", help="Environment variable holding the API key."
    )
    parser.add_argument("--session-header", default="x-session-id")
    parser.add_argument(
        "--session-id", default="", help="Defaults to a new ID for every run."
    )
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--extra-header", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        session = load_session(args.fixture)
    except (OSError, FixtureError) as exc:
        print(f"invalid fixture: {exc}", file=sys.stderr)
        return 2
    if args.api_key_env and not os.environ.get(args.api_key_env):
        print(f"{args.api_key_env} is not set", file=sys.stderr)
        return 2
    if not args.session_id:
        args.session_id = f"agent-session-fixture-{session.id}-{int(time.time())}"
    rows = replay(session, args)
    report = {
        "fixture": session.id,
        "session_id": args.session_id,
        "dry_run": args.dry_run,
        "tools": len(session.tools),
        "stable_prefix_bytes": stable_prefix_bytes(session),
        "turns": rows,
        "summary": summarize(rows),
    }
    print(json.dumps(report, indent=2))
    return 0 if all(row.get("success", True) for row in rows) else 1


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


if __name__ == "__main__":
    raise SystemExit(main())
