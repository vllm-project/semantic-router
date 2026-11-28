"""Route-replay cases built from router replay records (AG3).

The J3 pilot rebuilt its replay workload at *prompt* granularity because
its replay join was prompt-keyed; agentic traffic needs the conversation
back. `cases_from_replays` is the productized rebuild with an explicit
granularity:

- `session`: one case per conversation identity (the conversation_id, else
  the session_id), carrying that conversation's requests in replay order
  (turn_index, then record timestamp, then request id) with the recorded
  request messages, so a candidate router can be driven through the exact
  conversation and judged on routing only.
- `prompt`: one case per unique prompt - the J3 convention (Claude Code
  resends the same conversation prompts across turns).
- `request`: one case per request record.

Dedup rules are written into the returned provenance (records in, cases
out, the share of repeated keys), and an agentic source defaults to the
session granularity with a duplicate-rate warning above 50%. Records
without a readable conversation identity or request body are skipped with
their reason - missing evidence is reported, never guessed.

Pure functions, no I/O, deterministic: same records -> same cases bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

GRANULARITIES = ("session", "prompt", "request")
# AG3: an agentic source defaults to the session granularity; a duplicate
# share above this threshold warns that the granularity may be too fine.
DEFAULT_DUPLICATE_WARN_PCT = 50.0
USER_TURN_PHASE = "user_turn"
TOOL_LOOP_PHASE = "tool_loop"


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _request_body_messages(record: Mapping[str, Any]) -> list[Any] | None:
    """The recorded request's chat messages, or None when unreadable."""
    body = record.get("request_body")
    if isinstance(body, str):
        try:
            body = json.loads(body)
        except json.JSONDecodeError:
            return None
    if not isinstance(body, Mapping):
        return None
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return None
    return messages


def _user_text(messages: Sequence[Any]) -> str:
    """The latest user text in a message list (the prompt key of J3)."""
    for message in reversed(messages):
        if isinstance(message, Mapping) and message.get("role") == "user":
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content
    return ""


def request_phase_of(messages: Sequence[Any]) -> str:
    """Mirror of the Router's tool-loop rule over the request's messages.

    The replay record itself carries no phase; the phase of a replayed
    request is a property of its own messages (a tool result last, an
    assistant message carrying tool calls, or a user message right after a
    tool result means the conversation sits inside a tool loop).
    """
    if not messages or not isinstance(messages[-1], Mapping):
        return USER_TURN_PHASE
    latest = messages[-1]
    role = latest.get("role")
    if role == "tool" or (role == "assistant" and latest.get("tool_calls")):
        return TOOL_LOOP_PHASE
    if role == "user" and len(messages) > 1:
        previous = messages[-2]
        if isinstance(previous, Mapping) and previous.get("role") == "tool":
            return TOOL_LOOP_PHASE
    return USER_TURN_PHASE


def cases_from_replays(
    records: Sequence[Mapping[str, Any]],
    granularity: str,
    *,
    duplicate_warn_pct: float = DEFAULT_DUPLICATE_WARN_PCT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build route-replay cases at the declared granularity.

    Returns (cases, provenance). The provenance carries the dedup rules and
    their effects (records in, cases out, the duplicate share and whether
    the >50% warning fired) so a route-replay record can name exactly how
    its cases were built. Deterministic: cases keep first-seen order.
    """
    if granularity not in GRANULARITIES:
        raise ValueError(
            f"granularity must be one of {list(GRANULARITIES)}, got {granularity!r}"
        )
    skipped = {"no_identity": 0, "no_messages": 0, "no_prompt": 0}
    cases: list[dict[str, Any]] = []
    key_records = 0  # records that map to a case key at this granularity

    if granularity == "session":
        conversations: dict[str, list[Mapping[str, Any]]] = {}
        for record in records:
            conversation = record.get("conversation_id") or record.get("session_id")
            if not isinstance(conversation, str) or not conversation.strip():
                skipped["no_identity"] += 1
                continue
            conversations.setdefault(conversation, []).append(record)
        key_records = sum(len(v) for v in conversations.values())
        for conversation, conv_records in conversations.items():
            ordered = sorted(
                conv_records,
                key=lambda r: (
                    r.get("turn_index") if isinstance(r.get("turn_index"), int) else 0,
                    str(r.get("timestamp") or ""),
                    str(r.get("request_id") or ""),
                ),
            )
            requests = []
            for record in ordered:
                messages = _request_body_messages(record)
                prompt = record.get("prompt")
                prompt = prompt if isinstance(prompt, str) else ""
                if messages is None:
                    if not prompt.strip():
                        skipped["no_messages"] += 1
                        continue
                    # a record trimmed of its request body still rebuilds: the
                    # recorded prompt replays as the single user message
                    messages = [{"role": "user", "content": prompt}]
                requests.append(
                    {
                        "prompt": _user_text(messages) or prompt,
                        "messages": messages,
                        "phase": request_phase_of(messages),
                    }
                )
            if not requests:
                continue  # every body unreadable: provenance carries the skips
            prompt = next((req["prompt"] for req in requests if req["prompt"]), "")
            cases.append(
                {
                    "case_id": "sess-" + _short_hash(conversation),
                    "session_id": conversation,
                    "prompt": prompt,
                    "requests": requests,
                }
            )
        dedup_rule = (
            "one case per conversation identity (conversation_id, else "
            "session_id); requests ordered by (turn_index, record ts, request id)"
        )
    elif granularity == "request":
        for record in records:
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                skipped["no_prompt"] += 1
                continue
            request_id = str(record.get("request_id") or "")
            key = f"{prompt}\x00{request_id}" if request_id else prompt
            cases.append(
                {
                    "case_id": "req-" + _short_hash(key),
                    "prompt": prompt,
                    "request_id": request_id,
                }
            )
        key_records = len(cases)
        dedup_rule = "one case per request record"
    else:  # prompt granularity: unique prompts (the J3 convention)
        seen: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for record in records:
            prompt = record.get("prompt")
            if not isinstance(prompt, str) or not prompt.strip():
                skipped["no_prompt"] += 1
                continue
            if prompt not in seen:
                seen[prompt] = {
                    "case_id": "uq-" + _short_hash(prompt),
                    "prompt": prompt,
                }
                order.append(prompt)
        cases = [seen[prompt] for prompt in order]
        key_records = sum(
            1
            for record in records
            if isinstance(record.get("prompt"), str) and record["prompt"].strip()
        )
        dedup_rule = "one case per unique prompt (J3 convention)"

    duplicate_pct = (1.0 - len(cases) / key_records) * 100.0 if key_records > 0 else 0.0
    provenance = {
        "granularity": granularity,
        "records": len(records),
        "key_records": key_records,
        "cases": len(cases),
        "dedup_rule": dedup_rule,
        "duplicate_pct": round(duplicate_pct, 2),
        "duplicate_warning": duplicate_pct > duplicate_warn_pct,
        "skipped": skipped,
    }
    return cases, provenance


def load_route_cases(path: str) -> list[dict[str, Any]]:
    """Load a route-replay case list (JSONL) and validate both shapes.

    Session cases carry a non-empty `requests` list of message lists;
    prompt/request cases carry a non-empty `prompt`. `case_id` is required
    either way.
    """
    cases_path = Path(path)
    if not cases_path.is_file():
        raise ValueError(f"route-replay cases file not found: {path}")
    cases: list[dict[str, Any]] = []
    for lineno, raw_line in enumerate(
        cases_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not raw_line.strip():
            continue
        try:
            raw = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"route-replay cases line {lineno}: invalid JSON: {exc}"
            ) from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("case_id"), str):
            raise ValueError(f"route-replay cases line {lineno}: case_id required")
        requests = raw.get("requests")
        if isinstance(requests, list):
            if not requests:
                raise ValueError(
                    f"route-replay cases line {lineno}: session case needs "
                    "a non-empty requests list"
                )
            for index, request in enumerate(requests):
                messages = (
                    request.get("messages") if isinstance(request, dict) else None
                )
                if not isinstance(messages, list) or not messages:
                    raise ValueError(
                        f"route-replay cases line {lineno}: request {index} "
                        "needs a non-empty messages list"
                    )
        elif not isinstance(raw.get("prompt"), str) or not raw["prompt"].strip():
            raise ValueError(
                f"route-replay cases line {lineno}: prompt required for "
                "single-request cases"
            )
        cases.append(raw)
    if not cases:
        raise ValueError(f"route-replay cases file is empty: {path}")
    return cases
