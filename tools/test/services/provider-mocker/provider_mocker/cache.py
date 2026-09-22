"""Session-isolated synthetic cache usage for deterministic Messages fixtures."""

import hashlib
import json
from collections import OrderedDict
from typing import Any

MAX_CACHE_SESSIONS = 32
MAX_CACHE_PREFIXES = 128


class SessionCacheTracker:
    def __init__(self):
        self._seen = OrderedDict()

    def mark(self, session_id: str, prefix: str) -> bool:
        if session_id not in self._seen:
            if len(self._seen) >= MAX_CACHE_SESSIONS:
                self._seen.popitem(last=False)
            self._seen[session_id] = OrderedDict()
        self._seen.move_to_end(session_id)
        bucket = self._seen[session_id]
        seen = prefix in bucket
        bucket[prefix] = None
        bucket.move_to_end(prefix)
        if len(bucket) > MAX_CACHE_PREFIXES:
            bucket.popitem(last=False)
        return seen


def has_cache_control(body: dict[str, Any]) -> bool:
    """Return True when any block in the request carries a ``cache_control`` marker."""
    system = body.get("system")
    if isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and "cache_control" in block:
                return True
    for message in body.get("messages", []) or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    for tool in body.get("tools", []) or []:
        if isinstance(tool, dict) and "cache_control" in tool:
            return True
    return False


def cache_prefix_hash(body: dict[str, Any]) -> str:
    """Hash the cacheable prefix of a request.

    Anthropic's prompt-cache contract keys on the request prefix up to
    and including the last block bearing ``cache_control``. The fixture
    walks ``tools``, ``system``, then each message's ``content``
    blocks in order, and stops at the last marker. Everything past
    that marker (subsequent turns, the final user query) is treated as
    cache-irrelevant. The hash is opaque; only equality matters.
    """
    prefix_parts: list[Any] = []

    tool_prefix = _tools_prefix(body.get("tools") or [])
    if tool_prefix is not None:
        prefix_parts.append({"tools": tool_prefix})

    system = body.get("system")
    if _should_include_system(body, system, tool_prefix is not None):
        prefix_parts.append({"system": system})

    message_prefix = _messages_prefix(body.get("messages") or [])
    if message_prefix is not None:
        prefix_parts.append({"messages": message_prefix})

    digest = hashlib.sha256(
        json.dumps(prefix_parts, sort_keys=True, default=str).encode("utf-8")
    )
    return digest.hexdigest()


def apply_cache_usage(
    response: dict[str, Any],
    request_had_cache_control: bool,
    prefix_seen: bool,
) -> dict[str, Any]:
    """Populate ``cache_creation_input_tokens`` and ``cache_read_input_tokens``.

    Mutates and returns ``response``. The Anthropic contract says:
    on the first request with a given cache prefix, the whole input
    counts as a creation; on subsequent requests, the same input
    counts as a read. The fixture derives both counters from ``input_tokens``.
    """
    if not request_had_cache_control:
        return response
    usage = response.get("usage")
    if not isinstance(usage, dict):
        return response
    input_tokens = usage.get("input_tokens", 0)
    if prefix_seen:
        usage.setdefault("cache_creation_input_tokens", 0)
        usage["cache_read_input_tokens"] = input_tokens
    else:
        usage["cache_creation_input_tokens"] = input_tokens
        usage.setdefault("cache_read_input_tokens", 0)
    return response


def _tools_prefix(tools: list[Any]) -> list[Any] | None:
    """Return tools up to and including the last cache_control marker, or None."""
    last_marker = -1
    for idx, tool in enumerate(tools):
        if isinstance(tool, dict) and "cache_control" in tool:
            last_marker = idx
    return tools[: last_marker + 1] if last_marker >= 0 else None


def _should_include_system(
    body: dict[str, Any], system: Any, tools_have_marker: bool
) -> bool:
    """The system block joins the prefix if any later cached block exists."""
    if system is None:
        return False
    if tools_have_marker:
        return True
    if isinstance(system, list) and any(
        isinstance(block, dict) and "cache_control" in block for block in system
    ):
        return True
    return _messages_have_marker(body)


def _messages_prefix(messages: list[Any]) -> list[Any] | None:
    """Return messages truncated to the last cache_control marker, or None."""
    cutoff_message = -1
    cutoff_block = -1
    for m_idx, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for b_idx, block in enumerate(content):
            if isinstance(block, dict) and "cache_control" in block:
                cutoff_message = m_idx
                cutoff_block = b_idx
    if cutoff_message < 0:
        return None
    truncated: list[Any] = list(messages[:cutoff_message])
    last = messages[cutoff_message]
    if isinstance(last, dict) and isinstance(last.get("content"), list):
        partial = dict(last)
        partial["content"] = last["content"][: cutoff_block + 1]
        truncated.append(partial)
    else:
        truncated.append(last)
    return truncated


def _messages_have_marker(body: dict[str, Any]) -> bool:
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "cache_control" in block:
                    return True
    return False
