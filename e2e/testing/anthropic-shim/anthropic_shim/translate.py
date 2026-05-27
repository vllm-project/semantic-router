"""Pure translation helpers used by the proxy.

The functions here are deliberately framework-free so they can be
exercised by unit tests without spinning up a server or a model.

Three concerns are handled:

1. ``join_system_array`` flattens a Messages-API ``system`` array of
   ``TextBlockParam`` entries into a single newline-separated string.
   llama-server otherwise concatenates the text fields with no
   separator, which destroys word boundaries.
2. ``join_tool_result_content`` does the same for the ``content`` array
   of a ``tool_result`` block.
3. ``apply_cache_usage`` post-processes a Messages response and
   synthesises ``cache_creation_input_tokens`` /
   ``cache_read_input_tokens`` based on whether the inbound request
   carried ``cache_control`` markers and whether this session has seen
   the same request-body prefix before.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _is_text_block(block: Any) -> bool:
    return (
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def join_system_array(body: dict[str, Any]) -> dict[str, Any]:
    """Collapse ``system`` array into a single newline-joined string.

    Mutates and returns ``body``. No-op when ``system`` is missing or
    already a string. Non-text blocks are silently dropped (the
    Messages API permits only text blocks in ``system`` today).
    """
    system = body.get("system")
    if not isinstance(system, list):
        return body
    texts = [block["text"] for block in system if _is_text_block(block)]
    body["system"] = "\n".join(texts)
    return body


def join_tool_result_content(body: dict[str, Any]) -> dict[str, Any]:
    """Collapse every ``tool_result.content`` array into a single string.

    Walks all message blocks and rewrites ``tool_result`` entries whose
    ``content`` is an array of text blocks. Non-array contents
    (already a string, or absent) are left untouched. Non-text array
    entries are skipped, matching llama-server's expectation that
    ``tool_result.content`` carries only text.
    """
    messages = body.get("messages")
    if not isinstance(messages, list):
        return body
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            tr_content = block.get("content")
            if not isinstance(tr_content, list):
                continue
            texts = [item["text"] for item in tr_content if _is_text_block(item)]
            block["content"] = "\n".join(texts)
    return body


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
    and including the last block bearing ``cache_control``. The shim
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


def apply_cache_usage(
    response: dict[str, Any],
    request_had_cache_control: bool,
    prefix_seen: bool,
) -> dict[str, Any]:
    """Populate ``cache_creation_input_tokens`` and ``cache_read_input_tokens``.

    Mutates and returns ``response``. The Anthropic contract says:
    on the first request with a given cache prefix, the whole input
    counts as a creation; on subsequent requests, the same input
    counts as a read. llama-server reports neither field today, so the
    shim fills them in from ``input_tokens``.
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
