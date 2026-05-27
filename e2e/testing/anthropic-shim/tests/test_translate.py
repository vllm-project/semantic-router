"""Unit tests for the translation helpers.

Three gap-translation cases (system-array join, tool_result array join,
cache token synthesis) plus edge cases (nil-safety, mixed content,
malformed payloads).
"""

from __future__ import annotations

import copy

from anthropic_shim.translate import (
    apply_cache_usage,
    cache_prefix_hash,
    has_cache_control,
    join_system_array,
    join_tool_result_content,
)


def test_join_system_array_collapses_text_blocks_with_newline() -> None:
    body = {
        "system": [
            {"type": "text", "text": "You are a helpful assistant."},
            {"type": "text", "text": "Be very concise."},
        ]
    }
    join_system_array(body)
    assert body["system"] == "You are a helpful assistant.\nBe very concise."


def test_join_system_array_passes_through_string() -> None:
    body = {"system": "already a string"}
    join_system_array(body)
    assert body["system"] == "already a string"


def test_join_system_array_missing_is_noop() -> None:
    body: dict = {"messages": []}
    join_system_array(body)
    assert "system" not in body


def test_join_system_array_drops_non_text_blocks() -> None:
    body = {
        "system": [
            {"type": "text", "text": "Hello"},
            {"type": "image", "source": {"type": "base64"}},
            {"type": "text", "text": "World"},
        ]
    }
    join_system_array(body)
    assert body["system"] == "Hello\nWorld"


def test_join_tool_result_content_collapses_text_blocks() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "abc",
                        "content": [
                            {"type": "text", "text": "first line"},
                            {"type": "text", "text": "second line"},
                        ],
                    }
                ],
            }
        ]
    }
    join_tool_result_content(body)
    block = body["messages"][0]["content"][0]
    assert block["content"] == "first line\nsecond line"


def test_join_tool_result_content_preserves_string_form() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "abc",
                        "content": "already a string",
                    }
                ],
            }
        ]
    }
    original = copy.deepcopy(body)
    join_tool_result_content(body)
    assert body == original


def test_join_tool_result_content_handles_mixed_blocks() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "irrelevant"},
                    {
                        "type": "tool_result",
                        "tool_use_id": "abc",
                        "content": [{"type": "text", "text": "one"}],
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "def",
                        "content": [
                            {"type": "text", "text": "two"},
                            {"type": "text", "text": "three"},
                        ],
                    },
                ],
            }
        ]
    }
    join_tool_result_content(body)
    blocks = body["messages"][0]["content"]
    assert blocks[1]["content"] == "one"
    assert blocks[2]["content"] == "two\nthree"


def test_join_tool_result_content_missing_messages_is_noop() -> None:
    body: dict = {}
    join_tool_result_content(body)
    assert body == {}


def test_has_cache_control_detects_system_block_marker() -> None:
    body = {
        "system": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}},
        ]
    }
    assert has_cache_control(body) is True


def test_has_cache_control_detects_message_block_marker() -> None:
    body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Hello",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            }
        ]
    }
    assert has_cache_control(body) is True


def test_has_cache_control_detects_tool_marker() -> None:
    body = {
        "tools": [
            {"name": "calc", "input_schema": {}, "cache_control": {"type": "ephemeral"}}
        ]
    }
    assert has_cache_control(body) is True


def test_has_cache_control_negative_when_absent() -> None:
    body = {"system": "plain", "messages": [{"role": "user", "content": "hi"}]}
    assert has_cache_control(body) is False


def test_apply_cache_usage_creation_on_first_request() -> None:
    response = {"usage": {"input_tokens": 42, "output_tokens": 7}}
    apply_cache_usage(response, request_had_cache_control=True, prefix_seen=False)
    assert response["usage"]["cache_creation_input_tokens"] == 42
    assert response["usage"]["cache_read_input_tokens"] == 0


def test_apply_cache_usage_read_on_repeat_request() -> None:
    response = {"usage": {"input_tokens": 42, "output_tokens": 7}}
    apply_cache_usage(response, request_had_cache_control=True, prefix_seen=True)
    assert response["usage"]["cache_creation_input_tokens"] == 0
    assert response["usage"]["cache_read_input_tokens"] == 42


def test_apply_cache_usage_noop_without_cache_control() -> None:
    response = {"usage": {"input_tokens": 42, "output_tokens": 7}}
    apply_cache_usage(response, request_had_cache_control=False, prefix_seen=True)
    assert "cache_creation_input_tokens" not in response["usage"]
    assert "cache_read_input_tokens" not in response["usage"]


def test_apply_cache_usage_handles_missing_usage() -> None:
    response: dict = {"id": "msg_1"}
    apply_cache_usage(response, request_had_cache_control=True, prefix_seen=False)
    assert response == {"id": "msg_1"}


def test_cache_prefix_hash_stable_across_repeat_requests() -> None:
    body_a = {
        "system": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": "first turn"}],
    }
    body_b = {
        "system": [
            {"type": "text", "text": "Hello", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "ack"},
            {"role": "user", "content": "second turn"},
        ],
    }
    # body_b appends new turns after the cached prefix; hashes should match
    # because the cache_control marker is in `system`, so the prefix is just
    # the system block.
    assert cache_prefix_hash(body_a) == cache_prefix_hash(body_b)


def test_cache_prefix_hash_differs_when_prefix_changes() -> None:
    body_a = {
        "system": [
            {"type": "text", "text": "Hello A", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [],
    }
    body_b = {
        "system": [
            {"type": "text", "text": "Hello B", "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [],
    }
    assert cache_prefix_hash(body_a) != cache_prefix_hash(body_b)
