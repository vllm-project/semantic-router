from provider_mocker.chat_request import ChatMessage, ChatRequest
from provider_mocker.chat_wire import chat_contains, chat_requests_mock_tool


def test_chat_tool_marker_in_cached_text_part():
    req = ChatRequest(
        model="m",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "text",
                        "text": "Use lookup __mock_tool_call__",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            )
        ],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    assert chat_contains(req, "__mock_tool_call__")
    assert chat_requests_mock_tool(req)


def test_chat_marker_ignores_nontext_parts():
    req = ChatRequest(
        model="m",
        messages=[
            ChatMessage(
                role="user",
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/__mock_tool_call__.png"
                        },
                    }
                ],
            )
        ],
        tools=[{"type": "function", "function": {"name": "lookup"}}],
    )
    assert not chat_contains(req, "__mock_tool_call__")
    assert not chat_requests_mock_tool(req)
