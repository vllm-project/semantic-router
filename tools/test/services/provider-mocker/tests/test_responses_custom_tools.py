"""Provider-native custom tool fixtures used by the public Responses E2E profile."""

import json

import httpx
import pytest
from provider_mocker.app import create_app

MARKER = "__mock_responses_custom_tool__"
HISTORY_INPUT = "*** Begin Patch\n+hello\n*** End Patch"
PROVIDER_INPUT = "*** Begin Patch\n+provider\n*** End Patch"


def sse_events(body: str) -> list[dict]:
    return [
        json.loads(line[6:])
        for line in body.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize("backend", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_custom_tool_fixture_preserves_native_shape_and_history(backend, stream):
    app = create_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://fixture"
    ) as client:
        if backend == "chat":
            path = "/v1/chat/completions"
            request = {
                "model": "native-test",
                "stream": stream,
                "messages": [
                    {"role": "user", "content": MARKER},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "custom",
                                "custom": {
                                    "name": "apply_patch",
                                    "input": HISTORY_INPUT,
                                },
                            }
                        ],
                    },
                    {"role": "tool", "tool_call_id": "call_1", "content": "Success"},
                ],
                "tools": [{"type": "custom", "custom": {"name": "apply_patch"}}],
            }
        else:
            path = "/v1/responses"
            request = {
                "model": "native-test",
                "stream": stream,
                "input": [
                    {"role": "user", "content": MARKER},
                    {
                        "type": "custom_tool_call",
                        "call_id": "call_1",
                        "name": "apply_patch",
                        "input": HISTORY_INPUT,
                    },
                    {
                        "type": "custom_tool_call_output",
                        "call_id": "call_1",
                        "output": "Success",
                    },
                ],
                "tools": [{"type": "custom", "name": "apply_patch"}],
            }
        response = await client.post(path, json=request)
        assert response.status_code == 200
        observed = await client.get("/debug/last-request")
        assert observed.json()["body"] == request

        if not stream:
            payload = response.json()
            if backend == "chat":
                assert payload["choices"][0]["message"]["tool_calls"] == [
                    {
                        "id": "call_mock_patch",
                        "type": "custom",
                        "custom": {"name": "apply_patch", "input": PROVIDER_INPUT},
                    }
                ]
            else:
                assert payload["output"] == [
                    {
                        "type": "custom_tool_call",
                        "id": "item_mock_patch",
                        "call_id": "call_mock_patch",
                        "name": "apply_patch",
                        "input": PROVIDER_INPUT,
                        "status": "completed",
                    }
                ]
            return

        events = sse_events(response.text)
        if backend == "chat":
            calls = [
                call
                for event in events
                for choice in event["choices"]
                for call in choice["delta"].get("tool_calls", [])
            ]
            assert calls[0] == {"index": 0, "id": "call_mock_patch"}
            assert calls[1]["type"] == "custom"
            assert calls[1]["custom"]["name"] == "apply_patch"
            assert (
                calls[1]["custom"]["input"] + calls[2]["custom"]["input"]
                == PROVIDER_INPUT
            )
            assert events[-1]["choices"][0]["finish_reason"] == "tool_calls"
            assert response.text.endswith("data: [DONE]\n\n")
        else:
            assert [event["type"] for event in events] == [
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.custom_tool_call_input.delta",
                "response.custom_tool_call_input.delta",
                "response.custom_tool_call_input.done",
                "response.output_item.done",
                "response.completed",
            ]
            assert events[2]["item"]["input"] == ""
            assert events[3]["delta"] + events[4]["delta"] == PROVIDER_INPUT
            assert events[5]["input"] == PROVIDER_INPUT
            assert events[6]["item"]["input"] == PROVIDER_INPUT
            assert events[7]["response"]["output"][0]["input"] == PROVIDER_INPUT
