import json

from provider_mocker.messages_wire import build_message, stream_message


def test_cache_cost_fixture_has_disjoint_aggregate_and_tiers():
    for aggregate_only in (False, True):
        prompt = "__mock_provider_cache_cost__"
        if aggregate_only:
            prompt += " __mock_cache_aggregate__"
        response = build_message(
            {"model": "claude", "messages": [{"role": "user", "content": prompt}]}
        )
        usage = response["usage"]
        assert usage["input_tokens"] == 10
        assert usage["cache_read_input_tokens"] == 10
        assert usage["cache_creation_input_tokens"] == 80
        assert usage["output_tokens"] == 2
        if aggregate_only:
            assert "cache_creation" not in usage
        else:
            assert usage["cache_creation"] == {
                "ephemeral_5m_input_tokens": 30,
                "ephemeral_1h_input_tokens": 50,
            }
        events = [
            json.loads(frame.split("data: ")[1]) for frame in stream_message(response)
        ]
        assert events[0]["message"]["usage"] == {**usage, "output_tokens": 0}
        assert events[-2]["usage"] == usage


def test_cache_read_heavy_fixture_reads_without_writes():
    response = build_message(
        {
            "model": "claude",
            "messages": [
                {
                    "role": "user",
                    "content": "__mock_provider_cache_cost__ __mock_cache_read_heavy__",
                }
            ],
        }
    )
    assert response["usage"] == {
        "input_tokens": 10,
        "cache_read_input_tokens": 90,
        "cache_creation_input_tokens": 0,
        "output_tokens": 2,
    }
