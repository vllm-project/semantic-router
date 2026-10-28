"""Provider fixtures expose live response decorations only for targeted probes."""

from provider_mocker.messages_wire import build_message
from provider_mocker.responses_wire import build_responses_response


def test_anthropic_diagnostics_are_probe_scoped() -> None:
    base = {"model": "claude-test", "messages": [{"role": "user", "content": "hello"}]}
    assert "diagnostics" not in build_message(base)
    decorated = build_message(
        {
            **base,
            "messages": [{"role": "user", "content": "__mock_anthropic_diagnostics__"}],
        }
    )
    assert decorated["diagnostics"] == {
        "cache_miss_reason": {"type": "tools_changed", "cache_missed_input_tokens": 4}
    }


def test_responses_decorations_are_probe_scoped() -> None:
    base = {"model": "gpt-5-nano", "input": "hello"}
    plain, _ = build_responses_response(base)
    assert "billing" not in plain

    decorated, _ = build_responses_response(
        {**base, "input": "__mock_responses_decorations__"}
    )
    assert decorated["access_programs"] is None
    assert decorated["billing"] == {"payer": "openai"}
    assert decorated["frequency_penalty"] == 0.0
    assert decorated["presence_penalty"] == 0.0
    assert decorated["tool_usage"] == {"web_search": {"num_requests": 0}}
