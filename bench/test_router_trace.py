from router_trace import routing_headers, trace_members, warning_codes


def test_trace_members_unquote_strings_and_join_inner_lists():
    trace = (
        'config_hash="abc", confidence=0.91, algorithm=multi_factor, '
        'signals=("keyword:code" "language:en");scored, '
        'reasoning="say \\"hi\\" 100%25", system_prompt'
    )

    assert trace_members(trace) == {
        "config_hash": "abc",
        "confidence": "0.91",
        "algorithm": "multi_factor",
        "signals": "keyword:code,language:en",
        "reasoning": 'say "hi" 100%',
        "system_prompt": "?1",
    }


def test_warning_codes_leave_out_parameters():
    value = 'protocol;action=dropped;field="tools", hallucination, truncated;count=2'

    assert warning_codes(value) == ["protocol", "hallucination", "truncated"]


def test_routing_headers_place_trace_members_beside_the_headers():
    headers = {
        "X-VSR-Selected-Model": "qwen",
        "X-VSR-Trace": (
            "session_phase=tool_loop, "
            'signals=("conversation:multi_turn" "conversation:tools" "fact_check:needs")'
        ),
    }

    fields = routing_headers(headers)

    assert fields["x-vsr-selected-model"] == "qwen"
    assert fields["session_phase"] == "tool_loop"
    assert fields["matched_conversation"] == "multi_turn,tools"
    assert fields["matched_fact_check"] == "needs"
    assert "signals" not in fields
