"""Keep the optional-tool probe's routing expectations aligned with the Chat wire."""

from types import SimpleNamespace

from live_speed_smoke import probe_optional_tools


def test_disabled_tools_make_the_fast_text_backend_eligible():
    cases = {}

    def capture(label, messages, lanes, model, finish, **options):
        cases[label] = (messages, model, finish, options)

    probe_optional_tools(capture, SimpleNamespace(fast="small", strong="large"))
    for stem in ("greeting", "explicit"):
        messages, model, finish, options = cases[f"{stem}-none"]
        assert messages == cases[f"{stem}-auto"][0]
        assert options["tools"] == cases[f"{stem}-auto"][3]["tools"]
        assert options["tool_choice"] == "none"
        assert options["parallel_tool_calls"] is False
        assert (model, finish) == ("small", "stop")
        assert cases[f"{stem}-auto"][1] == "large"
    assert cases["named"][1:3] == ("large", "tool_calls")
