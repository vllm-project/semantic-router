import re

from cli.api_registry import API_NAMES, PATTERNS, served_path_regex
from cli.models import Listener


def test_prefixed_api_is_also_served_under_its_prefix():
    regex = re.compile(
        served_path_regex(("messages", "chat_completions"), {"messages": "/anthropic"})
    )
    for path in (
        "/v1/messages",
        "/v1/messages/count_tokens",
        "/anthropic/v1/messages",
        "/anthropic/v1/messages/count_tokens",
        "/v1/models",
    ):
        assert regex.fullmatch(path), path
    for path in (
        "/anthropic/v1/messagesx",
        "/anthropicx/v1/messages",
        "/anthropic/v1/chat/completions",
        "/anthropic/anthropic/v1/messages",
    ):
        assert not regex.fullmatch(path), path


def test_no_prefix_changes_nothing():
    assert served_path_regex(("messages",)) == served_path_regex(("messages",), {})


def test_listener_reports_prefixes_by_canonical_api_name():
    listener = Listener(
        name="p", port=1, apis={"anthropic": {"prefix": "/anthropic"}, "openai": {}}
    )
    assert listener.api_prefixes() == {"messages": "/anthropic"}


def test_bad_prefix_is_refused():
    for bad in ("/", "anthropic", "/anthropic/", "/a b"):
        try:
            Listener(name="p", port=1, apis={"anthropic": {"prefix": bad}})
        except ValueError:
            continue
        raise AssertionError(bad)


def test_every_api_is_served_under_its_prefix_and_nowhere_else():
    samples = {
        "chat_completions": ["/v1/chat/completions", "/openai/v1/chat/completions"],
        "responses": [
            "/v1/responses",
            "/v1/responses/resp_1",
            "/v1/responses/resp_1/input_items",
            "/openai/responses",
        ],
        "messages": ["/v1/messages", "/v1/messages/count_tokens"],
        "images": ["/v1/images/generations"],
        "systemone": [
            "/v1/systemone",
            "/v1/systemone/batch",
            "/v1/systemone/models",
            "/v1/decisions",
        ],
    }
    assert set(samples) == set(API_NAMES), "add the new API to samples"
    prefixes = {name: f"/m-{name}" for name in API_NAMES}
    regex = re.compile(served_path_regex(tuple(API_NAMES), prefixes))
    for name, paths in samples.items():
        for path in paths:
            assert re.fullmatch(PATTERNS[name], path), path
            assert regex.fullmatch(path), path
            assert regex.fullmatch(prefixes[name] + path), (name, path)
            for other, prefix in prefixes.items():
                if other != name:
                    assert not regex.fullmatch(prefix + path), (other, name, path)
            misses = [
                prefixes[name] + "x" + path,
                prefixes[name] + path + "x",
                prefixes[name] * 2 + path,
            ]
            if path.endswith("/resp_1"):
                del misses[1]  # resp_1x is another response id
            for miss in misses:
                assert not regex.fullmatch(miss), miss
