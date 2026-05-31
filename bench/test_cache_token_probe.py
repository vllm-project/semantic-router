import importlib.util
from argparse import Namespace
from pathlib import Path

CACHED_TOKEN_SAMPLE = 42
EXPECTED_REQUESTS = 3
EXPECTED_CACHED_TOKENS = 60
EXPECTED_CACHED_RATIO = 0.2
EXPECTED_FIELD_RATE = 1.0
EXPECTED_RPS = 1.5
EXPECTED_METADATA_REPEATS = 2
CHAT_PROMPT_TOKENS = 100
CHAT_COMPLETION_TOKENS = 3
RESPONSES_PROMPT_TOKENS = 111
RESPONSES_COMPLETION_TOKENS = 7
CHAT_COMPLETIONS_CACHE_SOURCE = "usage.prompt_tokens_details.cached_tokens"
RESPONSES_CACHE_SOURCE = "usage.input_tokens_details.cached_tokens"
ROOT_CACHE_SOURCE = "usage.cached_tokens"


def load_probe_module():
    path = Path(__file__).with_name("cache_token_probe.py")
    spec = importlib.util.spec_from_file_location("cache_token_probe", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def response(cached=None, prompt_tokens=100, usage=None):
    usage = (
        {"prompt_tokens": prompt_tokens, "completion_tokens": CHAT_COMPLETION_TOKENS}
        if usage is None
        else usage
    )
    if cached is not None:
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return {"usage": usage, "model": "test-model"}


def row(probe, cached=None, success=True):
    return {
        "success": success,
        "status": 200 if success else 503,
        "latency_ms": 10.0,
        "prompt_tokens": 100,
        "completion_tokens": 3,
        "cached_tokens": int(cached or 0),
        "cached_token_field_present": cached is not None,
        "cached_token_source": (
            CHAT_COMPLETIONS_CACHE_SOURCE if cached is not None else ""
        ),
        "error": "" if success else "injected",
    }


def validation_args(**overrides):
    values = {
        "min_success_rate": 1.0,
        "min_cached_token_reporting": "missing",
        "min_cached_token_field_rate": 0.0,
        "min_cached_prompt_ratio": 0.0,
    }
    values.update(overrides)
    return Namespace(**values)


def test_cached_token_extraction_handles_missing_zero_and_positive_values():
    probe = load_probe_module()

    assert probe.cached_token_field_present(response()) is False
    assert probe.cached_tokens(response()) == 0
    assert probe.cached_token_field_present(response(0)) is True
    assert probe.cached_tokens(response(0)) == 0
    assert probe.cached_token_field_present(response(CACHED_TOKEN_SAMPLE)) is True
    assert probe.cached_tokens(response(CACHED_TOKEN_SAMPLE)) == CACHED_TOKEN_SAMPLE
    assert (
        probe.cached_token_source(response(CACHED_TOKEN_SAMPLE))
        == CHAT_COMPLETIONS_CACHE_SOURCE
    )
    responses_usage = {
        "input_tokens": 100,
        "output_tokens": 3,
        "input_tokens_details": {"cached_tokens": CACHED_TOKEN_SAMPLE},
    }
    assert probe.cached_token_field_present(response(usage=responses_usage)) is True
    assert probe.cached_tokens(response(usage=responses_usage)) == CACHED_TOKEN_SAMPLE
    assert (
        probe.cached_token_source(response(usage=responses_usage))
        == RESPONSES_CACHE_SOURCE
    )
    root_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 3,
        "cached_tokens": CACHED_TOKEN_SAMPLE,
    }
    assert probe.cached_token_field_present(response(usage=root_usage)) is True
    assert probe.cached_tokens(response(usage=root_usage)) == CACHED_TOKEN_SAMPLE
    assert probe.cached_token_source(response(usage=root_usage)) == ROOT_CACHE_SOURCE


def test_usage_extraction_handles_chat_and_responses_token_names():
    probe = load_probe_module()

    assert probe.usage_value(response(), probe.PROMPT_TOKEN_KEYS) == CHAT_PROMPT_TOKENS
    assert (
        probe.usage_value(response(), probe.COMPLETION_TOKEN_KEYS)
        == CHAT_COMPLETION_TOKENS
    )
    responses_usage = {
        "input_tokens": RESPONSES_PROMPT_TOKENS,
        "output_tokens": RESPONSES_COMPLETION_TOKENS,
        "input_tokens_details": {"cached_tokens": 11},
    }
    assert (
        probe.usage_value(response(usage=responses_usage), probe.PROMPT_TOKEN_KEYS)
        == RESPONSES_PROMPT_TOKENS
    )
    assert (
        probe.usage_value(response(usage=responses_usage), probe.COMPLETION_TOKEN_KEYS)
        == RESPONSES_COMPLETION_TOKENS
    )


def test_cache_reporting_state_distinguishes_missing_zero_and_positive():
    probe = load_probe_module()

    assert probe.cache_reporting_state([row(probe), row(probe)]) == "missing"
    assert probe.cache_reporting_state([row(probe, 0), row(probe, 0)]) == (
        "reported_zero"
    )
    assert probe.cache_reporting_state([row(probe, 0), row(probe, 7)]) == "positive"
    assert probe.cache_reporting_state([row(probe, 7, success=False)]) == "missing"


def test_summary_reports_cached_ratio_and_reporting_state():
    probe = load_probe_module()
    rows = [row(probe, 0), row(probe, 20), row(probe, 40)]

    summary = probe.summarize(rows, elapsed_seconds=2.0, label="router")

    assert summary["requests"] == EXPECTED_REQUESTS
    assert summary["success_rate"] == 1.0
    assert summary["cached_tokens"] == EXPECTED_CACHED_TOKENS
    assert summary["cached_prompt_ratio"] == EXPECTED_CACHED_RATIO
    assert summary["cached_token_reporting"] == "positive"
    assert summary["cached_token_field_rate"] == EXPECTED_FIELD_RATE
    assert summary["cached_token_source_counts"] == {
        CHAT_COMPLETIONS_CACHE_SOURCE: EXPECTED_REQUESTS
    }
    assert summary["probe_kind"] == probe.CACHE_PROBE_KIND
    assert summary["probe_repeats"] == EXPECTED_REQUESTS
    assert summary["requests_per_second"] == EXPECTED_RPS


def test_summary_records_repeated_prefix_probe_metadata():
    probe = load_probe_module()
    rows = [row(probe, 0), row(probe, 20)]

    summary = probe.summarize(rows, elapsed_seconds=2.0, label="router")

    assert summary["probe_kind"] == "repeated-prefix-cache-token-probe"
    assert summary["probe_repeats"] == EXPECTED_METADATA_REPEATS
    assert summary["stable_prefix_chars"] == len(probe.probe_prompt())
    assert summary["stable_prefix_sha256"] == probe.probe_prompt_hash()
    assert summary["unique_suffix_pattern"] == "probe_turn_index"


def test_validate_summary_gates_positive_cache_evidence():
    probe = load_probe_module()
    rows = [row(probe, 0), row(probe, 20), row(probe, 40)]
    summary = probe.summarize(rows, elapsed_seconds=2.0, label="router")

    failures = probe.validate_summary(
        validation_args(
            min_cached_token_reporting="positive",
            min_cached_token_field_rate=1.0,
            min_cached_prompt_ratio=0.1,
        ),
        summary,
        "router",
    )

    assert failures == []


def test_validate_summary_fails_missing_or_zero_cache_evidence():
    probe = load_probe_module()
    missing = probe.summarize([row(probe), row(probe)], 1.0, "router")
    reported_zero = probe.summarize([row(probe, 0), row(probe, 0)], 1.0, "router")

    assert probe.validate_summary(
        validation_args(
            min_cached_token_reporting="reported_zero",
            min_cached_token_field_rate=1.0,
        ),
        missing,
        "router",
    ) == [
        "router: cached_token_reporting missing < reported_zero",
        "router: cached_token_field_rate 0.0 < 1.0",
    ]
    assert probe.validate_summary(
        validation_args(
            min_cached_token_reporting="positive",
            min_cached_prompt_ratio=0.01,
        ),
        reported_zero,
        "router",
    ) == [
        "router: cached_token_reporting reported_zero < positive",
        "router: cached_prompt_ratio 0.0 < 0.01",
    ]


def test_validate_summary_fails_low_success_rate():
    probe = load_probe_module()
    summary = probe.summarize(
        [row(probe, 0), row(probe, 0, success=False)], 1.0, "router"
    )

    assert probe.validate_summary(validation_args(), summary, "router") == [
        "router: success_rate 0.5 < 1.0"
    ]
