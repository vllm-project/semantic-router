import importlib.util
from pathlib import Path

CACHED_TOKEN_SAMPLE = 42
EXPECTED_REQUESTS = 3
EXPECTED_CACHED_TOKENS = 60
EXPECTED_CACHED_RATIO = 0.2
EXPECTED_RPS = 1.5


def load_probe_module():
    path = Path(__file__).with_name("cache_token_probe.py")
    spec = importlib.util.spec_from_file_location("cache_token_probe", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def response(cached=None, prompt_tokens=100):
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": 3}
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
        "error": "" if success else "injected",
    }


def test_cached_token_extraction_handles_missing_zero_and_positive_values():
    probe = load_probe_module()

    assert probe.cached_token_field_present(response()) is False
    assert probe.cached_tokens(response()) == 0
    assert probe.cached_token_field_present(response(0)) is True
    assert probe.cached_tokens(response(0)) == 0
    assert probe.cached_token_field_present(response(CACHED_TOKEN_SAMPLE)) is True
    assert probe.cached_tokens(response(CACHED_TOKEN_SAMPLE)) == CACHED_TOKEN_SAMPLE


def test_cache_reporting_state_distinguishes_missing_zero_and_positive():
    probe = load_probe_module()

    assert probe.cache_reporting_state([row(probe), row(probe)]) == "missing"
    assert probe.cache_reporting_state([row(probe, 0), row(probe, 0)]) == (
        "reported_zero"
    )
    assert probe.cache_reporting_state([row(probe, 0), row(probe, 7)]) == "positive"


def test_summary_reports_cached_ratio_and_reporting_state():
    probe = load_probe_module()
    rows = [row(probe, 0), row(probe, 20), row(probe, 40)]

    summary = probe.summarize(rows, elapsed_seconds=2.0, label="router")

    assert summary["requests"] == EXPECTED_REQUESTS
    assert summary["success_rate"] == 1.0
    assert summary["cached_tokens"] == EXPECTED_CACHED_TOKENS
    assert summary["cached_prompt_ratio"] == EXPECTED_CACHED_RATIO
    assert summary["cached_token_reporting"] == "positive"
    assert summary["requests_per_second"] == EXPECTED_RPS
