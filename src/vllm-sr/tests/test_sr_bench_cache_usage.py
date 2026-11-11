from __future__ import annotations

import copy
from http import HTTPStatus
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from cli.sr_bench.engine import Engine
from cli.sr_bench.store import Store
from cli.sr_bench.transport import (
    CACHE_OBSERVATION_VERSION,
    normalize_usage,
    observe_cache_usage,
    usage_presence,
)
from test_sr_bench import Target, manifest, wait_run
from test_sr_bench import target as target  # noqa: PLC0414 - pytest fixture re-export


@pytest.mark.parametrize("cached", [0, 40])
def test_verified_aliases_are_one_observation_without_changing_accounting(
    cached: int,
) -> None:
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": cached, "cache_write_tokens": 10},
        "input_tokens_details": {"cached_tokens": cached},
        "prompt_cache_hit_tokens": cached,
        "prompt_cache_miss_tokens": 100 - cached,
    }
    before = copy.deepcopy(raw)
    accounting = normalize_usage(raw)
    presence = usage_presence(raw)
    observation = observe_cache_usage(raw=raw)
    assert observation["version"] == CACHE_OBSERVATION_VERSION
    assert observation["qualification"] == "qualified"
    assert observation["source"] == "usage.prompt_tokens_details.cached_tokens"
    assert observation["cached_tokens"] == cached
    assert observation["usage"] == {
        "input_tokens": 90 - cached,
        "cached_input_tokens": cached,
        "cache_write_tokens": 10,
        "output_tokens": 5,
    }
    assert raw == before
    assert normalize_usage(raw) == accounting
    assert usage_presence(raw) == presence


@pytest.mark.parametrize(
    "fields",
    [
        {"prompt_tokens_details": {"cached_tokens": 40}, "prompt_cache_hit_tokens": 41},
        {
            "prompt_tokens_details": {"cached_tokens": 40},
            "input_tokens_details": {"cached_tokens": 41},
        },
        {"prompt_tokens_details": {"cached_tokens": 40}, "cached_tokens": 41},
        {"prompt_cache_hit_tokens": 40, "prompt_cache_miss_tokens": 61},
    ],
)
def test_conflicting_cache_evidence_never_selects_a_convenient_alias(
    fields: dict[str, Any],
) -> None:
    raw = {"prompt_tokens": 100, "completion_tokens": 5, **fields}
    observation = observe_cache_usage(raw=raw)
    assert observation["qualification"] == "conflicting"
    assert observation["state"] is None
    assert observation["cached_tokens"] is None
    assert observation["usage"] is None
    assert observation["reasons"]


@pytest.mark.parametrize(
    "raw",
    [
        {"prompt_tokens": "100", "prompt_cache_hit_tokens": 40},
        {"prompt_tokens": 20, "prompt_cache_hit_tokens": 40},
        {"completion_tokens": -1, "prompt_cache_hit_tokens": 40},
        {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_cache_hit_tokens": 40,
            "prompt_cache_miss_tokens": "60",
        },
        {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": {"cached_tokens": 40},
            "prompt_cache_hit_tokens": False,
        },
        {
            "prompt_tokens": 100,
            "completion_tokens": 5,
            "prompt_tokens_details": "invalid",
            "prompt_cache_hit_tokens": 40,
        },
    ],
)
def test_malformed_partial_and_duplicate_values_remain_malformed(
    raw: dict[str, Any],
) -> None:
    observation = observe_cache_usage(raw=raw)
    assert observation["qualification"] == "malformed"
    assert observation["usage"] is None
    assert observation["reasons"]


@pytest.mark.parametrize("missing", ["prompt_tokens", "completion_tokens"])
def test_partial_usage_keeps_known_cache_value_without_qualified_totals(
    missing: str,
) -> None:
    raw = {
        "prompt_tokens": 100,
        "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 40},
    }
    raw.pop(missing)
    observation = observe_cache_usage(raw=raw)
    assert observation["field_present"] is True
    assert observation["cached_tokens"] == 40
    assert observation["state"] == "positive"
    assert observation["qualification"] == "incomplete"
    assert observation["usage"] is None
    assert normalize_usage(raw) is None


@pytest.mark.parametrize("cached", [0, 40])
def test_legacy_root_value_is_only_raw_compatibility_evidence(cached: int) -> None:
    raw = {"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": cached}
    observation = observe_cache_usage(raw=raw)
    assert observation["field_present"] is True
    assert observation["values"] == {"usage.cached_tokens": cached}
    assert observation["qualification"] == "unqualified"
    assert observation["cached_tokens"] is None
    assert observation["usage"] is None
    assert normalize_usage(raw)["cached_input_tokens"] == 0
    assert usage_presence(raw) == (False, False)


@pytest.mark.parametrize(
    ("events", "qualification", "cached", "cost"),
    [
        (
            [{"prompt_tokens": 100, "completion_tokens": 5}],
            "missing",
            0,
            0.000115,
        ),
        (
            [
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_cache_hit_tokens": 40,
                }
            ],
            "qualified",
            0,
            0.000115,
        ),
        (
            [{"prompt_tokens": 100, "completion_tokens": 5, "cached_tokens": 40}],
            "unqualified",
            0,
            0.000115,
        ),
        (
            [
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "prompt_cache_hit_tokens": 41,
                }
            ],
            "conflicting",
            40,
            0.000079,
        ),
        (
            [
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_cache_hit_tokens": "40",
                },
                {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "prompt_tokens_details": {"cached_tokens": 40},
                },
            ],
            "qualified",
            40,
            0.000079,
        ),
        (
            [{"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 40}}],
            "incomplete",
            None,
            None,
        ),
    ],
)
def test_context_saves_response_observations_without_repricing_binary_calls(
    tmp_path: Path,
    target: ThreadingHTTPServer,
    events: list[dict[str, Any]],
    qualification: str,
    cached: int | None,
    cost: float | None,
) -> None:
    target.usage_events = events
    store = Store(tmp_path)
    engine = Engine(store)
    run = engine.start(manifest(target, cost_policy="capability_only"))
    try:
        assert wait_run(store, run["id"])["status"] == "completed"
        call = store.calls(run["id"])[0]
        observation = call["provider_cache_observation"]
        assert call["response_status"] == HTTPStatus.OK
        assert observation["qualification"] == qualification
        assert call["raw_usage"] == events[-1]
        assert call["usage"] == normalize_usage(events[-1])
        assert call["cache_read_reported"] == any(
            usage_presence(raw)[0] for raw in events
        )
        if cached is None:
            assert call["usage"] is None
            assert call["cost_usd"] is None
        else:
            assert call["usage"]["cached_input_tokens"] == cached
            assert call["cost_usd"] == pytest.approx(cost)
        if len(events) > 1:
            assert observation["event_issues"] == {"malformed": 1}
        streams = list((store.root / "runs" / run["id"]).glob("*/*.sse"))
        assert len(streams) == 1
        text = streams[0].read_text()
        assert all(
            json_fragment in text
            for json_fragment in ('"prompt_tokens": 100', '"usage":')
        )
        assert store.results(run["id"])[0]["correct"] is True
    finally:
        engine.threads[run["id"]].join(timeout=5)
        assert not engine.threads[run["id"]].is_alive()
        store.db.close()


class FailedTarget(Target):
    def do_POST(self) -> None:  # noqa: N802 - HTTP handler callback
        self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)


def test_failed_http_response_retains_the_fault_branch_status_contract(
    tmp_path: Path, target: ThreadingHTTPServer
) -> None:
    target.RequestHandlerClass = FailedTarget
    store = Store(tmp_path)
    engine = Engine(store)
    run = engine.start(manifest(target, cost_policy="capability_only"))
    try:
        assert wait_run(store, run["id"])["status"] == "failed"
        call = store.calls(run["id"])[0]
        assert call["status"] == "failed"
        assert call["response_status"] == HTTPStatus.SERVICE_UNAVAILABLE
        assert call["usage"] is None
        assert call["provider_cache_observation"]["cached_tokens"] is None
    finally:
        engine.threads[run["id"]].join(timeout=5)
        assert not engine.threads[run["id"]].is_alive()
        store.db.close()
