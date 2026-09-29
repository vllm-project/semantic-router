from types import SimpleNamespace

import pandas as pd

from bench.reasoning.router_accounting import (
    cost_summary,
    metrics_delta,
    parse_metrics_text,
    response_accounting,
)
from bench.reasoning.router_reason_bench_multi_dataset import analyze_results

METRICS_BEFORE = """
# HELP llm_model_cost_total The total cost attributed to each LLM model
# TYPE llm_model_cost_total counter
llm_model_cost_total{currency="USD",model="cloud"} 0.5
llm_model_routing_latency_seconds_sum 1.0
llm_model_routing_latency_seconds_count 10
"""

METRICS_AFTER = """
llm_model_cost_total{currency="USD",model="cloud"} 0.75
llm_model_cost_total{currency="USD",model="mid"} 0.01
llm_model_routing_latency_seconds_bucket{le="0.005"} 12
llm_model_routing_latency_seconds_sum 1.5
llm_model_routing_latency_seconds_count 14
"""


def make_row(model, finish_reason, cost, latency, reasoning=None, success=True):
    return {
        "mode_label": "Router_NR",
        "category": "Science",
        "is_correct": True,
        "response_time": 1.0,
        "success": success,
        "prompt_tokens": 100,
        "completion_tokens": 10,
        "total_tokens": 110,
        "selected_model": model,
        "finish_reason": finish_reason,
        "reasoning_tokens": reasoning,
        "cached_tokens": None,
        "routing_latency_ms": latency,
        "cost": cost,
        "cost_currency": "USD" if cost is not None else "",
        "cache_hit": False,
    }


def test_response_accounting_reads_router_headers_and_usage():
    usage = SimpleNamespace(
        completion_tokens_details=SimpleNamespace(reasoning_tokens=120),
        prompt_tokens_details=SimpleNamespace(cached_tokens=8),
    )
    headers = {
        "x-vsr-selected-model": "cloud",
        "x-vsr-routing-latency-ms": "0.412",
        "x-vsr-cost": "0.000054",
        "x-vsr-cost-currency": "USD",
    }

    fields = response_accounting(headers, usage, "length")

    assert fields["selected_model"] == "cloud"
    assert fields["routing_latency_ms"] == 0.412
    assert fields["cost"] == 0.000054
    assert fields["cost_currency"] == "USD"
    assert fields["reasoning_tokens"] == 120
    assert fields["cached_tokens"] == 8
    assert fields["finish_reason"] == "length"


def test_response_accounting_leaves_blanks_without_router_headers():
    fields = response_accounting({}, None, "stop")

    assert fields["selected_model"] == ""
    assert fields["cost"] is None
    assert fields["routing_latency_ms"] is None
    assert fields["reasoning_tokens"] is None


def test_metrics_delta_reports_cost_per_model_and_mean_routing_latency():
    delta = metrics_delta(
        parse_metrics_text(METRICS_BEFORE), parse_metrics_text(METRICS_AFTER)
    )

    assert delta["cost_by_model"] == {"cloud": {"USD": 0.25}, "mid": {"USD": 0.01}}
    assert delta["routing_decisions"] == 4
    assert delta["routing_latency_ms_mean"] == 125.0


def test_cost_summary_counts_priced_requests_per_currency():
    summary = cost_summary([0.002, 0.004, 0.01, None], ["USD", "USD", "EUR", ""])

    assert summary["total"] == {"EUR": 0.01, "USD": 0.006}
    assert summary["priced_requests_by_currency"] == {"EUR": 1, "USD": 2}
    assert summary["mean_per_priced_request"] == {"EUR": 0.01, "USD": 0.003}
    assert summary["unpriced_requests"] == 1


def test_analyze_results_reports_split_cost_latency_and_truncation():
    rows = [
        make_row("local", "stop", cost=0.0, latency=0.4),
        make_row("local", "stop", cost=0.0, latency=0.6),
        make_row("cloud", "length", cost=0.002, latency=0.5, reasoning=300),
        make_row("", "", cost=None, latency=None, success=False),
    ]

    analysis = analyze_results(pd.DataFrame(rows))

    assert analysis["selected_model_counts"] == {"cloud": 1, "local": 2}
    assert analysis["truncated_responses"] == 1
    assert analysis["reasoning_tokens_total"] == 300
    assert analysis["cost"]["total"] == {"USD": 0.002}
    assert analysis["cost"]["priced_requests"] == 3
    assert analysis["routing_latency_ms"]["p50"] == 0.5


def test_analyze_results_without_router_columns_is_unchanged():
    rows = [make_row("local", "stop", cost=None, latency=None)]
    for key in (
        "selected_model",
        "finish_reason",
        "reasoning_tokens",
        "cached_tokens",
        "routing_latency_ms",
        "cost",
        "cost_currency",
        "cache_hit",
    ):
        rows[0].pop(key)

    analysis = analyze_results(pd.DataFrame(rows))

    assert "selected_model_counts" not in analysis
    assert analysis["overall_accuracy"] == 1.0
