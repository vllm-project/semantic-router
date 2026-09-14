"""Routing and cost fields for router benchmark reports.

Per-request values come from router response headers. Per-run totals can also
be read from the router's Prometheus endpoint, which works on routers that
predate the per-request headers.
"""

from __future__ import annotations

import re
import statistics
import urllib.request
from collections.abc import Iterable, Mapping
from typing import Any

SELECTED_MODEL_HEADER = "x-vsr-selected-model"
ROUTING_LATENCY_HEADER = "x-vsr-routing-latency-ms"
COST_HEADER = "x-vsr-cost"
COST_CURRENCY_HEADER = "x-vsr-cost-currency"
CACHE_HIT_HEADER = "x-vsr-cache-hit"

COST_METRIC = "llm_model_cost_total"
ROUTING_LATENCY_METRIC = "llm_model_routing_latency_seconds"
COST_BASIS = "configured pricing (x-vsr-cost), not a provider bill"

_METRIC_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{([^}]*)\})?\s+(\S+)")
_METRIC_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

MetricSamples = dict[tuple[str, tuple[tuple[str, str], ...]], float]


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _field(container: Any, name: str) -> Any:
    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(name)
    return getattr(container, name, None)


def usage_detail(usage: Any, group: str, name: str) -> int | None:
    """Read a nested usage count such as completion_tokens_details.reasoning_tokens."""
    value = _field(_field(usage, group), name)
    return int(value) if isinstance(value, (int, float)) else None


def response_accounting(
    headers: Mapping[str, str] | None, usage: Any, finish_reason: Any
) -> dict[str, Any]:
    """Per-request routing, cost and truncation fields; missing values stay blank."""
    headers = headers or {}
    return {
        "selected_model": headers.get(SELECTED_MODEL_HEADER) or "",
        "finish_reason": finish_reason or "",
        "reasoning_tokens": usage_detail(
            usage, "completion_tokens_details", "reasoning_tokens"
        ),
        "cached_tokens": usage_detail(usage, "prompt_tokens_details", "cached_tokens"),
        "routing_latency_ms": parse_float(headers.get(ROUTING_LATENCY_HEADER)),
        "cost": parse_float(headers.get(COST_HEADER)),
        "cost_currency": headers.get(COST_CURRENCY_HEADER) or "",
        "cache_hit": (headers.get(CACHE_HIT_HEADER) or "").lower() == "true",
    }


def count_values(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def split_summary(models: list[str]) -> dict[str, Any]:
    """Routing split in the same shape as the agentic benchmarks' selected_model_counts."""
    counts = count_values(models)
    return {
        "selected_model_counts": counts,
        "selected_model_share": {
            model: round(count / len(models), 4) for model, count in counts.items()
        },
        "unattributed_requests": len(models) - sum(counts.values()),
    }


def _percentile(ordered: list[float], pct: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct / 100
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def value_summary(values: Iterable[float | None]) -> dict[str, Any]:
    ordered = sorted(value for value in values if value is not None)
    if not ordered:
        return {"samples": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "samples": len(ordered),
        "mean": round(statistics.fmean(ordered), 3),
        "p50": round(_percentile(ordered, 50), 3),
        "p95": round(_percentile(ordered, 95), 3),
        "max": round(ordered[-1], 3),
    }


def cost_summary(costs: list[float | None], currencies: list[str]) -> dict[str, Any]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for cost, currency in zip(costs, currencies, strict=True):
        if cost is None:
            continue
        key = currency or "unknown"
        totals[key] = totals.get(key, 0.0) + cost
        counts[key] = counts.get(key, 0) + 1
    priced = sum(counts.values())
    return {
        "basis": COST_BASIS,
        "total": {key: round(value, 8) for key, value in sorted(totals.items())},
        "mean_per_priced_request": {
            key: round(value / counts[key], 8) for key, value in sorted(totals.items())
        },
        "priced_requests_by_currency": dict(sorted(counts.items())),
        "priced_requests": priced,
        "unpriced_requests": len(costs) - priced,
    }


def parse_metrics_text(text: str) -> MetricSamples:
    samples: MetricSamples = {}
    for line in text.splitlines():
        match = _METRIC_SAMPLE.match(line)
        if not match or not match.group(1).startswith(
            (COST_METRIC, ROUTING_LATENCY_METRIC)
        ):
            continue
        value = parse_float(match.group(3))
        if value is None:
            continue
        labels = tuple(sorted(_METRIC_LABEL.findall(match.group(2) or "")))
        samples[(match.group(1), labels)] = value
    return samples


def scrape_metrics(url: str, timeout: float = 10.0) -> MetricSamples:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return parse_metrics_text(response.read().decode("utf-8", errors="replace"))


def metrics_delta(before: MetricSamples, after: MetricSamples) -> dict[str, Any]:
    """What the router recorded between two scrapes, including any other traffic."""
    cost_by_model: dict[str, dict[str, float]] = {}
    for (name, labels), value in after.items():
        if name != COST_METRIC:
            continue
        spent = value - before.get((name, labels), 0.0)
        if spent <= 0:
            continue
        label_map = dict(labels)
        model_costs = cost_by_model.setdefault(label_map.get("model", ""), {})
        model_costs[label_map.get("currency", "")] = round(spent, 8)

    def diff(suffix: str) -> float:
        key = (ROUTING_LATENCY_METRIC + suffix, ())
        return after.get(key, 0.0) - before.get(key, 0.0)

    decisions = diff("_count")
    return {
        "cost_basis": COST_BASIS.replace("x-vsr-cost", COST_METRIC),
        "cost_by_model": dict(sorted(cost_by_model.items())),
        "routing_decisions": int(decisions),
        "routing_latency_ms_mean": (
            round(diff("_sum") / decisions * 1000, 3) if decisions > 0 else None
        ),
    }
