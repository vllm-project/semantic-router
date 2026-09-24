"""Optional public-safe Prometheus counter snapshots around throughput waves."""

from __future__ import annotations

import math
import re
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .transport import OPENER, _consume

METRIC_NAMES = {
    "decision_runtime_row_preparation_duration_seconds_total": "row_preparation_seconds",
    "decision_runtime_row_preparations_total": "row_preparations",
    "decision_runtime_physical_batches_total": "physical_batches",
    "decision_runtime_physical_batch_rows_total": "physical_batch_rows",
}
BUCKET_NAME = "decision_runtime_physical_batch_size_bucket"
GRAPH_EVENT_NAME = "decision_runtime_qwen_rocm_graph_events_total"
GRAPH_EVENTS = ("capture", "replay", "fallback")
HTTP_OK = 200
LINE = re.compile(
    r"(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    r"(?:\s+[0-9]+)?\s*\Z"
)
LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="([^"\\]*)"')


class MetricsError(ValueError):
    """Only an error code is exposed; never an endpoint or response body."""


def validate_metrics_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as error:
        raise MetricsError("metrics_url_invalid") from error
    if (
        url.strip() != url
        or parts.scheme not in {"http", "https"}
        or not parts.hostname
        or port == 0
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.path != "/metrics"
    ):
        raise MetricsError("metrics_url_invalid")
    return url


@dataclass(frozen=True)
class MetricSnapshot:
    sha256: str
    counters: dict[str, float]
    buckets: dict[str, float]
    graph_events: dict[str, float]

    def public_record(self) -> dict[str, object]:
        return {
            "response_sha256": self.sha256,
            "counters": self.counters,
            "physical_batch_size_buckets": self.buckets,
            "graph_events": self.graph_events,
        }


@dataclass(frozen=True)
class MetricCapture:
    arm: str
    question_count: int
    state_count: int
    concurrency: int
    round: int
    before: MetricSnapshot | None
    after: MetricSnapshot | None
    error_code: str | None

    def delta(
        self,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]] | None:
        if self.before is None or self.after is None or self.error_code is not None:
            return None
        counters = {
            key: self.after.counters[key] - self.before.counters[key]
            for key in METRIC_NAMES.values()
        }
        if set(self.before.buckets) != set(self.after.buckets):
            return None
        if set(self.before.graph_events) != set(GRAPH_EVENTS) or set(
            self.after.graph_events
        ) != set(GRAPH_EVENTS):
            return None
        buckets = {
            key: self.after.buckets[key] - self.before.buckets[key]
            for key in self.before.buckets
        }
        graph_events = {
            key: self.after.graph_events[key] - self.before.graph_events[key]
            for key in GRAPH_EVENTS
        }
        if any(
            value < 0
            for value in (*counters.values(), *buckets.values(), *graph_events.values())
        ):
            return None
        return counters, buckets, graph_events

    def public_record(self) -> dict[str, object]:
        delta = self.delta()
        return {
            "arm": self.arm,
            "question_count": self.question_count,
            "state_count": self.state_count,
            "concurrency": self.concurrency,
            "round": self.round,
            "before": self.before.public_record() if self.before else None,
            "after": self.after.public_record() if self.after else None,
            "delta": (
                {
                    "counters": delta[0],
                    "physical_batch_size_buckets": delta[1],
                    "graph_events": delta[2],
                }
                if delta is not None
                else None
            ),
            "error_code": self.error_code
            or (
                "metrics_counter_reset"
                if self.before and self.after and delta is None
                else None
            ),
        }


def _parse_snapshot(body: bytes, sha256: str, model: str) -> MetricSnapshot:
    try:
        lines = body.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise MetricsError("metrics_invalid_text") from error
    counters: dict[str, float] = {}
    buckets: dict[str, float] = {}
    graph_events = dict.fromkeys(GRAPH_EVENTS, 0.0)
    observed_graph_events: set[str] = set()
    for line in lines:
        if not line or line.startswith("#"):
            continue
        name = line.split("{", 1)[0].split(" ", 1)[0]
        if name not in METRIC_NAMES and name not in (BUCKET_NAME, GRAPH_EVENT_NAME):
            continue
        match = LINE.fullmatch(line)
        if match is None:
            raise MetricsError("metrics_invalid_text")
        label_text = match.group("labels") or ""
        label_pairs = LABEL.findall(label_text)
        labels = dict(label_pairs)
        if name == GRAPH_EVENT_NAME and (
            len(label_pairs) != 2
            or len(labels) != 2
            or ",".join(f'{key}="{value}"' for key, value in label_pairs) != label_text
        ):
            raise MetricsError("metrics_invalid_graph_events")
        if labels.get("model") != model:
            continue
        value = float(match.group("value"))
        if not math.isfinite(value) or value < 0:
            raise MetricsError("metrics_invalid_value")
        if name == BUCKET_NAME:
            le = labels.get("le")
            if le is None or le in buckets:
                raise MetricsError("metrics_invalid_histogram")
            buckets[le] = value
        elif name == GRAPH_EVENT_NAME:
            event = labels.get("event")
            if (
                set(labels) != {"model", "event"}
                or event not in GRAPH_EVENTS
                or event in observed_graph_events
                or not value.is_integer()
            ):
                raise MetricsError("metrics_invalid_graph_events")
            observed_graph_events.add(event)
            graph_events[event] = value
        else:
            key = METRIC_NAMES[name]
            if key in counters:
                raise MetricsError("metrics_duplicate_counter")
            counters[key] = value
    if set(counters) != set(METRIC_NAMES.values()):
        raise MetricsError("metrics_missing_counter")
    if "+Inf" not in buckets:
        raise MetricsError("metrics_missing_histogram")
    return MetricSnapshot(
        sha256=sha256,
        counters=counters,
        buckets=buckets,
        graph_events=graph_events,
    )


def read_snapshot(
    url: str, token: str | None, model: str, timeout: float
) -> MetricSnapshot:
    headers = {"Accept": "text/plain"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with OPENER.open(request, timeout=timeout) as response:
            if response.status != HTTP_OK:
                raise MetricsError("metrics_http_status")
            body, sha256 = _consume(response)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise MetricsError("metrics_transport_error") from error
    if body is None:
        raise MetricsError("metrics_response_too_large")
    return _parse_snapshot(body, sha256, model)


def summarize_captures(
    captures: list[MetricCapture],
    *,
    requested: bool,
    successful_decisions: int,
    failed_workflows: int,
) -> dict[str, Any]:
    if not requested:
        return {"status": "not_requested"}
    errors = Counter(
        item.public_record()["error_code"]
        for item in captures
        if item.public_record()["error_code"] is not None
    )
    if errors or not captures:
        return {"status": "incomplete", "errors": dict(sorted(errors.items()))}
    deltas = [item.delta() for item in captures]
    if any(delta is None for delta in deltas):
        return {"status": "incomplete", "errors": {"metrics_counter_reset": 1}}
    counter_totals = {
        key: sum(delta[0][key] for delta in deltas if delta is not None)
        for key in METRIC_NAMES.values()
    }
    bucket_keys = set().union(*(delta[1] for delta in deltas if delta is not None))
    bucket_totals = {
        key: sum(delta[1].get(key, 0) for delta in deltas if delta is not None)
        for key in sorted(bucket_keys)
    }
    graph_totals = {
        key: sum(delta[2][key] for delta in deltas if delta is not None)
        for key in GRAPH_EVENTS
    }
    batches = counter_totals["physical_batches"]
    valid_denominator = successful_decisions > 0 and failed_workflows == 0
    return {
        "status": "complete",
        "rounds": len(captures),
        "counter_deltas": counter_totals,
        "physical_batch_size_bucket_deltas": bucket_totals,
        "graph_event_deltas": graph_totals,
        "normalization_decisions": successful_decisions if valid_denominator else None,
        "row_preparation_seconds_per_decision": (
            counter_totals["row_preparation_seconds"] / successful_decisions
            if valid_denominator
            else None
        ),
        "row_preparations_per_decision": (
            counter_totals["row_preparations"] / successful_decisions
            if valid_denominator
            else None
        ),
        "physical_batch_rows_per_decision": (
            counter_totals["physical_batch_rows"] / successful_decisions
            if valid_denominator
            else None
        ),
        "observed_rows_per_physical_batch": (
            counter_totals["physical_batch_rows"] / batches if batches > 0 else None
        ),
    }


def telemetry_comparison(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    if old["status"] != "complete" or new["status"] != "complete":
        return {"available": False, "reason": "both_arms_require_complete_metrics"}
    old_prep = old["row_preparation_seconds_per_decision"]
    new_prep = new["row_preparation_seconds_per_decision"]
    old_batch = old["observed_rows_per_physical_batch"]
    new_batch = new["observed_rows_per_physical_batch"]
    return {
        "available": True,
        "old_over_new_row_preparation_seconds_per_decision": (
            old_prep / new_prep if old_prep is not None and new_prep else None
        ),
        "new_over_old_observed_rows_per_physical_batch": (
            new_batch / old_batch if new_batch is not None and old_batch else None
        ),
    }
