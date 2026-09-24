"""Small dependency-free Prometheus collector for the Decision runtime."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable

from .scheduler import SchedulerSnapshot

PHYSICAL_BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
QWEN_ROCM_GRAPH_EVENTS = frozenset({"capture", "replay", "fallback"})


class RuntimeMetrics:
    """Track bounded-label HTTP and inference counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._http_count: dict[tuple[str, str], int] = defaultdict(int)
        self._http_seconds: dict[tuple[str, str], float] = defaultdict(float)
        self._evaluation_count: dict[tuple[str, str], int] = defaultdict(int)
        self._evaluation_seconds: dict[tuple[str, str], float] = defaultdict(float)
        self._row_preparations: dict[str, int] = defaultdict(int)
        self._row_preparation_seconds: dict[str, float] = defaultdict(float)
        self._physical_batches: dict[str, int] = defaultdict(int)
        self._physical_batch_rows: dict[str, int] = defaultdict(int)
        self._physical_batch_seconds: dict[str, float] = defaultdict(float)
        self._physical_batch_sizes: dict[tuple[str, int | str], int] = defaultdict(int)
        self._qwen_rocm_graph_events: dict[tuple[str, str], int] = defaultdict(int)

    def record_http(self, route: str, status: int, elapsed_seconds: float) -> None:
        key = (route, str(status))
        with self._lock:
            self._http_count[key] += 1
            self._http_seconds[key] += elapsed_seconds

    def record_evaluation(
        self, model: str, outcome: str, elapsed_seconds: float
    ) -> None:
        key = (model, outcome)
        with self._lock:
            self._evaluation_count[key] += 1
            self._evaluation_seconds[key] += elapsed_seconds

    def record_row_preparation(self, model: str, elapsed_seconds: float) -> None:
        """Count one logical request's tokenizer/input preparation attempt."""

        with self._lock:
            self._row_preparations[model] += 1
            self._row_preparation_seconds[model] += elapsed_seconds

    def record_physical_batch(
        self, model: str, rows: int, elapsed_seconds: float
    ) -> None:
        """Count one actual model forward with bounded batch-size labels."""

        bucket = next(
            (limit for limit in PHYSICAL_BATCH_SIZE_BUCKETS if rows <= limit),
            "+Inf",
        )
        with self._lock:
            self._physical_batches[model] += 1
            self._physical_batch_rows[model] += rows
            self._physical_batch_seconds[model] += elapsed_seconds
            self._physical_batch_sizes[(model, bucket)] += 1

    def record_qwen_rocm_graph_event(self, model: str, event: str) -> None:
        """Count one bounded Qwen ROCm graph outcome without request content."""

        if event not in QWEN_ROCM_GRAPH_EVENTS:
            raise ValueError("unsupported Qwen ROCm graph event")
        with self._lock:
            self._qwen_rocm_graph_events[(model, event)] += 1

    def render(self, scheduler: Iterable[SchedulerSnapshot]) -> str:
        with self._lock:
            http_count = dict(self._http_count)
            http_seconds = dict(self._http_seconds)
            evaluation_count = dict(self._evaluation_count)
            evaluation_seconds = dict(self._evaluation_seconds)
            row_preparations = dict(self._row_preparations)
            row_preparation_seconds = dict(self._row_preparation_seconds)
            physical_batches = dict(self._physical_batches)
            physical_batch_rows = dict(self._physical_batch_rows)
            physical_batch_seconds = dict(self._physical_batch_seconds)
            physical_batch_sizes = dict(self._physical_batch_sizes)
            graph_events = dict(self._qwen_rocm_graph_events)

        lines = [
            "# HELP decision_runtime_http_requests_total HTTP requests by route and status.",
            "# TYPE decision_runtime_http_requests_total counter",
        ]
        for (route, status), value in sorted(http_count.items()):
            labels = f'route="{_escape(route)}",status="{_escape(status)}"'
            lines.append(f"decision_runtime_http_requests_total{{{labels}}} {value}")
        lines.extend(
            [
                "# HELP decision_runtime_http_request_duration_seconds_total Cumulative HTTP request duration.",
                "# TYPE decision_runtime_http_request_duration_seconds_total counter",
            ]
        )
        for (route, status), value in sorted(http_seconds.items()):
            labels = f'route="{_escape(route)}",status="{_escape(status)}"'
            lines.append(
                f"decision_runtime_http_request_duration_seconds_total{{{labels}}} {value:.9f}"
            )
        lines.extend(
            [
                "# HELP decision_runtime_evaluations_total Evaluations by model and outcome.",
                "# TYPE decision_runtime_evaluations_total counter",
            ]
        )
        for (model, outcome), value in sorted(evaluation_count.items()):
            labels = f'model="{_escape(model)}",outcome="{_escape(outcome)}"'
            lines.append(f"decision_runtime_evaluations_total{{{labels}}} {value}")
        lines.extend(
            [
                "# HELP decision_runtime_evaluation_duration_seconds_total Cumulative evaluation duration.",
                "# TYPE decision_runtime_evaluation_duration_seconds_total counter",
            ]
        )
        for (model, outcome), value in sorted(evaluation_seconds.items()):
            labels = f'model="{_escape(model)}",outcome="{_escape(outcome)}"'
            lines.append(
                f"decision_runtime_evaluation_duration_seconds_total{{{labels}}} {value:.9f}"
            )

        for metric, values, help_text, numeric in (
            (
                "row_preparations_total",
                row_preparations,
                "Logical Decision row-preparation attempts.",
                False,
            ),
            (
                "row_preparation_duration_seconds_total",
                row_preparation_seconds,
                "Cumulative tokenizer and row-preparation duration.",
                True,
            ),
            (
                "physical_batches_total",
                physical_batches,
                "Actual physical Decision model forwards.",
                False,
            ),
            (
                "physical_batch_rows_total",
                physical_batch_rows,
                "Rows submitted to actual physical Decision model forwards.",
                False,
            ),
            (
                "physical_batch_duration_seconds_total",
                physical_batch_seconds,
                "Cumulative physical model-forward duration.",
                True,
            ),
        ):
            name = f"decision_runtime_{metric}"
            lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} counter"))
            for model, value in sorted(values.items()):
                rendered = f"{value:.9f}" if numeric else str(value)
                lines.append(f'{name}{{model="{_escape(model)}"}} {rendered}')

        histogram = "decision_runtime_physical_batch_size"
        lines.extend(
            (
                f"# HELP {histogram} Physical model-forward sizes in rows.",
                f"# TYPE {histogram} histogram",
            )
        )
        for model, count in sorted(physical_batches.items()):
            cumulative = 0
            for bucket in (*PHYSICAL_BATCH_SIZE_BUCKETS, "+Inf"):
                cumulative += physical_batch_sizes.get((model, bucket), 0)
                lines.append(
                    f'{histogram}_bucket{{model="{_escape(model)}",le="{bucket}"}} '
                    f"{cumulative}"
                )
            labels = f'model="{_escape(model)}"'
            lines.append(f"{histogram}_count{{{labels}}} {count}")
            lines.append(f"{histogram}_sum{{{labels}}} {physical_batch_rows[model]}")

        graph_metric = "decision_runtime_qwen_rocm_graph_events_total"
        lines.extend(
            (
                f"# HELP {graph_metric} Qwen ROCm qualified capture, "
                "used replay, and eager fallback events.",
                f"# TYPE {graph_metric} counter",
            )
        )
        for (model, event), value in sorted(graph_events.items()):
            labels = f'model="{_escape(model)}",event="{event}"'
            lines.append(f"{graph_metric}{{{labels}}} {value}")

        snapshots = tuple(scheduler)
        for metric, attribute, help_text in (
            ("running", "running", "Currently running evaluations."),
            ("queued", "queued", "Currently queued evaluations."),
        ):
            name = f"decision_runtime_scheduler_{metric}"
            lines.extend([f"# HELP {name} {help_text}", f"# TYPE {name} gauge"])
            for snapshot in snapshots:
                lines.append(
                    f'{name}{{model="{_escape(snapshot.model)}"}} '
                    f"{getattr(snapshot, attribute)}"
                )
        return "\n".join(lines) + "\n"


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
