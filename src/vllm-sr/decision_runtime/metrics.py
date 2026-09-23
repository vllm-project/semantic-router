"""Small dependency-free Prometheus collector for the Decision runtime."""

from __future__ import annotations

import threading
from collections import defaultdict
from collections.abc import Iterable

from .scheduler import SchedulerSnapshot


class RuntimeMetrics:
    """Track bounded-label HTTP and inference counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._http_count: dict[tuple[str, str], int] = defaultdict(int)
        self._http_seconds: dict[tuple[str, str], float] = defaultdict(float)
        self._evaluation_count: dict[tuple[str, str], int] = defaultdict(int)
        self._evaluation_seconds: dict[tuple[str, str], float] = defaultdict(float)

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

    def render(self, scheduler: Iterable[SchedulerSnapshot]) -> str:
        with self._lock:
            http_count = dict(self._http_count)
            http_seconds = dict(self._http_seconds)
            evaluation_count = dict(self._evaluation_count)
            evaluation_seconds = dict(self._evaluation_seconds)

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
