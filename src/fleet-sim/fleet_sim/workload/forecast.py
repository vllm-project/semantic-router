"""Content-free workload archetype aggregate forecasting.

The forecast schema intentionally stores only low-cardinality aggregate windows.
It can be converted into FleetSim mixture scenarios for offline/nearline
capacity backtests, but it is not a production scaler contract.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from .mixture import (
    CompositionWindow,
    MixtureScenario,
    MixtureValidationError,
    WorkloadArchetype,
    _load_cdf_source,
    load_mixture_scenario,
)
from .trace import TraceWorkload

SCHEMA_VERSION = "fleet-sim.workload-archetype-forecast/v1alpha1"
TAXONOMY_VERSION = "fleet-sim.workload-archetype-taxonomy/v1alpha1"
WEIGHT_TOLERANCE = 1e-6

WINDOW_DIMENSIONS = frozenset(
    {
        "archetype_weights",
        "model_class",
        "region",
        "slo_class",
    }
)
WINDOW_SCHEMA_FIELDS = frozenset(
    {
        "archetype_weights",
        "duration_s",
        "model_class",
        "p50_total_tokens",
        "p95_total_tokens",
        "p99_ttft_ms",
        "redacted",
        "region",
        "request_count",
        "slo_class",
        "start_s",
        "total_tokens",
        "uncertainty",
        "weights",
    }
)

DEFAULT_FORBIDDEN_DIMENSIONS = (
    "account_id",
    "caller_id",
    "customer_id",
    "domain",
    "host",
    "prompt",
    "prompt_text",
    "raw_prompt",
    "raw_response",
    "request_id",
    "response",
    "session_id",
    "tenant_id",
    "url",
    "user_id",
)


class ForecastValidationError(ValueError):
    """Raised when a workload forecast aggregate fails validation."""


@dataclass(frozen=True)
class ForecastPrivacyPolicy:
    """Privacy limits for aggregate forecast windows."""

    min_requests_per_window: int = 50
    allowed_dimensions: tuple[str, ...] = (
        "archetype_weights",
        "model_class",
        "region",
        "slo_class",
    )
    forbidden_dimensions: tuple[str, ...] = DEFAULT_FORBIDDEN_DIMENSIONS
    content_free: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ForecastPrivacyPolicy:
        data = data or {}
        return cls(
            min_requests_per_window=int(data.get("min_requests_per_window", 50)),
            allowed_dimensions=tuple(
                str(item)
                for item in data.get(
                    "allowed_dimensions",
                    (
                        "archetype_weights",
                        "model_class",
                        "region",
                        "slo_class",
                    ),
                )
            ),
            forbidden_dimensions=tuple(
                str(item)
                for item in data.get(
                    "forbidden_dimensions", DEFAULT_FORBIDDEN_DIMENSIONS
                )
            ),
            content_free=bool(data.get("content_free", True)),
        )


@dataclass(frozen=True)
class AggregateWindow:
    """Observed aggregate demand for one privacy-safe time bucket."""

    start_s: float
    duration_s: float
    request_count: int
    total_tokens: int
    archetype_weights: dict[str, float]
    model_class: str
    slo_class: str
    region: str
    p50_total_tokens: int | None = None
    p95_total_tokens: int | None = None
    p99_ttft_ms: float | None = None
    uncertainty: dict[str, Any] = field(default_factory=dict)
    redacted: bool = False
    extra_fields: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AggregateWindow:
        weights = data.get("archetype_weights", data.get("weights", {}))
        extra_fields = tuple(sorted(set(data) - WINDOW_SCHEMA_FIELDS))
        return cls(
            start_s=float(data.get("start_s", 0.0)),
            duration_s=float(data.get("duration_s", 0.0)),
            request_count=int(data.get("request_count", 0)),
            total_tokens=int(data.get("total_tokens", 0)),
            archetype_weights={str(k): float(v) for k, v in dict(weights).items()},
            model_class=str(data.get("model_class", "")),
            slo_class=str(data.get("slo_class", "")),
            region=str(data.get("region", "")),
            p50_total_tokens=(
                int(data["p50_total_tokens"])
                if data.get("p50_total_tokens") is not None
                else None
            ),
            p95_total_tokens=(
                int(data["p95_total_tokens"])
                if data.get("p95_total_tokens") is not None
                else None
            ),
            p99_ttft_ms=(
                float(data["p99_ttft_ms"])
                if data.get("p99_ttft_ms") is not None
                else None
            ),
            uncertainty=dict(data.get("uncertainty") or {}),
            redacted=bool(data.get("redacted", False)),
            extra_fields=extra_fields,
        )

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def arrival_rate(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.request_count / self.duration_s

    @property
    def tokens_per_request(self) -> float:
        if self.request_count <= 0:
            return 0.0
        return self.total_tokens / self.request_count


@dataclass(frozen=True)
class ForecastedWindow:
    """Predicted aggregate demand for one future time bucket."""

    start_s: float
    duration_s: float
    request_count: int
    total_tokens: int
    archetype_weights: dict[str, float]
    source_method: str
    model_class: str
    slo_class: str
    region: str
    p50_total_tokens: int | None = None
    p95_total_tokens: int | None = None
    p99_ttft_ms: float | None = None
    uncertainty: dict[str, Any] = field(default_factory=dict)

    @property
    def end_s(self) -> float:
        return self.start_s + self.duration_s

    @property
    def arrival_rate(self) -> float:
        if self.duration_s <= 0:
            return 0.0
        return self.request_count / self.duration_s

    @property
    def tokens_per_request(self) -> float:
        if self.request_count <= 0:
            return 0.0
        return self.total_tokens / self.request_count


ForecastWindowLike = AggregateWindow | ForecastedWindow


@dataclass(frozen=True)
class WorkloadForecastScenario:
    """Versioned workload archetype forecast/backtest fixture."""

    id: str
    version: str
    source_mixture_path: str
    aggregate_windows: tuple[AggregateWindow, ...]
    schema_version: str = SCHEMA_VERSION
    taxonomy_version: str = TAXONOMY_VERSION
    privacy: ForecastPrivacyPolicy = field(default_factory=ForecastPrivacyPolicy)
    holdout_windows: tuple[AggregateWindow, ...] = ()
    forecast_horizon_windows: int = 1
    generated_at_s: float | None = None
    max_staleness_s: float = 180.0
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    base_dir: Path | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        base_dir: str | Path | None = None,
    ) -> WorkloadForecastScenario:
        holdout = tuple(
            AggregateWindow.from_dict(dict(item))
            for item in data.get("holdout_windows", ())
        )
        horizon = int(data.get("forecast_horizon_windows", len(holdout) or 1))
        return cls(
            schema_version=str(data.get("schema_version", "")),
            taxonomy_version=str(data.get("taxonomy_version", "")),
            id=str(data.get("id", "")),
            version=str(data.get("version", "")),
            description=data.get("description"),
            source_mixture_path=str(
                data.get("source_mixture", data.get("source_mixture_path", ""))
            ),
            privacy=ForecastPrivacyPolicy.from_dict(data.get("privacy")),
            aggregate_windows=tuple(
                AggregateWindow.from_dict(dict(item))
                for item in data.get("aggregate_windows", ())
            ),
            holdout_windows=holdout,
            forecast_horizon_windows=horizon,
            generated_at_s=(
                float(data["generated_at_s"])
                if data.get("generated_at_s") is not None
                else None
            ),
            max_staleness_s=float(data.get("max_staleness_s", 180.0)),
            metadata=dict(data.get("metadata") or {}),
            base_dir=Path(base_dir) if base_dir is not None else None,
        )

    def resolve_source_mixture_path(self) -> Path:
        path = Path(self.source_mixture_path)
        if path.is_absolute() or self.base_dir is None:
            return path
        return self.base_dir / path

    def load_source_mixture(self) -> MixtureScenario:
        return load_mixture_scenario(self.resolve_source_mixture_path())

    def stale_reason(self, now_s: float | None) -> str | None:
        if now_s is None:
            return None
        if self.generated_at_s is None:
            return "generated_at_s is missing"
        age_s = now_s - self.generated_at_s
        if age_s > self.max_staleness_s:
            return (
                f"forecast is stale: age={age_s:.0f}s exceeds "
                f"max_staleness_s={self.max_staleness_s:.0f}s"
            )
        return None


def load_forecast_scenario(
    path: str | Path,
    validate: bool = True,
) -> WorkloadForecastScenario:
    """Load a workload archetype forecast/backtest JSON file."""

    scenario_path = Path(path)
    data = json.loads(scenario_path.read_text())
    scenario = WorkloadForecastScenario.from_dict(data, base_dir=scenario_path.parent)
    if validate:
        validate_forecast_scenario(scenario, raw_data=data)
    return scenario


def validate_forecast_scenario(
    scenario: WorkloadForecastScenario,
    raw_data: dict[str, Any] | None = None,
) -> None:
    """Validate a forecast scenario and raise ForecastValidationError on errors."""

    errors: list[str] = []
    if raw_data is not None:
        errors.extend(_privacy_key_errors(raw_data, scenario.privacy))
    if scenario.schema_version != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION!r}, "
            f"got {scenario.schema_version!r}"
        )
    if scenario.taxonomy_version != TAXONOMY_VERSION:
        errors.append(
            f"taxonomy_version must be {TAXONOMY_VERSION!r}, "
            f"got {scenario.taxonomy_version!r}"
        )
    if not scenario.id:
        errors.append("scenario id is required")
    if not scenario.version:
        errors.append("scenario version is required")
    if not scenario.source_mixture_path:
        errors.append("source_mixture is required")
    if scenario.privacy.min_requests_per_window <= 0:
        errors.append("privacy.min_requests_per_window must be positive")
    if not scenario.privacy.content_free:
        errors.append("privacy.content_free must be true")
    if scenario.max_staleness_s <= 0:
        errors.append("max_staleness_s must be positive")
    if scenario.forecast_horizon_windows <= 0:
        errors.append("forecast_horizon_windows must be positive")
    if not scenario.aggregate_windows:
        errors.append("aggregate_windows must not be empty")
    if scenario.holdout_windows and scenario.forecast_horizon_windows > len(
        scenario.holdout_windows
    ):
        errors.append(
            "forecast_horizon_windows exceeds available holdout_windows: "
            f"horizon={scenario.forecast_horizon_windows} "
            f"holdout_windows={len(scenario.holdout_windows)}"
        )

    expected_ids: set[str] | None = None
    if scenario.source_mixture_path:
        try:
            expected_ids = set(scenario.load_source_mixture().archetype_ids)
        except (OSError, json.JSONDecodeError, MixtureValidationError) as exc:
            errors.append(f"failed to load source_mixture: {exc}")

    _validate_windows(
        "aggregate_windows",
        scenario.aggregate_windows,
        scenario.privacy,
        expected_ids,
        errors,
    )
    _validate_windows(
        "holdout_windows",
        scenario.holdout_windows,
        scenario.privacy,
        expected_ids,
        errors,
    )

    if errors:
        raise ForecastValidationError("\n".join(errors))


def forecast_to_mixture_scenario(
    source_mixture: MixtureScenario,
    windows: Sequence[ForecastWindowLike],
    base_lam: float,
    scenario_id: str,
    version: str,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> MixtureScenario:
    """Convert aggregate forecast windows into a reproducible mixture scenario."""

    if base_lam <= 0:
        raise ValueError("base_lam must be positive")
    if not windows:
        raise ValueError("windows must not be empty")
    source_cdfs = _source_archetype_cdfs(source_mixture)
    schedule = tuple(
        CompositionWindow(
            start_s=window.start_s,
            duration_s=window.duration_s,
            weights=dict(window.archetype_weights),
            arrival_rate_multiplier=max(window.arrival_rate / base_lam, 1e-9),
            token_scale=_window_token_scale(window, source_cdfs),
        )
        for window in windows
    )
    return MixtureScenario(
        schema_version=source_mixture.schema_version,
        id=scenario_id,
        version=version,
        seed=source_mixture.seed,
        description=description,
        archetypes=source_mixture.archetypes,
        composition_schedule=schedule,
        metadata=dict(metadata or {}),
        base_dir=source_mixture.base_dir,
    )


def detect_aggregate_diagnostics(
    history: Sequence[AggregateWindow],
    actual: Sequence[AggregateWindow],
    burst_threshold: float = 1.5,
    drift_threshold: float = 0.25,
    oscillation_sign_changes: int = 2,
) -> tuple[str, ...]:
    """Return burst/drift/oscillation diagnostics from aggregate windows."""

    if not history or not actual:
        return ()
    diagnostics: list[str] = []
    historical_rate = _mean([window.arrival_rate for window in history])
    actual_rates = [window.arrival_rate for window in actual]
    if historical_rate > 0 and max(actual_rates) >= historical_rate * burst_threshold:
        diagnostics.append(
            "burst detected in holdout aggregates; verify proactive capacity "
            "against reactive fallback"
        )

    drift = max(
        _weight_l1(history[-1].archetype_weights, window.archetype_weights)
        for window in actual
    )
    if drift >= drift_threshold:
        diagnostics.append(
            f"taxonomy drift detected: max holdout weight L1={drift:.3f}"
        )

    rates = [window.arrival_rate for window in history[-3:]] + actual_rates
    sign_changes = _sign_changes(rates)
    if sign_changes >= oscillation_sign_changes:
        diagnostics.append(
            f"oscillation risk detected: {sign_changes} demand direction changes"
        )
    return tuple(diagnostics)


def _validate_windows(
    label: str,
    windows: Sequence[AggregateWindow],
    privacy: ForecastPrivacyPolicy,
    expected_ids: set[str] | None,
    errors: list[str],
) -> None:
    allowed_dimensions = {_normalise_key(item) for item in privacy.allowed_dimensions}
    last_end = None
    for idx, window in enumerate(windows):
        item = f"{label}[{idx}]"
        if window.extra_fields:
            errors.append(
                f"{item}: fields are not allowed by forecast window schema: "
                f"{list(window.extra_fields)}"
            )
        if window.start_s < 0:
            errors.append(f"{item}: start_s must be non-negative")
        if window.duration_s <= 0:
            errors.append(f"{item}: duration_s must be positive")
        if last_end is not None and window.start_s < last_end:
            errors.append(f"{item}: windows must not overlap")
        last_end = window.end_s
        if (
            not window.redacted
            and window.request_count < privacy.min_requests_per_window
        ):
            errors.append(
                f"{item}: request_count must be >= "
                f"privacy.min_requests_per_window={privacy.min_requests_per_window}"
            )
        if window.total_tokens <= 0:
            errors.append(f"{item}: total_tokens must be positive")
        if window.p50_total_tokens is not None and window.p50_total_tokens <= 0:
            errors.append(f"{item}: p50_total_tokens must be positive")
        if window.p95_total_tokens is not None and window.p95_total_tokens <= 0:
            errors.append(f"{item}: p95_total_tokens must be positive")
        if (
            window.p50_total_tokens is not None
            and window.p95_total_tokens is not None
            and window.p50_total_tokens > window.p95_total_tokens
        ):
            errors.append(f"{item}: p50_total_tokens must be <= p95_total_tokens")
        if window.p99_ttft_ms is not None and window.p99_ttft_ms <= 0:
            errors.append(f"{item}: p99_ttft_ms must be positive")
        for dim_name, dim_value in (
            ("model_class", window.model_class),
            ("slo_class", window.slo_class),
            ("region", window.region),
        ):
            if _normalise_key(dim_name) not in allowed_dimensions:
                errors.append(
                    f"{item}: dimension {dim_name!r} is not allowed by "
                    "privacy.allowed_dimensions"
                )
            if not dim_value:
                errors.append(f"{item}: {dim_name} is required")
        if "archetype_weights" not in allowed_dimensions:
            errors.append(
                f"{item}: dimension 'archetype_weights' is not allowed by "
                "privacy.allowed_dimensions"
            )
        _validate_weight_map(
            window.archetype_weights,
            expected_ids,
            f"{item}.archetype_weights",
            errors,
        )
        _validate_uncertainty(window.uncertainty, item, errors)


def _validate_weight_map(
    weights: dict[str, float],
    expected_ids: set[str] | None,
    label: str,
    errors: list[str],
) -> None:
    if not weights:
        errors.append(f"{label}: weights must not be empty")
        return
    if expected_ids is not None:
        missing = sorted(expected_ids - set(weights))
        extra = sorted(set(weights) - expected_ids)
        if missing:
            errors.append(f"{label}: missing weights for {missing}")
        if extra:
            errors.append(f"{label}: unknown archetype ids {extra}")
    for archetype_id, weight in weights.items():
        if weight < 0:
            errors.append(f"{label}: {archetype_id} weight must be non-negative")
    total = sum(weights.values())
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        errors.append(f"{label}: weights must sum to 1.0, got {total:.6f}")


def _validate_uncertainty(
    uncertainty: dict[str, Any], label: str, errors: list[str]
) -> None:
    for key, value in uncertainty.items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                if float(subvalue) < 0:
                    errors.append(f"{label}.uncertainty.{key}.{subkey} is negative")
        elif isinstance(value, (int, float)) and float(value) < 0:
            errors.append(f"{label}.uncertainty.{key} is negative")


def _privacy_key_errors(
    raw_data: dict[str, Any],
    privacy: ForecastPrivacyPolicy,
) -> list[str]:
    forbidden = {_normalise_key(item) for item in privacy.forbidden_dimensions}
    errors = []
    for path, key in _walk_keys(raw_data):
        if _normalise_key(key) in forbidden:
            errors.append(
                f"privacy violation: high-cardinality/content field {key!r} at {path}"
            )
    return errors


def _source_archetype_cdfs(
    source_mixture: MixtureScenario,
) -> dict[str, list[tuple[int, float]]]:
    return {
        archetype.id: _cdf_for_archetype(archetype, source_mixture)
        for archetype in source_mixture.archetypes
    }


def _cdf_for_archetype(
    archetype: WorkloadArchetype,
    source_mixture: MixtureScenario,
) -> list[tuple[int, float]]:
    if archetype.source.kind == "cdf":
        return _load_cdf_source(archetype.source, source_mixture.base_dir)
    trace = TraceWorkload(
        str(archetype.source.resolve_path(source_mixture.base_dir)),
        fmt=archetype.source.format or "semantic_router",
        max_reqs=archetype.source.max_reqs,
        field_map=archetype.source.field_map,
        model_id_field=archetype.source.model_id_field,
        default_model_id=archetype.source.default_model_id,
    )
    totals = sorted(req.total_tokens() for _, req in trace.generate())
    if not totals:
        raise ForecastValidationError(
            f"{archetype.id}: trace source produced no requests"
        )
    return _empirical_cdf(totals)


def _window_token_scale(
    window: ForecastWindowLike,
    source_cdfs: dict[str, list[tuple[int, float]]],
) -> float:
    baseline_cdf = _aggregate_cdf(source_cdfs, window.archetype_weights)
    baseline_mean = _cdf_mean(baseline_cdf)
    baseline_p50 = _cdf_quantile(baseline_cdf, 0.50)
    baseline_p95 = _cdf_quantile(baseline_cdf, 0.95)
    ratios = []
    if window.tokens_per_request > 0 and baseline_mean > 0:
        ratios.append(window.tokens_per_request / baseline_mean)
    p50_total_tokens = getattr(window, "p50_total_tokens", None)
    if p50_total_tokens is not None and baseline_p50 > 0:
        ratios.append(float(p50_total_tokens) / baseline_p50)
    p95_total_tokens = getattr(window, "p95_total_tokens", None)
    if p95_total_tokens is not None and baseline_p95 > 0:
        ratios.append(float(p95_total_tokens) / baseline_p95)
    if not ratios:
        return 1.0
    return max(*ratios, 1e-9)


def _aggregate_cdf(
    cdfs: dict[str, list[tuple[int, float]]],
    weights: dict[str, float],
) -> list[tuple[int, float]]:
    points = sorted(
        {threshold for archetype_id in weights for threshold, _ in cdfs[archetype_id]}
    )
    aggregate = [
        (
            threshold,
            min(
                1.0,
                max(
                    0.0,
                    sum(
                        weights[archetype_id] * _cdf_eval(cdfs[archetype_id], threshold)
                        for archetype_id in weights
                    ),
                ),
            ),
        )
        for threshold in points
    ]
    if aggregate:
        aggregate[-1] = (aggregate[-1][0], 1.0)
    return aggregate


def _cdf_eval(cdf: list[tuple[int, float]], threshold: int) -> float:
    previous = 0.0
    for token_threshold, fraction in cdf:
        if threshold < token_threshold:
            return previous
        previous = fraction
    return previous


def _cdf_mean(cdf: list[tuple[int, float]]) -> float:
    total = 0.0
    previous_fraction = 0.0
    for threshold, fraction in cdf:
        probability = max(0.0, fraction - previous_fraction)
        total += threshold * probability
        previous_fraction = fraction
    return total


def _cdf_quantile(cdf: list[tuple[int, float]], quantile: float) -> float:
    for threshold, fraction in cdf:
        if fraction >= quantile:
            return float(threshold)
    return float(cdf[-1][0]) if cdf else 0.0


def _empirical_cdf(totals: list[int]) -> list[tuple[int, float]]:
    n = len(totals)
    cdf: list[tuple[int, float]] = []
    last = None
    for idx, total in enumerate(totals, start=1):
        if total == last and idx < n:
            continue
        cdf.append((total, idx / n))
        last = total
    cdf[-1] = (cdf[-1][0], 1.0)
    return cdf


def _walk_keys(value: Any, prefix: str = "$") -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_str = str(key)
            path = f"{prefix}.{key_str}"
            found.append((path, key_str))
            found.extend(_walk_keys(child, path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            found.extend(_walk_keys(child, f"{prefix}[{idx}]"))
    return found


def _normalise_key(value: str) -> str:
    return value.lower().replace("-", "_")


def _weight_l1(left: dict[str, float], right: dict[str, float]) -> float:
    ids = set(left) | set(right)
    return sum(
        abs(left.get(archetype_id, 0.0) - right.get(archetype_id, 0.0))
        for archetype_id in ids
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sign_changes(values: Sequence[float]) -> int:
    signs = []
    for left, right in pairwise(values):
        delta = right - left
        if abs(delta) <= 1e-9:
            continue
        signs.append(1 if delta > 0 else -1)
    return sum(1 for left, right in pairwise(signs) if left != right)
