"""Server-reducible assessment of a frozen repeated closed-loop load protocol."""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from pydantic import Field, model_validator

from cli.evaluation.capacity_statistics import (
    arithmetic_mean,
    one_sided_wilson_upper,
    sample_coefficient_of_variation,
)
from cli.evaluation.constants import SCHEMA_VERSION
from cli.evaluation.contract_primitives import StrictModel
from cli.evaluation.contracts import CapacityLoadProtocol, CapacitySLO
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import canonical_ordered_float_sum, percentile

CapacityFailureReason = Literal[
    "required_concurrency",
    "warmup_errors",
    "latency_p95",
    "measurement_cluster_coverage",
    "error_rate_cluster_stability",
    "error_rate_upper_bound",
    "throughput",
    "throughput_scaling",
    "throughput_stability",
    "latency_stability",
]


class CapacityProfileRepetition(StrictModel):
    concurrency: int = Field(gt=0)
    repetition: int = Field(gt=0)
    requests: int = Field(gt=0)
    successes: int = Field(ge=0)
    errors: int = Field(ge=0)
    elapsed_seconds: float = Field(gt=0, allow_inf_nan=False)
    throughput_rps: float = Field(gt=0, allow_inf_nan=False)
    latency_p95_ms: float = Field(ge=0, allow_inf_nan=False)
    error_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    error_rate_upper_bound: float = Field(ge=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_counts(self) -> CapacityProfileRepetition:
        if self.successes + self.errors != self.requests:
            raise ValueError("capacity repetition counts do not sum to requests")
        if self.throughput_rps != self.requests / self.elapsed_seconds:
            raise ValueError(
                "capacity repetition throughput does not match elapsed time"
            )
        if self.error_rate != self.errors / self.requests:
            raise ValueError("capacity repetition error rate does not match counts")
        if self.error_rate_upper_bound != one_sided_wilson_upper(
            self.errors, self.requests
        ):
            raise ValueError(
                "capacity repetition error bound does not match its independent cluster"
            )
        return self


class CapacityProfileLevel(StrictModel):
    concurrency: int = Field(gt=0)
    warmup_requests: int = Field(gt=0)
    warmup_errors: int = Field(ge=0)
    warmup_elapsed_seconds: float = Field(gt=0, allow_inf_nan=False)
    measurement_requests: int = Field(gt=0)
    successes: int = Field(ge=0)
    errors: int = Field(ge=0)
    elapsed_seconds: float = Field(gt=0, allow_inf_nan=False)
    throughput_rps: float = Field(gt=0, allow_inf_nan=False)
    throughput_cv: float = Field(ge=0, allow_inf_nan=False)
    latency_p50_ms: float = Field(ge=0, allow_inf_nan=False)
    latency_p95_ms: float = Field(ge=0, allow_inf_nan=False)
    latency_p99_ms: float = Field(ge=0, allow_inf_nan=False)
    latency_p95_cv: float = Field(ge=0, allow_inf_nan=False)
    error_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    error_rate_upper_bound: float = Field(ge=0, le=1, allow_inf_nan=False)
    measurement_cluster_count: int = Field(ge=3, le=5)
    error_rate_cluster_range: float = Field(ge=0, le=1, allow_inf_nan=False)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    runtime_cost_usd: float = Field(ge=0, allow_inf_nan=False)
    repetitions: tuple[CapacityProfileRepetition, ...] = Field(
        min_length=3,
        max_length=5,
    )
    throughput_scaling_efficiency: float | None = Field(
        default=None, ge=0, allow_inf_nan=False
    )
    warmup_passed: bool
    latency_slo_passed: bool
    cluster_coverage_passed: bool
    error_rate_stability_passed: bool
    error_slo_passed: bool
    throughput_slo_passed: bool
    scaling_slo_passed: bool
    throughput_stability_passed: bool
    latency_stability_passed: bool
    qualified: bool

    @model_validator(mode="after")
    def validate_internal_reduction(self) -> CapacityProfileLevel:
        if self.warmup_errors > self.warmup_requests:
            raise ValueError("capacity warmup errors exceed requests")
        if self.successes + self.errors != self.measurement_requests:
            raise ValueError("capacity measurement counts do not sum to requests")
        if tuple(row.repetition for row in self.repetitions) != tuple(
            range(1, len(self.repetitions) + 1)
        ):
            raise ValueError("capacity repetitions must use contiguous identities")
        if any(row.concurrency != self.concurrency for row in self.repetitions):
            raise ValueError("capacity repetitions do not bind their level")
        if sum(row.requests for row in self.repetitions) != self.measurement_requests:
            raise ValueError("capacity repetitions do not cover the measurement window")
        if sum(row.successes for row in self.repetitions) != self.successes:
            raise ValueError("capacity repetition successes do not match the level")
        if sum(row.errors for row in self.repetitions) != self.errors:
            raise ValueError("capacity repetition errors do not match the level")
        throughputs = tuple(row.throughput_rps for row in self.repetitions)
        p95_values = tuple(row.latency_p95_ms for row in self.repetitions)
        error_rates = tuple(row.error_rate for row in self.repetitions)
        expected = (
            arithmetic_mean(throughputs),
            sample_coefficient_of_variation(throughputs),
            sample_coefficient_of_variation(p95_values),
            arithmetic_mean(error_rates),
            max(row.error_rate_upper_bound for row in self.repetitions),
            len(self.repetitions),
            max(error_rates) - min(error_rates),
        )
        observed = (
            self.throughput_rps,
            self.throughput_cv,
            self.latency_p95_cv,
            self.error_rate,
            self.error_rate_upper_bound,
            self.measurement_cluster_count,
            self.error_rate_cluster_range,
        )
        if observed != expected:
            raise ValueError("capacity level statistics do not match repetitions")
        return self


class CapacitySLOAssessment(StrictModel):
    qualified_concurrency: int | None = Field(default=None, gt=0)
    saturation_concurrency: int | None = Field(default=None, gt=0)
    slo_headroom: int
    verdict: Literal["pass", "fail"]
    failure_reasons: tuple[CapacityFailureReason, ...]

    @model_validator(mode="after")
    def validate_verdict(self) -> CapacitySLOAssessment:
        if (self.slo_headroom >= 0) != (self.verdict == "pass"):
            raise ValueError("capacity verdict does not match SLO headroom")
        if (self.verdict == "pass") != (len(self.failure_reasons) == 0):
            raise ValueError("capacity failures do not match its verdict")
        return self


class CapacityProfile(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    kind: Literal["repeated-closed-loop-capacity"] = "repeated-closed-loop-capacity"
    protocol: CapacityLoadProtocol
    levels: tuple[CapacityProfileLevel, ...] = Field(min_length=2, max_length=8)
    slo: CapacitySLO
    assessment: CapacitySLOAssessment

    @model_validator(mode="after")
    def validate_reduction(self) -> CapacityProfile:
        if tuple(level.concurrency for level in self.levels) != (
            self.protocol.concurrency_levels
        ):
            raise ValueError("capacity profile levels do not match the frozen protocol")
        envelope_open = True
        previous: CapacityProfileLevel | None = None
        for level in self.levels:
            if len(level.repetitions) != self.protocol.repetitions_per_level or any(
                row.requests != self.protocol.measurement_requests_per_repetition
                for row in level.repetitions
            ):
                raise ValueError(
                    "capacity profile does not contain the frozen measurement window"
                )
            if level.warmup_requests != (
                level.concurrency * self.protocol.warmup_request_multiplier
            ):
                raise ValueError("capacity profile does not contain the frozen warmup")
            scaling = _scaling_efficiency(previous, level)
            expected_flags = _level_flags(
                level,
                self.slo,
                self.protocol,
                scaling,
                envelope_open,
            )
            observed_flags = (
                level.throughput_scaling_efficiency,
                level.warmup_passed,
                level.latency_slo_passed,
                level.cluster_coverage_passed,
                level.error_rate_stability_passed,
                level.error_slo_passed,
                level.throughput_slo_passed,
                level.scaling_slo_passed,
                level.throughput_stability_passed,
                level.latency_stability_passed,
                level.qualified,
            )
            if observed_flags != expected_flags:
                raise ValueError("capacity level decisions do not match observations")
            if not level.qualified:
                envelope_open = False
            previous = level
        expected_assessment = _assessment(self.levels, self.slo)
        if self.assessment != expected_assessment:
            raise ValueError("capacity assessment does not match measured levels")
        return self


def _scaling_efficiency(
    previous: CapacityProfileLevel | None,
    current: CapacityProfileLevel,
) -> float | None:
    if previous is None:
        return None
    throughput_growth = current.throughput_rps / previous.throughput_rps
    concurrency_growth = current.concurrency / previous.concurrency
    return throughput_growth / concurrency_growth


def _level_flags(
    level: CapacityProfileLevel,
    slo: CapacitySLO,
    protocol: CapacityLoadProtocol,
    scaling: float | None,
    envelope_open: bool,
) -> tuple[
    float | None,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
]:
    warmup_passed = level.warmup_errors == 0
    latency_passed = level.latency_p95_ms <= slo.max_latency_p95_ms
    cluster_coverage_passed = (
        level.measurement_cluster_count
        >= protocol.minimum_measurement_clusters_per_level
    )
    error_rate_stable = (
        level.error_rate_cluster_range <= protocol.max_error_rate_cluster_range
    )
    error_passed = level.error_rate_upper_bound <= slo.max_error_rate
    throughput_passed = (
        level.concurrency < slo.required_concurrency
        or level.throughput_rps >= slo.min_throughput_rps
    )
    scaling_passed = scaling is None or scaling >= slo.min_throughput_scaling_efficiency
    throughput_stable = level.throughput_cv <= protocol.max_throughput_cv
    latency_stable = level.latency_p95_cv <= protocol.max_latency_p95_cv
    qualified = (
        envelope_open
        and warmup_passed
        and latency_passed
        and cluster_coverage_passed
        and error_rate_stable
        and error_passed
        and throughput_passed
        and scaling_passed
        and throughput_stable
        and latency_stable
    )
    return (
        scaling,
        warmup_passed,
        latency_passed,
        cluster_coverage_passed,
        error_rate_stable,
        error_passed,
        throughput_passed,
        scaling_passed,
        throughput_stable,
        latency_stable,
        qualified,
    )


def _failure_reasons(
    levels: tuple[CapacityProfileLevel, ...],
    slo: CapacitySLO,
    qualified_concurrency: int | None,
) -> tuple[CapacityFailureReason, ...]:
    if qualified_concurrency is not None and qualified_concurrency >= (
        slo.required_concurrency
    ):
        return ()
    target = next(
        (level for level in levels if level.concurrency >= slo.required_concurrency),
        None,
    )
    if target is None:
        return ("required_concurrency",)
    reasons: list[CapacityFailureReason] = []
    checks = (
        (target.warmup_passed, "warmup_errors"),
        (target.latency_slo_passed, "latency_p95"),
        (target.cluster_coverage_passed, "measurement_cluster_coverage"),
        (target.error_rate_stability_passed, "error_rate_cluster_stability"),
        (target.error_slo_passed, "error_rate_upper_bound"),
        (target.throughput_slo_passed, "throughput"),
        (target.scaling_slo_passed, "throughput_scaling"),
        (target.throughput_stability_passed, "throughput_stability"),
        (target.latency_stability_passed, "latency_stability"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    if not reasons:
        reasons.append("required_concurrency")
    return tuple(reasons)


def _assessment(
    levels: tuple[CapacityProfileLevel, ...],
    slo: CapacitySLO,
) -> CapacitySLOAssessment:
    qualified = max(
        (level.concurrency for level in levels if level.qualified),
        default=None,
    )
    headroom = (qualified or 0) - slo.required_concurrency
    saturation = next(
        (level.concurrency for level in levels if not level.qualified),
        None,
    )
    reasons = _failure_reasons(levels, slo, qualified)
    return CapacitySLOAssessment(
        qualified_concurrency=qualified,
        saturation_concurrency=saturation,
        slo_headroom=headroom,
        verdict="pass" if headroom >= 0 else "fail",
        failure_reasons=reasons,
    )


def _single_batch_value(
    rows: list[ExecutionRecord],
    field: Literal["throughput_rps", "load_elapsed_seconds"],
) -> float:
    values = {getattr(row, field) for row in rows}
    if None in values or len(values) != 1:
        raise ValueError(f"capacity {field} must bind one complete batch")
    value = next(iter(values))
    if value is None:
        raise ValueError(f"capacity {field} must bind one complete batch")
    return value


def _require_exact_indices(rows: list[ExecutionRecord], expected: int) -> None:
    indices = sorted(row.load_request_index for row in rows)
    if indices != list(range(expected)):
        raise ValueError("capacity batch request indices are incomplete or duplicated")


def _validated_capacity_batches(
    records: list[ExecutionRecord],
) -> dict[tuple[int, str, int], list[ExecutionRecord]]:
    capacity = [row for row in records if row.track_id == "capacity"]
    if not capacity:
        raise ValueError("capacity profile requires request evidence")
    if any(
        row.concurrency is None
        or row.load_phase is None
        or row.load_repetition is None
        or row.load_request_index is None
        or row.success is None
        or row.latency_ms is None
        for row in capacity
    ):
        raise ValueError("capacity request evidence lacks load coordinates")
    batches: dict[tuple[int, str, int], list[ExecutionRecord]] = defaultdict(list)
    for row in capacity:
        assert row.concurrency is not None
        assert row.load_phase is not None
        assert row.load_repetition is not None
        batches[(row.concurrency, row.load_phase, row.load_repetition)].append(row)
    return dict(batches)


def _pop_warmup(
    batches: dict[tuple[int, str, int], list[ExecutionRecord]],
    concurrency: int,
    protocol: CapacityLoadProtocol,
) -> tuple[list[ExecutionRecord], float]:
    rows = batches.pop((concurrency, "warmup", 0), [])
    expected = concurrency * protocol.warmup_request_multiplier
    if len(rows) != expected:
        raise ValueError("capacity warmup does not match the frozen protocol")
    _require_exact_indices(rows, expected)
    return rows, _single_batch_value(rows, "load_elapsed_seconds")


def _pop_measurement_repetitions(
    batches: dict[tuple[int, str, int], list[ExecutionRecord]],
    concurrency: int,
    protocol: CapacityLoadProtocol,
) -> tuple[tuple[CapacityProfileRepetition, ...], list[ExecutionRecord]]:
    repetitions: list[CapacityProfileRepetition] = []
    measurement_rows: list[ExecutionRecord] = []
    for repetition in range(1, protocol.repetitions_per_level + 1):
        rows = batches.pop((concurrency, "measurement", repetition), [])
        expected = protocol.measurement_requests_per_repetition
        if len(rows) != expected:
            raise ValueError("capacity repetition does not match the frozen protocol")
        _require_exact_indices(rows, expected)
        latencies = [row.latency_ms for row in rows]
        assert all(value is not None for value in latencies)
        p95 = percentile([float(value) for value in latencies], 0.95)
        assert p95 is not None
        successes = sum(row.success is True for row in rows)
        errors = len(rows) - successes
        repetitions.append(
            CapacityProfileRepetition(
                concurrency=concurrency,
                repetition=repetition,
                requests=len(rows),
                successes=successes,
                errors=errors,
                elapsed_seconds=_single_batch_value(rows, "load_elapsed_seconds"),
                throughput_rps=_single_batch_value(rows, "throughput_rps"),
                latency_p95_ms=p95,
                error_rate=errors / len(rows),
                error_rate_upper_bound=one_sided_wilson_upper(errors, len(rows)),
            )
        )
        measurement_rows.extend(rows)
    return tuple(repetitions), measurement_rows


def _unqualified_capacity_level(
    concurrency: int,
    warmup: list[ExecutionRecord],
    warmup_elapsed: float,
    repetitions: tuple[CapacityProfileRepetition, ...],
    measurement_rows: list[ExecutionRecord],
) -> CapacityProfileLevel:
    latencies = [row.latency_ms for row in measurement_rows]
    assert all(value is not None for value in latencies)
    numeric_latencies = [float(value) for value in latencies]
    p50 = percentile(numeric_latencies, 0.50)
    p95 = percentile(numeric_latencies, 0.95)
    p99 = percentile(numeric_latencies, 0.99)
    assert p50 is not None and p95 is not None and p99 is not None
    successes = sum(row.success is True for row in measurement_rows)
    errors = len(measurement_rows) - successes
    throughputs = tuple(row.throughput_rps for row in repetitions)
    repetition_p95s = tuple(row.latency_p95_ms for row in repetitions)
    repetition_error_rates = tuple(row.error_rate for row in repetitions)
    return CapacityProfileLevel(
        concurrency=concurrency,
        warmup_requests=len(warmup),
        warmup_errors=sum(row.success is not True for row in warmup),
        warmup_elapsed_seconds=warmup_elapsed,
        measurement_requests=len(measurement_rows),
        successes=successes,
        errors=errors,
        elapsed_seconds=sum(row.elapsed_seconds for row in repetitions),
        throughput_rps=arithmetic_mean(throughputs),
        throughput_cv=sample_coefficient_of_variation(throughputs),
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        latency_p99_ms=p99,
        latency_p95_cv=sample_coefficient_of_variation(repetition_p95s),
        error_rate=arithmetic_mean(repetition_error_rates),
        error_rate_upper_bound=max(row.error_rate_upper_bound for row in repetitions),
        measurement_cluster_count=len(repetitions),
        error_rate_cluster_range=(
            max(repetition_error_rates) - min(repetition_error_rates)
        ),
        input_tokens=sum(row.input_tokens or 0 for row in measurement_rows),
        output_tokens=sum(row.output_tokens or 0 for row in measurement_rows),
        runtime_cost_usd=canonical_ordered_float_sum(
            row.runtime_cost or 0 for row in measurement_rows
        ),
        repetitions=repetitions,
        throughput_scaling_efficiency=None,
        warmup_passed=False,
        latency_slo_passed=False,
        cluster_coverage_passed=False,
        error_rate_stability_passed=False,
        error_slo_passed=False,
        throughput_slo_passed=False,
        scaling_slo_passed=False,
        throughput_stability_passed=False,
        latency_stability_passed=False,
        qualified=False,
    )


def _qualified_capacity_level(
    level: CapacityProfileLevel,
    slo: CapacitySLO,
    protocol: CapacityLoadProtocol,
    previous: CapacityProfileLevel | None,
    envelope_open: bool,
) -> CapacityProfileLevel:
    flags = _level_flags(
        level,
        slo,
        protocol,
        _scaling_efficiency(previous, level),
        envelope_open,
    )
    fields = (
        "throughput_scaling_efficiency",
        "warmup_passed",
        "latency_slo_passed",
        "cluster_coverage_passed",
        "error_rate_stability_passed",
        "error_slo_passed",
        "throughput_slo_passed",
        "scaling_slo_passed",
        "throughput_stability_passed",
        "latency_stability_passed",
        "qualified",
    )
    return level.model_copy(update=dict(zip(fields, flags, strict=True)))


def build_capacity_profile(
    records: list[ExecutionRecord],
    slo: CapacitySLO,
    protocol: CapacityLoadProtocol,
) -> CapacityProfile:
    """Reduce typed request rows into an independently reproducible G7 envelope."""

    batches = _validated_capacity_batches(records)
    levels: list[CapacityProfileLevel] = []
    envelope_open = True
    previous: CapacityProfileLevel | None = None
    for concurrency in protocol.concurrency_levels:
        warmup, warmup_elapsed = _pop_warmup(batches, concurrency, protocol)
        repetitions, measurement_rows = _pop_measurement_repetitions(
            batches, concurrency, protocol
        )
        unqualified = _unqualified_capacity_level(
            concurrency,
            warmup,
            warmup_elapsed,
            repetitions,
            measurement_rows,
        )
        level = _qualified_capacity_level(
            unqualified,
            slo,
            protocol,
            previous,
            envelope_open,
        )
        if not level.qualified:
            envelope_open = False
        levels.append(level)
        previous = level

    if batches:
        raise ValueError("capacity evidence contains an undeclared load batch")
    frozen = tuple(levels)
    return CapacityProfile(
        protocol=protocol,
        levels=frozen,
        slo=slo,
        assessment=_assessment(frozen, slo),
    )
