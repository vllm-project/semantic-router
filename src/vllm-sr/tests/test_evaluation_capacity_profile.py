from __future__ import annotations

import json
from pathlib import Path

import pytest
from cli.evaluation.builtin_executors import (
    LiveRuntimeExecutor,
    NormalizedLiveExecutor,
)
from cli.evaluation.capacity_profile import CapacityProfile, build_capacity_profile
from cli.evaluation.capacity_statistics import one_sided_wilson_upper
from cli.evaluation.contracts import (
    CapacityLoadProtocol,
    CapacitySLO,
)
from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.evidence_level import run_evidence_level, track_evidence_level
from cli.evaluation.evidence_source_ids import LIVE_CAPACITY_EVIDENCE_SOURCE_ID
from cli.evaluation.metric_capacity import capacity_metrics
from evaluation_contract_test_support import default_capacity_load_protocol
from pydantic import ValidationError

_CLUSTER_PARITY_FIXTURE = (
    Path(__file__).parent / "fixtures" / "capacity_cluster_metric_parity.v1.json"
)


def _slo(
    *,
    required_concurrency: int = 2,
    max_error_rate: float = 0.05,
) -> CapacitySLO:
    return CapacitySLO(
        required_concurrency=required_concurrency,
        max_latency_p95_ms=100,
        max_error_rate=max_error_rate,
        min_throughput_rps=10,
        min_throughput_scaling_efficiency=0.8,
    )


def _row(
    *,
    concurrency: int,
    phase: str,
    repetition: int,
    index: int,
    requests: int,
    throughput: float,
    success: bool = True,
    latency: float = 50,
    runtime_cost: float = 0,
) -> ExecutionRecord:
    attempt = f"capacity-c{concurrency}-{phase[0]}{repetition}-q{index}"
    return ExecutionRecord(
        id=attempt,
        track_id="capacity",
        case_id="case-1",
        attempt_id=attempt,
        status="succeeded" if success else "failed",
        success=success,
        concurrency=concurrency,
        latency_ms=latency,
        throughput_rps=throughput,
        load_elapsed_seconds=requests / throughput,
        load_phase=phase,  # type: ignore[arg-type]
        load_repetition=repetition,
        load_request_index=index,
        runtime_cost=runtime_cost,
        evidence_kind=LIVE_CAPACITY_EVIDENCE_SOURCE_ID,
    )


def _records(
    protocol: CapacityLoadProtocol,
    *,
    throughputs: dict[int, tuple[float, ...]] | None = None,
    failed_measurements: set[tuple[int, int, int]] = frozenset(),
    costs: tuple[float, ...] = (),
) -> list[ExecutionRecord]:
    rows: list[ExecutionRecord] = []
    cost_index = 0
    for concurrency in protocol.concurrency_levels:
        warmup_count = concurrency * protocol.warmup_request_multiplier
        for index in range(warmup_count):
            rows.append(
                _row(
                    concurrency=concurrency,
                    phase="warmup",
                    repetition=0,
                    index=index,
                    requests=warmup_count,
                    throughput=float(concurrency * 10),
                )
            )
        values = (
            throughputs[concurrency]
            if throughputs is not None
            else tuple(
                float(concurrency * 10) for _ in range(protocol.repetitions_per_level)
            )
        )
        for repetition, throughput in enumerate(values, 1):
            count = protocol.measurement_requests_per_repetition
            for index in range(count):
                cost = costs[cost_index] if cost_index < len(costs) else 0
                cost_index += 1
                rows.append(
                    _row(
                        concurrency=concurrency,
                        phase="measurement",
                        repetition=repetition,
                        index=index,
                        requests=count,
                        throughput=throughput,
                        success=(concurrency, repetition, index)
                        not in failed_measurements,
                        runtime_cost=cost,
                    )
                )
    return rows


def test_capacity_profile_qualifies_a_repeated_stable_slo_envelope() -> None:
    protocol = default_capacity_load_protocol(2)
    profile = build_capacity_profile(_records(protocol), _slo(), protocol)

    assert profile.assessment.verdict == "pass"
    assert profile.assessment.qualified_concurrency == 2
    assert profile.assessment.slo_headroom == 0
    assert profile.assessment.failure_reasons == ()
    assert profile.levels[1].measurement_requests == 300
    assert profile.levels[1].error_rate_upper_bound < 0.05
    assert profile.levels[1].measurement_cluster_count == 3
    assert profile.levels[1].error_rate_cluster_range == 0
    assert profile.levels[1].throughput_scaling_efficiency == pytest.approx(1)
    assert all(level.qualified for level in profile.levels)


def test_live_capacity_records_publish_the_sealed_e5_level() -> None:
    records = _records(default_capacity_load_protocol(2))
    sealed = [
        record.model_copy(update={"broker_receipt": f"sha256:{index:064x}"})
        for index, record in enumerate(records, 1)
    ]

    executor = LiveRuntimeExecutor.contract
    assert track_evidence_level("live", executor, "capacity", records) == "E0"
    assert track_evidence_level("live", executor, "capacity", sealed) == "E5"
    assert run_evidence_level("live", executor, ("capacity",), sealed) == "E5"
    assert (
        track_evidence_level(
            "live", NormalizedLiveExecutor.contract, "capacity", sealed
        )
        == "E5"
    )


def test_capacity_profile_fails_at_the_scaling_saturation_boundary() -> None:
    protocol = default_capacity_load_protocol(2)
    profile = build_capacity_profile(
        _records(protocol, throughputs={1: (10, 10, 10), 2: (12, 12, 12)}),
        _slo(),
        protocol,
    )

    assert profile.assessment.verdict == "fail"
    assert profile.assessment.qualified_concurrency == 1
    assert profile.assessment.saturation_concurrency == 2
    assert profile.assessment.failure_reasons == ("throughput_scaling",)


def test_capacity_profile_uses_worst_cluster_bound_not_pooled_requests() -> None:
    protocol = default_capacity_load_protocol(2)
    profile = build_capacity_profile(
        _records(protocol, failed_measurements={(2, 1, 0)}),
        _slo(max_error_rate=0.02),
        protocol,
    )

    target = profile.levels[1]
    assert one_sided_wilson_upper(1, 300) < 0.02
    assert target.error_rate < 0.01
    assert target.error_rate_upper_bound == one_sided_wilson_upper(1, 100)
    assert target.error_rate_upper_bound > 0.02
    assert target.error_slo_passed is False
    assert profile.assessment.failure_reasons == ("error_rate_upper_bound",)


def test_capacity_profile_gates_cross_cluster_error_instability() -> None:
    protocol = default_capacity_load_protocol(2)
    profile = build_capacity_profile(
        _records(
            protocol,
            failed_measurements={(2, 1, index) for index in range(6)},
        ),
        _slo(max_error_rate=0.5),
        protocol,
    )

    target = profile.levels[1]
    assert target.error_rate_cluster_range == 0.06
    assert target.error_rate_stability_passed is False
    assert profile.assessment.failure_reasons == ("error_rate_cluster_stability",)


def test_capacity_cluster_reducer_matches_shared_go_python_fixture() -> None:
    fixture = json.loads(_CLUSTER_PARITY_FIXTURE.read_text(encoding="utf-8"))
    assert fixture["schema_version"] == "capacity-cluster-metric-parity.v1"
    protocol = default_capacity_load_protocol(2)
    assert protocol.confidence_level == fixture["confidence_level"]
    assert (
        protocol.minimum_measurement_clusters_per_level
        == fixture["minimum_measurement_clusters_per_level"]
    )
    assert (
        protocol.max_error_rate_cluster_range == fixture["max_error_rate_cluster_range"]
    )
    failed_measurements = {
        (level["concurrency"], cluster["load_repetition"], index)
        for level in fixture["levels"]
        for cluster in level["clusters"]
        for index in range(cluster["errors"])
    }
    records = _records(protocol, failed_measurements=failed_measurements)
    profile = build_capacity_profile(
        records,
        _slo(required_concurrency=1, max_error_rate=0.5),
        protocol,
    )
    assert profile.assessment.verdict == "pass"
    assert profile.assessment.qualified_concurrency == 1
    assert profile.assessment.saturation_concurrency == 2

    for level, expected_level in zip(profile.levels, fixture["levels"], strict=True):
        expected = expected_level["expected"]
        assert level.concurrency == expected_level["concurrency"]
        assert level.measurement_cluster_count == expected["measurement_cluster_count"]
        assert level.error_rate == pytest.approx(expected["error_rate"])
        assert level.error_rate_upper_bound == pytest.approx(
            expected["error_rate_upper_bound"]
        )
        assert level.error_rate_cluster_range == pytest.approx(
            expected["error_rate_cluster_range"]
        )
        for repetition, expected_cluster in zip(
            level.repetitions, expected_level["clusters"], strict=True
        ):
            assert expected_cluster["load_phase"] == "measurement"
            assert repetition.repetition == expected_cluster["load_repetition"]
            assert repetition.requests == expected_cluster["requests"]
            assert repetition.errors == expected_cluster["errors"]
            assert repetition.error_rate == pytest.approx(
                expected_cluster["error_rate"]
            )
            assert repetition.error_rate_upper_bound == pytest.approx(
                expected_cluster["error_rate_upper_bound"]
            )

    metrics = {metric.id: metric for metric in capacity_metrics(records, profile)}
    summary = fixture["expected_summary"]
    for metric_id in (
        "capacity.error_rate",
        "capacity.success_rate",
        "capacity.error_rate_upper_bound",
        "capacity.error_rate_cluster_range_max",
        "capacity.measurement_cluster_count_min",
    ):
        expected_value = summary[metric_id.removeprefix("capacity.")]
        assert metrics[metric_id].value == pytest.approx(expected_value)
    assert (
        metrics["capacity.error_rate"].sample_count
        == summary["measurement_cluster_count"]
    )
    assert (
        metrics["capacity.error_rate_upper_bound"].sample_count
        == summary["measurement_cluster_count"]
    )
    assert metrics["capacity.error_rate_cluster_range_max"].sample_count == len(
        fixture["levels"]
    )


def test_recorded_capacity_does_not_infer_cluster_statistics_from_requests() -> None:
    records = _records(default_capacity_load_protocol(2))
    metrics = {metric.id: metric for metric in capacity_metrics(records, None)}

    for metric_id in (
        "capacity.success_rate",
        "capacity.error_rate",
        "capacity.error_rate_upper_bound",
        "capacity.error_rate_cluster_range_max",
        "capacity.measurement_cluster_count_min",
    ):
        assert metrics[metric_id].value is None
        assert metrics[metric_id].sample_count == 0


def test_capacity_profile_rejects_unstable_repetitions() -> None:
    protocol = default_capacity_load_protocol(2)
    profile = build_capacity_profile(
        _records(protocol, throughputs={1: (10, 10, 10), 2: (10, 20, 10)}),
        _slo(),
        protocol,
    )

    assert profile.levels[1].throughput_cv > protocol.max_throughput_cv
    assert profile.levels[1].throughput_stability_passed is False
    assert "throughput_stability" in profile.assessment.failure_reasons


def test_capacity_protocol_rejects_a_tiny_measurement_window() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 100"):
        CapacityLoadProtocol(
            concurrency_levels=(1, 2),
            warmup_request_multiplier=2,
            measurement_requests_per_repetition=2,
            repetitions_per_level=3,
            confidence_level=0.95,
            max_throughput_cv=0.2,
            max_latency_p95_cv=0.2,
        )


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("assessment", "slo_headroom"), 9, "assessment does not match"),
        (("levels", 1, "qualified"), False, "decisions do not match"),
        (("levels", 1, "error_rate_upper_bound"), 0.0, "statistics do not match"),
        (("levels", 1, "repetitions", 1, "requests"), 99, "counts do not sum"),
    ),
)
def test_capacity_profile_rejects_tampered_derived_evidence(
    path: tuple[str | int, ...], value: object, message: str
) -> None:
    protocol = default_capacity_load_protocol(2)
    document = build_capacity_profile(_records(protocol), _slo(), protocol).model_dump(
        mode="json", exclude_none=False
    )
    target: object = document
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]

    with pytest.raises(ValidationError, match=message):
        CapacityProfile.model_validate(document)


def test_capacity_cost_uses_record_ordered_binary64_sum() -> None:
    protocol = default_capacity_load_protocol(2)
    costs = (1e16, 1.0, 1.0)
    expected = 0.0
    for value in costs:
        expected += value

    profile = build_capacity_profile(
        _records(protocol, costs=costs),
        _slo(required_concurrency=1),
        protocol,
    )

    assert profile.levels[0].runtime_cost_usd == expected
