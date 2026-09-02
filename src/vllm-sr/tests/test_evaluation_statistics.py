from __future__ import annotations

from cli.evaluation.evidence import ExecutionRecord
from cli.evaluation.metric_core import metric_analysis_provenance
from cli.evaluation.reporting import EvaluationMetric
from cli.evaluation.statistics import (
    attach_confidence_intervals,
    bootstrap_interval,
)


def _record(case: int, track: str, **updates: object) -> ExecutionRecord:
    return ExecutionRecord(
        id=f"{track}-{case}",
        track_id=track,
        case_id=f"case-{case}",
        attempt_id="attempt-0",
        status="succeeded",
        **updates,
    )


def _metric(
    metric_id: str,
    value: float,
    sample_count: int,
    track_id: str,
    unit: str,
) -> EvaluationMetric:
    return EvaluationMetric(
        id=metric_id,
        name=metric_id,
        track_id=track_id,
        value=value,
        unit=unit,
        direction="higher_is_better",
        sample_count=sample_count,
        analysis_provenance=metric_analysis_provenance(
            metric_id, observed_exclusions=0
        ),
    )


def test_bootstrap_interval_is_deterministic_and_rejects_singleton_claim() -> None:
    first = bootstrap_interval(
        [1.0, 2.0, 3.0, 4.0], lambda rows: sum(rows) / len(rows), seed=7
    )
    second = bootstrap_interval(
        [1.0, 2.0, 3.0, 4.0], lambda rows: sum(rows) / len(rows), seed=7
    )
    assert first == second
    assert first is not None
    assert first[0] <= 2.5 <= first[1]
    assert bootstrap_interval([1.0], lambda rows: rows[0], seed=7) is None


def test_fraction_metric_gets_wilson_not_zero_width_interval() -> None:
    decorated = attach_confidence_intervals(
        [_metric("routing.coverage", 1.0, 20, "routing", "fraction")],
        [],
        seed=42,
    )
    interval = decorated[0].confidence_interval
    assert interval is not None
    assert interval[0] < 1.0
    assert interval[1] == 1.0


def test_server_reduced_worst_arm_reliability_has_no_worker_interval() -> None:
    records = [
        _record(case, "model_pool", arm_id="stable", success=True)
        for case in range(1, 5)
    ] + [
        _record(case, "model_pool", arm_id="weak", success=case % 2 == 0)
        for case in range(1, 5)
    ]
    metric = _metric(
        "model_pool.worst_arm_reliability",
        0.5,
        4,
        "model_pool",
        "fraction",
    )

    decorated = attach_confidence_intervals([metric], records, seed=17, resamples=400)

    assert decorated[0].confidence_interval is None


def test_worst_arm_reliability_interval_rejects_a_sparse_arm_matrix() -> None:
    records = [
        _record(1, "model_pool", arm_id="stable", success=True),
        _record(1, "model_pool", arm_id="weak", success=False),
        _record(2, "model_pool", arm_id="stable", success=True),
    ]
    metric = _metric(
        "model_pool.worst_arm_reliability",
        0.5,
        2,
        "model_pool",
        "fraction",
    )

    decorated = attach_confidence_intervals([metric], records, seed=17, resamples=400)

    assert decorated[0].confidence_interval is None


def test_unattested_joint_regret_omits_interval_while_latency_uses_case_evidence() -> (
    None
):
    records = [
        _record(1, "model_pool", arm_id="fast", success=True, quality=0.5),
        _record(1, "model_pool", arm_id="strong", success=True, quality=1.0),
        _record(1, "joint", quality=0.8, latency_ms=100),
        _record(2, "model_pool", arm_id="fast", success=True, quality=0.8),
        _record(2, "model_pool", arm_id="strong", success=True, quality=0.9),
        _record(2, "joint", quality=0.6, latency_ms=200),
        _record(3, "model_pool", arm_id="fast", success=True, quality=0.7),
        _record(3, "model_pool", arm_id="strong", success=True, quality=0.8),
        _record(3, "joint", quality=0.7, latency_ms=300),
    ]
    metrics = [
        _metric("joint.normalized_regret", 0.22, 3, "joint", "fraction"),
        _metric("joint.latency_p95_ms", 290, 3, "joint", "ms"),
    ]
    decorated = attach_confidence_intervals(metrics, records, seed=3, resamples=400)
    assert decorated[0].confidence_interval is None
    assert decorated[1].confidence_interval is not None


def test_unavailable_metric_never_receives_an_interval() -> None:
    metric = EvaluationMetric(
        id="capacity.slo_headroom",
        name="SLO headroom",
        track_id="capacity",
        value=None,
        unit="concurrency",
        direction="higher_is_better",
        sample_count=0,
        analysis_provenance=metric_analysis_provenance(
            "capacity.slo_headroom", observed_exclusions=0
        ),
    )
    decorated = attach_confidence_intervals([metric], [], seed=1)
    assert decorated[0].confidence_interval is None


def test_rich_track_metrics_use_their_actual_analysis_units() -> None:
    records = [
        _record(1, "model_pool", arm_id="fast", success=True, quality=0.5),
        _record(1, "model_pool", arm_id="strong", success=True, quality=1.0),
        _record(1, "joint", success=True, quality=0.8),
        _record(2, "model_pool", arm_id="fast", success=True, quality=0.8),
        _record(2, "model_pool", arm_id="strong", success=True, quality=0.9),
        _record(2, "joint", success=True, quality=0.6),
        _record(1, "agentic", success=True, trajectory_steps=2),
        _record(2, "agentic", success=True, trajectory_steps=6),
        _record(1, "multimodal", success=True, modality="image", quality=0.9),
        _record(2, "multimodal", success=False, modality="image", quality=0.2),
    ]
    metrics = [
        _metric("joint.oracle_capture_ratio", 0.733, 2, "joint", "fraction"),
        _metric("agentic.mean_trajectory_steps", 4.0, 2, "agentic", "steps"),
        _metric(
            "multimodal.image.support_rate",
            0.5,
            2,
            "multimodal",
            "fraction",
        ),
        _metric("multimodal.image.quality", 0.55, 2, "multimodal", "score"),
    ]

    decorated = attach_confidence_intervals(metrics, records, seed=9, resamples=400)

    assert all(metric.confidence_interval is not None for metric in decorated)
    assert (
        decorated[2].confidence_interval[0] < 0.5 < decorated[2].confidence_interval[1]
    )
