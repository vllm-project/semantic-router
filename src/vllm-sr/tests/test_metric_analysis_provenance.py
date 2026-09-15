from types import SimpleNamespace

import pytest
from cli.evaluation.metric_analysis_catalog import resolve_metric_analysis
from cli.evaluation.metric_core import build_metric, metric_analysis_provenance
from cli.evaluation.metrics import _bind_metric_analysis_provenance
from cli.evaluation.reporting import EvaluationMetric
from pydantic import ValidationError


def _metric_payload() -> dict[str, object]:
    return {
        "id": "routing.accuracy",
        "name": "Routing accuracy",
        "track_id": "routing",
        "value": 0.75,
        "unit": "fraction",
        "direction": "higher_is_better",
        "sample_count": 3,
        "analysis_provenance": metric_analysis_provenance(
            "routing.accuracy", observed_exclusions=1
        ),
    }


def test_metric_analysis_provenance_rejects_missing_and_forged_contracts() -> None:
    payload = _metric_payload()
    specification = resolve_metric_analysis("routing.accuracy").specification
    assert (
        EvaluationMetric.model_validate(payload).analysis_provenance.estimator_id
        == specification.estimator_id
        == "deterministic-routing-case-observed-ratio"
    )

    missing = dict(payload)
    missing.pop("analysis_provenance")
    with pytest.raises(ValidationError, match="analysis_provenance"):
        EvaluationMetric.model_validate(missing)

    forged = dict(payload)
    forged["analysis_provenance"] = {
        **payload["analysis_provenance"].model_dump(),
        "weighting": "uniform_repetition",
    }
    with pytest.raises(ValidationError, match="registered estimator"):
        EvaluationMetric.model_validate(forged)

    unknown = {**payload, "id": "routing.made_up_accuracy"}
    with pytest.raises(ValidationError, match="unknown evaluation metric id"):
        EvaluationMetric.model_validate(unknown)


def test_metric_draft_binds_unavailable_and_partial_units_without_track_inference() -> (
    None
):
    draft = build_metric(
        "routing.accuracy",
        "Routing accuracy",
        "routing",
        0.75,
        "fraction",
        "higher_is_better",
        3,
        planned_analysis_units=4,
    )
    assert not hasattr(draft, "analysis_provenance")
    records = [
        SimpleNamespace(track_id="routing", status="succeeded", case_id="a"),
        SimpleNamespace(track_id="routing", status="succeeded", case_id="b"),
        SimpleNamespace(track_id="routing", status="succeeded", case_id="c"),
        SimpleNamespace(track_id="routing", status="succeeded", case_id="d"),
        SimpleNamespace(track_id="routing", status="unavailable", case_id="e"),
    ]

    bound = _bind_metric_analysis_provenance([draft], records)

    assert bound[0].analysis_provenance.observed_exclusions == 2


def test_model_pool_arm_projection_counts_unavailable_cases_not_case_arm_rows() -> None:
    draft = build_metric(
        "model_pool.arm.fast.success_rate",
        "Fast success rate",
        "model_pool",
        1.0,
        "fraction",
        "higher_is_better",
        2,
        planned_analysis_units=2,
    )
    records = [
        SimpleNamespace(
            track_id="model_pool", status="unavailable", case_id="a", arm_id="fast"
        ),
        SimpleNamespace(
            track_id="model_pool", status="unavailable", case_id="a", arm_id="slow"
        ),
    ]

    bound = _bind_metric_analysis_provenance([draft], records)

    assert bound[0].analysis_provenance.observed_exclusions == 1


def test_capacity_projection_counts_unavailable_repetitions_once() -> None:
    draft = build_metric(
        "capacity.level.4.throughput_rps",
        "Throughput",
        "capacity",
        20.0,
        "requests/s",
        "higher_is_better",
        3,
        planned_analysis_units=3,
    )
    records = [
        SimpleNamespace(
            track_id="capacity",
            status="unavailable",
            case_id="a",
            concurrency=4,
            load_repetition=2,
        ),
        SimpleNamespace(
            track_id="capacity",
            status="unavailable",
            case_id="b",
            concurrency=4,
            load_repetition=2,
        ),
    ]

    bound = _bind_metric_analysis_provenance([draft], records)

    assert bound[0].analysis_provenance.observed_exclusions == 1


def test_dynamic_projection_uses_catalog_filters_and_rejects_missing_dimensions() -> (
    None
):
    draft = build_metric(
        "multimodal.image.quality",
        "Image quality",
        "multimodal",
        0.8,
        "score",
        "higher_is_better",
        2,
        planned_analysis_units=2,
    )
    records = [
        SimpleNamespace(
            track_id="multimodal",
            status="unavailable",
            case_id="image-case",
            modality="image",
        ),
        SimpleNamespace(
            track_id="multimodal",
            status="unavailable",
            case_id="audio-case",
            modality="audio",
        ),
    ]

    bound = _bind_metric_analysis_provenance([draft], records)

    assert bound[0].analysis_provenance.observed_exclusions == 1
    with pytest.raises(ValueError, match="without modality"):
        _bind_metric_analysis_provenance(
            [draft],
            [
                SimpleNamespace(
                    track_id="multimodal",
                    status="unavailable",
                    case_id="ambiguous-case",
                )
            ],
        )


def test_preference_propensity_missingness_is_an_observed_exclusion() -> None:
    draft = build_metric(
        "preference.self_normalized_ips_agreement",
        "Self-normalized IPS agreement",
        "preference",
        0.8,
        "fraction",
        "higher_is_better",
        3,
        planned_analysis_units=5,
    )

    bound = _bind_metric_analysis_provenance([draft], [])

    assert bound[0].analysis_provenance.observed_exclusions == 2
