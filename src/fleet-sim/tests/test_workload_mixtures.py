"""Tests for workload archetype mixture scenarios."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fleet_sim.workload import (
    MixtureSampler,
    MixtureScenario,
    MixtureValidationError,
    load_mixture_scenario,
    validate_mixture_scenario,
    validate_sample_distribution,
)

_DATA = Path(__file__).parent.parent / "fleet_sim" / "data"


def _scenario(name: str) -> MixtureScenario:
    return load_mixture_scenario(_DATA / f"workload_mixture_{name}.json")


def _signature(arrivals):
    return [
        (
            round(t, 6),
            req.l_in,
            req.l_out,
            req.category,
            req.archetype_id,
            req.slo_class,
            req.eligible_models,
            req.residency,
        )
        for t, req in arrivals
    ]


def test_nominal_fixture_loads_and_validates():
    scenario = _scenario("nominal")

    assert scenario.schema_version == "fleet-sim.workload-mixture/v1alpha1"
    assert scenario.id == "nominal-chat-multiturn-agent"
    assert scenario.nominal_weights == {
        "interactive-chat": 0.6,
        "multiturn-chat": 0.25,
        "agent-heavy": 0.15,
    }


def test_bad_weight_normalization_fails():
    data = json.loads((_DATA / "workload_mixture_nominal.json").read_text())
    data["archetypes"][0]["weight"] = 0.7
    scenario = MixtureScenario.from_dict(data, base_dir=_DATA)

    with pytest.raises(MixtureValidationError, match=r"weights must sum to 1\.0"):
        validate_mixture_scenario(scenario)


def test_invalid_cdf_fails_closed(tmp_path):
    bad_cdf = tmp_path / "bad_cdf.json"
    bad_cdf.write_text(json.dumps([[10, 0.7], [5, 1.0]]))
    scenario = MixtureScenario.from_dict(
        {
            "schema_version": "fleet-sim.workload-mixture/v1alpha1",
            "id": "bad-cdf",
            "version": "test",
            "seed": 1,
            "archetypes": [
                {
                    "id": "bad",
                    "version": "test",
                    "source": {"kind": "cdf", "path": "bad_cdf.json"},
                    "arrival_process": {"kind": "poisson"},
                    "slo_class": "interactive",
                    "model_eligibility": ["test-model"],
                    "residency": ["global"],
                    "weight": 1.0,
                }
            ],
        },
        base_dir=tmp_path,
    )

    with pytest.raises(MixtureValidationError, match="strictly increasing"):
        validate_mixture_scenario(scenario)


def test_fixed_mixture_sampling_is_reproducible():
    scenario = _scenario("nominal")

    first = MixtureSampler(scenario).generate(lam=80, n_requests=200)
    second = MixtureSampler(scenario).generate(lam=80, n_requests=200)
    changed_seed = MixtureSampler(replace(scenario, seed=scenario.seed + 1)).generate(
        lam=80,
        n_requests=200,
    )

    assert _signature(first) == _signature(second)
    assert _signature(first) != _signature(changed_seed)


def test_fixed_mixture_distribution_report_passes_for_large_sample():
    scenario = _scenario("nominal")
    arrivals = MixtureSampler(scenario).generate(lam=400, n_requests=8_000)

    report = validate_sample_distribution(
        arrivals,
        scenario,
        weight_tolerance=0.03,
        cdf_tolerance=0.08,
    )

    assert report.ok, report.errors
    assert report.aggregate_cdf_max_error is not None
    assert report.quantiles["p95"] >= report.quantiles["p50"]


def test_distribution_report_catches_sampling_errors():
    scenario = _scenario("nominal")
    arrivals = MixtureSampler(scenario).generate(lam=200, n_requests=1_000)
    for _, req in arrivals:
        req.archetype_id = "interactive-chat"

    report = validate_sample_distribution(arrivals, scenario, weight_tolerance=0.05)

    assert not report.ok
    assert any("observed weight" in error for error in report.errors)


def test_small_samples_warn_instead_of_failing_distribution_checks():
    scenario = _scenario("nominal")
    arrivals = MixtureSampler(scenario).generate(lam=20, n_requests=12)

    report = validate_sample_distribution(arrivals, scenario)

    assert report.ok
    assert any("sample size" in warning for warning in report.warnings)


def test_time_varying_fixture_shifts_and_validates_distribution():
    scenario = _scenario("drift")
    arrivals = MixtureSampler(scenario).generate_until(lam=80, duration_s=120)

    first_window = [req for t, req in arrivals if t < 60]
    second_window = [req for t, req in arrivals if t >= 60]

    def share(requests, archetype_id: str) -> float:
        return sum(1 for req in requests if req.archetype_id == archetype_id) / len(
            requests
        )

    assert (
        share(second_window, "agent-heavy") > share(first_window, "agent-heavy") + 0.20
    )

    report = validate_sample_distribution(
        arrivals,
        scenario,
        weight_tolerance=0.04,
        cdf_tolerance=0.08,
    )
    assert report.ok, report.errors


def test_request_metadata_is_stamped_from_archetype():
    scenario = _scenario("nominal")
    arrivals = MixtureSampler(scenario).generate(lam=200, n_requests=200)
    agent_req = next(req for _, req in arrivals if req.archetype_id == "agent-heavy")

    assert agent_req.slo_class == "batch"
    assert agent_req.eligible_models == ("llama-3.1-70b", "deepseek-v3")
    assert agent_req.residency == ("us",)


def test_trace_source_preserves_explicit_row_fields(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": 1.0,
                        "prompt_tokens": 100,
                        "generated_tokens": 20,
                        "selected_model": "trace-model-a",
                        "category": "code",
                        "complexity": "hard",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": 2.0,
                        "prompt_tokens": 40,
                        "generated_tokens": 10,
                        "selected_model": "trace-model-b",
                        "category": "rag",
                        "complexity": "easy",
                    }
                ),
            ]
        )
    )
    scenario = MixtureScenario.from_dict(
        {
            "schema_version": "fleet-sim.workload-mixture/v1alpha1",
            "id": "trace-mixture",
            "version": "test",
            "seed": 7,
            "archetypes": [
                {
                    "id": "trace-archetype",
                    "version": "test",
                    "source": {
                        "kind": "trace",
                        "path": "trace.jsonl",
                        "format": "semantic_router",
                    },
                    "arrival_process": {"kind": "poisson"},
                    "slo_class": "interactive",
                    "model_eligibility": ["trace-model-a", "trace-model-b"],
                    "residency": ["global"],
                    "weight": 1.0,
                }
            ],
        },
        base_dir=tmp_path,
    )

    arrivals = MixtureSampler(scenario).generate(lam=10, n_requests=2)

    first_req = arrivals[0][1]
    assert first_req.model_id == "trace-model-a"
    assert first_req.category == "code"
    assert first_req._complexity == "hard"
    assert first_req.archetype_id == "trace-archetype"
