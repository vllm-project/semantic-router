"""Distribution checks for workload mixture sampled request streams."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..core.request import Request
from .mixture import MixtureScenario, _load_cdf_source, validate_mixture_scenario

MIN_SAMPLE_SIZE = 100


@dataclass(frozen=True)
class MixtureValidationReport:
    """Distribution validation summary for a sampled mixture request stream."""

    sample_size: int
    expected_weights: dict[str, float]
    observed_weights: dict[str, float]
    weight_error_by_archetype: dict[str, float]
    quantiles: dict[str, int]
    aggregate_cdf_max_error: float | None = None
    component_cdf_max_error: dict[str, float] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


def validate_sample_distribution(
    arrivals: list[tuple[float, Request]],
    scenario: MixtureScenario,
    weight_tolerance: float = 0.05,
    cdf_tolerance: float = 0.10,
    min_sample_size: int = MIN_SAMPLE_SIZE,
) -> MixtureValidationReport:
    """Validate sampled request weights and CDF reproduction."""

    validate_mixture_scenario(scenario)
    requests = [req for _, req in arrivals]
    sample_size = len(requests)
    expected_weights = _expected_weights_for_sample(arrivals, scenario)
    observed_weights = _observed_weights(requests, scenario.archetype_ids)
    errors: list[str] = []
    warnings: list[str] = []

    if sample_size < min_sample_size:
        warnings.append(
            f"sample size {sample_size} is below min_sample_size={min_sample_size}; "
            "distribution errors are warnings-only"
        )

    weight_errors = _weight_errors(
        expected_weights,
        observed_weights,
        sample_size,
        min_sample_size,
        weight_tolerance,
        errors,
    )
    quantiles = _quantiles([req.total_tokens() for req in requests])
    aggregate_error = None
    component_errors: dict[str, float] = {}
    if all(archetype.source.kind == "cdf" for archetype in scenario.archetypes):
        cdfs = {
            archetype.id: _load_cdf_source(archetype.source, scenario.base_dir)
            for archetype in scenario.archetypes
        }
        aggregate_error, component_errors = _cdf_errors(
            requests,
            scenario,
            cdfs,
            expected_weights,
            sample_size,
            min_sample_size,
            cdf_tolerance,
            errors,
        )
    else:
        warnings.append("CDF reproduction checks skipped for non-CDF archetype sources")

    return MixtureValidationReport(
        sample_size=sample_size,
        expected_weights=expected_weights,
        observed_weights=observed_weights,
        weight_error_by_archetype=weight_errors,
        quantiles=quantiles,
        aggregate_cdf_max_error=aggregate_error,
        component_cdf_max_error=component_errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def _weight_errors(
    expected_weights: dict[str, float],
    observed_weights: dict[str, float],
    sample_size: int,
    min_sample_size: int,
    weight_tolerance: float,
    errors: list[str],
) -> dict[str, float]:
    weight_errors: dict[str, float] = {}
    for archetype_id, expected in expected_weights.items():
        observed = observed_weights.get(archetype_id, 0.0)
        delta = abs(observed - expected)
        weight_errors[archetype_id] = delta
        if sample_size >= min_sample_size and delta > weight_tolerance:
            errors.append(
                f"{archetype_id}: observed weight {observed:.4f} differs from "
                f"expected {expected:.4f} by {delta:.4f}"
            )
    return weight_errors


def _cdf_errors(
    requests: list[Request],
    scenario: MixtureScenario,
    cdfs: dict[str, list[tuple[int, float]]],
    expected_weights: dict[str, float],
    sample_size: int,
    min_sample_size: int,
    cdf_tolerance: float,
    errors: list[str],
) -> tuple[float, dict[str, float]]:
    aggregate_error = _aggregate_cdf_error(requests, cdfs, expected_weights)
    if sample_size >= min_sample_size and aggregate_error > cdf_tolerance:
        errors.append(
            f"aggregate CDF max error {aggregate_error:.4f} exceeds "
            f"cdf_tolerance={cdf_tolerance:.4f}"
        )
    component_errors: dict[str, float] = {}
    for archetype_id, cdf in cdfs.items():
        component_requests = [
            req for req in requests if req.archetype_id == archetype_id
        ]
        if len(component_requests) < min_sample_size:
            continue
        component_error = _single_cdf_error(
            [req.total_tokens() for req in component_requests],
            cdf,
        )
        component_errors[archetype_id] = component_error
        if component_error > cdf_tolerance:
            errors.append(
                f"{archetype_id}: component CDF max error "
                f"{component_error:.4f} exceeds cdf_tolerance={cdf_tolerance:.4f}"
            )
    return aggregate_error, component_errors


def _expected_weights_for_sample(
    arrivals: list[tuple[float, Request]],
    scenario: MixtureScenario,
) -> dict[str, float]:
    if not scenario.composition_schedule:
        return scenario.nominal_weights
    weighted: dict[str, float] = dict.fromkeys(scenario.archetype_ids, 0.0)
    windows = sorted(scenario.composition_schedule, key=lambda item: item.start_s)
    matched = 0
    for arrival, _ in arrivals:
        window = next(
            (item for item in windows if item.start_s <= arrival <= item.end_s),
            None,
        )
        if window is None:
            continue
        matched += 1
        for archetype_id, weight in window.weights.items():
            weighted[archetype_id] += weight
    total = sum(weighted.values())
    if matched == 0 or total <= 0:
        return scenario.nominal_weights
    return {archetype_id: value / total for archetype_id, value in weighted.items()}


def _observed_weights(
    requests: list[Request],
    archetype_ids: tuple[str, ...],
) -> dict[str, float]:
    counts = dict.fromkeys(archetype_ids, 0)
    for req in requests:
        if req.archetype_id in counts:
            counts[req.archetype_id] += 1
    total = len(requests)
    if total == 0:
        return dict.fromkeys(archetype_ids, 0.0)
    return {archetype_id: count / total for archetype_id, count in counts.items()}


def _quantiles(lengths: list[int]) -> dict[str, int]:
    if not lengths:
        return {}
    sorted_lengths = sorted(lengths)

    def pick(q: float) -> int:
        index = min(
            len(sorted_lengths) - 1, max(0, round(q * (len(sorted_lengths) - 1)))
        )
        return sorted_lengths[index]

    return {"p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99)}


def _aggregate_cdf_error(
    requests: list[Request],
    cdfs: dict[str, list[tuple[int, float]]],
    weights: dict[str, float],
) -> float:
    lengths = [req.total_tokens() for req in requests]
    if not lengths:
        return 0.0
    points = sorted({threshold for cdf in cdfs.values() for threshold, _ in cdf})
    max_error = 0.0
    for point in points:
        observed = sum(1 for length in lengths if length <= point) / len(lengths)
        expected = sum(
            weights[archetype_id] * _cdf_value(cdf, point)
            for archetype_id, cdf in cdfs.items()
        )
        max_error = max(max_error, abs(observed - expected))
    return max_error


def _single_cdf_error(lengths: list[int], cdf: list[tuple[int, float]]) -> float:
    if not lengths:
        return 0.0
    max_error = 0.0
    for threshold, expected in cdf:
        observed = sum(1 for length in lengths if length <= threshold) / len(lengths)
        max_error = max(max_error, abs(observed - expected))
    return max_error


def _cdf_value(cdf: list[tuple[int, float]], token_length: int) -> float:
    for threshold, fraction in cdf:
        if token_length <= threshold:
            return fraction
    return 1.0
