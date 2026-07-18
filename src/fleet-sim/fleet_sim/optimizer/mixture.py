"""FleetOptimizer reporting for workload mixture scenarios."""

from __future__ import annotations

from ..workload.mixture import (
    CompositionWindow,
    MixtureScenario,
    MixtureValidationError,
    WorkloadArchetype,
    _load_cdf_source,
    validate_mixture_scenario,
)
from ..workload.trace import TraceWorkload
from . import analytical
from .base import FleetOptimizer, SweepResult
from .mixture_models import (
    MixtureCaseResult,
    MixtureOptimizationError,
    MixtureOptimizationReport,
    MixtureStressCase,
    RobustMixtureRecommendation,
)


def evaluate_mixture_scenario(
    optimizer: FleetOptimizer,
    scenario: MixtureScenario,
    lam: float,
    gammas: list[float] | None = None,
    n_sim_requests: int = 0,
    verify_top_n: int = 0,
    max_total_gpus: int | None = None,
    fail_on_infeasible: bool = False,
    verbose: bool = False,
) -> MixtureOptimizationReport:
    """Evaluate FleetOptimizer sensitivity across mixture-derived stress cases."""

    if lam <= 0:
        raise ValueError("lam must be positive")
    if max_total_gpus is not None and max_total_gpus <= 0:
        raise ValueError("max_total_gpus must be positive")
    validate_mixture_scenario(scenario)
    gammas = gammas or [round(1.0 + 0.1 * k, 1) for k in range(11)]

    archetype_cdfs = _archetype_cdfs(scenario)
    cases = _stress_cases(scenario, archetype_cdfs, lam)
    case_results = [
        _evaluate_case(
            optimizer,
            case,
            gammas,
            n_sim_requests,
            verify_top_n,
            max_total_gpus,
            verbose,
        )
        for case in cases
    ]
    _annotate_sensitivity(case_results)
    robust = _robust_recommendation(
        optimizer,
        [case for case in cases if case.kind != "aggregate_cdf_baseline"],
        gammas,
        max_total_gpus,
    )
    diagnostics = _diagnostics(case_results, robust)
    report = MixtureOptimizationReport(
        scenario_id=scenario.id,
        scenario_version=scenario.version,
        lam=lam,
        cases=tuple(case_results),
        robust_recommendation=robust,
        diagnostics=tuple(diagnostics),
    )
    if fail_on_infeasible and not report.ok:
        raise MixtureOptimizationError(
            "; ".join(report.diagnostics) or "mixture infeasible"
        )
    return report


def aggregate_mixture_cdf(
    cdfs: dict[str, list[tuple[int, float]]],
    weights: dict[str, float],
) -> list[tuple[int, float]]:
    """Collapse weighted archetype CDFs into one aggregate marginal CDF."""

    missing = sorted(set(weights) - set(cdfs))
    if missing:
        raise MixtureValidationError(f"missing CDFs for archetypes: {missing}")
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise MixtureValidationError(f"weights must sum to 1.0, got {total:.6f}")
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
                        weights[archetype_id]
                        * analytical.cdf_eval(cdfs[archetype_id], threshold)
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


def _archetype_cdfs(scenario: MixtureScenario) -> dict[str, list[tuple[int, float]]]:
    return {
        archetype.id: _cdf_for_archetype(archetype, scenario)
        for archetype in scenario.archetypes
    }


def _cdf_for_archetype(
    archetype: WorkloadArchetype,
    scenario: MixtureScenario,
) -> list[tuple[int, float]]:
    if archetype.source.kind == "cdf":
        return _load_cdf_source(archetype.source, scenario.base_dir)
    trace = TraceWorkload(
        str(archetype.source.resolve_path(scenario.base_dir)),
        fmt=archetype.source.format or "semantic_router",
        max_reqs=archetype.source.max_reqs,
        field_map=archetype.source.field_map,
        model_id_field=archetype.source.model_id_field,
        default_model_id=archetype.source.default_model_id,
    )
    totals = sorted(req.total_tokens() for _, req in trace.generate())
    if not totals:
        raise MixtureValidationError(
            f"{archetype.id}: trace source produced no requests"
        )
    return _empirical_cdf(totals)


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


def _stress_cases(
    scenario: MixtureScenario,
    cdfs: dict[str, list[tuple[int, float]]],
    lam: float,
) -> list[MixtureStressCase]:
    cases = [
        _case(
            "aggregate-cdf",
            "aggregate_cdf_baseline",
            lam,
            scenario.nominal_weights,
            cdfs,
            "Collapsed aggregate CDF baseline at nominal weights.",
        ),
        _case(
            "nominal-mixture",
            "nominal_mixture",
            lam,
            scenario.nominal_weights,
            cdfs,
            "Versioned nominal workload mixture.",
        ),
    ]
    for archetype_id in scenario.archetype_ids:
        weights = {
            item: 1.0 if item == archetype_id else 0.0
            for item in scenario.archetype_ids
        }
        cases.append(
            _case(
                f"archetype:{archetype_id}",
                "individual_archetype",
                lam,
                weights,
                cdfs,
                f"All traffic follows archetype {archetype_id}.",
            )
        )
    for index, window in enumerate(scenario.composition_schedule):
        cases.append(_window_case(index, window, scenario, cdfs, lam))
    return cases


def _window_case(
    index: int,
    window: CompositionWindow,
    scenario: MixtureScenario,
    cdfs: dict[str, list[tuple[int, float]]],
    lam: float,
) -> MixtureStressCase:
    return _case(
        f"window:{index}",
        "composition_window",
        lam * window.arrival_rate_multiplier,
        window.weights,
        cdfs,
        (
            f"Composition window {index} from {window.start_s:.0f}s to "
            f"{window.end_s:.0f}s."
        ),
    )


def _case(
    case_id: str,
    kind: str,
    lam: float,
    weights: dict[str, float],
    cdfs: dict[str, list[tuple[int, float]]],
    description: str,
) -> MixtureStressCase:
    return MixtureStressCase(
        id=case_id,
        kind=kind,
        lam=lam,
        weights=dict(weights),
        cdf=aggregate_mixture_cdf(cdfs, weights),
        description=description,
    )


def _evaluate_case(
    optimizer: FleetOptimizer,
    case: MixtureStressCase,
    gammas: list[float],
    n_sim_requests: int,
    verify_top_n: int,
    max_total_gpus: int | None,
    verbose: bool,
) -> MixtureCaseResult:
    report = optimizer.optimize(
        cdf=case.cdf,
        lam=case.lam,
        gammas=gammas,
        n_sim_requests=n_sim_requests,
        verify_top_n=verify_top_n,
        verbose=verbose,
    )
    best = report.best_simulated or report.best_analytical
    baseline = next((item for item in report.analytical if item.gamma == 1.0), None)
    reason = _infeasible_reason(best, max_total_gpus)
    return MixtureCaseResult(
        case_id=case.id,
        kind=case.kind,
        lam=case.lam,
        weights=case.weights,
        best=best,
        baseline=baseline,
        sweep=tuple(report.analytical + report.simulated),
        infeasible_reason=reason,
    )


def _infeasible_reason(
    best: SweepResult | None, max_total_gpus: int | None
) -> str | None:
    if best is None:
        return "FleetOptimizer produced no candidates"
    if not best.slo_met:
        return (
            f"best candidate violates SLO: short={best.p99_ttft_short_ms:.1f}ms "
            f"long={best.p99_ttft_long_ms:.1f}ms"
        )
    if max_total_gpus is not None and best.total_gpus > max_total_gpus:
        return (
            f"requires {best.total_gpus} GPUs, exceeds max_total_gpus={max_total_gpus}"
        )
    return None


def _annotate_sensitivity(results: list[MixtureCaseResult]) -> None:
    nominal = next(
        (case for case in results if case.case_id == "nominal-mixture"), None
    )
    if nominal is None or nominal.best is None:
        return
    base = nominal.best.annualised_cost_kusd
    for case in results:
        if case.best is None:
            continue
        delta = case.best.annualised_cost_kusd - base
        case.annual_cost_delta_vs_nominal_kusd = delta
        case.cost_sensitivity_pct = (delta / base * 100.0) if base > 0 else 0.0


def _robust_recommendation(
    optimizer: FleetOptimizer,
    cases: list[MixtureStressCase],
    gammas: list[float],
    max_total_gpus: int | None,
) -> RobustMixtureRecommendation | None:
    candidates: list[RobustMixtureRecommendation] = []
    for gamma in gammas:
        per_case = [
            (
                case,
                optimizer.sweep_analytical(case.cdf, case.lam, [gamma], verbose=False)[
                    0
                ],
            )
            for case in cases
        ]
        if any(not result.slo_met for _, result in per_case):
            continue
        n_s = max(result.n_s for _, result in per_case)
        n_l = max(result.n_l for _, result in per_case)
        total = n_s + n_l
        if max_total_gpus is not None and total > max_total_gpus:
            continue
        cost_hr = (
            n_s * optimizer.gpu_short.cost_per_hr + n_l * optimizer.gpu_long.cost_per_hr
        )
        worst_case, worst = max(
            per_case,
            key=lambda item: max(item[1].p99_ttft_short_ms, item[1].p99_ttft_long_ms),
        )
        candidates.append(
            RobustMixtureRecommendation(
                gamma=gamma,
                n_s=n_s,
                n_l=n_l,
                total_gpus=total,
                cost_per_hr=cost_hr,
                annualised_cost_kusd=cost_hr * 8760 / 1000,
                worst_case_id=worst_case.id,
                worst_p99_ttft_short_ms=worst.p99_ttft_short_ms,
                worst_p99_ttft_long_ms=worst.p99_ttft_long_ms,
                slo_met=True,
            )
        )
    if candidates:
        return min(candidates, key=lambda item: item.cost_per_hr)
    return None


def _diagnostics(
    cases: list[MixtureCaseResult],
    robust: RobustMixtureRecommendation | None,
) -> list[str]:
    diagnostics = [
        f"{case.case_id}: {case.infeasible_reason}"
        for case in cases
        if case.infeasible_reason
    ]
    worst = max(
        (case for case in cases if case.best is not None),
        key=lambda case: case.best.annualised_cost_kusd,
        default=None,
    )
    if worst is not None:
        diagnostics.append(
            f"worst-case mixture by cost is {worst.case_id} at "
            f"${worst.best.annualised_cost_kusd:.1f}K/yr"
        )
    if robust is None:
        diagnostics.append("no robust recommendation satisfies all stress cases")
    return diagnostics
