"""Three-state verdict for the live tier (Sub-issue 3).

Ported from InferRSI's evaluation semantics, adapted to sr-bench evidence:

- INCONCLUSIVE: evidence is incomplete (an interrupted trial, missing
  metrics, an unfinished guard evaluation, an unevaluable constraint, a
  changed snapshot hash). This outranks FAIL - a broken run must never be
  recorded as "the recipe is worse".
- FAIL: evidence complete but the candidate does not beat the baseline by
  at least `min_improvement_pct`, OR a guard floor is violated (safety
  veto), OR an economic constraint is violated (E).
- PASS: evidence complete, guard floors hold, constraints hold,
  improvement reaches the bar.

The verdict is a pure function of its inputs: deterministic, unit-testable,
no I/O.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICTS = (VERDICT_PASS, VERDICT_FAIL, VERDICT_INCONCLUSIVE)


@dataclass(frozen=True)
class GuardResult:
    """One guard floor check (e.g. pii-recall-floor)."""

    name: str
    metric: str
    value: float | None  # None = the guard evaluation did not complete
    floor: float | None

    @property
    def evaluated(self) -> bool:
        return self.value is not None and self.floor is not None

    @property
    def passed(self) -> bool:
        return self.evaluated and self.value >= self.floor


@dataclass(frozen=True)
class VerdictResult:
    verdict: str
    reasons: list[str] = field(default_factory=list)
    improvement_pct: float | None = None
    constraints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "reasons": list(self.reasons),
            "improvement_pct": self.improvement_pct,
            "constraints": [dict(c) for c in self.constraints],
        }


def _improvement_pct(baseline: float, candidate: float, direction: str) -> float:
    if direction == "minimize":
        if baseline == 0:
            return 0.0 if candidate == 0 else -100.0
        return (baseline - candidate) / baseline * 100.0
    if baseline == 0:
        return 0.0 if candidate == 0 else 100.0
    return (candidate - baseline) / baseline * 100.0


@dataclass(frozen=True)
class PairedStats:
    """Paired comparison statistics (M2), sr-bench compare `comparisons[]` shape.

    `quality_delta` is the paired quality difference (candidate - baseline) on
    the 0-1 accuracy scale; both intervals are [low, high] 95% bounds computed
    by sr-bench (weighted-paired Hoeffding, fixed-seed bootstrap). The live
    tier reads and records them verbatim - it never recomputes a compare
    payload. `compute_paired_stats` exists for the drivers whose evidence
    carries per-case diffs instead of a compare payload (model_eval C7,
    SWE-bench AG2) and produces the same fields so results are comparable.

    `repetitions` counts the merged pairs (AB/BA orderings or repeated runs
    of the same case set) and `paired_observations` their total case count;
    a single unmerged comparison keeps the defaults (1 / None, where None
    means "equals paired_cases"). `paired_cases` stays the distinct-case
    count the min_cases bar is applied to.
    """

    paired_cases: int
    quality_delta: float
    quality_delta_ci95: tuple[float, float] | None
    quality_delta_bootstrap_ci95: tuple[float, float] | None
    wins: int
    losses: int
    ties: int
    repetitions: int = 1
    paired_observations: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paired_cases": self.paired_cases,
            "quality_delta": self.quality_delta,
            "quality_delta_ci95": (
                list(self.quality_delta_ci95) if self.quality_delta_ci95 else None
            ),
            "quality_delta_bootstrap_ci95": (
                list(self.quality_delta_bootstrap_ci95)
                if self.quality_delta_bootstrap_ci95
                else None
            ),
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "repetitions": self.repetitions,
            "paired_observations": self.paired_observations,
        }


_BOOTSTRAP_SEED = 20260918  # same fixed seed as sr-bench compare
_BOOTSTRAP_SAMPLES = 2000


def compute_paired_stats(
    diffs: Sequence[float],
    *,
    groups: dict[str, Sequence[float]] | None = None,
    weights: dict[str, float] | None = None,
    alpha: float = 0.05,
    bootstrap_samples: int = _BOOTSTRAP_SAMPLES,
    seed: int = _BOOTSTRAP_SEED,
) -> PairedStats:
    """Paired statistics in the sr-bench compare field shape (M2).

    `diffs` are per-case paired quality differences (candidate - baseline)
    on a bounded scale. With `groups` (benchmark -> per-case diffs) and
    `weights` (benchmark -> weight, normalized over the present groups) the
    sr-bench fixed-weight macro protocol is reproduced exactly: the
    Hoeffding radius is sqrt(2 * ln(2 / alpha) * sum(w_b^2 / n_b)) and the
    bootstrap resamples per group with the same fixed seed, so a driver's
    output is directly comparable to an sr-bench compare entry. Without
    groups the plain mean over `diffs` is used (one implicit group).
    """
    diffs = list(diffs)
    if not diffs:
        raise ValueError("compute_paired_stats needs at least one paired diff")
    for d in diffs:
        if isinstance(d, bool) or not isinstance(d, (int, float)):
            raise ValueError("paired diffs must be numbers")
    if groups is not None:
        if weights is None:
            raise ValueError("groups and weights must be provided together")
        missing = sorted(set(groups) - set(weights))
        if missing:
            raise ValueError(f"weights missing for groups: {missing}")
        covered = sum(len(values) for values in groups.values())
        if covered != len(diffs):
            raise ValueError(
                f"groups cover {covered} cases but {len(diffs)} diffs were given"
            )
        weight_sum = sum(weights[b] for b in groups)
        if weight_sum <= 0:
            raise ValueError("group weights must sum to a positive number")
        normalized = {b: weights[b] / weight_sum for b in groups}

        def resample(rng: random.Random) -> float:
            return sum(
                normalized[b]
                * statistics.mean(rng.choices(groups[b], k=len(groups[b])))
                for b in groups
            )

        delta = sum(normalized[b] * statistics.mean(groups[b]) for b in groups)
        squared = sum(normalized[b] ** 2 / len(groups[b]) for b in groups)
    else:

        def resample(rng: random.Random) -> float:
            return statistics.mean(rng.choices(diffs, k=len(diffs)))

        delta = statistics.mean(diffs)
        squared = 1.0 / len(diffs)
    radius = math.sqrt(2.0 * math.log(2.0 / alpha) * squared)
    ci = (max(-1.0, delta - radius), min(1.0, delta + radius))
    rng = random.Random(seed)
    samples = sorted(resample(rng) for _ in range(bootstrap_samples))
    # sr-bench indexes the 2.5% / 97.5% quantiles of its 2000 samples as
    # [49] / [1949]; the same offsets generalize to any sample count.
    low_i = max(0, int(0.025 * bootstrap_samples) - 1)
    high_i = min(bootstrap_samples - 1, int(0.975 * bootstrap_samples) - 1)
    return PairedStats(
        paired_cases=len(diffs),
        quality_delta=delta,
        quality_delta_ci95=ci,
        quality_delta_bootstrap_ci95=(samples[low_i], samples[high_i]),
        wins=sum(1 for d in diffs if d > 0),
        losses=sum(1 for d in diffs if d < 0),
        ties=sum(1 for d in diffs if d == 0),
    )


@dataclass(frozen=True)
class WorkloadGuardResult:
    """M5: one workload's auto "not worse" guard (contract guard_pp).

    The guard reads that workload's paired quality-delta CI lower bound:
    it must not fall below -guard_pp percentage points (the candidate may
    lose at most guard_pp pp on this workload). No paired evidence keeps
    the guard unevaluated - missing evidence is never counted as a loss.
    """

    workload: str
    guard_pp: float
    ci_low: float | None

    @property
    def evaluated(self) -> bool:
        return self.ci_low is not None

    @property
    def passed(self) -> bool:
        return self.evaluated and self.ci_low >= -(self.guard_pp / 100.0)


def workload_guard_results(
    guard_pp: dict[str, float],
    paired: dict[str, PairedStats | None],
) -> list[WorkloadGuardResult]:
    """M5: per-workload auto "not worse" guards from per-workload paired
    statistics (one sr-bench compare entry or per-driver PairedStats each).
    A workload without paired evidence stays unevaluated."""
    results: list[WorkloadGuardResult] = []
    for name in sorted(guard_pp):
        stats = paired.get(name)
        ci_low = (
            stats.quality_delta_ci95[0]
            if stats is not None and stats.quality_delta_ci95 is not None
            else None
        )
        results.append(
            WorkloadGuardResult(workload=name, guard_pp=guard_pp[name], ci_low=ci_low)
        )
    return results


def weighted_macro_average(
    measured: dict[str, float | None],
    weights: dict[str, float],
) -> float:
    """M5: the objective's target metric over tune workloads - the weighted
    macro average. Workloads without a measurement are excluded (missing
    evidence is never counted as zero) and the weights renormalize over the
    measured workloads."""
    present = {name: value for name, value in measured.items() if value is not None}
    if not present:
        raise ValueError(
            "weighted macro average needs at least one measured workload "
            "(missing evidence is never counted as zero)"
        )
    for name, value in present.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"workload {name!r} metric must be a finite number")
        weight = weights.get(name)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(f"workload {name!r} needs a positive macro-average weight")
    total = sum(float(weights[name]) for name in present)
    return sum(float(weights[name]) * float(v) for name, v in present.items()) / total


def resolve_constraint(
    constraint: dict[str, Any],
    references: dict[str, float | None] | None = None,
) -> dict[str, Any]:
    """Status and resolved bound for one constraint evidence entry (E).

    Shared by the live verdict and the replay cost proxy (E2): the replay
    tier feeds the same evidence shape with the routed_cost_index proxy
    substituted for the live sr-bench measurement.

    `constraint` carries the declared fields (name/metric/op/bound/
    tolerance_pct plus M4's tolerance_pp/factor) and the measured
    `baseline` and `candidate` values (None = unknown, e.g. sr-bench
    serializes unknown spend as null - unknown usage must never count as
    zero). A "baseline" bound resolves against the measured baseline
    value; a "ref:strong"/"ref:weak" bound (M4) resolves against the
    named reference arm's value via `references`, multiplied by `factor`
    when declared. tolerance_pct relaxes the bound multiplicatively and
    tolerance_pp (M4) additively in the permissive direction. A missing
    reference value is unevaluated (INCONCLUSIVE), never zero.
    """
    name = constraint["name"]
    op = constraint["op"]
    bound = constraint["bound"]
    tolerance = float(constraint.get("tolerance_pct", 0.0))
    baseline = constraint.get("baseline")
    candidate = constraint.get("candidate")
    out = dict(constraint)
    out["bound_value"] = None
    out["status"] = "not_evaluated"
    if candidate is None:
        return out
    ref_role = None
    if bound == "ref:strong":
        ref_role = "strong"
    elif bound == "ref:weak":
        ref_role = "weak"
    if ref_role is not None:
        ref_value = (references or {}).get(ref_role)
        if ref_value is None:
            out["not_evaluated_reason"] = (
                f"constraint {name} not evaluable: reference arm {ref_role!r} "
                f"for {constraint['metric']!r} is unknown (missing evidence, "
                "never counted as zero)"
            )
            return out
        factor = constraint.get("factor")
        bound_value = float(ref_value) * (float(factor) if factor is not None else 1.0)
        out["ref_value"] = float(ref_value)
    elif bound == "baseline":
        if baseline is None:
            return out
        bound_value = float(baseline)
    else:
        bound_value = float(bound)
    # lte allows a worst case above the bound, gte below it: tolerance_pct
    # relaxes multiplicatively, tolerance_pp (M4) by absolute percentage
    # points (the contract refuses declaring both).
    bound_value = (
        bound_value * (1 + tolerance / 100.0)
        if op == "lte"
        else bound_value * (1 - tolerance / 100.0)
    )
    tolerance_pp = float(constraint.get("tolerance_pp", 0.0) or 0.0)
    if tolerance_pp:
        bound_value += tolerance_pp / 100.0 if op == "lte" else -(tolerance_pp / 100.0)
    out["bound_value"] = bound_value
    violated = candidate > bound_value if op == "lte" else candidate < bound_value
    out["status"] = "violated" if violated else "pass"
    if violated:
        out["reason"] = (
            f"constraint {name} violated (candidate {candidate:.6f} {'>' if op == 'lte' else '<'} bound {bound_value:.6f})"
        )
    return out


def evidence_quality_reasons(
    arm: str,
    *,
    truncation_rate: float | None,
    missing_result_rate: float | None,
    no_answer_rate: float | None,
    max_truncation_pct: float,
    max_no_answer_pct: float,
) -> list[str]:
    """M3: incomplete-evidence reasons when an arm's output quality degrades.

    A truncated output, a case with no recorded result, or a missing
    answer is unreliable evidence, not a quality loss: a rate beyond its
    ceiling turns the verdict INCONCLUSIVE. Missing results and missing
    answers are distinct failures and are reported separately; both share
    the `max_no_answer_pct` ceiling because either way the case
    contributes no judged answer. Unknown rates (the payload carries no
    output diagnostics, or sr-bench publishes no per-case no-answer
    marker) are not penalized - missing evidence is reported, never
    guessed as zero.
    """
    reasons: list[str] = []
    if truncation_rate is not None and truncation_rate * 100.0 > max_truncation_pct:
        reasons.append(
            f"{arm} truncation rate {truncation_rate * 100.0:.2f}% exceeds max "
            f"{max_truncation_pct:.2f}%"
        )
    if (
        missing_result_rate is not None
        and missing_result_rate * 100.0 > max_no_answer_pct
    ):
        reasons.append(
            f"{arm} missing-result rate {missing_result_rate * 100.0:.2f}% exceeds "
            f"max {max_no_answer_pct:.2f}%"
        )
    if no_answer_rate is not None and no_answer_rate * 100.0 > max_no_answer_pct:
        reasons.append(
            f"{arm} no-answer rate {no_answer_rate * 100.0:.2f}% exceeds max "
            f"{max_no_answer_pct:.2f}%"
        )
    return reasons


def evaluate_verdict(
    *,
    baseline_metric: float | None,
    candidate_metric: float | None,
    direction: str,
    min_improvement_pct: float,
    guards: list[GuardResult],
    incomplete_reasons: list[str],
    constraints: list[dict[str, Any]] | None = None,
    stats: PairedStats | None = None,
    min_cases: int = 1,
    min_improvement_pp: float | None = None,
    references: dict[str, float | None] | None = None,
) -> VerdictResult:
    """Decide PASS / FAIL / INCONCLUSIVE from live-tier evidence.

    `incomplete_reasons` lists everything that makes the evidence unreliable
    (timeout, missing metric, snapshot mismatch, ...). Guards that did not
    complete and constraints that cannot be evaluated also count as
    incomplete evidence, not as failures.

    Guards are safety vetoes and outrank constraints (E): when both a guard
    and a constraint are violated the guard reasons are listed first.

    M2: when paired statistics are available they replace the point-estimate
    comparison entirely - PASS needs the CI lower bound at or above the
    `min_improvement_pp` bar, FAIL means the upper bound is below it, and a
    spanning interval or an undersized sample is INCONCLUSIVE. A relative
    `min_improvement_pct` cannot be applied to an absolute interval, so
    paired evidence with a relative bar stays undecided. Without paired
    statistics a declared pp bar has nothing to apply to (INCONCLUSIVE) and
    the point-estimate path applies `min_improvement_pct` unchanged.

    M4: constraint bounds may pin a reference arm (`ref:strong`/`ref:weak`);
    `references` carries those arms' measured values, and a missing
    reference value keeps the constraint unevaluated (INCONCLUSIVE).
    """
    reasons = list(incomplete_reasons)
    unevaluated = [g.name for g in guards if not g.evaluated]
    if unevaluated:
        reasons.append("guard evaluation incomplete: " + ", ".join(unevaluated))
    resolved_constraints = [
        resolve_constraint(c, references) for c in (constraints or [])
    ]
    for constraint in resolved_constraints:
        if constraint["status"] == "not_evaluated":
            reasons.append(
                constraint.get("not_evaluated_reason")
                or (
                    f"constraint {constraint['name']} not evaluable: "
                    f"{'candidate' if constraint.get('candidate') is None else 'baseline'} "
                    f"{constraint['metric']!r} is unknown (missing evidence, never counted as zero)"
                )
            )
    if reasons:
        return VerdictResult(
            VERDICT_INCONCLUSIVE, reasons, constraints=resolved_constraints
        )

    if baseline_metric is None:
        return VerdictResult(
            VERDICT_INCONCLUSIVE,
            ["baseline metric missing"],
            constraints=resolved_constraints,
        )
    if candidate_metric is None:
        return VerdictResult(
            VERDICT_INCONCLUSIVE,
            ["candidate metric missing"],
            constraints=resolved_constraints,
        )

    improvement = _improvement_pct(baseline_metric, candidate_metric, direction)
    failed_guards = [g for g in guards if not g.passed]
    violated_constraints = [
        c for c in resolved_constraints if c["status"] == "violated"
    ]
    if failed_guards:
        details = "; ".join(
            f"{g.name}: {g.value:.4f} < floor {g.floor:.4f}" for g in failed_guards
        )
        # safety veto first; any simultaneously violated constraint is still listed
        return VerdictResult(
            VERDICT_FAIL,
            [f"guard floor violated ({details})"]
            + [c["reason"] for c in violated_constraints],
            improvement_pct=round(improvement, 6),
            constraints=resolved_constraints,
        )
    if violated_constraints:
        return VerdictResult(
            VERDICT_FAIL,
            [c["reason"] for c in violated_constraints],
            improvement_pct=round(improvement, 6),
            constraints=resolved_constraints,
        )
    # M2: paired decision - with paired statistics the CI replaces the
    # point estimates entirely; safety vetoes above still outrank it.
    if stats is not None:
        return _paired_decision(
            stats,
            direction=direction,
            min_cases=min_cases,
            min_improvement_pp=min_improvement_pp,
            min_improvement_pct=min_improvement_pct,
            improvement=improvement,
            constraints=resolved_constraints,
        )
    if min_improvement_pp is not None:
        return VerdictResult(
            VERDICT_INCONCLUSIVE,
            [
                "objective.min_improvement_pp requires paired CI evidence "
                "(sr-bench compare comparisons[]); this comparison carries "
                "no paired statistics"
            ],
            improvement_pct=round(improvement, 6),
            constraints=resolved_constraints,
        )
    if improvement < min_improvement_pct:
        return VerdictResult(
            VERDICT_FAIL,
            [f"improvement {improvement:.2f}% below min {min_improvement_pct:.2f}%"],
            improvement_pct=round(improvement, 6),
            constraints=resolved_constraints,
        )
    return VerdictResult(
        VERDICT_PASS,
        [f"improvement {improvement:.2f}% >= min {min_improvement_pct:.2f}%"],
        improvement_pct=round(improvement, 6),
        constraints=resolved_constraints,
    )


def _paired_decision(
    stats: PairedStats,
    *,
    direction: str,
    min_cases: int,
    min_improvement_pp: float | None,
    min_improvement_pct: float,
    improvement: float,
    constraints: list[dict[str, Any]],
) -> VerdictResult:
    """M2: decide from the paired CI instead of the point estimates.

    Guard and constraint vetoes are decided before this runs; what remains
    is whether the paired interval clears the absolute bar.
    """

    def undecided(reasons: list[str]) -> VerdictResult:
        return VerdictResult(
            VERDICT_INCONCLUSIVE,
            reasons,
            improvement_pct=round(improvement, 6),
            constraints=constraints,
        )

    if stats.paired_cases < min_cases:
        return undecided(
            [
                f"insufficient paired sample: {stats.paired_cases} paired cases "
                f"< min_cases {min_cases}"
            ]
        )
    if stats.quality_delta_ci95 is None:
        return undecided(
            ["paired comparison carries no confidence interval; cannot decide"]
        )
    if direction != "maximize":
        return undecided(
            [
                "paired CI evidence is defined for maximize objectives only; "
                f"objective.direction is {direction!r}"
            ]
        )
    if min_improvement_pp is None:
        return undecided(
            [
                "paired CI evidence requires objective.min_improvement_pp "
                "(absolute percentage points); min_improvement_pct "
                f"{min_improvement_pct:g} is relative and cannot be applied to "
                "an absolute interval"
            ]
        )
    low, high = stats.quality_delta_ci95
    threshold = min_improvement_pp / 100.0
    if low >= threshold:
        return VerdictResult(
            VERDICT_PASS,
            [
                f"paired CI lower bound {low * 100:.2f}pp >= min "
                f"{min_improvement_pp:.2f}pp ({stats.paired_cases} paired cases, "
                f"delta {stats.quality_delta * 100:.2f}pp)"
            ],
            improvement_pct=round(improvement, 6),
            constraints=constraints,
        )
    if high < threshold:
        return VerdictResult(
            VERDICT_FAIL,
            [
                f"paired CI upper bound {high * 100:.2f}pp below min "
                f"{min_improvement_pp:.2f}pp ({stats.paired_cases} paired cases, "
                f"delta {stats.quality_delta * 100:.2f}pp)"
            ],
            improvement_pct=round(improvement, 6),
            constraints=constraints,
        )
    return undecided(
        [
            f"paired CI [{low * 100:.2f}pp, {high * 100:.2f}pp] spans the threshold "
            f"{min_improvement_pp:.2f}pp; not enough evidence to decide"
        ]
    )
