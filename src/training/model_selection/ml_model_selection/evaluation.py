#!/usr/bin/env python3
"""Held-out evaluation of a selector against fixed baselines.

A selector is scored on test-split queries against the strongest, cheapest,
global-best, random and (optionally) current-router baselines, and against the
oracle that always picks the best recorded candidate. Baselines learn only from
the train split, so a selector is never compared against a policy that has seen
the test queries.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from objective import SelectorObjective
from query_outcome_set import QueryOutcomeSet, _digest

# A selector sees the query text and the models it may pick from, never the outcomes.
# It returns one of those models, or None to abstain.
Selector = Callable[[str, tuple[str, ...]], str | None]

ORACLE = "oracle"


def _category(snapshot: QueryOutcomeSet) -> str:
    return snapshot.category


@dataclass(frozen=True)
class Metrics:
    """Averages over the queries a policy answered; abstentions only lower coverage."""

    queries: int
    answered: int
    coverage: float
    success_rate: float
    mean_quality: float
    mean_latency_ms: float
    mean_cost: float
    mean_utility: float
    mean_regret: float


@dataclass(frozen=True)
class PolicyReport:
    name: str
    overall: Metrics
    slices: Mapping[str, Metrics]
    route_share: Mapping[str, float]


@dataclass
class _Totals:
    queries: int = 0
    answered: int = 0
    successes: int = 0
    quality: float = 0.0
    latency_ms: float = 0.0
    cost: float = 0.0
    utility: float = 0.0
    regret: float = 0.0

    def finish(self) -> Metrics:
        n = self.answered
        return Metrics(
            queries=self.queries,
            answered=n,
            coverage=n / self.queries if self.queries else 0.0,
            success_rate=self.successes / n if n else 0.0,
            mean_quality=self.quality / n if n else 0.0,
            mean_latency_ms=self.latency_ms / n if n else 0.0,
            mean_cost=self.cost / n if n else 0.0,
            mean_utility=self.utility / n if n else 0.0,
            mean_regret=self.regret / n if n else 0.0,
        )


def _by_ref(snapshot: QueryOutcomeSet) -> dict[str, object]:
    return {o.model_ref: o for o in snapshot.outcomes}


def evaluate_policy(
    name: str,
    select: Selector | None,
    snapshots: Sequence[QueryOutcomeSet],
    objective: SelectorObjective,
    *,
    slice_key: Callable[[QueryOutcomeSet], str] = _category,
) -> PolicyReport:
    """Score `select` on `snapshots`; `None` scores the oracle.

    Regret is the oracle's score minus the chosen candidate's, both under
    `objective.score`, so a failed pick costs its full utility. O(queries x candidates).
    """
    if not snapshots:
        raise ValueError("no queries to evaluate")

    overall = _Totals()
    slices: defaultdict[str, _Totals] = defaultdict(_Totals)
    routed: defaultdict[str, int] = defaultdict(int)

    for snapshot in snapshots:
        best_ref, best_score = objective.best(snapshot)
        outcomes = _by_ref(snapshot)
        if select is None:
            chosen_ref = best_ref
        else:
            chosen_ref = select(snapshot.query, snapshot.model_refs)
        group = slices[slice_key(snapshot)]
        overall.queries += 1
        group.queries += 1
        if chosen_ref is None:
            continue
        if chosen_ref not in outcomes:
            raise ValueError(
                f"{name}: picked {chosen_ref!r}, not a candidate of query {snapshot.query_id}"
            )
        picked = outcomes[chosen_ref]
        score = objective.score(picked)
        routed[chosen_ref] += 1
        for totals in (overall, group):
            totals.answered += 1
            totals.successes += 1 if picked.success else 0
            totals.quality += float(picked.quality)
            totals.latency_ms += float(picked.latency_ms)
            totals.cost += float(picked.cost)
            totals.utility += score
            totals.regret += best_score - score

    answered = overall.answered
    share = {ref: count / answered for ref, count in sorted(routed.items())}
    return PolicyReport(
        name=name,
        overall=overall.finish(),
        slices={key: totals.finish() for key, totals in sorted(slices.items())},
        route_share=share,
    )


def _mean_by_model(
    train: Sequence[QueryOutcomeSet], value: Callable[[object], float]
) -> dict[str, float]:
    sums: defaultdict[str, float] = defaultdict(float)
    counts: defaultdict[str, int] = defaultdict(int)
    for snapshot in train:
        for outcome in snapshot.outcomes:
            sums[outcome.model_ref] += value(outcome)
            counts[outcome.model_ref] += 1
    return {ref: sums[ref] / counts[ref] for ref in sums}


def _pick_by_stat(stats: Mapping[str, float], *, higher_is_better: bool) -> Selector:
    """Best eligible model by a train-split statistic; a model unseen in train ranks last."""
    sign = -1.0 if higher_is_better else 1.0

    def select(query: str, eligible: tuple[str, ...]) -> str | None:
        if not eligible:
            return None
        return min(
            eligible,
            key=lambda ref: (ref not in stats, sign * stats.get(ref, 0.0), ref),
        )

    return select


def _random_pick(seed: int) -> Selector:
    """Uniform over the eligible models, fixed by the query and seed, not by call order."""

    def select(query: str, eligible: tuple[str, ...]) -> str | None:
        if not eligible:
            return None
        ordered = sorted(eligible)
        return ordered[int(_digest(query, str(seed)), 16) % len(ordered)]

    return select


def baseline_selectors(
    train: Sequence[QueryOutcomeSet],
    objective: SelectorObjective,
    *,
    seed: int = 0,
    current_router: Selector | None = None,
) -> dict[str, Selector]:
    """The comparison policies, fitted on `train` only. O(train queries x candidates)."""
    if not train:
        raise ValueError("baselines need a non-empty train split")
    selectors: dict[str, Selector] = {
        "strongest": _pick_by_stat(
            _mean_by_model(train, lambda o: float(o.quality)), higher_is_better=True
        ),
        "cheapest": _pick_by_stat(
            _mean_by_model(train, lambda o: float(o.cost)), higher_is_better=False
        ),
        "global_best": _pick_by_stat(
            _mean_by_model(train, objective.score), higher_is_better=True
        ),
        "random": _random_pick(seed),
    }
    if current_router is not None:
        selectors["current_router"] = current_router
    return selectors


def evaluate_against_baselines(
    selector: Selector,
    train: Sequence[QueryOutcomeSet],
    test: Sequence[QueryOutcomeSet],
    objective: SelectorObjective,
    *,
    selector_name: str = "selector",
    seed: int = 0,
    current_router: Selector | None = None,
    slice_key: Callable[[QueryOutcomeSet], str] = _category,
) -> dict[str, PolicyReport]:
    """Report the selector, every baseline and the oracle on the same test queries."""
    baselines: Final = baseline_selectors(
        train, objective, seed=seed, current_router=current_router
    )
    if selector_name in baselines or selector_name == ORACLE:
        raise ValueError(f"selector_name {selector_name!r} is reserved for a baseline")
    policies: dict[str, Selector | None] = {selector_name: selector, **baselines}
    policies[ORACLE] = None
    return {
        name: evaluate_policy(name, select, test, objective, slice_key=slice_key)
        for name, select in policies.items()
    }
