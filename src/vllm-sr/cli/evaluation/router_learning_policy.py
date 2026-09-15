"""Production routing_sampling arithmetic; paired with the Go runtime contract tests."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass
class ArmState:
    quality_seed: float = 0.5
    seed_weight: float = 2.0
    good_fit: int = 0
    underpowered: int = 0
    overprovisioned: int = 0
    failed: int = 0
    latency_ewma: float = 0.0
    cache_hit_ewma: float = 0.0
    cache_write_ewma: float = 0.0
    input_cost_multiplier_ewma: float = 0.0
    updated: bool = False
    successes: int = 0
    failures: int = 0

    @property
    def observations(self) -> int:
        return self.good_fit + self.underpowered + self.overprovisioned + self.failed


@dataclass(frozen=True)
class CandidateScore:
    model: str
    score: float
    posterior_mean: float
    predicted_quality: float
    cost_penalty: float
    overuse_penalty: float
    reliability_penalty: float
    latency_adjustment: float
    cache_adjustment: float
    cold_start: bool


def clamp01(value: float) -> float:
    return min(1.0, max(0.0, value)) if math.isfinite(value) else 0.0


def cost_penalty(cost: float, max_cost: float, candidate_set: str) -> float:
    if max_cost <= 0:
        return 0.0
    multiplier = {"tier": 0.06, "global": 0.10}.get(candidate_set, 0.04)
    return multiplier * clamp01(cost / max_cost)


def score_candidate(
    model: str,
    state: ArmState,
    catalog_cost_penalty: float,
    is_base: bool,
    sample: Callable[[float, float], float] | None,
) -> CandidateScore:
    seed, weight = state.quality_seed, state.seed_weight
    alpha = weight * seed + state.good_fit + 1.0
    beta = weight * (1.0 - seed) + state.underpowered + 1.0
    mean = alpha / (alpha + beta)
    predicted = mean if sample is None else sample(alpha, beta)
    cost = catalog_cost_penalty + 0.03 * clamp01(state.input_cost_multiplier_ewma)
    total = state.observations + 1
    overuse = 0.03 * state.overprovisioned / total
    reliability = 0.10 * state.failed / total
    latency = -0.02 * clamp01(state.latency_ewma)
    cache = 0.02 * clamp01(state.cache_hit_ewma)
    score = predicted - cost - overuse - reliability + latency + cache
    if is_base:
        score += 0.001
    return CandidateScore(
        model,
        score,
        mean,
        predicted,
        cost,
        overuse,
        reliability,
        latency,
        cache,
        not state.updated,
    )


def select_winner(
    candidates: Iterable[CandidateScore],
    base_model: str,
    candidate_set: str,
    use_sampling: bool,
) -> CandidateScore:
    scores = sorted(candidates, key=lambda row: (-row.score, row.model))
    winner = scores[0]
    # Production scoreByModel uses the best candidate if the base is absent.
    base = next((row for row in scores if row.model == base_model), winner)
    margin = {"tier": 0.03, "global": 0.08}.get(candidate_set, 0.0)
    extra_cost = (
        0.0
        if not base_model or winner.model == base_model
        else max(0.0, winner.cost_penalty - base.cost_penalty)
    )
    if winner.model != base_model and winner.score < base.score + margin + extra_cost:
        winner = base
    if use_sampling:
        winner = next((row for row in scores if row.cold_start), winner)
    return winner


def update_ewma(previous: float, observed: float) -> float:
    if observed < 0:
        return previous
    if previous <= 0:
        return observed
    return previous * 0.8 + observed * 0.2


def apply_outcome(state: ArmState, verdict: str, weight: float) -> None:
    count = 1 if weight < 1 else int(weight)
    setattr(state, verdict, getattr(state, verdict) + count)
    state.updated = True


def apply_telemetry(
    state: ArmState,
    *,
    latency_seconds: float | None = None,
    cache_hit_ratio: float | None = None,
    cache_write_pressure: float = 0.0,
    input_cost_multiplier: float | None = None,
    provider_failed: bool = False,
) -> None:
    if latency_seconds is not None:
        state.latency_ewma = update_ewma(state.latency_ewma, latency_seconds)
    if cache_hit_ratio is not None:
        state.cache_hit_ewma = update_ewma(state.cache_hit_ewma, cache_hit_ratio)
        state.cache_write_ewma = update_ewma(
            state.cache_write_ewma, cache_write_pressure
        )
    if input_cost_multiplier is not None:
        state.input_cost_multiplier_ewma = update_ewma(
            state.input_cost_multiplier_ewma, input_cost_multiplier
        )
    if provider_failed:
        state.failed += 1
    state.updated = True
