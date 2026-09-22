#!/usr/bin/env python3
"""Versioned selector objective: one scoring rule for labels and for models.

Scoring is currently written four times. Label generation weighs a candidate
against the slowest one in its own query group, while KNN, KMeans, SVM and MLP
each use an absolute scale, and three of them ignore the configured weight and
hard-code 0.9/0.1. A candidate's score therefore depends on which call site
asked, and a group-relative rule cannot be evaluated at inference at all, where
one candidate arrives with no group to compare against.

This module defines the objective once, absolutely, with cost and failure as
first-class terms, and stamps it with an id so an artifact records the rule it
was trained under.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass

OBJECTIVE_VERSION = 1

# Latency and cost enter as saturating ratios, so a candidate an order of
# magnitude past the scale still scores worse than one just past it.
DEFAULT_LATENCY_SCALE_MS = 10_000.0
DEFAULT_COST_SCALE = 1.0

_DIGEST_CHARS = 16


@dataclass(frozen=True)
class SelectorObjective:
    """What "better" means, as configuration rather than four literals."""

    quality_weight: float = 0.9
    latency_weight: float = 0.1
    cost_weight: float = 0.0
    # Subtracted outright from a failed candidate, so failure is never a
    # quality trade rather than a disqualification.
    failure_penalty: float = 1.0
    latency_scale_ms: float = DEFAULT_LATENCY_SCALE_MS
    cost_scale: float = DEFAULT_COST_SCALE
    version: int = OBJECTIVE_VERSION

    def __post_init__(self) -> None:
        for name in (
            "quality_weight",
            "latency_weight",
            "cost_weight",
            "failure_penalty",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.latency_scale_ms <= 0 or self.cost_scale <= 0:
            raise ValueError("latency_scale_ms and cost_scale must be positive")
        if self.quality_weight + self.latency_weight + self.cost_weight <= 0:
            raise ValueError(
                "at least one of quality, latency or cost must carry weight"
            )

    @property
    def objective_id(self) -> str:
        """Identity of this rule, so a trained artifact pins the objective it used."""
        parts = (
            f"v{self.version}",
            f"q{self.quality_weight!r}",
            f"l{self.latency_weight!r}",
            f"c{self.cost_weight!r}",
            f"f{self.failure_penalty!r}",
            f"ls{self.latency_scale_ms!r}",
            f"cs{self.cost_scale!r}",
        )
        digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=16)
        return digest.hexdigest()[:_DIGEST_CHARS]

    def score(self, outcome) -> float:
        """Score one CandidateOutcome on its own, with no reference to its peers."""
        speed = 1.0 / (
            1.0 + max(0.0, float(outcome.latency_ms)) / self.latency_scale_ms
        )
        thrift = 1.0 / (1.0 + max(0.0, float(outcome.cost)) / self.cost_scale)
        total = (
            self.quality_weight * float(outcome.quality)
            + self.latency_weight * speed
            + self.cost_weight * thrift
        )
        if not outcome.success:
            total -= self.failure_penalty
        return total

    def rank(self, snapshot) -> list[tuple[str, float]]:
        """Candidates best first, ties broken by model_ref so the order is reproducible."""
        scored = [(o.model_ref, self.score(o)) for o in snapshot.outcomes]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))

    def best(self, snapshot) -> tuple[str, float]:
        """The winning candidate for one query."""
        ranked = self.rank(snapshot)
        if not ranked:
            raise ValueError("snapshot has no candidate outcomes")
        return ranked[0]


def label_snapshots(
    snapshots: Sequence, objective: SelectorObjective | None = None
) -> dict[str, tuple[str, float]]:
    """Best candidate per query, keyed by query_id rather than raw query text."""
    rule = objective or SelectorObjective()
    return {snapshot.query_id: rule.best(snapshot) for snapshot in snapshots}
