# ruff: noqa: PLR2004
# ruff: noqa: PLR2004
"""Synthetic bandit workloads for MAB algorithm comparison.

Each workload generates a deterministic reward sequence given a seed. The
sequence is hashed (sha256) so consumers can verify byte-identical inputs
across algorithms — the same byte-identical principle PR #2269 enforces for
fusion arms via cached panels.

A workload exposes:
    n_arms          number of arms
    horizon         number of rounds
    context_dim     None for non-contextual; integer for contextual
    optimal_arm(t)  index of the optimal arm at round t (drift-aware)
    sample(t, arm)  reward draw for choosing `arm` at round t
    context(t)      context vector at round t, or None
    sha256()        hash of the full reward / context sequence under this seed

Workloads are pure functions of (seed, params); no global state. All sampling
uses a per-workload random.Random instance keyed by `seed`.
"""

from __future__ import annotations

import hashlib
import random
import struct
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkloadResult:
    """Pre-generated reward / context sequence for one workload + seed pair."""

    name: str
    seed: int
    n_arms: int
    horizon: int
    context_dim: int | None
    rewards: list[list[float]]  # rewards[t][arm] — full information for regret
    contexts: list[list[float]] | None  # contexts[t] — None for non-contextual
    optimal_arms: list[int]  # optimal_arms[t] — index of best arm at round t
    sha256: str

    def context_at(self, t: int) -> list[float] | None:
        if self.contexts is None:
            return None
        return self.contexts[t]

    def reward_at(self, t: int, arm: int) -> float:
        return self.rewards[t][arm]

    def optimal_at(self, t: int) -> int:
        return self.optimal_arms[t]


# ---- registry --------------------------------------------------------------


def generate(name: str, seed: int) -> WorkloadResult:
    """Generate a workload by name. Raises ValueError on unknown name."""
    if name not in _REGISTRY:
        raise ValueError(f"unknown workload: {name!r} (known: {sorted(_REGISTRY)})")
    return _REGISTRY[name](seed)


def names() -> list[str]:
    return sorted(_REGISTRY.keys())


# ---- workload definitions --------------------------------------------------


def _bernoulli_seq(
    name: str,
    seed: int,
    true_means_at: list[list[float]],
    horizon: int,
) -> WorkloadResult:
    """Bernoulli rewards from a sequence of true-mean vectors (one per round)."""
    rng = random.Random(seed)
    n_arms = len(true_means_at[0])
    rewards = [
        [1.0 if rng.random() < mu else 0.0 for mu in row] for row in true_means_at
    ]
    optimal_arms = [max(range(n_arms), key=lambda a: row[a]) for row in true_means_at]
    return WorkloadResult(
        name=name,
        seed=seed,
        n_arms=n_arms,
        horizon=horizon,
        context_dim=None,
        rewards=rewards,
        contexts=None,
        optimal_arms=optimal_arms,
        sha256=_hash_workload(name, seed, rewards, None),
    )


def _stationary_3arm(seed: int) -> WorkloadResult:
    horizon = 10000
    means = [[0.9, 0.5, 0.1]] * horizon
    return _bernoulli_seq("stationary_3arm", seed, means, horizon)


def _stationary_5arm_close(seed: int) -> WorkloadResult:
    horizon = 20000
    means = [[0.62, 0.60, 0.58, 0.56, 0.54]] * horizon
    return _bernoulli_seq("stationary_5arm_close", seed, means, horizon)


def _drift_at_t5000(seed: int) -> WorkloadResult:
    horizon = 10000
    before = [0.9, 0.5, 0.1]
    after = [0.1, 0.5, 0.9]
    means = [before if t < 5000 else after for t in range(horizon)]
    return _bernoulli_seq("drift_at_t5000", seed, means, horizon)


def _gradual_drift(seed: int) -> WorkloadResult:
    """Arm 0 decays from 0.9 to 0.3; arm 2 grows 0.1 to 0.7. Arm 1 fixed at 0.5."""
    horizon = 10000
    means: list[list[float]] = []
    for t in range(horizon):
        progress = t / (horizon - 1)
        means.append(
            [
                0.9 - 0.6 * progress,
                0.5,
                0.1 + 0.6 * progress,
            ]
        )
    return _bernoulli_seq("gradual_drift", seed, means, horizon)


def _contextual_2cluster(seed: int) -> WorkloadResult:
    """Two query clusters (one-hot context), each with a different best arm.

    Cluster A (context = [1, 0]): best arm = 0 (mean 0.85)
    Cluster B (context = [0, 1]): best arm = 2 (mean 0.85)
    Non-contextual algorithms see ~0.5 average and cannot beat random.
    """
    horizon = 10000
    n_arms = 3
    rng = random.Random(seed)
    contexts: list[list[float]] = []
    rewards: list[list[float]] = []
    optimal_arms: list[int] = []
    for _ in range(horizon):
        cluster = rng.randrange(2)
        context = [1.0, 0.0] if cluster == 0 else [0.0, 1.0]
        true_means = [0.85, 0.50, 0.15] if cluster == 0 else [0.15, 0.50, 0.85]
        rewards_row = [1.0 if rng.random() < mu else 0.0 for mu in true_means]
        optimal_arms.append(max(range(n_arms), key=lambda a, m=true_means: m[a]))
        contexts.append(context)
        rewards.append(rewards_row)
    return WorkloadResult(
        name="contextual_2cluster",
        seed=seed,
        n_arms=n_arms,
        horizon=horizon,
        context_dim=2,
        rewards=rewards,
        contexts=contexts,
        optimal_arms=optimal_arms,
        sha256=_hash_workload("contextual_2cluster", seed, rewards, contexts),
    )


def _cold_start_with_prior(seed: int) -> WorkloadResult:
    """Same as stationary_3arm but flagged for prior-injection in algorithms.

    The reward sequence is identical; only the metadata differs. Algorithms
    that support priors (e.g. routing_sampling reading category.ModelScores)
    should warm-start their state via the prior_hint exposed in the runner.
    """
    horizon = 10000
    means = [[0.9, 0.5, 0.1]] * horizon
    result = _bernoulli_seq("cold_start_with_prior", seed, means, horizon)
    return result


_REGISTRY = {
    "stationary_3arm": _stationary_3arm,
    "stationary_5arm_close": _stationary_5arm_close,
    "drift_at_t5000": _drift_at_t5000,
    "gradual_drift": _gradual_drift,
    "contextual_2cluster": _contextual_2cluster,
    "cold_start_with_prior": _cold_start_with_prior,
}


# ---- hashing ---------------------------------------------------------------


def _hash_workload(
    name: str,
    seed: int,
    rewards: Sequence[Sequence[float]],
    contexts: Sequence[Sequence[float]] | None,
) -> str:
    h = hashlib.sha256()
    h.update(name.encode("utf-8"))
    h.update(struct.pack(">q", seed))
    h.update(struct.pack(">q", len(rewards)))
    for row in rewards:
        for v in row:
            h.update(struct.pack(">d", v))
    if contexts is not None:
        h.update(b"|ctx|")
        for row in contexts:
            for v in row:
                h.update(struct.pack(">d", v))
    return h.hexdigest()
