# ruff: noqa: PLR2004
# ruff: noqa: PLR2004
"""MAB algorithms for the comparison harness.

All algorithms implement a uniform protocol:

    reset(n_arms, context_dim) -> None       # called before each experiment
    select(t, context) -> int                # returns arm index in [0, n_arms)
    update(arm, reward, context) -> None     # observe reward for chosen arm

`context` is None for non-contextual workloads, or a list of floats whose
length equals the workload's `context_dim`.

NOTE on RoutingSamplingPy: this is a Python port of the score formula in
src/semantic-router/pkg/extproc/router_learning_adaptation.go:313-373. It
matches the *deterministic* parts (alpha/beta updates, penalty/bonus
arithmetic) bit-for-bit; it does NOT byte-match the Go RNG output (Python's
random.Random is not Go's math/rand). The score formula is what the
comparison harness exercises.

NOTE on LinUCBPy / LinearThompsonPy: same relationship — Python ports of
the Go strategies in router_learning_linucb.go / router_learning_linear_thompson.go.
Same score formula and same matrix backend algorithm (ridge-regularised
posterior with Sherman-Morrison inverse and Cholesky sampling); different
RNG implementation. Both are non-contextual-aware when the workload does
not carry a context — they degrade to the ridge posterior mean of a
single, dimensionless feature (the constant 1) which is uninteresting;
this is why we expect KEEP_LINUCB_CONTEXTUAL_ONLY, not KEEP_LINUCB.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol


class MABAlgorithm(Protocol):
    name: str

    def reset(self, n_arms: int, context_dim: int | None) -> None: ...

    def select(self, t: int, context: list[float] | None) -> int: ...

    def update(self, arm: int, reward: float, context: list[float] | None) -> None: ...


# ---- baseline algorithms ---------------------------------------------------


@dataclass
class UCB1:
    """UCB1 (Auer, Cesa-Bianchi & Fischer 2002).

    score(arm) = mean(arm) + c * sqrt(2 * ln(t) / n_arm)

    `c` defaults to 1.0 (the standard tuning); larger c -> more exploration.
    """

    c: float = 1.0
    name: str = "ucb1"
    _counts: list[int] = field(default_factory=list)
    _means: list[float] = field(default_factory=list)
    _t: int = 0

    def reset(self, n_arms: int, context_dim: int | None) -> None:
        self._counts = [0] * n_arms
        self._means = [0.0] * n_arms
        self._t = 0

    def select(self, t: int, context: list[float] | None) -> int:
        # Pull each arm once before applying UCB formula.
        for arm, n in enumerate(self._counts):
            if n == 0:
                return arm
        log_t = math.log(self._t) if self._t > 0 else 0.0
        scores = [
            self._means[a] + self.c * math.sqrt(2.0 * log_t / self._counts[a])
            for a in range(len(self._counts))
        ]
        return max(range(len(scores)), key=lambda a: scores[a])

    def update(self, arm: int, reward: float, context: list[float] | None) -> None:
        self._counts[arm] += 1
        self._t += 1
        # Incremental mean update.
        n = self._counts[arm]
        self._means[arm] += (reward - self._means[arm]) / n


@dataclass
class EpsilonGreedy:
    """Epsilon-greedy with exponential decay.

    At round t, epsilon(t) = max(eps_min, eps_0 * decay^(t/100)).
    """

    eps_0: float = 0.3
    decay: float = 0.99
    eps_min: float = 0.05
    seed: int = 0
    name: str = "epsilon_greedy"
    _counts: list[int] = field(default_factory=list)
    _means: list[float] = field(default_factory=list)
    _rng: random.Random | None = None

    def reset(self, n_arms: int, context_dim: int | None) -> None:
        self._counts = [0] * n_arms
        self._means = [0.0] * n_arms
        self._rng = random.Random(self.seed)

    def select(self, t: int, context: list[float] | None) -> int:
        eps = max(self.eps_min, self.eps_0 * (self.decay ** (t / 100.0)))
        rng = self._rng
        assert rng is not None
        if rng.random() < eps:
            return rng.randrange(len(self._counts))
        return max(range(len(self._counts)), key=lambda a: self._means[a])

    def update(self, arm: int, reward: float, context: list[float] | None) -> None:
        self._counts[arm] += 1
        n = self._counts[arm]
        self._means[arm] += (reward - self._means[arm]) / n


# ---- production baseline ---------------------------------------------------


@dataclass
class RoutingSamplingPy:
    """Python port of the production `routing_sampling` strategy.

    Mirrors src/semantic-router/pkg/extproc/router_learning_adaptation.go
    `scoreRoutingSamplingCandidates` (lines 313-373). The score formula:

        alpha = SeedWeight * QualitySeed + GoodFitCount + 1
        beta  = SeedWeight * (1-QualitySeed) + UnderpoweredCount + 1
        mean  = alpha / (alpha + beta)
        predicted = sample_beta(alpha, beta) if useSampling else mean

        score = predicted - costPenalty - overusePenalty - reliabilityPenalty
              + latencyAdjustment + cacheAdjustment

    For the synthetic harness:
      - QualitySeed is provided per-arm via `quality_seeds` (defaults to 0.5)
      - SeedWeight defaults to 0 (no prior) unless `seed_weight` is set
      - cost_per_arm / latency_per_arm / cache_hit_per_arm default to 0
      - All EWMA inputs decay according to the same factor used in Go
        (recordModelExperienceLocked uses ewma = 0.7*old + 0.3*new)

    This class does NOT replicate the Go random number generator; it samples
    from Beta using Python's `random.Random` for parity-of-distribution, not
    parity-of-bytes. Determinism is per-Python-seed.
    """

    use_sampling: bool = True
    seed_weight: float = 0.0
    quality_seeds: list[float] | None = None  # length = n_arms; None -> 0.5
    cost_per_arm: list[float] | None = None
    seed: int = 0
    name: str = "routing_sampling_py"
    _good: list[int] = field(default_factory=list)
    _under: list[int] = field(default_factory=list)
    _over: list[int] = field(default_factory=list)
    _failed: list[int] = field(default_factory=list)
    _latency_ewma: list[float] = field(default_factory=list)
    _cache_ewma: list[float] = field(default_factory=list)
    _input_cost_ewma: list[float] = field(default_factory=list)
    _quality: list[float] = field(default_factory=list)
    _rng: random.Random | None = None

    def reset(self, n_arms: int, context_dim: int | None) -> None:
        self._good = [0] * n_arms
        self._under = [0] * n_arms
        self._over = [0] * n_arms
        self._failed = [0] * n_arms
        self._latency_ewma = [0.0] * n_arms
        self._cache_ewma = [0.0] * n_arms
        self._input_cost_ewma = [0.0] * n_arms
        if self.quality_seeds is not None and len(self.quality_seeds) == n_arms:
            self._quality = [_clamp01(q) for q in self.quality_seeds]
        else:
            self._quality = [0.5] * n_arms
        self._rng = random.Random(self.seed)

    def select(self, t: int, context: list[float] | None) -> int:
        n = len(self._good)
        scores = [self._score(a) for a in range(n)]
        # Stable sort: highest score first; tie -> lowest arm index (matches Go's stable sort by model name).
        return max(range(n), key=lambda a: (scores[a], -a))

    def _score(self, arm: int) -> float:
        alpha = self.seed_weight * self._quality[arm] + self._good[arm] + 1
        beta = self.seed_weight * (1 - self._quality[arm]) + self._under[arm] + 1
        mean = alpha / (alpha + beta)
        if self.use_sampling:
            assert self._rng is not None
            predicted = _sample_beta(alpha, beta, self._rng)
        else:
            predicted = mean

        max_cost = max(self.cost_per_arm or [0.0]) if self.cost_per_arm else 0.0
        cost_penalty = self._cost_penalty(arm, max_cost) + 0.03 * _clamp01(
            self._input_cost_ewma[arm]
        )
        total = (
            self._good[arm] + self._under[arm] + self._over[arm] + self._failed[arm] + 1
        )
        overuse_penalty = 0.03 * self._over[arm] / total
        reliability_penalty = 0.10 * self._failed[arm] / total
        latency_adjustment = -0.02 * _clamp01(self._latency_ewma[arm])
        cache_adjustment = 0.02 * _clamp01(self._cache_ewma[arm])
        return (
            predicted
            - cost_penalty
            - overuse_penalty
            - reliability_penalty
            + latency_adjustment
            + cache_adjustment
        )

    def _cost_penalty(self, arm: int, max_cost: float) -> float:
        if max_cost <= 0 or not self.cost_per_arm:
            return 0.0
        # Mirrors costPenalty in router_learning_adaptation.go: 0.04 multiplier on
        # normalized cost-over-max within the candidate set.
        return 0.04 * (self.cost_per_arm[arm] / max_cost)

    def update(self, arm: int, reward: float, context: list[float] | None) -> None:
        # Map binary reward to good_fit / underpowered counts (as the Go side does
        # via outcome verdict ingestion at /v1/router/outcomes).
        if reward >= 0.5:
            self._good[arm] += 1
        else:
            self._under[arm] += 1
        # EWMA updates use the same decay (0.7 old / 0.3 new) as the Go runtime.
        # Synthetic harness has no real latency / cache signals, so leave EWMA at 0.


# ---- helpers ---------------------------------------------------------------


def _clamp01(value: float) -> float:
    if value < 0:
        return 0.0
    if value > 1:
        return 1.0
    return value


def _sample_beta(alpha: float, beta: float, rng: random.Random) -> float:
    """Beta(alpha, beta) sampler via the gamma-ratio identity.

    X = G_a / (G_a + G_b), G_a ~ Gamma(alpha, 1), G_b ~ Gamma(beta, 1).

    Uses Marsaglia-Tsang for shape >= 1; transformation Gamma(s) = Gamma(s+1) * U^(1/s)
    for shape < 1. Same algorithm family as Go's sampleGamma in
    router_learning_adaptation.go:599-618; not byte-equivalent (different RNGs)
    but distributionally equivalent.
    """
    if alpha <= 0 or beta <= 0:
        return alpha / (alpha + beta) if (alpha + beta) > 0 else 0.5
    x = _sample_gamma(alpha, rng)
    y = _sample_gamma(beta, rng)
    if x <= 0 or y <= 0:
        return alpha / (alpha + beta)
    return _clamp01(x / (x + y))


def _sample_gamma(shape: float, rng: random.Random) -> float:
    if shape < 1:
        u = rng.random()
        return _sample_gamma(shape + 1, rng) * (u ** (1.0 / shape))
    d = shape - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)
    while True:
        x = rng.gauss(0.0, 1.0)
        v = 1.0 + c * x
        if v <= 0:
            continue
        v = v * v * v
        u = rng.random()
        if u < 1 - 0.0331 * x * x * x * x:
            return d * v
        if math.log(u) < 0.5 * x * x + d * (1 - v + math.log(v)):
            return d * v


# ---- contextual bandit backend --------------------------------------------


class _LearningMatrix:
    """Ridge-regularised (A, b) state per arm with Sherman-Morrison inverse.

    Python analogue of learningMatrix in
    src/semantic-router/pkg/extproc/router_learning_matrix.go. Same update
    rule and Cholesky sampling; different RNG.
    """

    def __init__(self, dim: int, lam: float = 1.0):
        self.dim = dim
        self.a = [[lam if i == j else 0.0 for j in range(dim)] for i in range(dim)]
        self.inv = [
            [1.0 / lam if i == j else 0.0 for j in range(dim)] for i in range(dim)
        ]
        self.b = [0.0] * dim

    def theta(self) -> list[float]:
        out = [0.0] * self.dim
        for i in range(self.dim):
            row = self.inv[i]
            s = 0.0
            for j in range(self.dim):
                s += row[j] * self.b[j]
            out[i] = s
        return out

    def dot_theta(self, x: list[float]) -> float:
        theta = self.theta()
        return sum(v * theta[i] for i, v in enumerate(x))

    def quad_inv(self, x: list[float]) -> float:
        y = [0.0] * self.dim
        for i in range(self.dim):
            row = self.inv[i]
            s = 0.0
            for j in range(self.dim):
                s += row[j] * x[j]
            y[i] = s
        q = sum(v * y[i] for i, v in enumerate(x))
        return q if q > 0 else 0.0

    def update(self, x: list[float], reward: float) -> None:
        # y = A^{-1} x  (used by both A^{-1} update and denom)
        y = [0.0] * self.dim
        for i in range(self.dim):
            row = self.inv[i]
            s = 0.0
            for j in range(self.dim):
                s += row[j] * x[j]
            y[i] = s
        denom = 1.0 + sum(v * y[i] for i, v in enumerate(x))

        # A += x x^T
        for i in range(self.dim):
            row = self.a[i]
            xi = x[i]
            for j in range(self.dim):
                row[j] += xi * x[j]

        # b += r · x
        for i, v in enumerate(x):
            self.b[i] += reward * v

        # A^{-1} <- A^{-1} - (A^{-1} x x^T A^{-1}) / (1 + x^T A^{-1} x)
        if abs(denom) < 1e-12:
            return  # skip; recompute via Cholesky if this ever mattered in practice
        for i in range(self.dim):
            row = self.inv[i]
            yi = y[i]
            for j in range(self.dim):
                row[j] -= (yi * y[j]) / denom

    def sample_theta(self, sigma: float, rng: random.Random) -> list[float]:
        mean = self.theta()
        if sigma <= 0:
            return mean
        chol = _cholesky(self.inv, self.dim)
        if chol is None:
            return mean
        z = [rng.gauss(0.0, 1.0) for _ in range(self.dim)]
        out = list(mean)
        for i in range(self.dim):
            for j in range(i + 1):
                out[i] += sigma * chol[i][j] * z[j]
        return out


def _cholesky(m: list[list[float]], dim: int) -> list[list[float]] | None:
    """Lower-triangular Cholesky factor. Returns None if not positive-definite."""
    chol_l = [[0.0] * dim for _ in range(dim)]
    for i in range(dim):
        for j in range(i + 1):
            s = m[i][j]
            for k in range(j):
                s -= chol_l[i][k] * chol_l[j][k]
            if i == j:
                if s <= 0:
                    return None
                chol_l[i][i] = math.sqrt(s)
            else:
                chol_l[i][j] = s / chol_l[j][j]
    return chol_l


# ---- contextual bandit algorithms ------------------------------------------


@dataclass
class LinUCBPy:
    """Python port of the Go linUCB strategy in
    src/semantic-router/pkg/extproc/router_learning_linucb.go.

    score(m) = theta_m · x + alpha · sqrt(x^T A_m^{-1} x)

    When the workload provides no context, x is padded to `dim` zero-vector
    which collapses the algorithm to picking whichever arm has the ridge
    prior with highest theta. This is intentional: it mirrors the Go
    behaviour when the extractor produces empty features, and is the
    reason we expect KEEP_LINUCB_CONTEXTUAL_ONLY, not KEEP_LINUCB.
    """

    dim: int = 16
    alpha: float = 1.0
    lam: float = 1.0
    seed: int = 0
    name: str = "linucb_py"
    _arms: list[_LearningMatrix] = field(default_factory=list)

    def reset(self, n_arms: int, context_dim: int | None) -> None:
        # Feature width = provided context dim if the workload carries one, else self.dim.
        # Both Go and Python paths make this same choice: the extractor is authoritative.
        effective_dim = context_dim if context_dim is not None else self.dim
        self.dim = effective_dim
        self._arms = [_LearningMatrix(effective_dim, self.lam) for _ in range(n_arms)]

    def _feature(self, context: list[float] | None) -> list[float]:
        if context is None:
            # No workload context: use a bias-only "feature" (constant 1
            # in dim 0, zeros elsewhere). This makes every arm's theta
            # collapse toward its empirical mean — same as UCB1 without
            # the exploration term when alpha=0.
            x = [0.0] * self.dim
            if self.dim > 0:
                x[0] = 1.0
            return x
        if len(context) == self.dim:
            return list(context)
        # Length mismatch: pad or truncate.
        x = [0.0] * self.dim
        for i in range(min(len(context), self.dim)):
            x[i] = context[i]
        return x

    def select(self, t: int, context: list[float] | None) -> int:
        x = self._feature(context)
        best_arm = 0
        best_score = float("-inf")
        for a, arm in enumerate(self._arms):
            mean = arm.dot_theta(x)
            bonus = self.alpha * math.sqrt(arm.quad_inv(x)) if self.alpha > 0 else 0.0
            score = mean + bonus
            if score > best_score:
                best_score = score
                best_arm = a
        return best_arm

    def update(self, arm: int, reward: float, context: list[float] | None) -> None:
        x = self._feature(context)
        self._arms[arm].update(x, reward)


@dataclass
class LinearThompsonPy:
    """Python port of the Go linear_thompson strategy in
    router_learning_linear_thompson.go.

    theta_tilde ~ N(theta, sigma^2 · A^{-1})
    score(m)    = theta_tilde · x
    """

    dim: int = 16
    sigma: float = 0.3
    lam: float = 1.0
    seed: int = 0
    name: str = "linear_thompson_py"
    _arms: list[_LearningMatrix] = field(default_factory=list)
    _rng: random.Random | None = None

    def reset(self, n_arms: int, context_dim: int | None) -> None:
        effective_dim = context_dim if context_dim is not None else self.dim
        self.dim = effective_dim
        self._arms = [_LearningMatrix(effective_dim, self.lam) for _ in range(n_arms)]
        self._rng = random.Random(self.seed)

    def _feature(self, context: list[float] | None) -> list[float]:
        # Same policy as LinUCBPy — see comment there.
        if context is None:
            x = [0.0] * self.dim
            if self.dim > 0:
                x[0] = 1.0
            return x
        if len(context) == self.dim:
            return list(context)
        x = [0.0] * self.dim
        for i in range(min(len(context), self.dim)):
            x[i] = context[i]
        return x

    def select(self, t: int, context: list[float] | None) -> int:
        assert self._rng is not None
        x = self._feature(context)
        best_arm = 0
        best_score = float("-inf")
        for a, arm in enumerate(self._arms):
            theta_tilde = arm.sample_theta(self.sigma, self._rng)
            score = sum(v * theta_tilde[i] for i, v in enumerate(x))
            if score > best_score:
                best_score = score
                best_arm = a
        return best_arm

    def update(self, arm: int, reward: float, context: list[float] | None) -> None:
        x = self._feature(context)
        self._arms[arm].update(x, reward)


# ---- registry --------------------------------------------------------------


def build(name: str, seed: int = 0, **kwargs: object) -> MABAlgorithm:
    """Build an algorithm instance by name.

    Forwards `seed` only to algorithms that declare it as a field; deterministic
    algorithms (e.g. UCB1) are constructed without a seed.
    """
    if name not in _BUILDERS:
        raise ValueError(f"unknown algorithm: {name!r} (known: {sorted(_BUILDERS)})")
    cls = _BUILDERS[name]
    accepted = set(getattr(cls, "__dataclass_fields__", {}).keys())
    forwarded: dict[str, object] = {k: v for k, v in kwargs.items() if k in accepted}
    if "seed" in accepted:
        forwarded["seed"] = seed
    return cls(**forwarded)  # type: ignore[arg-type]


def names() -> list[str]:
    return sorted(_BUILDERS.keys())


_BUILDERS: dict[str, type] = {
    "ucb1": UCB1,
    "epsilon_greedy": EpsilonGreedy,
    "routing_sampling_py": RoutingSamplingPy,
    "linucb_py": LinUCBPy,
    "linear_thompson_py": LinearThompsonPy,
}
