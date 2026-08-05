# MAB Algorithm Comparison Harness

Synthetic, pure-Python benchmarking for multi-armed bandit algorithms used in
Router Learning. Complementary to `bench/agentic_routing_experiment.py`
(fixture-based engineering eval); this harness adds **academic** MAB metrics
that do not exist elsewhere in the codebase:

- cumulative regret
- optimal-arm selection rate
- non-stationary recovery time

## Why a separate harness?

`bench/agentic_routing_experiment.py --learning-architecture` answers
"does the router make the right routing call on these 7 fixtures?"
That is an **engineering** question.

This harness answers "given a synthetic bandit problem with known optimal
arms, does algorithm X explore-exploit better than algorithm Y, with paired
bootstrap CI evidence?" That is an **algorithm** question.

Both are needed; neither replaces the other. New algorithm proposals should
ship verdicts from both.

## Quick start

```bash
# Run all algorithms × workloads × 30 seeds (~5 minutes pure Python)
python3 -m bench.mab_comparison.runner \
  --seeds 30 \
  --out-dir .agent-harness/experiments/mab-comparison

# Build the verdict
python3 -m bench.mab_comparison.compare \
  --results-dir .agent-harness/experiments/mab-comparison \
  --json-out .agent-harness/experiments/mab-comparison/mab_verdict.json

# Optional: plot regret/arm-rate/recovery curves (requires matplotlib)
python3 -m bench.mab_comparison.plot \
  --results-dir .agent-harness/experiments/mab-comparison \
  --out-dir .agent-harness/experiments/mab-comparison/figures
```

Or via Make:

```bash
make bench-mab-comparison           # full run + verdict
make bench-mab-comparison-quick     # 3 seeds, smoke
```

## Workloads

| Name                       | Description                                                     |
|----------------------------|-----------------------------------------------------------------|
| `stationary_3arm`          | 3 arms, true means `[0.9, 0.5, 0.1]`, T=10000                   |
| `stationary_5arm_close`    | 5 arms, true means `[0.62, 0.60, 0.58, 0.56, 0.54]`, T=20000    |
| `drift_at_t5000`           | optimal flips from arm 0 to arm 2 at t=5000                     |
| `gradual_drift`            | arm 0 decays 0.9→0.3, arm 2 grows 0.1→0.7 linearly              |
| `contextual_2cluster`      | 2 clusters with one-hot context, each best at a different arm   |
| `cold_start_with_prior`    | same as stationary_3arm but flagged for prior-injection         |

Every workload is **byte-identical across algorithms** at fixed seed —
verified via SHA-256 hashing of the reward + context sequence (the same
principle PR #2269 enforces for fusion arms). This isolates algorithm
deltas from sampling-noise confounds.

## Algorithms

| Name                   | Type                          | Use                                |
|------------------------|-------------------------------|------------------------------------|
| `ucb1`                 | Non-contextual UCB            | Academic baseline (Auer 2002)      |
| `epsilon_greedy`       | Non-contextual epsilon-greedy | Sanity check / teaching baseline   |
| `routing_sampling_py`  | Beta-Bernoulli posterior      | **Production baseline** — Python port of `extproc/router_learning_adaptation.go:313-373` |

The Python port matches the **deterministic** parts of the Go score formula
(alpha/beta updates, penalty arithmetic, EWMA decay constants) exactly. It
does NOT byte-match the Go RNG output (Python `random` is not Go `math/rand`)
— the goal is parity-of-distribution, not parity-of-bytes. Test
`test_routing_sampling_py_score_*` enforces this.

## Adding a new algorithm

Implement the protocol in `algorithms.py`:

```python
@dataclass
class MyAlgorithm:
    name: str = "my_algorithm"
    # ... fields with defaults

    def reset(self, n_arms: int, context_dim: int | None) -> None:
        ...

    def select(self, t: int, context: list[float] | None) -> int:
        ...

    def update(self, arm: int, reward: float, context: list[float] | None) -> None:
        ...
```

Register it in `_BUILDERS` at the bottom of `algorithms.py`. Then re-run
`make bench-mab-comparison` and inspect the verdict for your new algorithm
against `routing_sampling_py` (the production baseline).

## Pre-registered KEEP/KILL rule

Applied in order, per algorithm pair (X, Y):

| Verdict                       | Condition                                                                  |
|-------------------------------|----------------------------------------------------------------------------|
| `KEEP_X`                      | X dominates Y on every workload (CI excludes 0 in X's favor everywhere)    |
| `KEEP_X_CONTEXTUAL_ONLY`      | X wins only on `contextual_2cluster`; ties on stationary workloads         |
| `KILL_X`                      | X significantly worse than Y on any workload, no compensating contextual win |
| `INCONCLUSIVE`                | Otherwise (typically n_seeds too small)                                    |

Pre-registration matters: the rule is decided **before** seeing data, so
results cannot be cherry-picked into a favorable narrative. This is the
same protocol PR #2269 used (`bench/grounded_fusion/compare_multiarm.py`).

## What this harness intentionally does NOT do

- ❌ Run real LLMs or talk to vLLM
- ❌ Read live Router Replay records (that is `--learning-architecture`)
- ❌ Affect any CI gate
- ❌ Modify the Go router runtime
- ❌ Pick the best algorithm for production deployment (that needs all
   four evidence layers — synthetic + fixture + replay + live)

## Files

```
bench/mab_comparison/
├── README.md         this file
├── workloads.py      synthetic bandit environments + sha256 hashing
├── algorithms.py     UCB1, EpsilonGreedy, RoutingSamplingPy
├── metrics.py        cumulative_regret, optimal_arm_rate, recovery_time, paired_bootstrap_ci
├── runner.py         (algorithm × workload × seed) -> JSONL
├── compare.py        paired bootstrap CIs + KEEP/KILL verdict
└── plot.py           regret / arm-rate / recovery figures (matplotlib)

bench/test_mab_comparison.py      pytest suite (17 tests, stdlib + pytest only)
```
