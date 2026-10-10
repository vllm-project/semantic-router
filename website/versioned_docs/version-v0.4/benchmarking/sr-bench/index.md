---
title: sr-bench 1.0
description: Compare real MoM and single-model capability, cost, latency and token usage on reusable frozen tasks.
---

# sr-bench 1.0

sr-bench compares a Mixture of Models (MoM) entrypoint with single models on the
same frozen tasks. The CLI and **Dashboard → Evaluation** share one durable
service, run IDs, results and reports. Use a small development slice to tune
routing, then evaluate a disjoint holdout before making a quality or savings
claim.

## Choose a scope

Each number below is a whole task count **per target**, not a model-call count.
Coding and agent tasks can make several calls; judges and user simulators add
separately reported costs.

| Benchmark ID | Capability | Smoke | Quick/dev | Standard/holdout |
| --- | --- | ---: | ---: | ---: |
| `mmlu-pro` | Knowledge across 14 subjects | 14 | 500 | 2,000 |
| `gpqa-diamond` | Scientific reasoning | 4 | 40 | 158 |
| `hle` | HLE text-only reasoning | 4 | 40 | 200 |
| `livecodebench` | Programming, cumulative v6 tasks | 2 | 30 | 150 |
| `scicode` | Scientific programming, whole main problems | 1 | 3 | 20 |
| `terminal-bench-2.1` | Sandboxed terminal tasks | 1 | 3 | 15 |
| `simpleqa-verified` | Factual correctness | 5 | 100 | 500 |
| `arc-agi-2` | Public evaluation puzzles, exact output grids | 2 | 12 | 80 |
| `tau3` | τ³ text interaction, three domains | 3 | 12 | 60 |
| **Total** | | **36** | **740** | **3,183** |

Run `vllm-sr benchmark catalog` for the installed adapter identities. Select a
capability slice when that answers the current question. A 500-question MMLU-Pro
slice is a development comparison, not the complete 12,032-question upstream
benchmark. A run without all nine benchmarks has no complete sr-bench score.

Preparation uses pinned sources, stable task IDs, content hashes, a recorded
seed and proportional stratification. Smoke is a subset of quick; standard is
disjoint from quick. Related units such as SciCode subproblems stay together.
Public tasks are not contamination-free. Previously seen GPQA labels require a
retest disclosure even when the local split is called holdout.

## Next

1. [Connect the shared worker](./shared-worker.md): set up the service that
   the CLI and Dashboard share.
2. [Prepare reusable tasks and targets](./tasks-and-targets.md): freeze
   datasets, reserve named evaluation history and register targets.
3. [Plan and run](./plan-and-run.md): freeze a manifest and run it within
   limits.
4. [Iterate with preview, replay and live evaluation](./iterate.md): tune
   routing on dev tasks, then evaluate a holdout.
5. [Read the results](./results.md): interpret reports, comparisons and Dashboard
   views, and recover unfinished work.
