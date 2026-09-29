---
title: Benchmarking
---

# Benchmarking

Use [sr-bench 1.0](sr-bench) to compare a real MoM entrypoint with single models
on reusable frozen tasks. The CLI and Dashboard share quality, cost, token,
latency and wall-time evidence. Begin with a bounded development slice and use
a disjoint holdout for acceptance.

| Question | Surface |
| --- | --- |
| Does Balance match the best single model while reducing subject cost? | [sr-bench 1.0](sr-bench) |
| How do I inspect routing and improve a recipe before live evaluation? | [Tune and verify a recipe](agent-evaluation-loop) |
| How do I publish measured model evidence into routing configuration? | [Custom evaluations](custom-evaluations) |
| Did a code change regress allocations or component latency? | [Component microbenchmarks](#component-microbenchmarks) |
| Which cache store or inference binding works better here? | [Backend comparisons](#backend-comparisons) |

Component benchmarks and specialized scripts under `bench/` remain developer
diagnostics. They do not produce a complete sr-bench score or replace a paired
live model comparison.

## Component microbenchmarks

The `perf/` package contains Go benchmarks for classification, decision
evaluation, response-cache operations, ExtProc processing, and Looper-family
paths. They do not need a running Router, but model-dependent suites require the
native libraries and benchmark model files.

```bash
make download-models-perf
make rust
make perf-bench-quick
```

Useful targets:

- `make perf-bench` runs the full component set.
- `make perf-bench-classification`, `make perf-bench-decision`,
  `make perf-bench-cache`, and `make perf-bench-looper` narrow the run.
- `make perf-check` records benchmark output and fails when a gated allocation
  or byte baseline regresses beyond its configured threshold.
- `make perf-compare` compares an existing `reports/bench-output.txt` without
  failing on the result.
- `make perf-profile-cpu` and `make perf-profile-mem` produce pprof data.

The regression gate uses `allocs/op` and `B/op` for pass/fail. `ns/op` is
reported as advisory because it varies with the runner. Performance CI is
selected for changes owned by the performance domain and is also available in
manual and nightly workflows; it is not run for every documentation or product
change.

See the repository's
[`perf/README.md`](https://github.com/vllm-project/semantic-router/blob/main/perf/README.md)
for baseline and profiling details.

## Backend comparisons

These targets start or build their own dependencies. Run them on the hardware
and container runtime you intend to evaluate:

```bash
# Response-cache stores
make benchmark-cache-comparison
make benchmark-hybrid-vs-milvus
make benchmark-redis
make benchmark-valkey

# Native inference implementations
make benchmark-openvino-classifier
make benchmark-openvino-embedding
make benchmark-openvino-vs-candle
```

Do not interpret a store or binding comparison as an end-to-end routing result.
Network placement, warmup, dataset shape, model files, and host contention can
change the outcome.

## Reporting results

For any number intended to guide a deployment or public claim, record:

- repository commit and complete Router configuration
- model, dataset, and dependency revisions
- hardware, driver, runtime, and backend topology
- exact command, warmup, concurrency, and sample count
- failures and excluded samples
- raw artifacts and the aggregation method

Compare alternatives on the same workload and environment. A QPS, latency,
accuracy, cost, or savings number without this context is a local observation,
not an expected property of Semantic Router.

Model-selection evaluation used during training is documented separately in
[Model Performance Evaluation](../training/model-performance-eval).
