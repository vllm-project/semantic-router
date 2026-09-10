---
title: Benchmarking
---

# Benchmarking

Choose a benchmark by the question you need to answer. Component benchmarks
measure Router code paths; evaluation suites measure end-to-end routing or model
quality; backend comparisons help evaluate interchangeable stores and inference
implementations. Their results are not directly comparable.

## Choose a suite

| Question | Suite | Starting point |
|----------|-------|----------------|
| Is a routing recipe, model pool, or combined candidate better than a frozen baseline? | Evaluation Plane | [Evaluation Plane](evaluation-plane) |
| Did a code change increase allocations or component latency? | Go microbenchmarks in `perf/` | `make perf-check` |
| Does routing preserve answer quality on reasoning datasets? | Reasoning evaluation in `bench/` | `vllm-semantic-router-bench compare --dataset arc-challenge` |
| Does a session-aware route remain stable across turns or faults? | Live agentic routing | `bench/agentic_routing_live_benchmark.py` |
| Can a routed model complete a multi-turn agent task? | Live agent task | `bench/agent_task_live_benchmark.py` |
| Does the backend report cached-input tokens through the Router? | Cache-token probe | `bench/cache_token_probe.py` |
| Did Router Learning behavior regress on deterministic fixtures? | Architecture evaluation | `make bench-router-learning` |
| How well does hallucination detection perform? | Hallucination evaluation | `make bench-hallucination` |
| Does grounding-aware fusion improve scored answers? | Grounded fusion | `bench/grounded_fusion/run_ab.sh` |
| Do I need formal Router Flow evaluation rather than a development smoke? | EvalScope-backed Router Flow suite | `bench/router_flow/real_eval/` |
| Did a native backend or GPU path change signal-extraction performance? | CPU/GPU comparison | `bench/cpu-vs-gpu/` |
| Which response-cache store or inference binding performs better here? | Backend Make targets | [Backend comparisons](#backend-comparisons) |

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

## End-to-end evaluation

Use the [Evaluation Plane](evaluation-plane) when the claim spans routing,
model-pool composition, live generation, agentic or multimodal behavior,
preference, safety, or capacity. It creates one versioned evidence bundle and
keeps component, system, and production evidence levels separate.

Install the benchmark package from the repository:

```bash
python -m pip install -e bench
```

Install `bench[real_eval]` only for the EvalScope-backed Router Flow suite.
Most live suites require a running OpenAI-compatible endpoint; some require
both the routed endpoint and a direct-backend baseline.

### Reasoning datasets

The packaged CLI supports MMLU, ARC, GPQA, TruthfulQA, CommonsenseQA, and
HellaSwag adapters. A comparison run needs explicit Router and direct-backend
endpoints when their defaults do not match your deployment:

```bash
vllm-semantic-router-bench compare \
  --dataset arc-challenge \
  --samples 20 \
  --router-endpoint http://localhost:8899/v1 \
  --vllm-endpoint http://localhost:8000/v1 \
  --vllm-model <served-model-name>
```

Treat small sample counts as smoke tests, not evidence of model quality.

### Session and agent workloads

Use the live routing scripts when the behavior under test depends on multiple
turns, stable identity, tool loops, backend faults, or Router headers. Each
script exposes threshold flags for a standalone local regression decision.

```bash
python3 bench/agentic_routing_live_benchmark.py --help
python3 bench/agent_task_live_benchmark.py --help
python3 bench/cache_token_probe.py --help
```

These scripts write run artifacts under `.agent-harness/experiments/` by
default. Keep generated reports out of public claims unless the report records
the source commit, configuration, endpoints or model revisions, workload,
sample count, exclusions, and acceptance thresholds.

These scripts are diagnostic and regression utilities. Their thresholds are
not Evaluation Campaign gates, and their artifacts do not implement the sealed
`evaluation-agent-task-ledger.v1` / `evaluation-agent-task-attempt.v1` contract
or the `live-fault-recovery` evidence contract. They therefore cannot claim
Evaluation agentic E5, G6, or Campaign qualification. Use the Evaluation
Plane's `live-agent-tasks` source for decision-grade repeated-task evidence;
that source can earn agentic E5 after server validation, but deliberately has
no Campaign gate and never qualifies G6.

### Router Learning

The Router Learning architecture evaluation is deterministic and does not need
a live endpoint:

```bash
make bench-router-learning
make bench-router-learning PROFILE=release
```

It checks fixture-derived metrics against the selected JSON profile. It is a
regression test for the represented scenarios, not a production traffic
benchmark.

### Specialized suites

- [`bench/hallucination/`](https://github.com/vllm-project/semantic-router/tree/main/bench/hallucination)
  evaluates detector and mitigation behavior against labeled data.
- [`bench/grounded_fusion/`](https://github.com/vllm-project/semantic-router/tree/main/bench/grounded_fusion)
  compares grounded-fusion configurations and may use rubric grading.
- [`bench/router_flow/real_eval/`](https://github.com/vllm-project/semantic-router/tree/main/bench/router_flow/real_eval)
  is the formal EvalScope path. `bench/router_flow/flow_eval.py` is a small
  development proxy and must not be presented as publishable benchmark data.
- [`bench/cpu-vs-gpu/`](https://github.com/vllm-project/semantic-router/tree/main/bench/cpu-vs-gpu)
  requires the documented accelerator, driver, container, and model setup.

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
