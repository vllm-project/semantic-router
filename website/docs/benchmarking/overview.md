---
title: Overview
---

# Benchmarking Overview

vLLM Semantic Router ships several benchmark suites that answer different
questions. This page is a map: it tells you which suite answers which question,
what each one needs, and how to start it. Each suite keeps its own README as the
source of truth for details — this page links out rather than duplicating them.

## The three layers

| Layer | Location | Answers | Runs in CI |
|-------|----------|---------|------------|
| Component micro-benchmarks | `perf/` | "Did my change make the router slower or allocate more?" | Yes, on every PR |
| End-to-end suites | `bench/` | "Does routing improve answer quality, safety, or session behavior?" | Router Learning eval only |
| Backend and store comparisons | Make targets | "Which cache store or inference backend should I use?" | No |

The first layer is Go benchmarks over individual components and needs no running
router. The second is Python suites that drive a live router, and usually an
LLM backend, over real datasets. The third is a set of Make targets that compare
interchangeable backends; these have no README of their own, so they are
documented here.

## Which benchmark answers my question?

| Your question | Use | Start with |
|---------------|-----|------------|
| Did my change regress latency or allocations? | `perf/` component benchmarks | `make perf-check` |
| Did my change regress the Looper? | `perf/` Looper benchmarks | `make perf-bench-looper` |
| Does routing preserve answer accuracy? | Reasoning datasets | `vllm-semantic-router-bench compare --dataset arc` |
| Is reasoning mode worth the cost? | Reasoning-mode eval | `vllm-semantic-router-bench reasoning-eval` |
| Does session-aware routing hold up under load or faults? | Live agentic routing | `bench/agentic_routing_live_benchmark.py` |
| Can the router complete multi-turn agent tasks? | Live agent task | `bench/agent_task_live_benchmark.py` |
| Are cached input tokens actually reported? | Cache-token probe | `bench/cache_token_probe.py` |
| Did Router Learning routing quality regress? | Router Learning architecture eval | `make bench-router-learning` |
| How good is hallucination detection? | Hallucination | `make bench-hallucination` |
| Does grounding-aware fusion improve answers? | Grounded fusion (DRACO) | `bench/grounded_fusion/run_ab.sh` |
| What are our publishable benchmark numbers? | Router Flow real eval | `bench/router_flow/real_eval/` |
| How much faster is GPU or Flash Attention? | CPU vs GPU | `bench/cpu-vs-gpu/` |
| Which cache store performs best? | Store comparisons | `make benchmark-hybrid-vs-milvus` |
| Is OpenVINO faster than Candle here? | Inference backend comparison | `make benchmark-openvino-vs-candle` |
| Is release evidence complete? | Session routing GA report | `bench/session_routing_ga_report.py` |

## Layer 1: component micro-benchmarks

Go benchmarks over classification, the decision engine, the semantic cache, the
ext_proc hot path, and the Looper. They need the Rust binding libraries and the
benchmark models, but no running router.

```bash
make download-models-perf
make rust
make perf-bench-quick
```

Common targets:

- `make perf-bench` — all component benchmarks
- `make perf-bench-quick` — faster iteration
- `make perf-bench-classification`, `-decision`, `-cache`, `-looper`, `-concurrency`
- `make perf-check` — run and fail on regression
- `make perf-compare` — compare against committed baselines, report only
- `make perf-baseline-update` — refresh baselines after an intended change
- `make perf-profile-cpu`, `make perf-profile-mem`, `make perf-flamegraph`
- `make perf-e2e` — full-stack performance tests via the e2e framework

### Why the gate is portable across machines

The PR regression gate blocks on **allocs/op and B/op**, which are determined by
the code path rather than the CPU. `ns/op` is reported but advisory only, since
absolute time depends on the runner. This is what makes the committed baselines
in `perf/testdata/baselines/` valid regardless of which machine recorded them,
and it means you can validate the gate locally without matching CI hardware.

Details: [`perf/README.md`](https://github.com/vllm-project/semantic-router/blob/main/perf/README.md).

## Layer 2: end-to-end suites

Python suites under `bench/`. Most need a running router (and Envoy), and an
OpenAI-compatible backend. Install the package first:

```bash
pip install -e bench
```

Some suites need extras: `bench[plotting]` for figures and
`bench[real_eval]` for the EvalScope benchmark adapters.

| Suite | Measures | Needs |
|-------|----------|-------|
| Reasoning datasets | Accuracy, token usage, and time per output token across MMLU-Pro, ARC, GPQA, TruthfulQA, CommonsenseQA, HellaSwag | OpenAI-compatible backend, Hugging Face access |
| Live agentic routing | Session continuity, latency percentiles, tool-loop and context-portability violations, failure recovery | Running router and Envoy |
| Live agent task | Deterministic multi-turn task completion, scored without a judge model | Running router, backend |
| Cache-token probe | Whether the backend reports cached input tokens through the router | Running router, backend |
| Router Learning architecture eval | Route correctness, switch rate, cost delta, learning overhead, gated by named profiles | Standard library only |
| Hallucination | Precision, recall, F1, and two-stage pipeline efficiency | Router, Envoy, vLLM backend |
| Grounded fusion (DRACO) | Whether grounding-aware fusion beats plain fusion on rubric-graded answers | Ollama, candle NLI model |
| Router Flow (`flow_eval.py`) | Development smoke and qualitative triage — explicitly **not** publishable data | Running router |
| Router Flow real eval | Publishable EvalScope rows (GPQA-D, LiveCodeBench, HLE, SciCode, and others) | Docker, EvalScope, usually an AMD eval host |
| CPU vs GPU | Signal-extraction latency across ONNX CPU/GPU, SDPA vs Flash Attention, BUFFERED vs STREAMED | AMD GPU with ROCm 7.0+, Docker |
| Session routing GA report | Whether the required release evidence exists and agrees | Summaries from the runs above |

Two suites deserve a caveat. `bench/router_flow/flow_eval.py` is a one-prompt-per-family
proxy and its own README says it is not publishable benchmark data; use
`bench/router_flow/real_eval/` for numbers that leave the repo. The Router Learning
architecture eval is fixture-based and standard-library-only, which makes it the
easiest suite to run anywhere.

Details: [`bench/README.md`](https://github.com/vllm-project/semantic-router/blob/main/bench/README.md)
and the per-suite READMEs under
[`bench/`](https://github.com/vllm-project/semantic-router/tree/main/bench).

## Layer 3: backend and store comparisons

These compare interchangeable backends and are driven entirely by Make targets.
They have no README of their own.

Cache and vector stores:

- `make benchmark-cache-comparison`
- `make benchmark-hybrid-only`, `make benchmark-hybrid-quick`
- `make benchmark-hybrid-vs-milvus`
- `make benchmark-redis`, `make benchmark-valkey`

Inference backends:

- `make benchmark-openvino-classifier`
- `make benchmark-openvino-embedding`
- `make benchmark-openvino-vs-candle`

The store benchmarks start their own container (for example `start-valkey`), so a
running container runtime is required. The OpenVINO targets first build the
OpenVINO C++ binding.

## Environment matrix

| Suite | Linux CI | Linux or macOS dev machine | GPU host |
|-------|----------|----------------------------|----------|
| `perf/` decision, ext_proc, Looper | Verified | Verified | Expected |
| `perf/` classification, cache | Verified | Blocked on macOS arm64, see below | Expected |
| Router Learning architecture eval | Verified, own workflow | Verified | Expected |
| Reasoning datasets | Not run | Needs a backend | Needs a backend |
| Live routing, agent task, cache-token | Not run | Needs a running router | Needs a running router |
| Hallucination | Not run | Needs router and vLLM backend | Recommended |
| Grounded fusion (DRACO) | Not run | Verified by suite author on macOS with Ollama | Expected |
| Router Flow real eval | Not run | Not practical | Required |
| CPU vs GPU | Not run | Not possible | Required, AMD ROCm |
| Store and backend comparisons | Not run | Needs a container runtime | Expected |

"Verified" means the command was run and observed in that environment while
writing this page; "expected" means it should work there but was not re-run.

## Known platform notes

### Classifier initialization hangs on macOS arm64

`init_lora_unified_classifier` can block indefinitely on macOS arm64 with the
mmbert32k merged models, which stalls `make perf-bench-quick` and the
classification benchmarks. Every thread parks in a condition variable with
almost no CPU consumed, and Go's test `-timeout` cannot interrupt it because the
goroutine is blocked inside a cgo call.

The decision, ext_proc, and Looper benchmarks are unaffected, so a macOS
contributor can still validate the regression gate for those suites:

```bash
cd perf
go test -bench='^BenchmarkEvaluateDecisions|^BenchmarkPrioritySelection' \
  -benchmem ./benchmarks/
```

Tracked in [issue #2440](https://github.com/vllm-project/semantic-router/issues/2440).

### Other notes

- `make build-openvino-binding` uses `nproc`, which is not present on macOS.
- Benchmarks that fail to load models usually need `make download-models-perf`
  and the Rust libraries on the library path; see the troubleshooting section of
  [`perf/README.md`](https://github.com/vllm-project/semantic-router/blob/main/perf/README.md).
- Suites whose baseline is empty are skipped by the regression gate rather than
  failed, so a missing model shows up as a skipped suite, not a red check.

## Where to go next

- [`perf/README.md`](https://github.com/vllm-project/semantic-router/blob/main/perf/README.md)
  — component benchmarks, profiling, thresholds, baseline lifecycle
- [`bench/README.md`](https://github.com/vllm-project/semantic-router/blob/main/bench/README.md)
  — reasoning datasets, live routing, GA evidence
- [`bench/hallucination/`](https://github.com/vllm-project/semantic-router/tree/main/bench/hallucination),
  [`bench/grounded_fusion/`](https://github.com/vllm-project/semantic-router/tree/main/bench/grounded_fusion),
  [`bench/router_flow/`](https://github.com/vllm-project/semantic-router/tree/main/bench/router_flow),
  [`bench/cpu-vs-gpu/`](https://github.com/vllm-project/semantic-router/tree/main/bench/cpu-vs-gpu)
  — per-suite detail
- [Model Performance Evaluation](../training/model-performance-eval.md) — model
  selection and training-time evaluation, which is distinct from the suites above
