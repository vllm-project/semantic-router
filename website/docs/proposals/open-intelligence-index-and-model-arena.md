---
title: Open Intelligence Index 1.0 and Unified Model Arena
description: A versioned five-benchmark intelligence contract and one public ranking surface for physical and virtual models.
created: 2026-09-09
status: Implemented
---

> **Status:** Implemented · **Created:** 2026-09-09 · **Tracks:**
> [#3577](https://github.com/vllm-project/semantic-router/issues/3577)

## Decision

vLLM Semantic Router defines one minimal, reproducible text-intelligence index
for both physical models and frozen Mixture-of-Models. The public Model Hub
Arena ranks only complete results produced by this exact contract. Benchmark
evidence that is valid but incomplete remains visible as coverage; it is never
turned into a synthetic score.

Version 1.0 intentionally measures five axes: broad reasoning, graduate-level
science, frontier knowledge, scientific programming, and agentic terminal
execution. Instruction following, long-context retrieval, competitive coding,
and repository repair remain useful additional evidence, but are not silently
folded into this index.

## Core benchmark pool

All five projects publish their evaluation implementation and benchmark data.
The repository pins the benchmark identity, metric, and eligible run profiles;
an evaluation record additionally preserves the exact model, reasoning effort,
harness, tools, run conditions, date, and source.

| Axis | Benchmark and metric | Weight | Open source |
| --- | --- | ---: | --- |
| Broad reasoning | [MMLU-Pro](https://github.com/TIGER-AI-Lab/MMLU-Pro) accuracy ([paper](https://arxiv.org/abs/2406.01574), [dataset](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)) | 20% | Apache-2.0 |
| Scientific reasoning | [GPQA Diamond](https://github.com/idavidrein/gpqa) accuracy ([paper](https://arxiv.org/abs/2311.12022), [dataset](https://huggingface.co/datasets/idavidrein/gpqa)) | 20% | MIT |
| Frontier reasoning | [Humanity's Last Exam](https://github.com/centerforaisafety/HLE) accuracy ([paper](https://arxiv.org/abs/2501.14249), [dataset](https://huggingface.co/datasets/cais/hle)) | 20% | MIT |
| Scientific programming | [SciCode](https://github.com/scicode-bench/SciCode) executable subproblem score ([paper](https://arxiv.org/abs/2407.13168), [dataset](https://huggingface.co/datasets/SciCode1/SciCode)) | 20% | Apache-2.0 |
| Agentic execution | [Terminal-Bench 2.1](https://github.com/harbor-framework/terminal-bench-2-1) resolved rate ([dataset](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1), [runner](https://github.com/harbor-framework/harbor)) | 20% | Apache-2.0 |

Terminal-Bench is an agent-and-model measurement. Comparisons therefore pin
the agent, tool contract, resource limits, task revision, and retry policy; the
model name alone is not enough to identify a result.

## Index contract

For model `m` at one named reasoning effort `e`, let each admitted raw metric
`x_i(m,e)` be in `[0, 1]`. Version 1.0 uses identity normalization and equal
weights:

```text
OpenIntelligenceIndex_1.0(m, e) = 100 * sum(0.20 * x_i(m, e))
```

Each component declares an ordered set of compatible evidence profiles. A
repository-run or independent profile is preferred where available; a
documented source-published standard profile is the fallback. The first exact
match is selected, and its record ID remains in the result provenance. Values
are never copied across models, checkpoints, quantizations, reasoning efforts,
benchmark revisions, tool modes, or agent harnesses.

The missing-data policy is `require_all`:

- five present components produce `status: available`, coverage `1.0`, and a
  score;
- one to four present components produce `status: partial`, a null score, and
  exact coverage plus the missing component list;
- zero present components produce `status: missing` and a null score.

There is no mean imputation, zero imputation, parameter-count proxy, partial
renormalization, or cross-effort fallback. This keeps two displayed scores
comparable and prevents a model from improving its index by omitting a weak
benchmark.

## Physical models and Mixture-of-Models

Physical and virtual models use the same benchmark definitions, component
weights, missing-data rule, and Arena ranking code. A virtual model is evaluated
through its frozen public endpoint on the full benchmark. Its score is never
estimated from member-model scores, routing ratios, or an oracle choice.

For paired analysis, the run manifest freezes the virtual recipe and records
per-task route choice, failures, tokens, latency, and cost. These measurements
explain a score and quantify savings; they are not ingredients of the
intelligence index itself.

## Arena and routing semantics

The Model Hub Arena is the first Hub section. It provides shareable scopes for
all models, open-weight models, and virtual models. Model details and every Hub
filter are also represented in the URL.

The Arena selects each model's highest complete effort result, displays that
effort, sorts by score, and assigns competition ranks. Models without a
complete result appear in a separate coverage queue ordered by coverage. This
queue exposes useful historical evidence without presenting unlike partial
means as a leaderboard.

At the current catalog snapshot, the Hub contains 89 Model Cards: 84 physical
and five virtual. Fifty-four are open-weight cards from 19 creators, covering
current and recent representative model lines. The index compiler emits 265
model-effort rows. Eight models currently have all five components and are
rankable; all eight are open-weight models. The other 81 cards stay visible with
their exact gaps. These counts are generated facts, not hard-coded UI copy.
The complete set is DeepSeek R1, DeepSeek V4 Flash, DeepSeek V4 Pro, Gemma 4
31B IT, Kimi K2.5, NVIDIA Nemotron 3 Ultra, NVIDIA Nemotron 3.5 Lightning, and
GPT-OSS 120B.

Routing consumes only an `available` result as its evidence-backed quality
prior. The existing multi-factor selector keeps price, latency, availability,
and load as independent objectives or constraints. A partial index can inform
coverage work and model inspection, but it cannot become route quality.

## Additional evidence and SWE-bench

The catalog keeps valid non-index measurements instead of deleting them.
SWE-bench Verified, IFEval, LiveCodeBench, LongBench v2, and other versioned
benchmarks remain individually filterable and source-linked in the benchmark
gallery.

SWE-bench Verified covers repository-level issue resolution, a capability not
directly measured by the five core components. It is retained as additional
evidence rather than a version 1.0 component because the reported result is a
joint property of the model and software agent, the 500-task Python suite is
substantially heavier to reproduce, and task validity has required continuing
audit. SWE-bench Live is the preferred successor candidate because it adds
newer and broader software tasks, but only an immutable, fully runnable
snapshot can enter a future index. A moving leaderboard cannot be an index
component.

## Benchmark evolution

Benchmark revisions are immutable identities. Terminal-Bench 2.1 remains the
only terminal component of index 1.0. A future Terminal-Bench major revision or
a frozen SWE-bench Live cohort creates a new index major version; it does not
replace a component in place and does not coexist with its predecessor inside
one score.

During migration, old benchmark records remain visible and both index versions
may be computed. One catalog release designates the new default only after the
new cohort has enough complete physical and virtual results. This preserves
historical auditability without comparing scores whose tasks changed.

## Implemented surfaces

- `config/catalog/resources/indices.yaml` owns the formula, profiles, weights,
  and missing-data rule.
- catalog generation computes one result matrix; the public snapshot retains
  available, partial, and missing rows, while the runtime projection admits
  only available routing priors.
- Go and Python evaluators use the same ordered-profile and missingness
  semantics; schemas and tests reject ambiguous profile definitions.
- the public Model Hub renders the unified Arena and its shareable URL state;
  selecting any physical or virtual model opens the same evidence detail flow.
