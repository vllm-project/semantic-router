---
title: Open Intelligence Architecture
description: A versioned capability hierarchy connecting reproducible evaluation, a unified Model Arena, and evidence-aware routing.
created: 2026-09-09
status: Implemented
---

> **Status:** Implemented in [PR #3634](https://github.com/vllm-project/semantic-router/pull/3634) · **Tracks:** [#3577](https://github.com/vllm-project/semantic-router/issues/3577)

## Decision

vLLM Semantic Router uses one versioned evaluation graph for three consumers:

1. the Model Hub preserves every valid benchmark result and its provenance;
2. the Arena renders Overall, capability, and benchmark rankings from that graph;
3. routing selects an Overall or capability index explicitly and consumes only an
   `available` result for the candidate's exact model and reasoning effort.

Physical and virtual models follow the same contract. Missing data remains visible
but is never estimated, silently reweighted, or copied across model variants.

## Capability roadmap

Version 1.0 is the active text-intelligence contract. Versions 1.5 and 2.0 are
design locks, not active catalog indices; each activates only after its benchmark
revisions, runners, scorers, and comparison cohort are frozen.

```text
Intelligence 1.0
├── General                 20%
│   └── MMLU-Pro           100%
├── Reasoning               40%
│   ├── GPQA Diamond        50%
│   └── Humanity's Last Exam 50%
├── Coding                  20%
│   ├── LiveCodeBench v6    50%
│   └── SciCode             50%
└── Agentic                 20%
    └── Terminal-Bench 2.1 100%
```

```text
Intelligence 1.5
├── General                 20%
│   └── MMLU-Pro           100%
├── Reasoning               40%
│   ├── GPQA Diamond        50%
│   └── Humanity's Last Exam 50%
├── Coding                  20%
│   ├── LiveCodeBench v6    50%
│   └── SciCode             50%
└── Agentic                 20%
    ├── Terminal-Bench 4.0  50%
    └── SWE-bench Live      50%  (immutable snapshot)
```

```text
Intelligence 2.0
├── General                 15%
│   └── MMLU-Pro           100%
├── Reasoning               30%
│   ├── GPQA Diamond        50%
│   └── Humanity's Last Exam 50%
├── Coding                  15%
│   ├── LiveCodeBench v6    50%
│   └── SciCode             50%
├── Agentic                 15%
│   ├── Terminal-Bench 4.0  40%
│   ├── SWE-bench Live      40%  (immutable snapshot)
│   └── CyberGym Level 1    20%
├── Multimodal              15%
│   ├── MMMU-Pro            50%
│   ├── MathVista           25%
│   └── OCRBench v1         25%
└── Safety                  10%  + eligibility gate
    ├── HarmBench Robustness 50%
    └── XSTest Safe Helpfulness 50%
```

CyberGym belongs to **Agentic / Security Engineering**: it measures autonomous
vulnerability reproduction in executable environments. It does not measure
whether a model behaves safely, so it is not part of Safety.

## Benchmark contract

### Active in 1.0

| Capability | Benchmark | What it measures | Public sources |
| --- | --- | --- | --- |
| General | MMLU-Pro | Broad multi-domain knowledge and reasoning | [repository](https://github.com/TIGER-AI-Lab/MMLU-Pro), [paper](https://arxiv.org/abs/2406.01574), [data](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| Reasoning | GPQA Diamond | Graduate-level scientific reasoning | [repository](https://github.com/idavidrein/gpqa), [paper](https://arxiv.org/abs/2311.12022), [data](https://huggingface.co/datasets/idavidrein/gpqa) |
| Reasoning | Humanity's Last Exam | Frontier, cross-domain closed-answer reasoning | [repository](https://github.com/centerforaisafety/HLE), [paper](https://arxiv.org/abs/2501.14249), [data](https://huggingface.co/datasets/cais/hle) |
| Coding | LiveCodeBench v6 | Recent competitive code generation | [repository](https://github.com/LiveCodeBench/LiveCodeBench), [paper](https://arxiv.org/abs/2403.07974), [data](https://huggingface.co/datasets/livecodebench/code_generation_lite) |
| Coding | SciCode | Executable scientific-programming problems | [repository](https://github.com/scicode-bench/SciCode), [paper](https://arxiv.org/abs/2407.13168), [data](https://huggingface.co/datasets/SciCode1/SciCode) |
| Agentic | Terminal-Bench 2.1 | Long-horizon work in a terminal environment | [tasks](https://github.com/harbor-framework/terminal-bench-2), [dataset](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1), [runner](https://github.com/harbor-framework/harbor) |

All six have public inputs, executable evaluation code, and a public scoring
path. Each catalog record still pins the exact benchmark revision, profile,
checkpoint, reasoning effort, harness, tools, run conditions, date, and source.
Terminal and agent benchmarks are joint measurements of a model and a frozen
agent harness; a model name alone never identifies such a result.

### Reserved for 1.5 and 2.0

| Version | Benchmark | Activation requirement | Public sources |
| --- | --- | --- | --- |
| 1.5 | Terminal-Bench 4.0 | Pin the `v4.0.0` tasks, Harbor version, agent, limits, retries, and environment | [repository](https://github.com/harbor-framework/terminal-bench), [release](https://github.com/harbor-framework/terminal-bench/releases/tag/v4.0.0), [dataset](https://huggingface.co/datasets/harborframework/terminal-bench) |
| 1.5 | SWE-bench Live | Pin one immutable verified snapshot, task IDs, images, scorer, agent, and retry policy | [repository](https://github.com/microsoft/SWE-bench-Live), [paper](https://arxiv.org/abs/2505.23419), [data](https://huggingface.co/SWE-bench-Live) |
| 2.0 | CyberGym Level 1 | Pin all Level-1 tasks, environment assets, agent, budget, and binary verifier | [repository](https://github.com/sunblaze-ucb/cybergym), [paper](https://arxiv.org/abs/2506.02548), [data](https://huggingface.co/datasets/sunblaze-ucb/cybergym) |
| 2.0 | MMMU-Pro | Pin the standard multimodal split and evaluator | [repository](https://github.com/MMMU-Benchmark/MMMU), [paper](https://arxiv.org/abs/2409.02813), [data](https://huggingface.co/datasets/MMMU/MMMU_Pro) |
| 2.0 | MathVista | Pin the public `testmini` split, prompt, extraction, and scorer | [repository](https://github.com/lupantech/MathVista), [paper](https://arxiv.org/abs/2310.02255), [data](https://huggingface.co/datasets/AI4Math/MathVista) |
| 2.0 | OCRBench v1 | Pin the 1,000 public items and deterministic v1 scorer; do not substitute private-test v2 | [repository](https://github.com/qywh2023/OCRbench), [paper](https://arxiv.org/abs/2305.07895), [data](https://huggingface.co/datasets/echo840/OCRBench) |
| 2.0 | HarmBench | Pin behaviors, attacks, generation budget, and an open classifier | [repository](https://github.com/centerforaisafety/HarmBench), [paper](https://arxiv.org/abs/2402.04249), [data](https://huggingface.co/datasets/walledai/HarmBench) |
| 2.0 | XSTest | Pin the public prompt set and open safe-compliance scorer | [repository](https://github.com/paul-rottger/xstest), [paper](https://arxiv.org/abs/2308.01263) |

“Open” here means an independent contributor can access the inputs, run the
evaluation, and reproduce the score with the pinned version. Before activation,
licenses and redistribution boundaries are recorded per artifact; a public
leaderboard without runnable inputs or scorer is not sufficient.

## Hierarchical score

Every raw metric is normalized to `[0, 1]`. A leaf capability is a weighted
mean of its benchmarks; Overall is a weighted mean of capability scores. For
1.0:

```text
General   = MMLU-Pro
Reasoning = 0.50 × GPQA Diamond + 0.50 × HLE
Coding    = 0.50 × LiveCodeBench + 0.50 × SciCode
Agentic   = Terminal-Bench 2.1

Intelligence 1.0 = 100 × (
    0.20 × General
  + 0.40 × Reasoning
  + 0.20 × Coding
  + 0.20 × Agentic
)
```

`require_all` applies at every node. A capability is available only when all
of its leaves are available for the same model and reasoning effort; Overall
is available only when all four capabilities are available. Otherwise the
result is `partial` or `missing`, its score is null, and coverage plus missing
components remain explicit. There is no zero/mean imputation, proxy score,
partial renormalization, or cross-effort borrowing.

This does not make incomplete evidence useless. A model missing General may
still have an available Coding score and participate in a Coding rank or a
coding-specific route. It simply cannot claim a comparable Overall score.

## Operator-owned evidence

Release evidence and deployment-local evidence use the same typed graph. An
operator defines new benchmark semantics, index DAGs, and model-linked
measurements under the top-level `evaluation`. Built-in benchmarks need no
redeclaration.

Operator resource IDs must be namespaced and versioned. They cannot shadow a
built-in benchmark or index. Definitions pin profile, metric range, direction,
normalization, component weights, and one explicit missing-data policy:

- `require_all` produces a score only with every component;
- `require_coverage` produces a score after a declared coverage threshold;
- `reported_only` produces a score from any reported component.

The latter two policies are deliberate operator index semantics, not implicit
imputation. Every result still exposes its coverage. A route can impose a
stricter `quality.min_coverage` than the index definition. A record for an
undeclared benchmark is preserved under `evaluation.records[]` but cannot
enter an index until its semantics are declared.

## Physical and virtual models

A virtual model is evaluated through a frozen endpoint over every task in the
same suite. The receipt additionally records recipe revision, per-task route,
failures, tokens, latency, and cost. Its intelligence score is never assembled
from member-model scores, routing shares, or an oracle choice.

## Arena

Both Hub surfaces render three independent views from the same catalog:

- **Overall** — one complete-case Intelligence leaderboard;
- **Capabilities** — General, Reasoning, Coding, and Agentic leaderboards;
- **Benchmarks** — the six raw 1.0 benchmark leaderboards.

Each view uses competition rank, exposes the selected reasoning effort, treats
physical and virtual models identically, and supports shareable scope and layer
state. The public Hub also preserves directory filters and model details in its
URL. Incomplete models appear wherever their evidence is valid instead of
appearing in a fabricated Overall rank.

Benchmarks are the Arena's third layer. The previous standalone benchmark
explorer is removed from both the public Hub and Dashboard so ranking, filters,
and URL state cannot diverge between two presentations of the same evidence.

The generated snapshot currently contains 101 model cards and 1,509 evaluation
records. Unique models with available 1.0 results are: General 35, Reasoning 86,
Coding 23, Agentic 77, and Overall 21. Overall therefore clears the initial
20-model target while keeping strict completeness. New GLM-5.3 and Qwen3.8
cards remain visible in their supported capability and benchmark views even
when a missing leaf prevents Overall eligibility.

## Routing

`multi_factor` can select any versioned Overall, capability, or operator index.
It separates hard eligibility from the optimization objective.

Balanced routing uses normalized weights:

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    quality:
      index: vllm-sr/coding@1.0.0
      on_missing: exclude
      min_coverage: 1.0
    weights:
      quality: 0.4
      latency: 0.2
      cost: 0.2
      load: 0.2
```

Accuracy-first and cost-first use the same generic lexicographic engine rather
than product-specific branches:

```yaml
# Accuracy-first: keep models within 3% of the best quality, then minimize cost.
objective:
  strategy: lexicographic
  priorities:
    - {factor: quality, tolerance: 0.03}
    - {factor: cost, tolerance: 0.05}
    - {factor: latency, tolerance: 0.05}
```

```yaml
# Cost-first: enforce a quality floor, then choose within the cheapest band.
quality:
  index: vllm-sr/intelligence@1.0.0
  on_missing: exclude
  min_coverage: 1.0
  min_score: 65
objective:
  strategy: lexicographic
  priorities:
    - {factor: cost, tolerance: 0.05}
    - {factor: quality, tolerance: 0.03}
    - {factor: latency, tolerance: 0.05}
```

The quality lookup uses the candidate's exact reasoning effort and accepts only
`status: available`:

- `exclude` removes candidates missing that index. The existing
  `on_no_candidates` policy applies if none remain.
- `disable_quality` keeps every candidate; if any candidate lacks the selected
  index, the selector disables quality for the entire candidate pool and uses
  only latency, cost, load, and configured SLOs. It never reweights one model
  differently from another.

A general decision can select `vllm-sr/intelligence@1.0.0`; a classified coding
decision can select `vllm-sr/coding@1.0.0`. A future benchmark is added as a
versioned leaf and composed into a new capability/index version, without adding
benchmark-specific branches to the router.

Cost uses the current input-token estimate and requested maximum output-token
budget with separate input/output prices. Hard SLOs and quality floors run
before either objective. Future safety-first and cybersecurity-first recipes
therefore add policy/eligibility gates and select the relevant versioned index;
they do not require another selection algorithm.

## Follow-up closure and roadmap

This implementation closes the current contract across configuration, runtime,
Dashboard, public Website, and documentation:

- custom benchmark definitions, index DAGs, model-linked records, validation,
  canonical round-trip, and exact-effort routing are one path;
- Balanced, Accuracy-first, and Cost-first are configurations of one selector;
- Overall, Capabilities, and Benchmarks are the only Arena layers on both Hub
  surfaces, with shareable Arena state and complete public-Hub URL state;
- configuration, evaluation, and algorithm guides publish the same fields and
  missing-data behavior.

The following work remains versioned follow-up, not hidden behavior in 1.0:

- **MoM:** publish Cost-first, Accuracy-first, Safety-first, and
  Cybersecurity-first virtual models beside the current Balanced recipe, then
  optimize frozen recipes from online outcomes plus evaluation, inference, and
  research evidence;
- **Evaluation:** activate 1.5 and 2.0 only after their open suites are pinned,
  expand capability indices and the Arena, and keep old benchmark/index versions
  queryable during migration;
- **Inference:** admit additional current models, including Kimi K3 and DSV4
  Flash Vision Exp, through the same model/provider/evidence contract;
- **Research:** pursue the routing program as nine concrete tracks:
  1. select models from large open and closed model pools;
  2. reuse KV cache across models when switching;
  3. determine which context to retain when switching models;
  4. improve SLMs by caching and reusing LLM reasoning traces for similar
     requests;
  5. learn failure patterns for self-improving distillation routing between
     SLMs and LLMs;
  6. route from model-internal latent statistics;
  7. forward embeddings or other representations beyond the prompt when
     routing to a model;
  8. use model collaboration for test-time scaling; and
  9. build a self-improving router with routing memory.

## Versioning and migration

Benchmark identity, profile, and index definition are immutable. Terminal-Bench
4.0 replaces 2.1 in Intelligence 1.5; their scores never coexist in one Agentic
node. Old evaluations and index versions remain queryable while the new version
builds a complete physical-and-virtual cohort. The catalog default changes only
after the replacement contract is reproducible and sufficiently covered.

The machine-readable 1.0 graph lives in
`config/catalog/resources/indices.yaml`; catalog generation is the sole scoring
implementation used by runtime and UI projections.
