---
title: Open Intelligence Index
sidebar_label: Intelligence Index
description: Versioned catalog evidence used by Model Arena and routing quality selection.
---

# Open Intelligence Index

The Open Intelligence Index is a catalog evidence graph shared by Model Hub,
Model Arena and routing. It preserves versioned reported measurements and
missing coverage. It is not the [sr-bench 1.0](sr-bench) live evaluation score.
Use sr-bench to measure new single-model/MoM outcomes, costs and latency; catalog
records do not establish a fresh result on your deployment.

## Intelligence 1.0

```text
Intelligence 1.0
├── General                 20%
│   └── MMLU-Pro           100%
├── Reasoning               40%
│   ├── GPQA Diamond        50%
│   └── HLE 1.0 text-only   50%
├── Coding                  20%
│   ├── LiveCodeBench v6    50%
│   └── SciCode             50%
└── Agentic                 20%
    └── Terminal-Bench 2.1 100%
```

| Capability | Benchmark | Metric | Source |
| --- | --- | --- | --- |
| General | MMLU-Pro | Accuracy | [repository](https://github.com/TIGER-AI-Lab/MMLU-Pro) · [paper](https://arxiv.org/abs/2406.01574) · [data](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) |
| Reasoning | GPQA Diamond | Accuracy | [repository](https://github.com/idavidrein/gpqa) · [paper](https://arxiv.org/abs/2311.12022) · [data](https://huggingface.co/datasets/idavidrein/gpqa) |
| Reasoning | HLE 1.0 text-only | Accuracy over the frozen 2,158-question text-only subset | [repository](https://github.com/centerforaisafety/HLE) · [paper](https://arxiv.org/abs/2501.14249) · [data](https://huggingface.co/datasets/cais/hle) |
| Coding | LiveCodeBench v6 | Pass@1, code generation | [repository](https://github.com/LiveCodeBench/LiveCodeBench) · [paper](https://arxiv.org/abs/2403.07974) · [data](https://huggingface.co/datasets/livecodebench/code_generation_lite) |
| Coding | SciCode | Executable subproblem score | [repository](https://github.com/scicode-bench/SciCode) · [paper](https://arxiv.org/abs/2407.13168) · [data](https://huggingface.co/datasets/SciCode1/SciCode) |
| Agentic | Terminal-Bench 2.1 | Resolved rate | [tasks](https://github.com/harbor-framework/terminal-bench-2) · [dataset](https://hub.harborframework.com/datasets/terminal-bench/terminal-bench-2-1) · [runner](https://github.com/harbor-framework/harbor) |

The benchmark inputs, runners, and scoring paths are public. Every admitted
record still identifies the exact version, profile, model checkpoint, reasoning
effort, harness, tools, run conditions, date, and source. Agentic results compare
frozen model-and-agent systems, not model names in isolation.
For HLE, a published `no-tools` label does not establish that multimodal items
were excluded. Those records stay visible but only explicit runs over the frozen
2,158 text-only questions enter Intelligence 1.0.

## Score and missing data

Raw metrics are normalized to `[0, 1]` before aggregation:

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

Every category and Overall uses `require_all`:

- `available` means every child exists for the same model and reasoning effort;
- `partial` has some children, a null score, exact coverage, and a missing list;
- `missing` has no admitted child and a null score.

Scores are never imputed, set to zero, renormalized over reported components,
or borrowed from another checkpoint or reasoning effort. A partial Overall can
still contain an available category score. That model can appear in the matching
category or benchmark ranking and can serve a category-aware route.

## Model Arena

The Arena provides three views over the same data:

1. Overall Intelligence rank;
2. General, Reasoning, Coding, and Agentic ranks;
3. the six raw benchmark ranks.

The highest available reasoning-effort result is shown per model, with that
effort exposed in the row. Physical and virtual models use identical ranking
logic. URL parameters preserve the layer, selected capability or benchmark,
model scope, and selected model so every view is shareable.

## Routing quality evidence

Select the capability appropriate for a decision instead of forcing every route
to use Overall:

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    quality:
      index: vllm-sr/coding@1.0.0
      on_missing: exclude
    weights:
      quality: 0.4
      latency: 0.2
      cost: 0.2
      load: 0.2
```

`exclude` admits only candidates with an available exact-effort result.
`disable_quality` keeps the full candidate pool but, if any candidate lacks the
selected index, removes quality from the entire comparison. Operational SLOs,
latency, cost, and load continue to work; candidate-local weight changes do not.

`quality.min_coverage` can require more evidence than an index's own missing-data
policy, and `quality.min_score` is a hard quality floor. This lets a deployment
route on a deliberately partial operator index without weakening the complete-case
1.0 Overall contract. See [Custom evaluations](custom-evaluations)
for the YAML contract and [Multi Factor](../tutorials/algorithm/selection/multi-factor)
for Balanced, Accuracy-first, and Cost-first objectives.

## New measurements

Evaluate a MoM through its actual routed endpoint using [sr-bench](sr-bench).
A model-pool score or per-case oracle does not establish the quality of the live
routing policy. Preserve dataset, model, recipe/configuration and price identities
with measured results before publishing an operator-owned evaluation record.

The catalog graph remains defined by
[`config/catalog/resources/indices.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/catalog/resources/indices.yaml).
Changing its weights or benchmark contract requires a new versioned identity.
