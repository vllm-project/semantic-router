---
title: Custom Evaluation Catalogs
sidebar_label: Custom Evaluations
description: Define versioned benchmark semantics, attach Model Card evidence, and route on operator-owned indices.
---

# Custom evaluation catalogs

Use `evaluation_catalog` when a deployment needs a benchmark or capability index
that is not built into the release. Definitions live at the top level;
measurements stay with the model under `routing.modelCards[].evaluations`.

## Define benchmark semantics and an index

The example combines one operator benchmark with the built-in GPQA Diamond leaf.
Its `require_coverage` policy allows a score after half of the component weight is
present, while preserving the exact coverage on every result.

```yaml
evaluation_catalog:
  benchmarks:
    - id: acme/clinical-reasoning@1.0.0
      display_name: ACME Clinical Reasoning
      domain: medical_reasoning
      source: https://benchmarks.example/clinical-reasoning/v1
      default_profile: heldout
      profiles:
        - id: heldout
          display_name: Held-out set
          description: Frozen v1 held-out cases with deterministic scoring.
      metrics:
        - id: accuracy
          unit: proportion
          direction: higher_is_better
          range: [0, 1]

  indices:
    - id: acme/clinical-quality@1.0.0
      display_name: Clinical Quality
      description: Clinical and scientific reasoning for ACME requests.
      aggregation: weighted_mean
      scale: [0, 100]
      missing:
        policy: require_coverage
        minimum: 0.5
      domains:
        medical_reasoning: 0.5
        scientific_reasoning: 0.5
      components:
        - benchmark: acme/clinical-reasoning@1.0.0
          metric: accuracy
          benchmark_profile: heldout
          weight: 0.5
          normalization: {type: identity}
        - benchmark: idavidrein/gpqa-diamond@1.0.0
          metric: accuracy
          benchmark_profiles: [independent-standard, published-standard]
          weight: 0.5
          normalization: {type: identity}
```

IDs must be lowercase, namespaced, and versioned, such as
`organization/resource@1.0.0`. An operator definition cannot replace a built-in
benchmark or index. Publish a new ID when tasks, profile, scorer, metric meaning,
normalization, or component weights change.

## Attach model evidence

The Model Card name must match the alias under `providers.models`:

```yaml
providers:
  models:
    - name: private-chat
      provider_model_id: private-chat-awq
      api_format: openai
      pricing:
        currency: USD
        prompt_per_1m: 0.20
        completion_per_1m: 0.80
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: model-gateway.example:8000
          protocol: http

routing:
  modelCards:
    - name: private-chat
      evaluations:
        - benchmark: acme/clinical-reasoning@1.0.0
          benchmark_profile: heldout
          reasoning_effort: high
          metrics: {accuracy: 0.74}
          measured_at: 2026-09-09
          source: https://benchmarks.example/runs/private-chat-clinical-v1
```

Built-in benchmark evidence uses the same measurement shape and does not need a
definition under `evaluation_catalog`. An undeclared benchmark result is retained
on the Model Card but is not indexable because its metric direction, range, and
profile semantics are unknown.

## Choose missing-data semantics

| Policy | Index becomes available when | Aggregation |
| --- | --- | --- |
| `require_all` | Coverage is 1.0 | All declared component weights |
| `require_coverage` | Coverage reaches `minimum` | Reported component weights |
| `reported_only` | Any component is present | Reported component weights |

No policy invents a missing score. `require_coverage` and `reported_only`
renormalize only because the operator explicitly chose that index contract;
coverage and missing components remain visible.

## Route on the custom index

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    quality:
      index: acme/clinical-quality@1.0.0
      on_missing: exclude
      min_coverage: 0.5
      min_score: 60
    objective:
      strategy: lexicographic
      priorities:
        - {factor: quality, tolerance: 0.03}
        - {factor: cost, tolerance: 0.05}
```

The Router reads only an `available` result for the candidate's exact reasoning
effort, then applies the route's coverage and score gates. See
[Multi Factor](../tutorials/algorithm/selection/multi-factor) for objective and
fallback behavior.
