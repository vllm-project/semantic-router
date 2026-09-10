---
title: Custom Evaluations
description: Add versioned benchmark definitions, model-linked records, and routing indices through one canonical evaluation section.
---

# Custom evaluations

Operator evaluation data has one canonical home:

```yaml
evaluation:
  benchmarks: [] # definitions not built into this release
  indices: []    # operator-owned index DAGs
  records: []    # measurements linked to Model Card identities
```

Built-in data uses the same logical model in the generated catalog snapshot:
benchmark definitions, index definitions, and model-linked evaluation records
are separate collections. Keeping measurements out of Model Cards avoids
duplicating identity metadata and lets one model have many benchmark runs,
profiles, and reasoning efforts.

The release-owned source is split by responsibility under `config/catalog`:
Model Cards in `resources/models/`, benchmark contracts in
`resources/benchmarks.yaml`, measurements in `resources/evaluations/`, and the
index DAG in `resources/indices.yaml`. Generation validates and embeds one
immutable snapshot. User `evaluation` data extends that snapshot for the local
deployment; it does not copy or override release-owned records.

## Add evidence for a custom model

For a custom model, omit `providers.models[].catalog`. The provider model name
is then its canonical Model Card identity and is also used by
`evaluation.records[].model`:

```yaml
version: v0.3

providers:
  models:
    - name: private-chat
      provider_model_id: private-chat-awq
      api_format: openai
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: model-gateway.example:8000
          protocol: http

evaluation:
  records:
    - model: private-chat
      benchmark: livecodebench/livecodebench@6.0.0
      benchmark_profile: independent-code-generation
      reasoning_effort: high
      metrics: {pass_at_1: 0.61}
      measured_at: 2026-09-09
      source: https://benchmarks.example/runs/private-chat-lcb6

routing:
  modelCards:
    - name: private-chat
      display_name: Private Chat
      capabilities: [chat, tools, coding]
```

For an alias backed by a built-in card, set `model` to the canonical
`providers.models[].catalog` ID, not the request-facing alias. Several aliases
for the same checkpoint therefore share the same model evidence.

The Evaluation Plane freezes the effective card behind each provider alias,
including built-in metadata and operator overrides. A catalog-backed live
target therefore does not require a duplicate `routing.modelCards` entry.

## Define a benchmark and index

Built-in benchmark IDs need no redeclaration. Define semantics only for an
operator-owned benchmark:

```yaml
evaluation:
  benchmarks:
    - id: acme/clinical-reasoning@1.0.0
      display_name: ACME Clinical Reasoning
      domain: medical_reasoning
      source: https://benchmarks.example/clinical-reasoning/v1
      default_profile: heldout
      profiles:
        - id: heldout
          display_name: Held-out set
          description: Frozen v1 cases with deterministic scoring.
      metrics:
        - id: accuracy
          unit: proportion
          direction: higher_is_better
          range: [0, 1]

  indices:
    - id: acme/clinical-quality@1.0.0
      display_name: Clinical Quality
      aggregation: weighted_mean
      scale: [0, 100]
      missing: {policy: require_coverage, minimum: 0.5}
      domains: {medical_reasoning: 0.5, scientific_reasoning: 0.5}
      components:
        - benchmark: acme/clinical-reasoning@1.0.0
          benchmark_profile: heldout
          metric: accuracy
          weight: 0.5
          normalization: {type: identity}
        - benchmark: idavidrein/gpqa-diamond@1.0.0
          benchmark_profiles: [independent-standard, published-standard]
          metric: accuracy
          weight: 0.5
          normalization: {type: identity}

  records:
    - model: private-chat
      benchmark: acme/clinical-reasoning@1.0.0
      benchmark_profile: heldout
      reasoning_effort: high
      metrics: {accuracy: 0.74}
```

Resource IDs must be lowercase, namespaced, and versioned. An operator
definition cannot shadow a built-in benchmark or index. Publish a new ID when
the tasks, profile, scorer, metric meaning, normalization, or weights change.

An undeclared namespaced benchmark record is retained and round-trips through
canonical export, but cannot enter an index until its metric range, direction,
and profiles are defined.

## Choose missing-data semantics

| Policy | Index becomes available when | Aggregation |
| --- | --- | --- |
| `require_all` | Coverage is 1.0 | All declared component weights |
| `require_coverage` | Coverage reaches `minimum` | Reported component weights |
| `reported_only` | Any component is present | Reported component weights |

No policy invents a missing score. Partial policies renormalize reported
weights only because the index definition explicitly requests it; coverage and
missing components remain visible.

## Route on the index

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

The Router uses an `available` result for the candidate's exact reasoning
effort, then applies route-level coverage and score gates. See
[Multi Factor](../tutorials/algorithm/selection/multi-factor) for balanced,
quality-first, and cost-first objectives.

In the Dashboard, open **Build → Models → Evaluation Records** to add or edit
records. Benchmark and index definitions remain explicit YAML because changing
their semantics creates a versioned scoring contract, not ordinary Model Card
metadata.
