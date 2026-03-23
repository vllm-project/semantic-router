---
sidebar_position: 1
---

# Projections

## Overview

`routing.projections` is the coordination layer between raw signal detection and final decision matching.

Signals answer "what matched?" Decisions answer "which route should win?" Projections fill the gap in between: "how should several signal results be coordinated into reusable routing facts?"

Use it when you need one of these behaviors:

- resolve one winner inside a competing domain or embedding lane
- combine several weak signals into one continuous routing score
- centralize threshold policy once and reuse the named result across many decisions

## What Problem Does It Solve?

Signals are intentionally narrow. A keyword rule, embedding rule, domain classifier, or context detector tells the router one local fact about the request. Decisions are intentionally boolean. They combine those facts into route selection rules.

That leaves a practical gap:

- several domain or embedding signals may match at the same time, but the route only wants one winner
- many weak signals may collectively mean "this request is hard" or "this answer needs verification"
- the same threshold story may need to be reused by many decisions without copying numeric logic everywhere

Without projections, that coordination logic gets pushed into decisions, duplicated across routes, and mixed back into the detector layer. `routing.projections` keeps that coordination as its own explicit layer.

## Relationship to Signals and Decisions

Think of the routing pipeline in three layers:

1. `routing.signals` extracts reusable facts from the request.
2. `routing.projections` coordinates or aggregates those facts.
3. `routing.decisions` matches boolean policy rules to choose a route.

More concretely:

- `partitions` coordinate existing `domain` or `embedding` matches and keep one winner
- `scores` aggregate matched signals into one numeric value
- `mappings` turn that numeric value into named projection outputs
- decisions continue to reference raw signals with their native types such as `domain`, `embedding`, or `keyword`
- decisions reference projection outputs only through `type: projection`

Two important boundaries from the current implementation:

- decisions do not reference partition names directly
- decisions do not reference score names directly

Only `mapping.outputs[*].name` becomes a decision-visible `projection(...)` target.

## Runtime Flow

In the current runtime, projections happen after signal extraction and before decision evaluation:

1. base signals run under `routing.signals`
2. `routing.projections.partitions` reduce competing `domain` or `embedding` matches
3. `routing.projections.scores` compute numeric values from matched signals and confidences
4. `routing.projections.mappings` emit named outputs such as `balance_reasoning` or `verification_required`
5. decisions combine raw signals plus those named outputs

That is why partitions feel "closer to signals", while mappings feel "closer to decisions".

## Current Contract

The repo uses one projection-first naming story across authoring and runtime:

- DSL authoring uses `PROJECTION partition`, `PROJECTION score`, and `PROJECTION mapping`
- canonical runtime config stores the same contract under `routing.projections.partitions`, `routing.projections.scores`, and `routing.projections.mappings`

The current implementation supports:

- partitions with `exclusive` or `softmax_exclusive`
- scores with `method: weighted_sum`
- mappings with `method: threshold_bands`
- optional mapping calibration with `method: sigmoid_distance`

## Workflow

1. Define reusable detectors under `routing.signals`.
2. Use [Partitions](./partitions) when one domain or embedding family should resolve to one winner before decisions read it.
3. Use [Scores](./scores) to combine matched signals into a weighted score.
4. Use [Mappings](./mappings) to turn that score into named routing bands.
5. Reference those bands from `routing.decisions[*].rules.conditions[*]` with `type: projection`.

## Balance Recipe Example

The maintained [`deploy/recipes/balance.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.yaml) recipe shows the intended pattern:

- `balance_domain_partition` resolves one winning domain across the maintained routing domains
- `balance_intent_partition` resolves one winning embedding intent lane
- `difficulty_score` blends context, structure, keyword, embedding, and complexity evidence
- `difficulty_band` converts that score into `balance_simple`, `balance_medium`, `balance_complex`, and `balance_reasoning`
- `verification_pressure` plus `verification_band` produce reusable verification outputs such as `verification_required`
- decisions such as `premium_legal` or `reasoning_math` combine raw `domain` matches with projection outputs

That recipe is important because it shows projections being reused across many routes, not just one toy example.

## Canonical Shape

```yaml
routing:
  signals:
    embeddings:
      - name: technical_support
        threshold: 0.75
        candidates: ["installation guide", "troubleshooting"]
      - name: account_management
        threshold: 0.72
        candidates: ["billing issue", "subscription change"]
    context:
      - name: long_context
        min_tokens: "4000"
        max_tokens: "200000"

  projections:
    partitions:
      - name: support_intents
        semantics: exclusive
        members: [technical_support, account_management]
        default: technical_support

    scores:
      - name: request_difficulty
        method: weighted_sum
        inputs:
          - type: embedding
            name: technical_support
            weight: 0.18
            value_source: confidence
          - type: context
            name: long_context
            weight: 0.18

    mappings:
      - name: request_band
        source: request_difficulty
        method: threshold_bands
        outputs:
          - name: support_fast
            lt: 0.25
          - name: support_escalated
            gte: 0.25
```

```dsl
PROJECTION partition support_intents {
  semantics: "exclusive"
  members: ["technical_support", "account_management"]
  default: "technical_support"
}

PROJECTION score request_difficulty {
  method: "weighted_sum"
  inputs: [
    { type: "embedding", name: "technical_support", weight: 0.18, value_source: "confidence" },
    { type: "context", name: "long_context", weight: 0.18 }
  ]
}

PROJECTION mapping request_band {
  source: "request_difficulty"
  method: "threshold_bands"
  outputs: [
    { name: "support_fast", lt: 0.25 },
    { name: "support_escalated", gte: 0.25 }
  ]
}
```

## When to Use

Use projections when:

- competing domain or embedding lanes should collapse to one winner
- route difficulty or verification pressure is spread across several weak signals
- multiple decisions should share the same tiering logic such as `simple / medium / complex`
- you want threshold policy in one place instead of copy-pasting it across route rules

## When Not to Use

Skip projections when:

- one raw signal already expresses the route condition clearly
- multiple matches should remain independently visible to decisions
- a decision can stay readable with ordinary boolean composition and does not need shared weighted logic

## Dashboard

The dashboard exposes the same projection contract directly:

- `Config -> Projections` manages partitions, scores, and mappings in canonical config form
- `Config -> Decisions` can reference mapping outputs with condition type `projection`
- `DSL -> Visual` shows `Projection Partitions`, `Projection Scores`, and `Projection Mappings` as editable entities alongside signals, routes, models, and plugins

For raw import/export, the DSL page still decompiles the current router YAML into routing-only DSL and recompiles the edited DSL back into canonical YAML.

## Next Steps

- Read [Partitions](./partitions) for exclusive domain or embedding winner selection.
- Read [Scores](./scores) for weighted aggregation over matched signals.
- Read [Mappings](./mappings) for named routing bands and `type: projection` decision references.
