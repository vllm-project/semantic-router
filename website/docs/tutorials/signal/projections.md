---
sidebar_position: 2
---

# Projections

`routing.projections` is the post-signal coordination layer for routing.

Use it when you need one of these behaviors:

- enforce a winner inside a declared domain or embedding partition
- combine learned and heuristic signals into one continuous score
- map that score into named routing bands that decisions can reference with `type: projection`

The repo keeps authoring and runtime naming intentionally asymmetric:

- DSL authoring uses `SIGNAL_GROUP`, `PROJECTION score`, and `PROJECTION mapping`
- canonical runtime config stores the same contract under `routing.projections.partitions`, `routing.projections.scores`, and `routing.projections.mappings`

## Workflow

1. Define reusable detectors under `routing.signals`.
2. Use `routing.projections.partitions` when one domain or embedding family should resolve to one winner.
3. Use `routing.projections.scores` to combine signal evidence into a weighted score.
4. Use `routing.projections.mappings` to turn that score into named routing bands.
5. Reference those bands from `routing.decisions[*].rules.conditions[*]` with `type: projection`.

## Canonical YAML

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

  decisions:
    - name: support_route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: embedding
            name: technical_support
          - type: projection
            name: support_escalated
```

## DSL

```dsl
SIGNAL embedding technical_support {
  threshold: 0.75
  candidates: ["installation guide", "troubleshooting"]
}

SIGNAL embedding account_management {
  threshold: 0.72
  candidates: ["billing issue", "subscription change"]
}

SIGNAL context long_context {
  min_tokens: "4000"
  max_tokens: "200000"
}

SIGNAL_GROUP support_intents {
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

ROUTE support_route {
  PRIORITY 100
  WHEN embedding("technical_support") AND projection("support_escalated")
  MODEL "qwen3-8b"
}
```

## Dashboard

The dashboard now exposes the whole projection contract directly:

- `Config -> Projections` manages partitions, scores, and mappings in canonical config form
- `Config -> Decisions` can reference mapping outputs with condition type `projection`
- `DSL -> Visual` shows `Signal Groups`, `Projection Scores`, and `Projection Mappings` as editable entities alongside signals, routes, models, and plugins

For raw import/export, the DSL page still decompiles the current router YAML into routing-only DSL and recompiles the edited DSL back into canonical YAML.

## Maintained Example

Use the maintained `balance` pair as the repo-native end-to-end example:

- [`deploy/recipes/balance.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.yaml)
- [`deploy/recipes/balance.dsl`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.dsl)

That recipe shows the intended pattern:

- `partitions` to resolve domain and intent winners
- `scores` to blend heuristic and learned signals
- `mappings` to emit named bands such as `balance_simple` and `verification_required`
- decisions that combine ordinary signals with `type: projection`
