# Retention Directives

## Overview

Use `EMIT retention` when a matched decision must also tell response-side
runtime surfaces what state should be kept, dropped, or treated as valuable for
future turns.

Retention is not another detector. Signals decide whether a route matched;
retention directives describe side effects after the route matched.

## Key Advantages

- Keeps retention policy next to the decision that produced it.
- Avoids hiding cache or session-retention behavior in extproc branches.
- Gives operators one typed contract for keep, drop, TTL, and prefix-retention
  hints.
- Lets the runtime consume safe parts immediately while preserving the rest as
  auditable log and trace attributes.

## What Problem Does It Solve?

Session-aware routing needs to decide more than the next model. It also needs to
control what evidence survives into later turns. Without an explicit directive,
cache writes, model-affinity hints, and prefix-cache preferences can drift into
separate runtime heuristics that are hard to audit.

`EMIT retention` solves that by making the decision's retention side effects
structured and reviewable.

## Runtime Scope

The current runtime consumer is intentionally narrow:

- `drop: true` skips the response-side semantic-cache write for the matched
  decision.
- It does not block semantic-cache reads for the current request.
- `ttl_turns`, `keep_current_model`, and `prefer_prefix_retention` are
  contract-only in this release: they round-trip through DSL/config and are
  emitted to logs/traces, but their business effects are reserved for follow-up
  runtime consumers.

## Retention Target Inventory

Besides the semantic-cache write, session-aware routing needs retention policy
for these state signals or runtime hints:

| Target | Why retention matters | Current status |
|--------|-----------------------|----------------|
| Semantic-cache response write | Prevents low-value, private, or unstable turns from becoming future cache hits. | Enforced by `drop: true`. |
| Cache-write lifetime | Keeps a reusable response only for a bounded number of future turns instead of using a wall-clock-only TTL. | `ttl_turns` is typed and observed; runtime consumption needs a turn-to-TTL mapping. |
| Current model affinity | Avoids bouncing a multi-turn session away from the model that owns the conversation context. | `keep_current_model` is typed and observed; the model-switch gate consumes this in a follow-up path. |
| Prefix or KV cache warmth | Protects expensive prompt prefixes or warm worker state when a follow-up turn is likely. | `prefer_prefix_retention` is typed and observed; provider/cache-manager eviction integration is follow-up work. |
| Turn and transition telemetry | Records turn index, selected model, token/cost totals, retry/quality trends, and model transitions so stay-vs-switch policy can be audited. | Produced by session telemetry and transition logging surfaces, not by this directive alone. |
| Conversation, tool, and replay history | Preserves the history needed for follow-up classification, tool retrieval, and offline lookup-table generation. | Owned by Response API, tool-history, and router-replay surfaces; retention directives should not duplicate those stores. |

That inventory is why the directive is named `retention` instead of
`semantic_cache`: semantic-cache write skipping is only the first runtime
consumer of a broader session-retention contract.

## DSL Example

```dsl
DECISION_TREE session_routing {
  IF pii("sensitive") {
    NAME "sensitive-turn"
    TIER 2
    MODEL "qwen3-8b" (reasoning = true)
    EMIT retention {
      drop: true
    }
  }
  ELSE IF conversation("follow_up") {
    NAME "follow-up-continuity"
    TIER 3
    MODEL "qwen3-32b" (reasoning = true)
    EMIT retention {
      keep_current_model: true
      prefer_prefix_retention: true
    }
  }
  ELSE {
    NAME "default-route"
    TIER 1
    MODEL "qwen3-8b" (reasoning = false)
  }
}
```

## Configuration

The compiled config stores retention directives under the matched decision:

```yaml
routing:
  decisions:
    - name: sensitive-turn
      rules:
        operator: AND
        conditions:
          - type: pii
            name: sensitive
      modelRefs:
        - model: qwen3-8b
          use_reasoning: true
      emits:
        - kind: retention
          retention:
            drop: true
```

## Field Reference

| Field | Type | Runtime behavior |
|-------|------|------------------|
| `drop` | boolean | When `true`, skips response-side semantic-cache writes for the matched decision. |
| `ttl_turns` | integer >= 0 | Contract-only and observed in logs/traces until turn-aware TTL consumption lands. |
| `keep_current_model` | boolean | Contract-only and observed until model-switch-gate consumption lands. |
| `prefer_prefix_retention` | boolean | Contract-only and observed until prefix/KV-cache retention integration lands. |

Validation rejects duplicate `EMIT retention` blocks on the same route, unknown
fields, invalid field types, negative `ttl_turns`, and contradictory
`drop: true` plus positive `ttl_turns`.

## When to Use

Use retention directives when:

- a route should answer normally but should not write the response to semantic
  cache, such as PII, secrets, or one-off personalized context
- a follow-up-heavy route should preserve continuity hints for later turns
- a decision needs auditable retention metadata before all runtime consumers are
  enabled

Do not use `EMIT retention` to replace route conditions, plugin configuration,
or provider-specific cache APIs.
