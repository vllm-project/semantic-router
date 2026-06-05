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

The runtime now consumes every retention field, with one scoring bias deferred:

- `drop: true` skips the response-side semantic-cache write for the matched
  decision. It does not block semantic-cache reads for the current request.
- `ttl_turns` (when `> 0`) overrides the decision/global semantic-cache TTL for
  this entry, scoping it to roughly that many future turns (turns are mapped to
  seconds at the cache-write seam; a configurable seconds-per-turn knob is
  follow-up work).
- `keep_current_model: true` forces the session-aware model-switch gate to stay
  on the current model, regardless of gate mode (shadow or enforce), as long as
  the current model is known and a valid candidate.
- `prefer_prefix_retention: true` is emitted to the inference pool as an
  `x-vsr-retention-prefer-prefix` response header; the session-aware scoring
  bias and provider/KV-cache eviction integration remain follow-up work.

Every explicitly set field is also emitted to the response as an
`x-vsr-retention-*` header (including an explicit `ttl_turns: 0`) and recorded
in logs/traces, so the pool and operators can audit the router's retention
intent. `drop` and positive `ttl_turns` are mutually exclusive (validation
rejects setting both).

## Retention Target Inventory

Besides the semantic-cache write, session-aware routing needs retention policy
for these state signals or runtime hints:

| Target | Why retention matters | Current status |
|--------|-----------------------|----------------|
| Semantic-cache response write | Prevents low-value, private, or unstable turns from becoming future cache hits. | Enforced by `drop: true`. |
| Cache-write lifetime | Keeps a reusable response only for a bounded number of future turns instead of using a wall-clock-only TTL. | `ttl_turns` overrides the per-entry semantic-cache TTL (turns mapped to seconds); a configurable seconds-per-turn knob is follow-up. |
| Current model affinity | Avoids bouncing a multi-turn session away from the model that owns the conversation context. | `keep_current_model` forces a stay via the model-switch gate in any mode. |
| Prefix or KV cache warmth | Protects expensive prompt prefixes or warm worker state when a follow-up turn is likely. | `prefer_prefix_retention` is emitted to the pool as a response header; scoring bias and provider/cache-manager eviction integration are follow-up work. |
| Turn and transition telemetry | Records turn index, selected model, token/cost totals, retry/quality trends, and model transitions so stay-vs-switch policy can be audited. | Produced by session telemetry and transition logging surfaces, not by this directive alone. |
| Conversation, tool, and replay history | Preserves the history needed for follow-up classification, tool retrieval, and offline lookup-table generation. | Owned by Response API, tool-history, and router-replay surfaces; retention directives should not duplicate those stores. |

That inventory is why the directive is named `retention` instead of
`semantic_cache`: semantic-cache write skipping is only the first runtime
consumer of a broader session-retention contract.

## DSL Round-Trip Scope

The example below uses `DECISION_TREE` for readability, but that syntax is an
authoring convenience. Compiled config stores flat `routing.decisions`, and
config-backed export/decompile paths emit flat `ROUTE` blocks rather than
reconstructing the original tree. Retention fields still round-trip through
DSL/config; the tree shape itself does not.

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
| `ttl_turns` | integer >= 0 | When `> 0`, overrides the matched decision's semantic-cache entry TTL (turns mapped to seconds). Emitted as `x-vsr-retention-ttl-turns` whenever explicitly set, including an explicit `0`. |
| `keep_current_model` | boolean | When `true`, forces the model-switch gate to keep the current model regardless of gate mode. Also emitted as `x-vsr-retention-keep-current-model`. |
| `prefer_prefix_retention` | boolean | Emitted to the pool as `x-vsr-retention-prefer-prefix`; session-aware scoring bias and KV-cache eviction integration are follow-up. |

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
