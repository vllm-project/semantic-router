# Retention Directives

## Overview

A retention directive adds bounded state-handling instructions to a matched
decision. It can skip a response-cache write, shorten that write's lifetime,
keep the current model for a session, or emit a prefix-retention hint.

Retention is an `emits` directive, not a signal, algorithm, or plugin.

## Key Advantages

- Keeps state-handling intent next to the route that produces it.
- Uses typed, bounded fields instead of ad hoc headers.
- Makes cache and session side effects visible in routing diagnostics.

## What Problem Does It Solve?

Some responses are private, short-lived, or valuable to a continuing session.
Keeping that policy beside the decision makes it reviewable and prevents broad
cache or affinity defaults from applying blindly.

## When to Use

Use retention when a matched route needs one of these side effects:

- do not write this response to the response cache
- apply a shorter cache lifetime for this response
- preserve the current model during a known session
- tell the inference pool that prompt-prefix retention is preferred

Do not use retention to match traffic, authorize a caller, redact content, or
configure a storage backend.

## Configuration

```yaml
routing:
  decisions:
    - name: sensitive-turn
      description: Drop retained state for sensitive turns.
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: pii
            name: restricted_pii
      modelRefs:
        - model: private-model
      emits:
        - kind: retention
          retention:
            drop: true
```

The equivalent DSL is:

```dsl
ROUTE sensitive-turn {
  PRIORITY 200
  WHEN pii("restricted_pii")
  MODEL "private-model"
  EMIT retention {
    drop: true
  }
}
```

## Fields and Runtime Effect

| Field | Effect | Important limit |
|---|---|---|
| `drop: true` | Skips the response-cache write for the matched decision | It does not prevent a cache read earlier in the same request |
| `ttl_turns` | Overrides the response-cache entry lifetime using the Router's turn-to-time mapping | It is a cache TTL hint, not durable conversation retention |
| `keep_current_model: true` | Forces the model-switch gate to stay on the known current model | It has no effect when session/current-model identity is unavailable |
| `prefer_prefix_retention: true` | Emits `x-vsr-retention-prefer-prefix` for the inference pool | The Router does not itself manage a provider's KV-cache eviction |

`drop: true` and a positive `ttl_turns` cannot be set together. A decision can
emit only one retention directive. Explicit values are exposed through bounded
`x-vsr-retention-*` response headers and routing diagnostics.

## Data and Security

The directive carries policy metadata, not prompt or response content. The
cache, session, replay, and provider surfaces it influences still need their
own authentication, encryption, tenant isolation, and retention settings.

The canonical schema is defined by
[`RetentionDirective`](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/config/decision_config.go).
