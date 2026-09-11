# Request Parameters

## Overview

`request_params` is a route-local plugin that validates and trims OpenAI Chat Completions request bodies before they are forwarded to backends.

## Key Advantages

- Caps expensive parameters (`max_tokens`, `n`) per route. `max_tokens_limit`
  is a completion-token ceiling: it injects when the client omitted a limit and
  otherwise composes with `modelRefs[].max_completion_tokens` and any Looper
  stage bound as strictest-wins.
- Blocks sensitive parameters such as `logprobs` / `top_logprobs` for tiers that should not expose token distributions.
- Optionally strips unknown top-level JSON fields to reduce surprise passthrough behavior.

## What Problem Does It Solve?

Model routing can restrict which backend serves a request, but clients can still amplify cost or extract logits via request parameters. This plugin enforces per-decision limits on the request body after a decision matches.

## When to Use

- a tier or route must not request logprobs or multiple completions
- you need a hard ceiling on `max_tokens` or `n` independent of client input.
  Pair `max_tokens_limit` with `routing.decisions[].modelRefs[].max_completion_tokens`
  when models should have different ceilings; dispatch keeps the strictest bound.
- unknown JSON fields should not be forwarded to backends

## Configuration

Add the plugin under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: request_params
    configuration:
      blocked_params:
        - logprobs
        - top_logprobs
      max_tokens_limit: 500
      max_n: 1
      strip_unknown: true
```

In DSL, the same plugin can appear as:

```dsl
PLUGIN request_params {
  blocked_params: ["logprobs", "top_logprobs"]
  max_tokens_limit: 500
  max_n: 1
  strip_unknown: true
}
```

This plugin enforces a bounded set of OpenAI Chat Completions fields. It is not
a general JSON-schema firewall and does not authorize a caller. `max_tokens_limit`
cannot raise a smaller client `max_completion_tokens` / `max_tokens` value, and
blocking those fields removes the client contribution before composition.
Test `strip_unknown` against clients that add provider-specific fields before
enabling it. See a complete example:
[`config/fragments/plugin/request-params/budget-tier.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/request-params/budget-tier.yaml).
