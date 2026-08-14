# Request Parameters

## Overview

`request_params` is a route-local plugin that validates and trims OpenAI Chat Completions request bodies before they are forwarded to backends.

It aligns to `config/fragments/plugin/request-params/budget-tier.yaml`.

## Key Advantages

- Caps expensive parameters (`max_tokens`, `n`) per route.
- Blocks sensitive parameters such as `logprobs` / `top_logprobs` for tiers that should not expose token distributions.
- Optionally strips unknown top-level JSON fields to reduce surprise passthrough behavior.

## What Problem Does It Solve?

Model routing can restrict which backend serves a request, but clients can still amplify cost or extract logits via request parameters. This plugin enforces per-decision limits on the request body after a decision matches.

## When to Use

- a tier or route must not request logprobs or multiple completions
- you need a hard ceiling on `max_tokens` or `n` independent of client input
- unknown JSON fields should not be forwarded to backends

## Configuration

Use this fragment under `routing.decisions[].plugins` (list of plugin entries):

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
a general JSON-schema firewall and does not authorize a caller. Test
`strip_unknown` against clients that add provider-specific fields before
enabling it. Maintained example:
[`config/fragments/plugin/request-params/budget-tier.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/request-params/budget-tier.yaml).
