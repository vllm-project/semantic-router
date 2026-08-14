# Response Jailbreak

## Overview

`response_jailbreak` is a route-local plugin for screening the model response before it is returned.

It aligns to `config/fragments/plugin/response-jailbreak/strict.yaml`.

## Key Advantages

- Adds a final response-side jailbreak check for sensitive routes.
- Keeps the action policy explicit in config.
- Complements request-side safety without replacing it.

## What Problem Does It Solve?

Even if the request routed correctly, the generated answer may still need a final safety gate. `response_jailbreak` gives the route that explicit output-screening step.

## When to Use

- a route needs a final response-side jailbreak screen
- output should be blocked or flagged via response headers before returning
- request-side screening alone is not enough for the workload

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugins:
  - type: response_jailbreak
    configuration:
      enabled: true
      threshold: 0.85
      action: block
```

This plugin processes generated response text with the configured prompt-guard
runtime. It adds latency and can produce false positives, so calibrate the
threshold and choose `block` versus header-only handling according to policy.
Maintained example:
[`config/fragments/plugin/response-jailbreak/strict.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-jailbreak/strict.yaml).
