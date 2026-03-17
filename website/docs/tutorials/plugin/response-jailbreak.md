# Response Jailbreak

## Overview

`response_jailbreak` is a route-local plugin for screening the model response before it is returned.

It aligns to `config/plugin/response-jailbreak/strict.yaml`.

## Key Advantages

- Adds a final response-side jailbreak check for sensitive routes.
- Keeps the action policy explicit in config.
- Complements request-side safety without replacing it.

## What Problem Does It Solve?

Even if the request routed correctly, the generated answer may still need a final safety gate. `response_jailbreak` gives the route that explicit output-screening step.

## When to Use

- a route needs a final response-side jailbreak screen
- output should be blocked or annotated before returning
- request-side screening alone is not enough for the workload

## Configuration

Use this fragment under `routing.decisions[].plugins`:

```yaml
plugin:
  type: response_jailbreak
  configuration:
    enabled: true
    threshold: 0.85
    action: block
```
