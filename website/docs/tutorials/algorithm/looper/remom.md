# ReMoM

## Overview

`remom` is a looper algorithm for breadth-controlled multi-model orchestration.

It aligns to `config/algorithm/looper/remom.yaml`.

## Key Advantages

- Coordinates several candidate models over a scheduled breadth pattern.
- Keeps intermediate-response behavior explicit.
- Useful for routes that need richer orchestration than simple escalation.

## What Problem Does It Solve?

Some routes should not only escalate, but also control how many models participate at each stage. `remom` gives the route that breadth schedule directly in config.

## When to Use

- one route should coordinate multiple models over several passes
- you need a configurable breadth schedule instead of one-step escalation
- intermediate responses should be included or excluded explicitly

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: remom
  remom:
    breadth_schedule: [3, 2, 1]
    model_distribution: round_robin
    temperature: 0.7
    max_concurrent: 3
    include_intermediate_responses: false
    on_error: skip
```
