# GMT Router

## Overview

`gmtrouter` is a selection algorithm for personalized routing based on prior history and a learned model.

It aligns to `config/algorithm/selection/gmtrouter.yaml`.

## Key Advantages

- Supports per-user or per-tenant personalization.
- Keeps the personalization model path explicit in config.
- Lets one route opt into history-aware selection without affecting others.

## What Problem Does It Solve?

Some workloads need selection to reflect prior user interactions instead of global averages. `gmtrouter` adds that learned personalization layer to the matched decision.

## When to Use

- the route should adapt to user or tenant history
- you have a trained selector artifact to load
- static or feedback-only ranking is not enough for the workload

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: gmtrouter
  gmtrouter:
    enable_personalization: true
    history_sample_size: 50
    model_path: models/gmtrouter.pt
```
