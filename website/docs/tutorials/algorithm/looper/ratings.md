# Ratings

## Overview

`ratings` is a looper algorithm that coordinates several candidates while reusing route-level ratings signals.

It aligns to `config/algorithm/looper/ratings.yaml`.

## Key Advantages

- Supports multi-model execution with a bounded concurrency cap.
- Keeps rating-aware orchestration local to one route.
- Makes error-handling behavior explicit.

## What Problem Does It Solve?

Some routes need more than one candidate to participate, but still need a controlled loop instead of an open-ended fan-out. `ratings` exposes that bounded multi-model coordination policy.

## When to Use

- more than one candidate should run inside the same route
- route-level ratings should influence the loop
- concurrency needs a hard upper bound

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: ratings
  ratings:
    max_concurrent: 3
    on_error: skip
```
