# Static

## Overview

`static` is the simplest selection algorithm: the route keeps its candidate list, and the selection policy stays fixed.

It aligns to `config/algorithm/selection/static.yaml`.

## Key Advantages

- Deterministic and easy to reason about.
- No learned selector state or runtime tuning.
- Good default when the candidate order is already intentional.

## What Problem Does It Solve?

Sometimes the route just needs a stable winner policy without extra scoring logic. `static` makes that explicit instead of leaving selection behavior implicit.

## When to Use

- one candidate should always win after the route matches
- model ordering is already curated outside the algorithm layer
- you want the simplest possible route-local selection policy

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: static
```
