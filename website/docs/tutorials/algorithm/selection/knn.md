# KNN

## Overview

`knn` is a selection algorithm for nearest-neighbor model selection.

It aligns to `config/algorithm/selection/knn.yaml`.

## Key Advantages

- Makes example-based routing explicit in config.
- Keeps the route-level policy compact.
- Works well when similar prompts should choose similar models.

## What Problem Does It Solve?

Some routing policies are easier to express as “pick the model that worked for similar prompts.” `knn` exposes that nearest-neighbor selector directly in the decision.

## When to Use

- you have historical prompt-to-model examples
- similar prompts should usually map to the same candidate model
- the route should use retrieval-style selection instead of fixed ranking

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: knn
```
