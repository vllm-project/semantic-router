# SVM

## Overview

`svm` is a selection algorithm for routes that use an SVM-based selector.

It aligns to `config/algorithm/selection/svm.yaml`.

## Key Advantages

- Exposes classic classifier-based selection in the same algorithm surface.
- Keeps the route config small when the selector logic lives in the model.
- Useful when a lightweight learned classifier is enough.

## What Problem Does It Solve?

For some workloads, a classic classifier is simpler to maintain than a larger learned selector or an online exploration policy. `svm` makes that classifier-backed route choice explicit.

## When to Use

- you have an SVM-based selector artifact for the route
- lightweight learned classification is enough for model choice
- you want learned selection without additional orchestration

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: svm
```
