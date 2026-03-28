# KMeans

## Overview

`kmeans` is a selection algorithm for routes that rely on a cluster-based selector.

It aligns to `config/algorithm/selection/kmeans.yaml`.

## Key Advantages

- Keeps cluster-based selection separate from decision rules.
- Minimal route config when clustering is handled by the selector itself.
- Works well when workloads naturally group into prompt clusters.

## What Problem Does It Solve?

When prompt traffic falls into a few recurring clusters, a cluster-based selector can choose the best model more cleanly than a hand-written priority rule. `kmeans` exposes that selector at the route level.

## When to Use

- you have a cluster-based selector for candidate models
- prompt traffic naturally groups into repeatable classes
- the route should use learned clusters instead of static ranking

## Configuration

Use this fragment inside `routing.decisions[].algorithm`:

```yaml
algorithm:
  type: kmeans
```
