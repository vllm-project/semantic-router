# Algorithm

## Overview

Latest algorithm tutorials mirror the fragment catalog under `config/algorithm/`.

Algorithms only matter after a decision matches and exposes multiple candidate models in `modelRefs`. The router then uses `decision.algorithm` to choose or coordinate those candidates.

## Key Advantages

- Separates route eligibility from model selection policy.
- Lets one decision keep several candidate models without inlining ranking logic.
- Supports both one-model ranking and multi-model orchestration.
- Mirrors the repo fragment tree exactly: one tutorial page per algorithm under `config/algorithm/selection/` and `config/algorithm/looper/`.

## What Problem Does It Solve?

Once a route matches, the router still needs a principled way to choose among candidate models. Without an algorithm layer, teams either hard-code one winner or duplicate ranking logic across routes.

Algorithms solve that by making the post-match selection policy explicit and reusable.

## When to Use

Use `algorithm/` when:

- `modelRefs` contains more than one candidate
- route policy depends on latency, feedback, semantic fit, or online exploration
- one decision should orchestrate several models instead of choosing exactly one
- you want model choice to evolve without changing the decision rule itself

## Configuration

In canonical v0.3 YAML, algorithms live inside each matched decision:

```yaml
routing:
  decisions:
    - name: computer-science-reasoning
      rules:
        operator: AND
        conditions:
          - type: domain
            name: "computer science"
      modelRefs:
        - model: qwen2.5:7b
        - model: qwen3:14b
      algorithm:
        type: router_dc
        router_dc:
          temperature: 0.07
```

The repo now keeps one tutorial page per algorithm.

### Selection Algorithms

- [Automix](./selection/automix)
- [Elo](./selection/elo)
- [GMT Router](./selection/gmtrouter)
- [Hybrid](./selection/hybrid)
- [KMeans](./selection/kmeans)
- [KNN](./selection/knn)
- [Latency Aware](./selection/latency-aware)
- [RL Driven](./selection/rl-driven)
- [Router DC](./selection/router-dc)
- [Static](./selection/static)
- [SVM](./selection/svm)

### Looper Algorithms

- [Confidence](./looper/confidence)
- [Ratings](./looper/ratings)
- [ReMoM](./looper/remom)
