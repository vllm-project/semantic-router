# Selection Algorithms

## Overview

Selection algorithms choose one model from the matched `modelRefs`.

This page aligns to `config/algorithm/selection/`.

## Key Advantages

- Keeps winner selection explicit inside the route.
- Supports deterministic, semantic, latency-aware, and feedback-aware ranking.
- Lets one route keep multiple candidates without hard-coding one default winner.
- Makes algorithm upgrades local to one route or fragment family.

## What Problem Does It Solve?

A matched decision often has several viable models. Without a selection policy, the router either picks a fixed default or relies on hidden heuristics.

Selection algorithms solve that by declaring how the router should rank and choose one candidate.

## When to Use

Use `selection/` when:

- one route has multiple candidate models
- only one model should serve each request
- latency, feedback, semantic fit, or personalization should influence the winner
- you want a reusable selection fragment under `config/algorithm/selection/`

## Configuration

Available fragment families:

| Algorithm | Fragment | Best for |
|-----------|----------|----------|
| `static` | `config/algorithm/selection/static.yaml` | deterministic routing |
| `latency_aware` | `config/algorithm/selection/latency-aware.yaml` | fastest acceptable model |
| `elo` | `config/algorithm/selection/elo.yaml` | feedback-driven ranking |
| `router_dc` | `config/algorithm/selection/router-dc.yaml` | query-model semantic matching |
| `automix` | `config/algorithm/selection/automix.yaml` | cost-quality tradeoff |
| `hybrid` | `config/algorithm/selection/hybrid.yaml` | combine ranking signals |
| `gmtrouter` | `config/algorithm/selection/gmtrouter.yaml` | personalized ranking |
| `kmeans` | `config/algorithm/selection/kmeans.yaml` | cluster-based routing |
| `knn` | `config/algorithm/selection/knn.yaml` | nearest-neighbor lookup |
| `rl_driven` | `config/algorithm/selection/rl-driven.yaml` | online exploration plus personalization |
| `svm` | `config/algorithm/selection/svm.yaml` | classic classifier-driven selection |

Simplified canonical example:

```yaml
routing:
  decisions:
    - name: computer_science_route
      rules:
        operator: AND
        conditions:
          - type: domain
            name: "computer science"
      modelRefs:
        - model: qwen2.5:7b
        - model: qwen3:14b
      algorithm:
        type: latency_aware
        latency_aware:
          tpot_percentile: 90
          ttft_percentile: 95
```
