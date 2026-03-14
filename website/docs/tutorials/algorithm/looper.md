# Looper Algorithms

## Overview

Looper algorithms coordinate more than one model after a decision matches.

This page aligns to `config/algorithm/looper/`.

## Key Advantages

- Supports escalation instead of a single winner-take-all choice.
- Lets one route coordinate several candidate models deliberately.
- Works well for confidence-based fallback and multi-pass reasoning.
- Keeps orchestration policy separate from the decision rule itself.

## What Problem Does It Solve?

Some routes should not stop at one model choice. They need escalation, voting, ratings reuse, or multi-model coordination.

Looper algorithms solve that by giving the route an execution policy for multiple candidates instead of a simple one-model selection.

## When to Use

Use `looper/` when:

- a route should escalate from smaller to larger models
- more than one model should participate in one execution path
- confidence or ratings should determine whether to continue
- you want orchestration fragments under `config/algorithm/looper/`

## Configuration

Available fragment families:

| Algorithm | Fragment | Best for |
|-----------|----------|----------|
| `confidence` | `config/algorithm/looper/confidence.yaml` | escalate until confidence is high enough |
| `ratings` | `config/algorithm/looper/ratings.yaml` | reuse rating signals in a loop |
| `remom` | `config/algorithm/looper/remom.yaml` | breadth-controlled multi-model execution |

Simplified canonical example:

```yaml
routing:
  decisions:
    - name: escalation_route
      modelRefs:
        - model: qwen2.5:3b
        - model: qwen3:14b
        - model: deepseek-r1:32b
      algorithm:
        type: confidence
        confidence:
          threshold: 0.72
          escalation_order: small_to_large
```
