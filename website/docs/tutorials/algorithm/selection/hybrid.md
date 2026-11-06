# Hybrid (removed)

## Overview

The `hybrid` selector was removed. Its quality and cost weighting is a
[Multi Factor](./multi-factor) weighting.

## What Problem Does It Solve?

It blended feedback experience, Router-DC similarity, AutoMix value and cost
into one candidate score. `multi_factor` weighs quality and cost; the
experience and Router-DC weights have no replacement and are dropped.

## When to Use

Do not configure it: a configuration that still names it fails at load. Use
[Multi Factor](./multi-factor), or [System One](./systemone) to choose by the
candidates' descriptions.

## Configuration

Run `vllm-sr config migrate` to rewrite it. `automix_weight` becomes the
quality weight and `cost_weight` the cost weight, read from the decision's
block, else from the removed `global.router.model_selection.hybrid` block; the
migration notes the weights it dropped:

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    weights: {quality: 0.2, cost: 0.1}
```
