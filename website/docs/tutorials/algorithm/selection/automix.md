# AutoMix (removed)

## Overview

The `automix` selector was removed. Its cost-quality tradeoff is a
[Multi Factor](./multi-factor) weighting.

## What Problem Does It Solve?

It chose a candidate by trading estimated quality against cost, steered by
`cost_quality_tradeoff`. `multi_factor` scores each candidate's quality and
cost directly; its learned POMDP policy and self-verification are not used.

## When to Use

Do not configure it: a configuration that still names it fails at load. Use
[Multi Factor](./multi-factor). To escalate only when an answer is not good
enough, use [Escalate](../looper/escalate).

## Configuration

Run `vllm-sr config migrate` to rewrite it. A tradeoff `t` becomes the weights
`{quality: 1 - t, cost: t}`, read from the decision's block, else from the
removed `global.router.model_selection.automix` block:

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    weights: {quality: 0.7, cost: 0.3}
```
