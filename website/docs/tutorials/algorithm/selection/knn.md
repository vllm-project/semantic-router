# KNN (removed)

## Overview

The `knn` selector was removed, with the other trained selectors (`kmeans`,
`svm`, `mlp`), their `global.router.model_selection.ml` settings and their
training pipeline. Use the [System One](./systemone) selector.

## What Problem Does It Solve?

It followed the models that performed well on the most similar recorded
requests, using a trained artifact. The `systemone` selector asks the decision
model which candidate should answer and needs no trained artifact.

## When to Use

Do not configure it: a configuration that still names it fails at load. Use
[System One](./systemone).

## Configuration

Run `vllm-sr config migrate` to rewrite it. A decision with two or more
candidates becomes the `systemone` selector, which chooses by the candidates'
descriptions; a decision with one candidate becomes `static`. Describe each
candidate in its model card, or in `algorithm.systemone.candidates`:

```yaml
algorithm:
  type: systemone
  systemone:
    instructions: Which candidate model best fits this request?
```
