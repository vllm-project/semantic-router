# MLP (removed)

## Overview

The `mlp` selector was removed, with the other trained selectors (`knn`,
`kmeans`, `svm`), their `global.router.model_selection.ml` settings and their
training pipeline. Use the [System One](./systemone) selector.

## What Problem Does It Solve?

It mapped a request to a candidate with a trained neural classifier. The
`systemone` selector asks the decision model which candidate should answer and
needs no trained artifact.

## When to Use

Do not configure it: a configuration that still names it fails at load. Use
[System One](./systemone).

## Configuration

Run `vllm-sr config migrate` to rewrite it. A decision with two or more
candidates becomes the `systemone` selector, which chooses by the candidates'
descriptions; a decision with one candidate becomes `static`. See
[KNN](./knn#configuration) for the migrated form.
