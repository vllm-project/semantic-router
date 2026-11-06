---
title: ML-Based Model Selection (removed)
sidebar_label: ML Model Selection
---

# ML-Based Model Selection (removed)

The trained model selectors (`knn`, `kmeans`, `svm` and `mlp`) were removed,
together with their `global.router.model_selection.ml` settings, the training
pipeline in `src/training/model_selection/ml_model_selection`, and the
Dashboard's ML setup pages.

The [System One selector](../tutorials/algorithm/selection/systemone) replaces
them: it asks the decision model which of a decision's candidates should answer,
by the candidates' descriptions, and needs no trained artifact or embedding
model. `vllm-sr config migrate` rewrites a decision that names a trained
selector as `systemone`, or as `static` when the decision has one candidate,
and removes the `ml` block.

To compare provider models offline, use
[provider model evaluation](./model-performance-eval). To measure a routing
policy against recorded traffic before changing it, use the counterfactual
evaluator in `src/training/model_selection/policy_eval`.
