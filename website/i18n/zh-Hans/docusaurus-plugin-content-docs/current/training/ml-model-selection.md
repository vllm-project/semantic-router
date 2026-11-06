---
is_mtpe: false
title: 基于 ML 的模型选择（已移除）
sidebar_label: ML 模型选择
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/training/ml-model-selection.md"
  outdated: false
---

# 基于 ML 的模型选择（已移除）

训练得到的模型选择器（`knn`、`kmeans`、`svm` 和 `mlp`）已被移除，它们的 `global.router.model_selection.ml` 设置、`src/training/model_selection/ml_model_selection` 中的训练流水线以及 Dashboard 的 ML 设置页面也一并移除。

[System One 选择器](../tutorials/algorithm/selection/systemone)取代了它们：它按候选的描述询问决策模型应由决策的哪个候选作答，不需要训练工件或嵌入模型。`vllm-sr config migrate` 会把使用训练选择器的决策改写为 `systemone`（只有一个候选时改写为 `static`），并删除 `ml` 块。

要离线比较提供商模型，请使用[提供商模型评估](./model-performance-eval)。要在修改路由策略之前用已记录的流量衡量它，请使用 `src/training/model_selection/policy_eval` 中的反事实评估器。
