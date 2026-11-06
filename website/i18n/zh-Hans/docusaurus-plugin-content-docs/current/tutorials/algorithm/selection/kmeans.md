---
is_mtpe: false
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/algorithm/selection/kmeans.md"
  outdated: false
---

# KMeans（已移除）

## 概览

`kmeans` 选择器已被移除，其他训练得到的选择器（`knn`、`svm`、`mlp`）、它们的 `global.router.model_selection.ml` 设置以及训练流水线也一并移除。请使用 [System One](./systemone) 选择器。

## 它解决什么问题？

它借助训练得到的工件，把请求发送给其最近的已学习流量簇所对应的模型。`systemone` 选择器向决策模型询问应由哪个候选作答，不需要训练工件。

## 何时使用

不要再配置它：仍然使用它的配置会在加载时失败。请使用 [System One](./systemone)。

## 配置

运行 `vllm-sr config migrate` 来改写它。有两个或更多候选的决策会变为按候选描述进行选择的 `systemone` 选择器；只有一个候选的决策会变为 `static`。迁移后的形式见 [K 近邻选择](./knn#配置)。
