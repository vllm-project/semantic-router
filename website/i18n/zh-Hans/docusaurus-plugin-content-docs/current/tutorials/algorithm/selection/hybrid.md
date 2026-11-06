---
is_mtpe: false
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/algorithm/selection/hybrid.md"
  outdated: false
---

# Hybrid（已移除）

## 概览

`hybrid` 选择器已被移除。它的质量与成本加权是一种 [Multi Factor](./multi-factor) 加权。

## 它解决什么问题？

它把反馈经验、Router-DC 相似度、AutoMix 价值和成本混合为一个候选分数。`multi_factor` 权衡质量和成本；经验权重和 Router-DC 权重没有替代项，会被丢弃。

## 何时使用

不要再配置它：仍然使用它的配置会在加载时失败。请使用 [Multi Factor](./multi-factor)，或使用 [System One](./systemone) 按候选的描述进行选择。

## 配置

运行 `vllm-sr config migrate` 来改写它。`automix_weight` 变为质量权重，`cost_weight` 变为成本权重，取自决策自身的块，否则取自已移除的 `global.router.model_selection.hybrid` 块；迁移会记录它丢弃的权重：

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    weights: {quality: 0.2, cost: 0.1}
```
