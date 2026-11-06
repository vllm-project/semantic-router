---
is_mtpe: false
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/algorithm/selection/automix.md"
  outdated: false
---

# AutoMix（已移除）

## 概览

`automix` 选择器已被移除。它的成本与质量取舍是一种 [Multi Factor](./multi-factor) 加权。

## 它解决什么问题？

它在 `cost_quality_tradeoff` 的引导下，权衡估计质量与成本来选择候选。`multi_factor` 直接为每个候选的质量和成本打分；不再使用它学习得到的 POMDP 策略和自我校验。

## 何时使用

不要再配置它：仍然使用它的配置会在加载时失败。请使用 [Multi Factor](./multi-factor)。如果只想在回答不够好时才升级，请使用 [Escalate](../looper/escalate)。

## 配置

运行 `vllm-sr config migrate` 来改写它。取舍值 `t` 变为权重 `{quality: 1 - t, cost: t}`，取自决策自身的块，否则取自已移除的 `global.router.model_selection.automix` 块：

```yaml
algorithm:
  type: multi_factor
  multi_factor:
    weights: {quality: 0.7, cost: 0.3}
```
