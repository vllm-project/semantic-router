---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/hybrid.md"
  outdated: true
---

# Hybrid

## 概览

`hybrid` 将多种排序信号合成为单一加权分数的选择算法。

对应 `config/algorithm/selection/hybrid.yaml`。

## 主要优势

- 混合多种选择器，而非只选一种。
- 权重显式、易审计。
- 支持在排序策略间渐进迁移。

## 解决什么问题？

单一选择器并不总够用。`hybrid` 让一条路由组合反馈、语义匹配与成本感知路由，而无需在别处重复组合逻辑。

## 何时使用

- 一条路由应组合多种排序信号
- 希望在旧选择器与新选择器间加权过渡
- 最终选择应同时反映质量与运维成本

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: hybrid
  hybrid:
    elo_weight: 0.4
    router_dc_weight: 0.4
    automix_weight: 0.2
    cost_weight: 0.1
```
