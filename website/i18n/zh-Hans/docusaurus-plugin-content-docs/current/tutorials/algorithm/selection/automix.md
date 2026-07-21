---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/automix.md"
  outdated: true
---

# Automix

## 概览

`automix` 用于在**核验质量、升级深度与成本**之间权衡的路由选择算法。

对应 `config/algorithm/selection/automix.yaml`。

## 主要优势

- 将成本与质量策略直接编码在路由中。
- 支持有界升级而非无限重试。
- 将核验行为局部在需要它的决策中。

## 解决什么问题？

部分路由应先偏好更便宜候选，但在核验置信度过低时仍应升级。`automix` 使该策略显式，而非硬编码在应用逻辑中。

## 何时使用

- 一条路由有多个成本与质量画像不同的候选模型
- 升级应在少量重试后停止
- 路由应保持成本意识，而非总选最强模型

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: automix
  automix:
    verification_threshold: 0.78
    max_escalations: 2
    cost_aware_routing: true
```
