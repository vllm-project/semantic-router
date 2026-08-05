---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/elo.md"
  outdated: true
---

# Elo

## 概览

`elo` 使用类 Elo 反馈分数对候选模型排序的选择算法。

对应 `config/algorithm/selection/elo.yaml`。

## 主要优势

- 复用历史反馈，而非仅当前请求启发式。
- 用少量参数即可调排序行为。
- 支持不同工作负载的类别感知加权。

## 解决什么问题？

模型质量随时间变化时，固定胜者过于僵化。`elo` 让路由偏好历史上在相似流量上表现稳定的候选。

## 何时使用

- 收集路由级反馈或质量对比
- 随更多对局到来排名应改进
- 单条路由有重复工作负载，适合评分体系

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: elo
  elo:
    initial_rating: 1200
    k_factor: 32
    category_weighted: true
    min_comparisons: 10
```
