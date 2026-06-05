---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/decision/single.md"
  outdated: false
---

# 单条件决策

## 概览

在**一个信号足以选定路由**时使用 `config/decision/single/`。

这是「单一权威检测器」路由最干净的入口。

## 主要优势

- 决策形态最小。
- 易读、易审计。
- 在加入更多布尔逻辑前是很好的基线。
- 让一条强信号独占路由，无需额外嵌套。

## 解决什么问题？

有些路由不需要布尔树。强行塞进更大的 `AND`/`OR` 会增加噪音，使简单策略难review。

`single/` 让路由聚焦在单一决定性匹配上。

## 何时使用

在以下情况使用 `single/`：

- 一个领域信号即权威
- 一个安全信号应立即拦截
- 一个偏好信号选择专用模型

## 配置

源片段：`config/decision/single/domain-business.yaml`

```yaml
routing:
  decisions:
    - name: business_route
      description: Route business and management questions.
      priority: 110
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
```

即使单条件也保持路由命名与可复用。若策略变复杂，可升级为 `and/`、`or/` 或 `composite/` 而不改变外层配置结构。
