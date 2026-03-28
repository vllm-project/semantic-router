---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/decision/composite.md"
  outdated: false
---

# 组合决策

## 概览

在策略需要**单条路由内嵌套 `AND`、`OR`、`NOT`** 时使用 `config/decision/composite/`。

适合业务逻辑与安全逻辑必须共存的真实生产策略。

## 主要优势

- 支持嵌套逻辑，无需把策略压成不可读的条件列表。
- 业务、运维与安全约束放在同一路由中。
- 复杂准入规则显式、可review。
- 避免仅因分支差异而复制多条相关路由。

## 解决什么问题？

扁平布尔规则在路由依赖多个独立分支、排除与升级路径时难以扩展。

`composite/` 用真实的匹配树表达策略，而不是强行简化形态。

## 何时使用

在以下情况使用 `composite/`：

- 领域路由需要紧急度或复杂度升级
- 生产安全策略必须排除不安全流量
- 一条路由在同一匹配树中结合业务与安全逻辑

## 配置

源片段：`config/decision/composite/priority-safe-escalation.yaml`

```yaml
routing:
  decisions:
    - name: priority_safe_escalation_route
      description: Combine AND, OR, and NOT for a realistic multi-signal routing case.
      priority: 160
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
          - operator: OR
            conditions:
              - type: keyword
                name: urgent_keywords
              - type: complexity
                name: needs_reasoning
          - operator: NOT
            conditions:
              - type: jailbreak
                name: prompt_injection
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: true
```

若需要嵌套逻辑，优先使用 `composite/` 片段，而不是把单一块扁平规则撑到不可读。
