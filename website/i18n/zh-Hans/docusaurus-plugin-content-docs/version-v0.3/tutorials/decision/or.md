---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/decision/or.md"
  outdated: false
---

# OR 决策

## 概览

在**多种等价信号匹配**应导向同一路由时使用 `config/decision/or/`。

`OR` 适用于多个独立信号应得到相同路由结果的情形。

## 主要优势

- 避免为同一路由重复写多条决策。
- 回退或共享策略路由更紧凑。
- 等价触发显式表达。
- 一个模型策略可覆盖多个主题或信号时很合适。

## 解决什么问题？

没有 `OR` 时，团队常为支持不同匹配条件而复制同一路由逻辑，导致漂移与后续策略变更风险。

`OR` 将等价触发合并为一条路由。

## 何时使用

在以下情况使用 `or/`：

- 两个领域共用同一模型策略
- 多种信号变体映射到同一回退路由
- 一个运维插件应对多种独立情况

## 配置

源片段：`config/decision/or/business-or-law.yaml`

```yaml
routing:
  decisions:
    - name: business_or_law_route
      description: Share one route across either business or law traffic.
      priority: 100
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
          - type: domain
            name: law
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
```

当**路由结果相同**但允许多种信号触发时使用 `OR`。
