---
translation:
  source_commit: "15a2f08f188b509dbe06132281a226d1cc6a665c"
  source_file: "docs/tutorials/decision/overview.md"
  outdated: true
---

# 决策

## 概览

信号告诉 Router 它检测到了什么。决策把这些检测变成路由策略：

- 哪条路由匹配
- 哪些模型是候选
- 是否启用推理
- 路由选定后运行哪些插件

## 主要优势

- 即使多个信号必须协作，也能保持路由策略可读。
- 让布尔逻辑显式且可审查。
- 将路由匹配与部署绑定、算法和插件分开。

## 解决什么问题？

没有决策层时，信号输出无法告诉路由器如何反应。团队最终会把路由逻辑散落在临时 if、模型默认值和插件接线里。

决策通过把命名信号变成带稳定优先级和候选模型的清晰路由策略，来解决这个问题。

## 何时使用

在以下情况使用决策：

- 路由应由一个或多个信号激活
- 同一模型策略应在多种信号组合中复用
- 路由优先级很重要
- 插件或算法应挂到匹配的路由上，而不是整个路由器

## 配置

在 v0.3 中，决策位于 `routing.decisions` 下：

```yaml
routing:
  decisions:
    - name: business_route
      description: Route business requests to the business model.
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

每个 `rules` 节点要么是叶子（`type` 和 `name`），要么是组合（`operator` 和 `conditions`）。运算符必须是 `AND`、`OR` 或 `NOT`；大小写和周围空白会被归一化，带条件的节点省略运算符时表示 `AND`。`NOT` 严格一元，恰好接受一个子条件；把 `NOT` 套在 `OR` 或 `AND` 外包一层即可得到 NOR 或 NAND。配置校验会拒绝任何其他运算符、子条件为零或多个的 `NOT`、同时混用叶子和组合字段的节点，以及除根 `AND`（显式的全匹配形式）之外任何没有子节点的组合。错误会点名决策和节点路径，例如
`decision "billing": rules.conditions[1]: NOT requires exactly one child condition, got 2`。

分类器失败会评估为 `Unknown`，而不是 `False`。`NOT Unknown` 仍是 `Unknown`；`False AND Unknown` 是 `False`，`True OR Unknown` 是 `True`。
当最终结果仍未知时，`rules.on_unknown` 选择 `no_match`、`match` 或 `fail_request`。如果省略，会保留现有通用分类器 `on_error` 和提示防护 `on_error` 行为，并且路由器会在启动时警告既未设置这些项的分类器条件。已应用的策略出现在 `x-vsr-applied-unknown-policy` 响应头和 `llm_decision_unknown_total{decision, policy}` 指标中；`fail_request` 的 503 消息会点名修复方式。

决策匹配与以下内容保持分离：

- `providers.models[]`，承载部署绑定
- `decision.algorithm`，在多个候选模型中选择
- `decision.plugins`，对匹配路由做后处理

选择能清楚表达策略的最小形态：

| 决策形态 | 最适合 | 指南 |
|----------------|----------|-------|
| 单条件 | 一个决定性信号 | [Single Condition](./single) |
| `AND` | 必须全部匹配的多个条件 | [AND 决策](./and) |
| `OR` | 由多个备选条件共享的一条路由 | [OR 决策](./or) |
| `NOT` | 显式排除或安全防护 | [NOT 决策](./not) |
| 复合 | `AND`、`OR` 和 `NOT` 的嵌套组合 | [Composite 决策](./composite) |
| 保留指令 | 决策匹配后的缓存或会话副作用 | [Retention Directives](./retention) |

当 `modelRefs` 包含多个候选时添加 [算法](../algorithm/overview)，当路由需要选择后行为时添加 [插件](../plugin/overview)。

## 运维边界

- 每个叶子都必须引用同一配方中声明的信号或投影输出。
- 多条决策匹配时，更高的 `priority` 获胜。请保留显式的无条件回退，或配置 `providers.defaults.model`。
- 决策名称和路由诊断可能变成运维元数据；不要在名称和描述中放入密钥或个人标识。
- 布尔逻辑是策略，不是认证。对访问敏感路由，通过 `authz` 服务和信号使用可信身份。
