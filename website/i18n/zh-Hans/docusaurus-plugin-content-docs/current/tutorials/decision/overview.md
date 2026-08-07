---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/decision/overview.md"
  outdated: true
---

# 决策教程

## 概览

最新决策教程与 `config/decision/` 下的布尔形态目录一致。

信号说明路由器**检测到了什么**；决策说明路由器**据此做什么**：

- 哪条路由命中
- 候选模型有哪些
- 是否启用推理
- 路由选定后运行哪些插件

## 主要优势

- 多个信号协作时仍能保持路由策略可读。
- 布尔逻辑显式、可review。
- 将路由匹配与部署绑定、算法、插件解耦。
- 直接映射到 `config/decision/` 下可复用的片段目录。

## 解决什么问题？

没有决策层时，信号输出无法告诉路由器如何反应。团队容易把路由逻辑散落在临时 if、模型默认和插件 wiring 里。

决策层把命名信号变成清晰的路由策略，带稳定优先级与候选模型。

## 何时使用

在以下情况使用 `decision/`：

- 一条路由应由一个或多个信号激活
- 同一模型策略要在多种信号组合下复用
- 路由优先级重要
- 插件或算法应挂在**已匹配的路由**上，而不是整台路由器

## 配置

在 v0.3 中，决策位于 `routing.decisions`：

```yaml
routing:
  decisions:
    - name: business_route
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

决策匹配与以下部分保持分离：

- `providers.models[]`：部署绑定
- `decision.algorithm`：在多个候选模型间选择
- `decision.plugins`：匹配路由后的后处理

按下方形态目录顺序，与片段树一致：

| 决策形态 | 片段示例 | 适用场景 | 教程 |
|----------|----------|----------|------|
| `single` | `config/decision/single/domain-business.yaml` | 单一决定性信号 | [单条件](./single) |
| `and` | `config/decision/and/urgent-business.yaml` | 多个信号必须同时满足 | [AND 决策](./and) |
| `or` | `config/decision/or/business-or-law.yaml` | 多种等价触发共用一路由 | [OR 决策](./or) |
| `not` | `config/decision/not/exclude-jailbreak.yaml` | 显式排除或安全门控 | [NOT 决策](./not) |
| `composite` | `config/decision/composite/priority-safe-escalation.yaml` | 嵌套的真实策略 | [组合决策](./composite) |

若 `modelRefs` 中有多个候选，请配合 [算法](../algorithm/overview)；若路由需要选中后的行为，请配合 [插件](../plugin/overview)。
