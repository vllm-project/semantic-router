---
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/tutorials/decision/overview.md"
  outdated: false
---

# 决策 {#decisions}

## 概览 {#overview}

信号告诉 Router 它检测到了什么。决策把这些检测变成路由策略：

- 哪条路由匹配
- 哪些模型是候选
- 是否启用推理
- 路由选定后运行哪些插件

## 主要优势 {#key-advantages}

- 即使多个信号必须协作，也能保持路由策略可读。
- 让布尔逻辑显式且可审查。
- 将路由匹配与部署绑定、算法和插件分开。

## 解决什么问题？ {#what-problem-does-it-solve}

没有决策层时，信号输出无法告诉路由器如何反应。团队最终会把路由逻辑散落在临时 if、模型默认值和插件接线里。

决策把具名信号变成带有稳定优先级和候选模型的清晰路由策略，从而解决这个问题。

## 何时使用 {#when-to-use}

在以下情况使用决策：

- 路由应由一个或多个信号激活
- 同一模型策略应在多种信号组合中复用
- 路由优先级很重要
- 插件或算法应挂到匹配的路由上，而不是整个路由器

## 配置 {#configuration}

决策位于 `recipes[].decisions` 下：

```yaml
recipes:
  - name: default
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

### 规则树 {#rule-trees}

- **节点。** 每个 `rules` 节点要么是叶子（`type` 和 `name`），要么是组合（`operator` 和 `conditions`）。
- **运算符。** `AND`、`OR` 或 `NOT`；大小写和周围空白会被归一化，带条件但未写运算符的节点表示 `AND`。
- **`NOT` 是一元的。** 它恰好接受一个子条件；把它套在 `OR` 或 `AND` 外面即可得到 NOR 或 NAND。
- **全匹配。** 没有子节点的组合只在根 `AND` 处有效，这是显式的全匹配形式。

其他形式都会被校验拒绝，例如同时混用叶子和组合字段的节点。每条错误都会点名决策和节点路径：

```text
decision "billing": rules.conditions[1]: NOT requires exactly one child condition, got 2
```

### 未知的信号 {#unknown-signals}

失败或超时作答的信号是未知的。默认情况下，它的条件评估为 `False`。有两个设置可以改变这一点：

| 设置 | 效果 |
| --- | --- |
| `rules.on_unknown` | 让未知条件保持 `Unknown`：`NOT Unknown` 仍是 `Unknown`，`False AND Unknown` 是 `False`，`True OR Unknown` 是 `True`。仍然未知的结果解析为 `no_match`、`match` 或 `fail_request`。 |
| `systemone` 条件上的 `on_error` | 把这一个条件解析为 `match` 或 `no_match`。不能与 `rules.on_unknown` 同时使用。 |

- 两者都未设置时，Router 会在启动时对该 `systemone` 条件发出警告。
- 已应用的策略出现在 `x-vsr-trace` 的 `on_unknown` 成员和 `llm_decision_unknown_total{decision, policy}` 指标中。
- `fail_request` 返回 503，原因码为 [`decision_unresolved`](../../api/router.md#routing-errors)，Router 的日志会点名修复方式。

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

当 `modelRefs` 包含多个候选时添加[算法](../algorithm/overview)，当路由需要选择后的行为时添加[插件](../plugin/overview)。

## 选择顺序 {#selection-order}

匹配和排序是两个独立的步骤。规则评估为真的每个决策都成为候选，然后由排序选出其中一个：

1. **先看层级。** `decision.tier` 是硬性的优先边界：层级低的胜出。未设置 `tier` 的决策属于第 0 层，排在所有设置了层级的决策之前。只要有匹配的决策层级大于 0，就会比较层级。
2. **再看配方的 strategy。** 它为同一层级内的决策排序；没有决策设置层级时，为全部决策排序。
3. **兜底决策最后。** 无论在哪种 strategy 下、优先级是多少，兜底决策都排在所有真正的匹配之后，因此无条件的回退始终是回退。

| `recipes[].strategy` | 排序键，依次为 |
| --- | --- |
| `priority`（默认） | 层级升序、兜底最后、优先级降序、置信度降序、名称升序 |
| `confidence` | 层级升序、兜底最后、置信度降序、优先级降序、名称升序 |

名称升序是最后的平局决胜规则，因此排序从不依赖映射或文件顺序。

### 置信度 {#confidence}

置信度只在可比较的池内排序：分层排序时是一个层级，否则是整个候选集。

- **证据来自由模型判断的叶子。** `systemone`、`domain`、`safety`、`jailbreak`、`complexity` 和 `turn_relation` 叶子报告其回答的概率。
- **策略叶子没有权重。** 关键词规则、`NOT` 防护、谓词、对话和元数据规则以及投影输出决定决策是否匹配，但不增加它的置信度。
- **每个决策一个分数。** 只有恰好一个证据叶子给出分数时，决策才可比较，因为多个叶子取平均会让决策在证据增多时分数反而下降。由 `on_error` 策略制造出来的匹配从不可比较。
- **否则按优先级。** 只有当池中每个非兜底成员都可比较时，池才按置信度排序。

在 `OR` 中，报告了分数的分支优先于没有分数的分支，因此额外匹配的门控只会增加支持，而不会抹去证据。

### 查看排序结果 {#read-the-ranking}

Eval API 在 `decision_ranking` 下报告一个请求是如何排序的：

- 运行的 strategy，以及胜出者来自哪个层级；
- 该池是否可比较，以及哪个决策使它不可比较；
- 排在胜出者之后的决策（`runner_up`），以及区分两者的排序键；
- 说明所比较数值的 `reason`，例如 `priority 150 > 100`。

[Decision Ranking Semantics](../../proposals/decision-ranking-semantics) 提案解释了这份契约的设计。

## 运维边界 {#operational-boundaries}

- 每个叶子都必须引用同一配方中声明的信号或投影输出。
- 排序遵循[选择顺序](#selection-order)。请保留显式的无条件回退，或配置 `providers.defaults.model`。
- 决策名称和路由诊断可能变成运维元数据；不要在名称和描述中放入密钥或个人标识。
- 布尔逻辑是策略，不是认证。对访问敏感的路由，请通过 `authz` 服务和信号使用可信身份。
