---
translation:
  source_commit: "ad233487"
  source_file: "docs/tutorials/learning/decision-adaptations.md"
  outdated: false
---

# Decision 自适应（Decision Adaptations）

## 概览

Decision 自适应允许匹配的 decision 控制全局 Router Learning 是否可以调整其提出的模型。

大多数 decision 会继承全局学习行为。仅当某个 decision 需要硬边界、仅观察式上线或小幅 Protection 调优时，才添加 `adaptations`。

## 主要优势

- 使策略边界紧邻其所属的 decision。
- 只需一个小型配置块，即可让敏感路由绕过学习。
- 在 Adaptation 或 Protection 可以影响流量之前，支持仅观察式上线。
- 允许某个 decision 使用比全局默认值更窄或更宽的 Adaptation 候选集。
- 允许某个 decision 调整稳定性权衡，而无需更改全局默认值。

## 解决什么问题？

全局学习很方便，但并非每个 decision 都应由在线状态进行调整。隐私、仅本地、安全、合规和运维路由通常需要硬边界。Decision 自适应让匹配的 decision 最终决定学习是否可以应用、观察或绕过。

## 何时使用

- 匹配的 decision 不得被在线学习更改。
- 希望在允许更改路由之前比较学习诊断。
- 某个 decision 应搜索整个路由 tier，而大多数 decision 仍局限于各自的 `modelRefs`。
- 某个 decision 需要比默认值更强或更弱的保护余量。
- 对于同一个 decision，Adaptation 和 Protection 需要使用不同的模式。

## 配置

使用 `bypass` 设置硬边界：

```yaml
routing:
  decisions:
    - name: local_privacy_policy
      modelRefs:
        - model: local-private-model
      adaptations:
        mode: bypass
```

当 Adaptation 和 Protection 应采用不同行为时，使用组件级控制：

```yaml
adaptations:
  adaptation:
    mode: observe
    candidate_set: tier
  protection:
    mode: apply
```

`adaptation.candidate_set` 是可选项。省略时，该 decision 会继承 `global.router.learning.adaptation.candidate_set`。

允许的模式：

| 模式 | 含义 |
| --- | --- |
| `apply` | 该组件可以影响最终路由。 |
| `observe` | 该组件会记录诊断，但不能更改最终路由。 |
| `bypass` | 该组件不会调整此 decision。 |

`adaptations.mode: bypass` 会覆盖组件级模式，并阻止 Adaptation 和 Protection 更改路由。

## Protection 调优

仅当某个 decision 需要与全局默认值不同的稳定性权衡时，才使用 decision 局部 Protection 调优：

```yaml
adaptations:
  protection:
    stability_weight: 1.5
    switch_margin: 0.10
```

较高的 `protection.stability_weight` 更偏向稳定性。较低的 `protection.stability_weight` 让 Adaptation 更容易切换模型。`switch_margin` 是针对此 decision 切换模型前所需的最小模型优势。
