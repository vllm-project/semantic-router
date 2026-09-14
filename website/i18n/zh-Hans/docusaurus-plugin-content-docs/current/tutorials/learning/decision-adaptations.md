---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/learning/decision-adaptations.md"
  outdated: false
---

# 决策自适应

## 概览

决策自适应让已匹配的决策控制全局路由学习能否调整其提议的模型。

大多数决策继承全局学习行为。仅当某个决策需要硬边界、仅观察的发布，或少量防护调优时，才添加 `adaptations`。

## 主要优势

- 把策略边界放在拥有它们的决策附近。
- 让敏感路由用一个小块绕过学习。
- 支持在自适应或防护影响流量之前进行仅观察发布。
- 让一个决策使用比全局默认更窄或更宽的自适应候选集。
- 让一个决策在不更改全局默认的情况下调整稳定性权衡。

## 解决什么问题？

全局学习很方便，但并非每个决策都应被在线状态调整。隐私、仅本地、安全、合规和运维路由常常需要硬边界。决策自适应让已匹配的决策最终决定学习是应用、观察还是绕过。

## 何时使用

- 已匹配的决策不得被在线学习更改。
- 希望在允许路由变更之前比较学习诊断。
- 某个决策应搜索整个路由 tier，而大多数决策留在自己的 `modelRefs` 内。
- 某个决策需要比默认更强或更弱的防护余量。
- 同一决策的自适应和防护需要不同模式。

## 配置

硬边界使用 `bypass`：

```yaml
routing:
  decisions:
    - name: local_privacy_policy
      description: Keep privacy-sensitive traffic on the local model.
      priority: 200
      modelRefs:
        - model: local-private-model
      adaptations:
        mode: bypass
```

当自适应和防护应表现不同时，使用组件级控制：

```yaml
adaptations:
  adaptation:
    mode: observe
    candidate_set: tier
  protection:
    mode: apply
```

`adaptation.candidate_set` 是可选的。省略时，该决策继承 `global.router.learning.adaptation.candidate_set`。

允许的模式：

| 模式 | 含义 |
| --- | --- |
| `apply` | 该组件可以影响最终路由。 |
| `observe` | 该组件记录诊断，但不能更改最终路由。 |
| `bypass` | 该组件不调整此决策。 |

`adaptations.mode: bypass` 会覆盖组件级模式，并阻止自适应和防护更改路由。

## 防护调优 {#protection-tuning}

仅当某个决策需要不同于全局默认的稳定性权衡时，才使用决策局部防护调优：

```yaml
adaptations:
  protection:
    stability_weight: 1.5
    switch_margin: 0.10
```

更高的 `protection.stability_weight` 更偏向稳定。更低的 `protection.stability_weight` 让自适应更容易切换模型。`switch_margin` 是此决策切换前所需的最小模型优势。
