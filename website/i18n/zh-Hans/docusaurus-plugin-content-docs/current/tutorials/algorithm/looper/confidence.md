---
is_mtpe: false
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/algorithm/looper/confidence.md"
  outdated: false
---

# 置信度（已移除）

## 概览

Looper `confidence` 算法已被移除。它的行为就是 [Escalate](./escalate) 的 Chat 路径。

## 它解决什么问题？

它按顺序尝试候选，在某个回答的 token 置信度达到阈值时停止。带 `accept.confidence` 的 `escalate` 做的是同一件事；`accept.systemone` 用决策模型的 `answer_check` 问题取代了它的 `automix_entailment` 校验服务器。

## 何时使用

不要再配置它：仍然使用它的配置会在加载时失败。请使用 [Escalate](./escalate)。

## 配置

运行 `vllm-sr config migrate` 来改写它。例如下面的决策

```yaml
algorithm:
  type: confidence
  confidence:
    confidence_method: hybrid
    threshold: 0.72
    escalation_order: small_to_large
```

会变为

```yaml
algorithm:
  type: escalate
  escalate:
    order: size
    accept:
      confidence: {method: hybrid, gte: 0.72}
```

其他字段见[从 confidence 和 cascade 迁移](./escalate#从-confidence-和-cascade-迁移)。
