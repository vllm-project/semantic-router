---
is_mtpe: false
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/algorithm/selection/prompt.md"
  outdated: false
---

# 提示词选择（已移除）

## 概览

`prompt` 选择器已被移除。[System One](./systemone) 选择器会向决策模型提出同样的问题。

## 它解决什么问题？

它让一个辅助模型按照运维人员的指令，从决策的候选中选择一个。`systemone` 选择器把同样的指令作为针对候选的类型化选择问题提给决策模型。

## 何时使用

不要再配置它：仍然使用它的配置会在加载时失败。请使用 [System One](./systemone)。

## 配置

运行 `vllm-sr config migrate` 来改写它。指令会被保留，`timeout_seconds` 变为 `timeout_ms`，辅助模型不再参与选择：如需询问决策模型以外的 System One 部署，请设置 `systemone.deployment`。

```yaml
algorithm:
  type: systemone
  systemone:
    instructions: Choose the model that best fits the request.
    timeout_ms: 5000
```
