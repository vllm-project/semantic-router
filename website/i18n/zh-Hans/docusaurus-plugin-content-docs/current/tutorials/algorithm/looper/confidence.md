---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/looper/confidence.md"
  outdated: true
---

# Confidence

## 概览

`confidence` 是循环算法：在候选模型间逐级升级，直到置信度足够高。

对应 `config/algorithm/looper/confidence.yaml`。

## 主要优势

- 支持由小到大升级，而非固定胜者。
- 停止条件显式。
- 仅在需要时用额外延迟换更高置信度。

## 解决什么问题？

部分路由应先试便宜候选，仅当答案不够自信时再升级。`confidence` 将升级策略独立为算法，而非塞进应用代码。

## 何时使用

- 路由应在多个候选模型间逐级尝试
- 由置信度决定是否继续
- 一旦某次响应足够好即停止

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: confidence
  confidence:
    confidence_method: hybrid
    threshold: 0.72
    escalation_order: small_to_large
    on_error: skip
```
