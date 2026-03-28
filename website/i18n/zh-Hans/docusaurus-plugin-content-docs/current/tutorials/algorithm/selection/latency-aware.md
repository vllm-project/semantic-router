---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/latency-aware.md"
  outdated: false
---

# Latency Aware

## 概览

`latency_aware` 按延迟分位数偏好**最快且可接受**候选的选择算法。

对应 `config/algorithm/selection/latency-aware.yaml`。

## 主要优势

- 在路由层保持延迟 SLO 可见。
- 平衡 TTFT 与 TPOT，而非单一指标。
- 适合响应性比绝对质量更重要的路由。

## 解决什么问题？

部分路由在多个候选都能回答时仍需满足延迟预算。`latency_aware` 让路由偏好满足预算的模型。

## 何时使用

- 路由有多个可行候选但有严格响应时间目标
- TTFT 与 TPOT 都应影响胜者
- 路由匹配后延迟是主要决胜因素

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: latency_aware
  latency_aware:
    tpot_percentile: 90
    ttft_percentile: 95
```
