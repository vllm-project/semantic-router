---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/decision/and.md"
  outdated: false
---

# AND 决策

## 概览

`AND` 决策仅在每个子条件都匹配时才匹配。把它用于需要多个独立事实的窄路由。

## 主要优势

- 通过要求多个检测器来降低误报。
- 适合升级和高级路由。
- 让复合要求保持显式，而不是藏在一个信号里。
- 产生可预测的路由边界。

## 解决什么问题？

单个信号常常匹配过宽。仅有领域往往不够，还需要紧急度、安全或复杂度上下文。

`AND` 通过要求所有必需信号一致后，路由才有资格，从而解决这个问题。

## 何时使用

在以下情况使用 `AND`：

- 领域和紧急度必须同时存在
- 领域和安全放行必须同时通过
- 偏好和复杂度应在升级前协作

## 配置

```yaml
routing:
  decisions:
    - name: urgent_business_route
      description: Match only when business intent and urgent language appear together.
      priority: 140
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
          - type: keyword
            name: urgent_keywords
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
```

当模型只应为一小段高置信度流量激活时，使用 `AND`。

每个引用的信号都必须在同一配方中声明。`AND` 会减少过宽匹配，但不会让概率信号变成权威判定。完整示例见：
[`config/fragments/decision/and/urgent-business.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/decision/and/urgent-business.yaml)。
