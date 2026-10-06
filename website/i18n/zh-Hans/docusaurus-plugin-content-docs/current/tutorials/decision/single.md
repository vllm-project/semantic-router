---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/decision/single.md"
  outdated: false
---

# 单条件决策

## 概览

单条件决策是最简单的路由策略：一个信号或投影输出决定路由是否有资格。

## 主要优势

- 最小的决策形态。
- 易于阅读、易于审计。
- 在添加更多布尔逻辑之前是好的基线。
- 让一个强信号拥有一条路由，而不需要额外嵌套。

## 解决什么问题？

有些路由不需要布尔树。把它们塞进更大的 `AND` 或 `OR` 结构只会增加噪音，并让简单策略更难审查。

单条件决策让路由聚焦于一次决定性匹配。

## 何时使用

在以下情况使用单条件决策：

- 一个领域信号是权威的
- 一个安全信号应立即阻断
- 一个偏好信号选择专用模型

## 配置

```yaml
routing:
  decisions:
    - name: business_route
      description: Route business and management questions.
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

即使只有一个条件，也请保持路由命名且可复用。如果策略以后变复杂，可以添加显式布尔组，而不改变周围的路由结构。

引用的信号必须在同一配方中声明。单个已学习信号仍是概率性的，因此对授权敏感路由使用可信身份或确定性策略。完整示例见：
[`config/fragments/decision/single/domain-business.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/decision/single/domain-business.yaml)。
