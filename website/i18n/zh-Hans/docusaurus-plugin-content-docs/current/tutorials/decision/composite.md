---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/decision/composite.md"
  outdated: false
---

# 复合决策

## 概览

复合决策在一条路由中嵌套 `AND`、`OR` 和 `NOT` 组。当业务、运维和安全要求必须一起评估时使用它。

## 主要优势

- 支持嵌套逻辑，而不把策略压平成难以阅读的条件。
- 把业务、运维和安全约束保留在一条路由中。
- 让复杂资格规则显式且可审查。
- 避免只因一个分支不同而复制相关路由。

## 解决什么问题？

一旦路由依赖多个独立分支、排除条件和升级路径，扁平布尔规则就难以扩展。

复合决策把策略编码成可读的匹配树，而不是强迫它变成扁平条件列表。

## 何时使用

在以下情况使用复合决策：

- 特定领域路由需要按紧急度或复杂度升级
- 生产安全策略必须排除不安全流量
- 一条路由在同一匹配树中同时组合业务逻辑和安全逻辑

## 配置

```yaml
routing:
  decisions:
    - name: priority_safe_escalation_route
      description: Combine AND, OR, and NOT for a realistic multi-signal routing case.
      priority: 160
      rules:
        operator: AND
        conditions:
          - type: domain
            name: business
          - operator: OR
            conditions:
              - type: keyword
                name: urgent_keywords
              - type: complexity
                name: needs_reasoning:hard
          - operator: NOT
            conditions:
              - type: jailbreak
                name: prompt_injection
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: true
```

如果决策需要嵌套逻辑，请保持分组显式，而不是把一个扁平规则块拉长到难以阅读。

嵌套应浅到足以审查和测试每个分支。信号结果可能是概率性的，因此复杂树不能替代授权或后端策略。完整示例见：
[`config/fragments/decision/composite/priority-safe-escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/decision/composite/priority-safe-escalation.yaml)。
