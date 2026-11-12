---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/decision/or.md"
  outdated: false
---

# OR 决策

## 概览

`OR` 决策在任一子条件匹配时匹配。当几种独立请求类型应共享同一路由结果时使用它。

## 主要优势

- 避免在多个决策中复制同一路由。
- 让回退或共享策略路由保持紧凑。
- 让等价匹配保持显式。
- 当一种模型策略覆盖多个主题或信号时效果好。

## 解决什么问题？

没有 `OR` 时，团队常常为了支持不同匹配条件而多次复制同一路由逻辑。这会造成漂移，也让后续策略变更更危险。

`OR` 通过把等价触发器折叠进一条路由来解决这个问题。

## 何时使用

在以下情况使用 `OR`：

- 两个领域共享同一模型策略
- 多个信号变体映射到一条回退路由
- 一个运维插件应对多种独立情况运行

## 配置

```yaml
routing:
  decisions:
    - name: business_or_law_route
      description: Share one route across either business or law traffic.
      priority: 100
      rules:
        operator: OR
        conditions:
          - type: domain
            name: business
          - type: domain
            name: law
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
```

当路由结果相同，但应允许多个信号触发它时，使用 `OR`。

任一子条件都可以让路由有资格，因此请把每个子条件当作独立路由条件来审计。完整示例见：
[`config/fragments/decision/or/business-or-law.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/decision/or/business-or-law.yaml)。
