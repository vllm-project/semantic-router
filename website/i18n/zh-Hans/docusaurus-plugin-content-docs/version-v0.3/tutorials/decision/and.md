---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/decision/and.md"
  outdated: false
---

# AND 决策

## 概览

在多个信号**必须全部匹配**路由才成立时使用 `config/decision/and/`。

`AND` 是收窄、高置信路由的标准形态。

## 主要优势

- 要求多个检测器同时命中，降低误报。
- 适合升级与高端路由。
- 复合条件显式，而不是藏在一个信号里。
- 路由边界可预测。

## 解决什么问题？

单一信号往往匹配过宽。仅有领域可能不够，还需要紧急度、安全或复杂度上下文。

`AND` 要求所有必要条件一致后再允许该路由生效。

## 何时使用

在以下情况使用 `and/`：

- 领域与紧急用语必须同时出现
- 领域与安全许可必须同时通过
- 偏好与复杂度在升级前应协同满足

## 配置

源片段：`config/decision/and/urgent-business.yaml`

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

当模型应仅在**窄而高置信**的流量片段上激活时使用 `AND`。
