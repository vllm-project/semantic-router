---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/decision/not.md"
  outdated: false
---

# NOT 决策

## 概览

在路由仅当**存在风险或不允许的信号缺失**时才匹配时使用 `config/decision/not/`。

`NOT` 是决策目录中最简单的排除规则。

## 主要优势

- 负面策略显式。
- 适合安全门控与高端路由排除。
- 排除逻辑留在路由图中，而不是藏到下游。
- 审计更容易，因为被否定的信号有命名。

## 解决什么问题？

有些路由应仅在已知风险信号**不存在**时运行。若排除是隐式的，审查者只能从下游行为推断。

`NOT` 把排除直接写进路由定义。

## 何时使用

在以下情况使用 `not/`：

- 必须排除已知越狱或含 PII 的流量
- 高端路由应避开不安全输入
- 升级前某冲突信号必须不存在

## 配置

源片段：`config/decision/not/exclude-jailbreak.yaml`

```yaml
routing:
  decisions:
    - name: safe_only_route
      description: Match only when the known prompt-injection signal is absent.
      priority: 70
      rules:
        operator: NOT
        conditions:
          - type: jailbreak
            name: prompt_injection
      modelRefs:
        - model: qwen2.5:3b
          use_reasoning: false
```

谨慎使用 `NOT`，并保持被排除的信号显式，否则决策难以审计。
