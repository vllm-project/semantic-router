---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/decision/not.md"
  outdated: false
---

# NOT 决策

## 概览

`NOT` 决策仅在其子条件不匹配时才匹配。用它在路由策略中把排除条件写清楚。

## 主要优势

- 让否定策略保持显式。
- 适合安全门和高级路由排除。
- 把排除逻辑放在路由图中，而不是藏在下游。
- 因为被拒绝的信号被直接点名，审计更容易。

## 解决什么问题？

有些路由只应在已知风险信号不存在时运行。如果该排除是隐式的，审查者只能从下游行为推断。

`NOT` 通过把排除直接写进路由定义来解决这个问题。

## 何时使用

在以下情况使用 `NOT`：

- 必须排除已知越狱或含 PII 的流量
- 高级路由应远离不安全输入
- 升级前必须没有冲突信号

## 配置

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

请谨慎使用 `NOT`，并保持被排除信号显式，否则决策会难以审计。

当子信号不可用或未触发时，`NOT` 也会匹配，因此不要把它当作内容安全的证明。对访问敏感路由，优先使用正向可信条件。完整示例见：
[`config/fragments/decision/not/exclude-jailbreak.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/decision/not/exclude-jailbreak.yaml)。
