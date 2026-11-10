---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/fact-check.md"
  outdated: false
---

# 事实核查信号 {#fact-check-signal}

## 概述 {#overview}

`fact_check` 识别答案依赖于需要核查的事实陈述的请求。它提出 `fact_check` [预设](../../../system-one/reference#preset-catalog)问题：“Does answering this request require checking facts?”，选项为 `FACT_CHECK_NEEDED` 和 `NO_FACT_CHECK_NEEDED`。它只标记需要核查，并不核查任何事实。

## 主要优势 {#key-advantages}

- 把事实核查与领域路由分开。
- 让决策为证据敏感的流量选择检索、更强的模型或答案检查。
- 让核查策略在路由图中可见，而不是藏在后面的插件里。

## 解决什么问题？ {#what-problem-does-it-solve}

并非每个请求都需要同等程度的依据。对所有流量一视同仁，会过度约束创作类请求，又对事实类请求保护不足。`fact_check` 在路由之前把两者区分开。

## 何时使用 {#when-to-use}

在以下情况使用 `fact_check`：

- 事实类请求需要检索、更严格的路由或有依据的答案检查；
- 创作或改写类请求应跳过昂贵的核查路径；
- 只有当请求需要事实时，[`response_guard`](../../plugin/response-guard) 规则才检查依据。

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      fact_check:
        - name: needs_fact_check
          description: Requests with factual claims that should be verified.
        - name: no_fact_check_needed
          description: Creative or rewriting requests.
          labels: [NO_FACT_CHECK_NEEDED]
    decisions:
      - name: grounded
        priority: 150
        rules:
          operator: AND
          conditions:
            - {type: fact_check, name: needs_fact_check}
        modelRefs:
          - model: grounded-model
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供 `{type: fact_check, name: ...}` 条件使用 |
| `labels` | `[FACT_CHECK_NEEDED]`（默认）或 `[NO_FACT_CHECK_NEEDED]` |
| `predicate` | 规则所选选项的最低概率；对于 `FACT_CHECK_NEEDED`，默认为模型的校准阈值（vela3-fast 上为 0.9999），否则为 0.5 |
| `deployment`、`timeout_ms` | 回答问题的部署，以及答案的截止时间 |

规则的名称不决定它的选项：没有 `labels` 的规则匹配 `FACT_CHECK_NEEDED`。每个选项按自己的概率匹配，所以模型不确定时，请求可能两条规则都不匹配。

vela3-fast 的事实核查适配器给出的概率接近 0 或 1，所以它的校准阈值接近 1。在其他模型上，请写一个你测量过的 `predicate`。

### 回答问题的模型 {#the-model-that-answers}

除非 `fact_check` 绑定或规则的 `deployment` 指定了其他部署，否则由决策模型回答所有事实核查规则；见[选择部署](../../../system-one/deployments)。vela3-fast、Vela 3.0 的 d3 规格和 Vela 2.0 有该问题的训练形式。它读取当前用户轮次。

完整示例见 [`config/fragments/signal/fact-check/needs-verification.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/fact-check/needs-verification.yaml)。
