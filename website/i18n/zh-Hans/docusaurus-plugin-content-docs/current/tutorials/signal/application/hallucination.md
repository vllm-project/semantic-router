---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/hallucination.md"
  outdated: false
---

# 幻觉信号 {#hallucination-signal}

## 概述 {#overview}

`hallucination` 找出模型答案中其依据上下文不支持的陈述。它在响应阶段提出 `hallucination` [预设](../../../system-one/reference#preset-catalog)问题，即针对答案、标签为 `unsupported` 的 span 问题。依据上下文是 RAG 检索到的文档和请求中的工具结果。

## 主要优势 {#key-advantages}

- 把检测无依据的陈述与 Router 如何处理它分开。
- 返回无依据陈述的 span，而不仅是一个判定。
- 无法检查的答案报告为未知，绝不当作已验证。
- 与 `fact_check` 结合时，只检查值得核对依据的答案。

## 解决什么问题？ {#what-problem-does-it-solve}

即使给了证据，模型也可能超出证据作答：一个错误的数字、一个虚构的名字、上下文从未提到的细节。`hallucination` 对照证据阅读答案，并把发现作为信号发布，由 [`response_guard`](../../plugin/response-guard) 据此行动。

## 何时使用 {#when-to-use}

在以下情况使用 `hallucination`：

- 基于工具结果或检索文档的答案必须先检查再信任；
- 决策应对无依据的答案加注说明或拦截；
- Router Replay 应记录答案是否被检查过，以及结果如何。

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      fact_check:
        - name: needs_fact_check
      hallucination:
        - name: ungrounded
    decisions:
      - name: grounded-qa
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: answer-model
        plugins:
          - type: response_guard
            configuration:
              rules:
                - when:
                    operator: AND
                    conditions:
                      - {type: fact_check, name: needs_fact_check}
                      - {type: hallucination, name: ungrounded}
                  action: annotate
                  details: true
              on_unknown: annotate
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供防护规则以 `{type: hallucination, name: ...}` 使用 |
| `predicate` | 无依据 span 的最低概率；默认由模型自己的 span 阈值决定 |
| `deployment`、`timeout_ms` | 回答问题的部署，以及答案的截止时间 |

当模型找到一个无依据的 span 时，规则匹配。上面的防护规则只检查需要事实的请求的答案：`AND` 在为假的条件处停止，所以 `needs_fact_check` 不匹配时不会提出这个问题。

### 阶段与结果 {#stage-and-outcomes}

这个问题在模型回答之后提出，所以决策无法读取它：直接或通过投影读取它的决策会在加载时被拒绝。只有 `response_guard` 从请求解析到的配方中读取它。当请求没有携带依据上下文、模型没有给出答案，或响应中没有可检查的文本时，规则为未知；由防护的 `on_unknown` 决定未知的规则怎么处理。

流式答案在流结束后检查。上面的守卫会扣住流直到检查完成，因为它的规则可能把说明写进答案（`details: true`）：客户端在答案检查完毕后才收到它，规则匹配时答案开头带有说明。规则只在 `x-vsr-warnings` 中标记答案或只记录结果的守卫会保持实时流式传输，并在流结束后记录结果。见[流式回答](../../plugin/response-guard.md#streamed-answers)。

### 回答问题的模型 {#the-model-that-answers}

除非 `hallucination` 绑定或规则的 `deployment` 指定了其他部署，否则由决策模型回答所有幻觉规则；见[选择部署](../../../system-one/deployments)。该问题需要 span：vela3-fast（默认）、Vela 3.0 的 d3 规格和 Vela 2.0 用它们自己的幻觉问题回答。解析到不支持 span 的部署（例如 d3 模型）的规则会以 `span_required` 加载失败。

这个问题会把请求、答案和依据上下文发给部署。完整示例见 [`config/fragments/signal/hallucination/grounded-answer.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/hallucination/grounded-answer.yaml)。
