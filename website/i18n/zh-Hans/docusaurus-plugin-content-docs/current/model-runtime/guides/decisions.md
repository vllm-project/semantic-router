---
title: 决策模型
description: 用大白话向决策模型提出你自己的路由问题，让它来挑回答的模型。
translation:
  source_commit: "abae8ff99df2fdab372f0fb6d032b305907b9f44"
  source_file: "docs/model-runtime/guides/decisions.md"
  outdated: false
is_mtpe: true
---

# 决策模型 {#decision-models}

决策模型用大白话回答你就一个请求提的问题，不用为每个问题训练一个分类器。router 在两个地方用它：

- [`decision` 信号](tutorials/signal/learned/decision.md) 提一个问题，按答案路由；
- [`decision` 选择算法](tutorials/algorithm/selection/decision.md) 问一条路由的模型里该由谁答。

## 问题的种类 {#kinds-of-questions}

| 类型 | 问什么 | 答案 |
| --- | --- | --- |
| `choice` | 这几个选项哪个合适？ | 选中的选项，以及每个选项的概率 |
| `noul` | 这是真的吗？ | 为是的概率 |
| `score` | 有多少，按有序刻度？ | 期望档位，以及每档的概率 |
| `set` | 这些标签哪些适用？（Vela 2.0） | 所有过阈值的标签 |
| `span` | 在文本哪里……？（Vela 2.0） | 文本的带标片段 |

一个请求里发给同一个模型的所有问题，一次调用送达、一起答。

## 提一个问题 {#ask-a-question}

```yaml
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: auto
routing:
  signals:
    decision:
      - name: needs_reasoning
        deployment: decision-kai
        question:
          type: noul
          instructions: Does answering this request need multi-step reasoning?
        predicate:
          gte: 0.7
        timeout_ms: 1000
      - name: request_kind
        deployment: decision-kai
        question:
          type: choice
          instructions: What kind of request is this?
          choices:
            - key: code
              description: Writing, reviewing or debugging code
            - key: math
              description: Mathematics or quantitative reasoning
            - key: chat
              description: Anything else
  decisions:
    - name: hard-code
      priority: 200
      rules:
        operator: AND
        on_unknown: no_match
        conditions:
          - type: decision
            name: request_kind
            label: code
          - type: decision
            name: needs_reasoning
      modelRefs:
        - model: large-coder
```

instructions 照你怎么问同事的写，每个选项用几个词描述清楚。选项短而具体，答案最可靠。

## 让它来挑模型 {#let-it-choose-the-model}

```yaml
routing:
  decisions:
    - name: code-route
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: decision
            name: request_kind
            label: code
      modelRefs:
        - model: qwen3-8b
        - model: qwen3-32b
      algorithm:
        type: decision
        decision:
          deployment: decision-kai
          instructions: Which model should answer this request?
          candidates:
            qwen3-8b: Fast general model for routine code
            qwen3-32b: Strong reasoning model for hard code
          timeout_ms: 1000
```

模型给每个候选的概率，就是它的选择分。模型没就绪或答晚了，`modelRefs` 里的第一个模型答。

## 选哪个决策模型 {#which-decision-model}

Decision 2.0 是默认家族：Kai-0.6B 在 CPU 上跑，大的尺寸在 GPU 上更准。Decision 1.0 的模型答同样的问题。Vela 2.0 还能答 `set` 和 `span` 问题，自带 PII 和无依据论断的现成问题，它的 0.3B 默认一次调用答掉 router 的内置信号（见[选模型](model-runtime/choose-a-model.md)）。见[选模型](model-runtime/choose-a-model.md#decision-models)。

## 验一下 {#check-it}

用 `/v1/decisions` 直接问模型同一个问题；见[快速上手](model-runtime/quickstart.md#3-send-a-request)。经过 router 时，`x-vsr-matched-decision-model` 列出匹配上的决策信号，`x-vsr-selected-model` 是选择器挑中的模型。
