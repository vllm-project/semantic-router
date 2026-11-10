---
title: 编写自定义问题
sidebar_label: 编写自定义问题
description: 把一个路由想法变成 systemone 规则，直接用模型验证，并调优它的 predicate。
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/system-one/custom-questions.md"
  outdated: false
---

# 编写自定义问题 {#write-custom-questions}

没有应用信号覆盖的路由想法，例如“这个请求和我们的产品有关吗？”或“这需要调用工具吗？”，可以写成一条 `systemone` 规则：一个用自然语言写的问题，由决策模型零样本回答，无需任何训练。本页从想法一路讲到调优好的规则；字段参考见 [System One 信号](../tutorials/signal/systemone)页面。

## 1. 选择问题类型 {#1-choose-the-question-type}

| 想法 | 类型 | 匹配依据 |
| --- | --- | --- |
| “这是 X 吗？” | `noul` | P(是) |
| “它属于这几种中的哪一种？”（只能选一个） | `choice` | 选中的选项，或某个选项的概率 |
| “X 的程度有多高？”（有序刻度） | `score` | 期望等级 |
| “这些中哪些适用？”（任意多个） | `set` | 某个标签的概率 |
| “X 在文本中的什么位置？” | `span` | 某个标签的 span；需要支持 span 的模型 |

需要互斥的几个 `noul` 问题，优先写成带 `other` 选项的一个 `choice`；选项有顺序的 `choice`，优先写成 `score`。

## 2. 编写规则 {#2-write-the-rule}

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: product_question
          question:
            type: choice
            instructions: Which part of the product is this request about?
            choices:
              - key: billing
                description: Invoices, payments, refunds or plan changes
              - key: exports
                description: Exporting data, reports or CSV files
              - key: other
                description: Anything else, including requests not about the product
          labels: [billing, exports]
        - name: needs_tools
          question:
            type: noul
            instructions: Does answering this request need a tool call?
          predicate: {gte: 0.7}
    decisions:
      - name: product-support
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: systemone, name: product_question}
        modelRefs:
          - model: support-model
```

没有 `deployment` 的规则向决策模型提问；见[选择部署](./deployments)。这条规则会和关于同一状态的其他所有问题一起加入请求阶段的调用，所以对已经在询问内置信号的请求来说，多一个问题几乎不增加延迟。

## 3. 用模型能读懂的方式措辞 {#3-word-it-for-a-model}

- **一次只问一件事。** “用户是否报告了缺陷？”比“用户是否报告了缺陷或提出了功能需求？”更好读。后者应拆成一个 `choice`。
- **描述每个选项。** 用几个词说明什么属于这个选项；选项相近时，也说明什么不属于：“Exporting data, reports or CSV files”。
- **覆盖全部情况。** 给 `choice` 加一个 `other` 选项承接其他一切，免得无关请求被迫选择你的某个选项。
- **说明模型读取什么。** 关于对话的问题要点明对话（“用户是否不止一次要求同样的修复？”），并读取对话（`input.history` 或 `input.previous_turns`）。
- **避免否定。** 问“这个请求无害吗？”或“这个请求有害吗？”，不要问“这个请求不是有害的吗？”。
- **等级要有序**，从最低开始，以两到五个为宜；期望等级从 0 开始计。

逐字使用训练过的内置问题（例如选项为 `benign` 和 `jailbreak` 的 `Is this a prompt injection or jailbreak attempt?`），会在 vela3-fast 上使用训练过的适配器。其他问题都是零样本回答；两种都可以。

## 4. 根据答案匹配 {#4-match-on-the-answer}

- `labels` 指定 `choice`、`set` 或 `span` 中哪些选项让规则匹配；条件仍可以用 `label` 指定任意选项。
- `predicate` 是答案必须满足的界限：`noul` 是 P(是)，`score` 是期望等级，其他类型是某个标签的概率。不写时，`noul` 在 0.5 处匹配，`choice` 按选中的选项匹配。
- 条件可以加上自己的 `predicate`，这样两个决策可以以不同强度读取同一条规则。

```yaml
recipes:
  - name: default
    signals:
      systemone:
        - name: product_question
          question:
            type: choice
            instructions: Which part of the product is this request about?
            choices:
              - key: billing
                description: Invoices, payments, refunds or plan changes
              - key: exports
                description: Exporting data, reports or CSV files
              - key: other
                description: Anything else
    decisions:
      - name: billing-sure
        priority: 200
        rules:
          operator: AND
          conditions:
            - type: systemone
              name: product_question
              label: billing
              predicate: {gte: 0.8}
        modelRefs:
          - model: billing-model
```

规则还会发布它的值 `systemone:product_question` 和 `systemone:product_question:<key>`，[投影得分](../tutorials/projection/scores)可以把它们与其他信号组合。

## 5. 直接用模型验证 {#5-check-it-against-the-model}

直接向模型提出这个问题。模型运行时提供的就是 Router 使用的同一个模型；[快速开始](../model-runtime/quickstart)介绍了如何安装：

```bash
vllm-srun serve vllm-sr/vela3-fast --device cpu --port 8100
curl -s localhost:8100/v1/decisions -H 'content-type: application/json' -d '{
  "state": "My March invoice charged me twice. Can I get one refunded?",
  "questions": {"product_question": {"type": "choice",
    "instructions": "Which part of the product is this request about?",
    "choices": [{"key": "billing", "description": "Invoices, payments, refunds or plan changes"},
                {"key": "exports", "description": "Exporting data, reports or CSV files"},
                {"key": "other", "description": "Anything else"}]}}
}'
```

通过 Router 时，`POST /api/v1/routing/preview` 会针对一个请求评估当前配方的所有信号，报告匹配的规则、它们的值和决策，而不调用任何模型；见[管理 API](../api/apiserver)。配置允许查看调试响应头的调用方，会在 `x-vsr-trace` 中看到匹配的规则。

## 6. 调优 predicate {#6-tune-the-predicate}

零样本问题的概率刻度取决于模型和措辞。在规则开始约束流量之前：

1. 在边界两侧各收集几十个真实请求。
2. 直接或通过预览针对每个请求提问。
3. 选出能保持你可接受的误报率的 `predicate`，写到规则上。
4. 更改措辞或决策模型后重复以上步骤。

约束高成本或高风险操作的规则，还应说明未知答案意味着什么：决策的 `rules.on_unknown`，或条件的 `on_error`。

## 替代已移除信号的问题 {#questions-that-replace-removed-signals}

`vllm-sr config migrate` 会替你写好这些问题；这里列出它们的形式，方便阅读和调整：

| 已移除 | 变成 |
| --- | --- |
| `classifiers` 规则 | 基于分类器标签的 `choice` |
| `preferences` | 一个基于各偏好的 `choice`，名为 `preference`；条件指定一个标签 |
| 文本 `embeddings` 规则 | 引用该规则示例的 `noul` |
| 带文本锚点的图片 `embeddings` 规则 | 带 `input.image` 的 `noul` 或 `choice` |
| `actions` | 名为 `action` 的 `choice`，选项为 generate、explain、fix、refactor、test 和 other |
| `kb` | 你基于知识库标签自己编写的 `set` 或 `choice` |

[迁移指南](../installation/migrate-to-v0-5)列出了所有已移除的形式。
