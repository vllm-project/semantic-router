---
title: 编写自定义问题
sidebar_label: 编写自定义问题
sidebar_position: 3
description: 把一个路由想法变成 systemone 规则，直接对模型验证它，并调整其谓词。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/system-one/use/custom-questions.md"
  outdated: false
---

# 编写自定义问题 {#write-custom-questions}

应用信号没有覆盖的路由想法，例如“这是关于我们产品的问题吗？”或“这需要调用工具吗？”，就是一条 `systemone` 规则：用自然语言写的问题，由决策模型零样本回答，无需任何训练。本页从想法讲到一条调好的规则；[System One 信号](../../tutorials/signal/systemone.md)页面是字段参考。

```yaml title="A question and the route it gates"
recipes:
  - name: default
    signals:
      systemone:
        - name: needs_tools
          question:
            type: noul
            instructions: Does answering this request need a tool call?
          predicate: {gte: 0.7}
    decisions:
      - name: tool-use
        priority: 100
        rules:
          operator: AND
          conditions:
            - {type: systemone, name: needs_tools}
        modelRefs:
          - model: tool-model
```

## 1. 选择问题类型 {#1-choose-the-question-type}

| 想法 | 类型 | 匹配依据 |
| --- | --- | --- |
| “这是 X 吗？” | `noul` | P(是) |
| “它属于这几类中的哪一类？”，恰好一类 | `choice` | 选中的选项，或某个选项的概率 |
| “X 的程度如何？”，在有序刻度上 | `score` | 期望等级 |
| “这些中哪些适用？”，任意数量 | `set` | 某个标签的概率 |
| “X 在文本的什么位置？” | `span` | 某个标签的 span；需要支持 span 的模型 |

:::tip
当几个 `noul` 问题本应互斥时，优先用带 `other` 选项的 `choice`；当选项本身有顺序时，优先用 `score` 而不是 `choice`。
:::

## 2. 编写规则 {#2-write-the-rule}

```yaml title="config.yaml"
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

没有 `deployment` 的规则会询问决策模型；参见[选择部署](../deploy/deployments.md)。这条规则与同一状态的其他问题一起加入请求阶段的那次调用，因此对于已经在询问内置信号的请求，多一个问题几乎不增加延迟。

## 3. 为模型措辞 {#3-word-it-for-a-model}

- **一次只问一件事。** “用户是否报告了 bug？”比“用户是否报告了 bug 或提出了功能需求？”更好。把后者拆成一个 `choice`。
- **描述每个选项。** 用几个词说明什么属于该选项；当选项相近时，也说明什么不属于：“导出数据、报表或 CSV 文件”。
- **覆盖全部情况。** 给 `choice` 加一个 `other` 选项表示其他一切，这样无关的请求不必硬选你的某个选项。
- **说明模型读取什么。** 关于对话的问题要点明对话（“用户是否不止一次要求同一个修复？”），并读取它（`input.history` 或 `input.previous_turns`）。
- **避免否定。** 问“这个请求无害吗？”或“这个请求有害吗？”，而不是“这个请求不是有害的吗？”。
- **等级按顺序排列**，从最低开始，两到五个；期望等级从 0 开始计。

:::note
逐字使用的训练过的内置问题，例如带 `benign` 和 `jailbreak` 选项的 `Is this a prompt injection or jailbreak attempt?`，会在 vela3-fast 训练过的适配器上运行。其他问题都以零样本方式运行；两者都没问题。
:::

## 4. 依据答案匹配 {#4-match-on-the-answer}

- `labels` 指定 `choice`、`set` 或 `span` 的哪些选项匹配该规则；`choice` 上的条件仍可用 `label` 指定任意选项，`set` 或 `span` 上的条件只能指定计入的标签。
- `predicate` 是答案必须满足的界限：`noul` 是 P(是)，`score` 是期望等级，其他是某个标签的概率。没有它时，`noul` 在 0.5 匹配，`choice` 匹配其选中的选项。
- 条件可以加上自己的 `predicate`，因此两个决策可以用不同强度读取同一条规则。

```yaml title="A condition with its own predicate"
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

这条规则还会发布它的值 `systemone:product_question` 和 `systemone:product_question:<key>`，[投影分数](../../tutorials/projection/scores.md)可以把它们与其他信号组合。

## 5. 直接对模型验证 {#5-check-it-against-the-model}

直接向模型提问。模型运行时提供的正是 Router 使用的同一个模型；[快速开始](../../model-runtime/get-started/quickstart.md)介绍了如何安装：

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

通过 Router，`POST /api/v1/routing/preview` 会针对一个请求评估当前 recipe 的每个信号，并报告匹配的规则、它们的值和决策，而不调用模型；参见[管理 API](../../api/apiserver.md)。配置允许查看调试头的调用方，可以在 `x-vsr-trace` 中看到匹配的规则。

## 6. 调整谓词 {#6-tune-the-predicate}

零样本问题的概率尺度取决于模型和措辞。在规则用于拦截流量之前：

1. 在边界两侧各收集几十个真实请求。
2. 直接或通过 preview 对每个请求提出该问题。
3. 选择能保持你可接受的误报率的 `predicate`，并写在规则上。
4. 更改措辞或决策模型时，重复以上步骤。

:::warning
用于拦截高成本或高风险操作的规则，还应说明未知答案的含义：决策的 `rules.on_unknown`，或条件的 `on_error`。
:::

## 替代已移除信号的问题 {#questions-that-replace-removed-signals}

`vllm-sr config migrate` 会为你写好这些；这里列出这些形式，便于阅读和调整：

| 已移除 | 变为 |
| --- | --- |
| `classifiers` 规则 | 覆盖该分类器标签的 `choice` |
| `preferences` | 一个覆盖各偏好的 `choice`，名为 `preference`；条件指定一个标签 |
| 文本 `embeddings` 规则 | 引用该规则示例的 `noul` |
| 带文本锚点的图像 `embeddings` 规则 | 带 `input.image` 的 `noul` 或 `choice` |
| `actions` | 覆盖 generate、explain、fix、refactor、test 和 other 的 `choice`，名为 `action` |
| `kb` | 你为知识库标签编写的 `set` 或 `choice` |

[迁移指南](../../installation/migrate-to-v0-5.md)列出了每一种已移除的形式。
