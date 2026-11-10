---
title: System One API
sidebar_label: API
description: System One API 的标准请求、批量形式、模型卡、错误和限制。
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/system-one/api.md"
  outdated: false
---

# System One API {#the-system-one-api}

Router 提问的每个 System One 模型都使用同一套 HTTP API，无论它由内置模型运行时提供、由你自己运行，还是由托管服务提供。Router 会替你调用它；本页用于手动检查模型、编写客户端，以及理解 Router 发出的内容。

| 端点 | 作用 |
| --- | --- |
| `POST /v1/systemone` | 回答请求。任何一个无法回答的问题都会让整个请求失败 |
| `POST /v1/decisions` | 模型运行时对同一调用的形式：请求体相同，失败的问题在 200 响应中带着自己的错误 |
| `GET /v1/models` | 列出运行时用来回答的模型及其模型卡；Router 或 Engine 前端在 `GET /v1/systemone/models` 列出其 System One 模型 |

[模型运行时参考](../model-runtime/reference#decisions)记录了运行时读取的每个字段，其 OpenAPI 契约是权威来源。下面的两种形式属于契约 4.0.0，运行时在 `GET /health` 中以 `api_version` 报告它。Router 拒绝任何其他契约的运行时：Router 和它的模型运行时来自同一个版本。Router 自己的公开 System One API 也接受这两种形式。

## 标准请求 {#the-standard-request}

标准请求是一个状态及关于它的问题：

```json title="POST /v1/systemone"
{
  "model": "vllm-sr/vela3-fast",
  "state": {
    "request": "Our export button fails with a 500 since this morning. Open a ticket.",
    "history": "How do I export my invoices as CSV?"
  },
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "choices": [
        {"key": "billing", "description": "Payments, invoices and refunds"},
        {"key": "engineering", "description": "Outages, errors and bugs"},
        {"key": "other", "description": "Anything else"}
      ]
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does the user report a problem that blocks their work?"
    }
  }
}
```

- `state` 是文本，或由命名部分组成的对象：`request`、`history`、`previous_turns`、`previous_answer`、`tool_results`、`candidate`、`answer` 和 `context`（见[状态及其组成部分](./concepts#the-state-and-its-parts)）。
- `questions` 可以包含任意数量的命名问题。答案按相同名称、相同顺序返回。
- `images` 和 `videos` 把当前轮次的媒体传给模型卡列出该模态的模型。
- 服务端只用一个模型回答时，`model` 可以省略。
- `options.deadline_ms` 约束这次调用；Router 根据规则的 `timeout_ms` 设置它。

要手动检查一个模型，用[模型运行时](../model-runtime/quickstart)启动它，再向它发送请求：

```bash
vllm-srun serve vllm-sr/vela3-fast --device cpu --port 8100
curl -s localhost:8100/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"request": "Our export button fails with a 500 since this morning. Open a ticket."},
  "questions": {"urgent": {"type": "noul",
    "instructions": "Does the user report a problem that blocks their work?"}}
}'
```

## 批量请求 {#the-batch-request}

批量请求是在一次调用中发送一组命名的标准请求，用于针对不同状态的问题：

```json title="POST /v1/systemone"
{
  "model": "vllm-sr/vela3-fast",
  "states": {
    "doc_1": {
      "state": {
        "request": "How long is the refund window?",
        "candidate": "Refunds are accepted within 30 days of purchase."
      },
      "questions": {
        "relevance": {
          "type": "noul",
          "instructions": "Does the candidate passage contain information that helps answer the request?"
        }
      }
    },
    "doc_2": {
      "state": {
        "request": "How long is the refund window?",
        "candidate": "Our offices close on public holidays."
      },
      "questions": {
        "relevance": {
          "type": "noul",
          "instructions": "Does the candidate passage contain information that helps answer the request?"
        }
      }
    }
  }
}
```

- `states` 的每一项都是一个标准请求：有自己的 `state`、自己的 `questions`，以及自己的 `images` 和 `videos`。问题名称在同一项内唯一，不同项之间可以重复。
- 批量请求的 `model`、`options` 和 `assets` 适用于每一项。
- 两种形式互斥。批量请求体没有顶层 `state`、`questions`、`images` 或 `videos`；混用两种形式的请求体，或 `states` 为空的批量请求，会以 400 `invalid_request` 被拒绝。
- 响应为 `{"model", "states", "usage"}`：每一项的答案以相同名称给出，与该项单独请求时得到的答案相同；`usage` 是所有项的总和。响应没有顶层 `answers`。

## 问题 {#questions}

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `type` | 全部 | `choice`、`noul`、`score`、`set` 或 `span`；模型卡声明时还有 `box` 和 `track` |
| `instructions` | 全部 | 用自然语言写的问题 |
| `choices` | `choice`、`noul` | 有序的选项，每个有 `key` 和 `description`；`noul` 的两个 key 是 `false` 和 `true` |
| `levels` | `score` | 等级描述，从低到高 |
| `labels` | `set`、`span` | 标签，每个有 `key` 和可选的 `description` |
| `threshold` | `set`、`span` | 替换模型自己的判定阈值；实际采用的值在 `thresholds` 中返回 |
| `head` | `span` | 在有 broad span head 的模型上为 `router` 或 `broad` |
| `preset` | 全部 | 模型自己定义的问题（例如 `pii`），列在其模型卡的 `presets` 中 |
| `over` | 全部 | 问题读取的状态部分 |
| `overflow` | 全部 | 在有扫描预算的模型上，`window`（分窗口完整读取长部分）或 `truncate`（只读前面的 token） |

`choices`、`levels` 和 `labels` 是 System One `criteria` 的有序形式；一个问题只能使用其中一种写法。问题只接受其类型的字段，其他字段会以 `invalid_question` 被拒绝。

## 响应 {#the-response}

```json title="Response"
{
  "model": "vllm-sr/vela3-fast",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "engineering",
      "probabilities": {"billing": 0.04, "engineering": 0.93, "other": 0.03},
      "confidence": 0.9
    },
    "urgent": {"type": "noul", "noul": 0.88}
  },
  "usage": {"input_tokens": 212, "output_tokens": 0}
}
```

| 类型 | 答案字段 |
| --- | --- |
| `choice` | `choice`、每个选项的 `probabilities`、`confidence` |
| `noul` | `noul`：P(是) |
| `score` | `score`（期望等级）、每个等级的 `probabilities`、`confidence`、`legend` |
| `set` | `sets.<question>`：每个标签的概率和选中的标签；组合出的 set 还会在 `<question>.<label>` 下为每个标签回答一个 `noul` |
| `span` | `spans.<question>`：每个 span 的 `label`、`start` 和 `end`（字符偏移，不含结尾）、`text` 和 `probability` |

批量响应在 `states.<name>` 下为每一项给出这些字段。

## 模型卡 {#model-cards}

运行时的 `GET /v1/models` 和远程 API 的 `GET <endpoint>/models` 列出每个模型及其模型卡。Router 在加载配置时（或挂接的运行时启动时）读取它，并据此检查每条规则：

| 模型卡字段 | Router 检查 |
| --- | --- |
| `question_types` | 部署能回答解析到它的规则所问的每种问题类型 |
| `presets` | 模型卡列出时，需要 span 的预设（`pii`、`halu`）由模型自己的问题回答 |
| `modalities` | 带 `input.image` 或 `input.video` 的规则解析到能读取该媒体的模型 |
| `limits` | 选项、标签和等级在模型的范围内；输入 token 限制 |

没有为该模型列出模型卡的远程 API，改由部署的 `capabilities` 描述（见[选择部署](./deployments#remote-system-one-apis)）。

## 错误和限制 {#errors-and-limits}

在 `POST /v1/systemone` 上，按请求顺序第一个无法回答的问题决定整个请求的响应。响应体的 `code` 为 `question_failed`，并给出失败的 `question`、它的 `question_error` 和 `reason`；在批量请求中，还以 `state` 给出该问题所在条目的名称：

| 问题错误 | 状态码 | 含义 |
| --- | --- | --- |
| `invalid_question` | 422 | 问题格式错误，或模型不支持该类型、字段或预设 |
| `max_length_exceeded`、`scan_budget_exceeded`、`invalid_input` | 400 | 输入超过模型能读取的长度，或格式错误 |
| `invalid_model_output` | 500 | 模型没有产生有效答案 |
| `unavailable` | 503 | 模型尚未就绪 |
| `deadline_exceeded` | 504 | 超过了 `options.deadline_ms` |

整体过大的请求以 413 `request_too_large` 拒绝，过载的服务端返回 429 `overloaded`。既不是标准请求也不是批量请求的请求体（例如把 `state` 或 `questions` 与 `states` 混用，或 `states` 为空）以 400 `invalid_request` 拒绝。每个问题的选项为 2 到 255 个、标签 1 到 255 个、等级 2 到 10 个；问题数量不设限制。

Router 收到错误时，这次调用中的问题都是未知的：由各规则或功能的失败策略决定其含义，而评审永远不会因为候选太多而让请求失败。
