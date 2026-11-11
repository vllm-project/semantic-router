---
title: System One API
sidebar_label: API
sidebar_position: 1
description: System One API（vllm-sr/5.0）的单个请求、批量端点、roles、媒体、模型卡、错误和限制。
is_mtpe: false
translation:
  source_commit: "6b782822edead98397acc6401112e6ec1e45af96"
  source_file: "docs/system-one/use/api.md"
  outdated: false
---

# System One API {#the-system-one-api}

每个 System One 模型都回答同一套 HTTP API，无论它由内置模型运行时提供、由你自己运行，还是由托管服务提供。Router 会替你调用它；本页用于手动检查模型、编写客户端，或了解 Router 发送了什么。

vllm-sr 使用 `vllm-sr/5.0`：已发布的 System One 标准 `systemone/0.2`，加上一组放在自然位置的扩展字段。任何标准客户端或 SDK 都可以直接使用；SDK 用户通过 `extra_body` 传递扩展字段。仍需服务旧契约客户端的服务器使用 `vllm-sr/4.0`，它同时接受两种形式；见 [System One schema](./schemas.md)。

## 试一试 {#try-it}

用[模型运行时](../../model-runtime/get-started/quickstart.md)运行一个模型，然后问它一个问题：

```bash
vllm-srun serve vllm-sr/vela3-fast --device cpu --port 8100
curl -s localhost:8100/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Our export button fails with a 500 since this morning. Open a ticket.",
  "questions": {"urgent": {"type": "noul",
    "instructions": "Does the user report a problem that blocks their work?"}}
}'
```

`urgent` 的答案是 P(是)，例如 `{"type": "noul", "noul": 0.88}`。

## 端点 {#endpoints}

| 端点 | 作用 |
| --- | --- |
| `POST /v1/systemone` | 回答一个请求。某个问题无法回答时，整个请求失败 |
| `POST /v1/systemone/batch` | 一次调用回答多个具名请求，每个请求的回答与单独发送时相同 |
| `POST /v1/decisions` | 同一调用的原生形式，供 Router 使用：内部请求形式，失败的问题在 200 中带自己的错误 |
| `GET /v1/models` | 列出服务器回答用的模型，每个带 `version`、`features` 和 `card` |

运行时在 `GET /health` 中以 `api_version` 报告其契约（`5.0.0`）。

:::note
Router 拒绝任何其他契约的运行时：Router 和它的模型运行时来自同一个版本。[HTTP API 参考](../../model-runtime/reference/http-api.md#decisions)记录了运行时读取的每个字段；其 OpenAPI 契约是事实来源，[扩展参考](../reference/extension-reference.md)列出了 vllm-sr 在标准之上增加的每个字段。
:::

## 单个请求 {#a-single-request}

一个请求是一个 state 以及关于它的问题：

```json title="POST /v1/systemone"
{
  "model": "vllm-sr/vela3-fast",
  "state": {
    "request": "Our export button fails with a 500 since this morning. Open a ticket.",
    "history": "How do I export my invoices as CSV?"
  },
  "roles": {"request": "user", "history": "context"},
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Payments, invoices and refunds",
        "engineering": "Outages, errors and bugs",
        "other": "Anything else"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Does the user report a problem that blocks their work?"
    }
  },
  "options": {"deadline_ms": 1500}
}
```

| 字段 | 含义 |
| --- | --- |
| `state` | 文本、对象或数组。它的键本身没有含义 |
| `roles` | 扩展。哪些 state 键是 `user` 的内容、`context`，或待检查的 `answer`，供区分角色的模型族（Vela）读取。没有 `roles` 时，整个 state 在所有模型族上都是用户内容 |
| `questions` | 任意数量的具名问题。答案按相同名称、相同顺序返回 |
| `assets` | 按占位符给出的图片和视频（见[媒体](#media)） |
| `model` | 服务器只用一个模型回答时可省略 |
| `options` | 扩展。`deadline_ms` 限定调用时长（Router 根据规则的 `timeout_ms` 设置），`return_meta` 增加 `meta`，`max_tokens` 是分窗读取长部分的模型的扫描预算 |

超出模型验证长度的请求发送标准请求头 `TypeSafe-Danger-Zone-Long-Context: true`。服务器对 schema 未定义的任何成员返回 400，并写明其名称。

模型族如何把对象 state 变成模型输入由它自己决定，就像对话模板：Decision 3 把它读作 `json.dumps` 文本，Vela 按角色读取各部分。对区分角色的模型族请发送 `roles`；Router 自己构建多部分 state 时，会发送其部分名称（`request`、`history`、`previous_turns`、`previous_answer`、`tool_results`、`candidate`、`answer`、`context`）一直以来的角色。

## 批量请求 {#the-batch-request}

批量请求是一次调用中的一组具名请求，用于关于不同 state 的问题：

```json title="POST /v1/systemone/batch"
{
  "model": "vllm-sr/vela3-fast",
  "requests": {
    "doc_1": {
      "state": {"request": "How long is the refund window?", "candidate": "Refunds are accepted within 30 days of purchase."},
      "roles": {"request": "user", "candidate": "context"},
      "questions": {"relevance": {"type": "noul", "instructions": "Does the candidate passage help answer the request?"}}
    },
    "doc_2": {
      "state": {"request": "How long is the refund window?", "candidate": "Our offices close on public holidays."},
      "roles": {"request": "user", "candidate": "context"},
      "questions": {"relevance": {"type": "noul", "instructions": "Does the candidate passage help answer the request?"}}
    }
  },
  "options": {"deadline_ms": 3000}
}
```

- `requests` 的每一项都是完整请求：自己的 `state`、`roles`、`questions` 和 `assets`。它可以重复批量的 `model`；`options` 属于整个批量。
- 响应为 `{"model", "responses", "usage"}`：每个请求的响应在相同名称下，与单独发送时相同，`usage` 为所有请求之和。
- 失败的问题使整个批量失败，错误的 `state` 是其请求的名称。`/v1/systemone` 只接受一个请求。

## 问题 {#questions}

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `type` | 全部 | 标准的 `choice`、`noul`、`score` 和 `box`，扩展类型 `set` 和 `span`，以及模型卡声明时的 `track` |
| `instructions` | 全部 | 用自然语言写的问题 |
| `criteria` | `choice`、`noul`、`score`、`set`、`span` | `choice` 的选项（键到描述的对象）、`noul` 的 `true` 和 `false`、`score` 的等级（数组，从低到高），或 `set` 与 `span` 的标签 |
| `preset` | 全部 | 扩展。模型训练过的问题（例如 `pii`），列在其模型卡的 `presets` 中；问题仍带有自己的 `instructions` 和 `criteria`，其他服务器按它们回答 |
| `threshold` | `set`、`span` | 扩展。替换模型自己的判定阈值；实际使用的值在 `thresholds` 中返回 |
| `head` | `span` | 扩展。`router` 或 `broad`，用于有 broad span head 的模型 |
| `over` | 全部 | 扩展。问题读取的 state 键，仅用于 Vela 模型：Vela 2.0 上任何问题都可用；Vela 3.0 上限于 `span`、预设问题和 vela3-fast 训练过的路由问题。Decision 1.0、2.0 和 3.0 以 `invalid_question` 拒绝 |
| `overflow` | 全部 | 扩展。`window`（分窗完整读取长部分）或 `truncate`（只读前面的 token），用于有扫描预算的模型 |
| `require_full_input` | 全部 | 扩展。除非完整读取全部输入，否则问题失败；此时答案带 `input_coverage: complete` |

## 媒体 {#media}

图片和视频是 `assets`，每个在一个占位符字符串下，占位符出现的位置就代表它。占位符可以在 state 中，也可以在模型卡允许的模型上出现在问题的 `instructions` 或 `criteria` 中。

Router 把一轮的媒体放在 state 前面：`"state": ["⟦img-1⟧", "⟦vid-1⟧", "What happens here?"]`。vllm-sr/5.0 服务器把这样的 state（依次为图片占位符、视频占位符，然后恰好一个其他元素）读作该元素，媒体在它前面，就像 Decision 3 读取一轮的图片那样。

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
| `set` | `sets.<question>`：每个标签的概率和选中的标签；组合式 set 还会在 `<question>.<label>` 下为每个标签回答一个 `noul` |
| `span` | `spans.<question>`：每个片段的 `label`、`start` 和 `end`（字符，end 不含）、`text` 和 `probability` |

## 模型卡 {#model-cards}

`GET /v1/models` 列出每个模型的标准字段 `name`、`description` 和 `release_date`，以及扩展字段 `version`（`5.0`）、`features`（它回答的问题类型）和 `card`。Router 在配置加载时或附加的运行时启动时读取模型卡，并据此检查每条规则：

| 模型卡字段 | Router 检查 |
| --- | --- |
| `question_types` | 部署回答指向它的规则所问的每种问题类型 |
| `presets` | 需要 span 的 preset（`pii`、`halu`）在模型卡列出时由模型自己的问题回答 |
| `modalities` | 带 `input.image` 或 `input.video` 的规则指向能读取该媒体的模型 |
| `limits` | 选项、标签和等级在模型的范围内；输入 token 上限 |

没有为该模型列出模型卡的远程 API，改由部署的 `capabilities` 描述（见[选择部署](../deploy/deployments.md#remote-system-one-apis)）。

## 错误和限制 {#errors-and-limits}

错误是标准的 `detail`，加上扩展的平铺字段 `code`、`question`、`state`（批量中为请求名称）、`reason`（固定词）和 `limit`。在 422 上，`detail` 是标准的列表，每项定位失败的问题：

```json title="422"
{"detail": [{"loc": ["body", "questions", "team"], "msg": "question \"team\" cannot be answered: the question is malformed", "type": "invalid_question"}],
 "code": "question_failed", "question": "team", "reason": "question_malformed"}
```

在 `POST /v1/systemone` 上，按请求顺序第一个无法回答的问题使整个请求以 `code` `question_failed` 失败：

| 问题错误 | 状态码 | 含义 |
| --- | --- | --- |
| `invalid_question` | 422 | 问题格式错误，或模型不回答该类型、字段或 preset |
| `max_length_exceeded`、`scan_budget_exceeded`、`invalid_input` | 400 | 输入长于模型能读取的长度，或格式错误 |
| `invalid_model_output` | 500 | 模型没有给出有效答案 |
| `unavailable` | 503 | 模型尚未就绪 |
| `deadline_exceeded` | 504 | 已超过 `options.deadline_ms` |

- 整体过大的请求以 413 `request_too_large` 拒绝；请求体最大 32 MiB。
- 过载的服务器返回 529 并带 `Retry-After`；正在启动的服务器返回 503 并带 `Retry-After`。
- 带有 schema 未定义成员的请求体以 400 `invalid_request` 拒绝，并写明其名称。
- 每个问题的选项为 2 到 255 个，标签 1 到 255 个，等级 2 到 10 个；问题数量没有上限。

Router 收到错误时，该调用的问题为未知：每条规则或功能的失败策略决定其含义。候选过多的 judge 永远不会使请求失败。
