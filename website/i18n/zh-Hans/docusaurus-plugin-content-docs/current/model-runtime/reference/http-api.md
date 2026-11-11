---
title: HTTP API 参考
sidebar_label: HTTP API
sidebar_position: 2
description: 模型运行时的 worker API：端点、响应与错误、长输入、分类、嵌入、重排、打包、决策，以及 System One 兼容性。
is_mtpe: true
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/model-runtime/reference/http-api.md"
  outdated: true
---

# HTTP API 参考 {#http-api-reference}

每个 `vllm-srun` worker 都提供同一套 HTTP API。Router 前端把其中的 System One 部分发布给客户端；其余部分是供你自己运行的运行时使用的 worker API。

## 端点 {#http-api}

每个请求都可以带自己的 `model`；一个进程服务多个模型时必须带。契约就是 `GET /openapi.yaml` 给的 OpenAPI 文档。

| 端点 | 用途 |
| --- | --- |
| `POST /v1/decisions` | 就一个状态答 Choice、Noul、Score、Set 和 Span 问题，失败的问题在其答案中返回。它读取运行时的内部请求形式（下文的形式加上 `roles`），与进程的 System One schema 无关。 |
| `POST /v1/systemone` | 一个按进程 System One schema 发送的请求，按 System One 的方式回答：有问题失败时整个请求失败（[System One 兼容性](#system-one-compatibility)）。 |
| `POST /v1/systemone/batch` | 一次调用中的多个具名请求，每个的回答与单独发送时相同，放在 `responses` 下。 |
| `POST /v1/classify` | 对文本、文本对或 grounded 答案跑分类头：标签概率、独立标签分或 token 片段。 |
| `POST /v1/embeddings` | 文本、图像、音频的 OpenAI 兼容嵌入，带 `dimensions` 和 `layer`。 |
| `POST /v1/rerank` | 就一个查询给文档打分。 |
| `POST /v1/bundle` | 上面几种的一次调用凑一批，对一个或多个模型。 |
| `GET /v1/models` | 每个模型是什么、服务什么、标签、上限、设备和自检状态，以及进程的 `limits`：一个 `/v1/bundle` 最多多少任务、请求体多大。配合 `/health` 和 `/health/live`，它还报告契约的 `api_version`。 |
| `GET /health` | 就绪：所有模型就绪返 200，否则 503。`status` 是 `starting`、`loading`、`warming`、`ready`、`degraded` 或 `failed`；服务多模型的进程会逐个模型列状态。 |
| `GET /health/live` | 存活：进程还在服务请求就返 200 带 `"status": "alive"`，模型就绪与否不论。 |
| `GET /metrics` | Prometheus 指标。 |

响应带答案和 `usage`。请求里加 `"options": {"return_meta": true}`，还能拿到 `meta`：答这次请求的 revision、模型摘要、`dtype`、`numerics`、引擎、设备和耗时。模型的答案从不受批处理影响时（每行单独跑和在任意批次里跑都一样，且使用发布精度），`numerics` 为 `exact`；一次前向里的其他行可能因舍入改变答案时为 `approximate`。`GET /v1/models` 报告模型的 `engine_args` 和 `numerics`。

`/v1/decisions`、`/v1/systemone`、`/v1/classify`、`/v1/embeddings`、`/v1/rerank` 和 `/v1/bundle` 的响应——出错的也算——带一个 `Server-Timing` 头，是 runtime 为这个请求花的自己的时间，单位毫秒：

```text
Server-Timing: parse;dur=0.021, tokenize;dur=0.153, queue;dur=0.008, forward;dur=4.871, post;dur=0.034, serialize;dur=0.019, total;dur=5.141
```

| 阶段 | 时间花在 |
| --- | --- |
| `parse` | 读取和解码请求体。 |
| `tokenize` | 校验、渲染和分词这个请求。 |
| `queue` | 等模型，在它第一次前向之前和两次前向之间。 |
| `forward` | 跑它的前向，读出包含在内。 |
| `post` | 组装答案。 |
| `serialize` | 编码响应。 |
| `total` | 从 handler 开始到响应可以发送。超出各阶段的部分是服务器自己的开销。 |

任务分发到多个模型的 bundle，报告的是最后答完那个模型的 `queue`、`forward` 和 `post`。客户端自己这次调用的时间减去 `total` 就是传输：它的编解码、连接和 HTTP 交换。router 把它作为每次调用的指标记下来。

完全服务不了的请求返 HTTP 错误，带 `{"error": {"code", "message"}}`：400 `invalid_request`、404 `model_not_found`、413 `request_too_large`、422 `unsupported_surface`（模型不服务这个端点）、429 `overloaded`、503 `not_ready`。前向计算显存不足时，请求独占这次前向返回 413 `request_too_large`，与其他请求共享时返回 429 `overloaded`。部署的所有 worker 都在忙时，Router 同样返回 429 `overloaded`；没有就绪 worker 时返回 503 `not_ready`；两者都带 `Retry-After`（1 秒和 5 秒）。Engine 模式下，调用超过 Router 的 30 秒预算时返回 504 `systemone_deadline_exceeded`。请求里单个条目失败（一个输入或一个问题）只带自己的错误码，不牵连其他条目。

## 长输入 {#long-inputs}

模型读一个输入，最多读到它的 token 预算：`options.max_tokens`，且不超过模型的 `max_input_tokens`。更长的输入，runtime 只把它分词到预算足以定答案的程度，所以一个输入花的内存和时间跟它的预算走，不跟它的长度走：

- 它在词边界处切文本（空白、标点、符号，或中日韩字符），保留切点之前那些词的 token。遇到分词器根本不拆的连续串时（base64、hex，或按空格分词的分词器遇到的无空格中文），它在字母或数字前切，并保留在切点前 1,024 个字符内结束的 token。
- 在 `reject` 下，长度超过「预算 × 模型单个 token 最多覆盖的字符数」的文本，不分词就直接失败。

这些 token 和读完整文本给出的是一样的，所以答案相同。runtime 第一次用一个分词器时，会在探针文本上检查每一种切法，切法检查不过关的文本整本读。会丢字符或折叠字符的分词器——比如 WordPiece 丢空白、把超过 100 字符的词当一个未知 token——从不在连续串内部切：这样的串整本读，好在串本身只有几个 token，代价很小。只读了一部分的输入，其 usage 按读到的 token 计（会超过预算），并加 `"tokens_lower_bound": true`。

按窗口读输入的模型，读到扫描预算为止，单位 token。`overflow: window` 的分类最多读 `max_tokens`。Vela 2.0 的问题把长的 state 部分在窗口里整读，上限是它卡片上的 `limits.max_scan_tokens`：CPU 上四个输入（Vela 2.0 0.3B 是 32,768 token），GPU 上 32 个。决策请求的 `options.max_tokens`，或 models 文件选项 `max_scan_tokens`，设另一个值。带 `overflow: truncate` 的问题只读该部分的前 `limits.truncate_tokens`：CPU 上一个输入，一次前向。Vela 3.0 的 evidence span 问题改为截到它自己的 32,768 token 窗口。两类问题共享它们的模型输入，和没有 `overflow` 时一样，除非某部分必须按窗口读。超过扫描预算的输入直接失败，不读：

| 条目错误 | 含义 |
| --- | --- |
| `max_length_exceeded` | 输入在 `reject` 下的 token 数超过其预算。 |
| `scan_budget_exceeded` | 输入是模型按窗口读的那种，且 token 数超过其扫描预算。它一点都没被读。 |

唯一的字节限额是 `--max-request-bytes`，作用于整个请求。

router 通过 Vela 2.0 用两种方式读长请求。路由问题（domain、fact check、feedback、modality、决策问题和决策模型选择器）截断。安全问题（jailbreak、safety、PII 和 hallucination）整读，上限是模型的扫描预算或部署的 `input.max_tokens`（带 `overflow: window`）。超出部分是没读到的内容：越狱或 PII 规则把它匹配为 `unscanned`。没赶上信号截止时间（`global.model_catalog.signal_timeout_ms`）的安全扫描没有答案：越狱或 PII 规则此时不匹配，除非它设置了 `on_unknown: match`。

## 分类 {#classify}

```json title="POST /v1/classify"
{
  "model": "vela-pii",
  "input": ["Hi, I'm Tom Baker (tom.baker@example.com)."],
  "options": {"overflow": "window", "window": {"tokens": 512, "overlap": 64}}
}
```

`input` 可以是字符串、字符串列表、`{"text", "text_pair"}` 对列表，或幻觉检查用的 `{"context", "question", "answer"}` 条目列表。`options.overflow` 是 `reject`、`truncate` 或 `window`。片段偏移按 Unicode 码点算，尾巴不含。

## 嵌入 {#embeddings}

```json title="POST /v1/embeddings"
{"model": "vela-embedding", "input": ["How do I reset my password?"], "dimensions": 256, "layer": 11}
```

模型声明了的话，`dimensions` 和 `layer` 用来选更小的向量或更早的层。带 `return_meta` 时，`meta.representation` 标明向量空间，不同设导向量可以分开存。

## 重排 {#rerank}

```json title="POST /v1/rerank"
{"model": "vela-reranker", "query": "How do I reset my password?",
 "documents": ["Open Settings, then Security.", "Our offices are closed on Sunday."], "top_n": 1}
```

## 打包 {#bundle}

```json title="POST /v1/bundle"
{
  "tasks": [
    {"id": "domain", "classify": {"model": "vela-domain", "input": ["What is 2 + 2?"]}},
    {"id": "kind", "decisions": {"model": "decision-edge", "state": "What is 2 + 2?",
      "questions": {"math": {"type": "noul", "instructions": "Is this a math question?"}}}}
  ],
  "options": {"deadline_ms": 80}
}
```

结果按任务顺序返回，每条都带它单独跑时会有的状态。

## 决策 {#decisions}

请求给一个 `state` 和它的 `questions`。一个问题的选项是 `criteria`，或 router 配置里表达同一件事的有序列表：

| 类型 | `criteria` | 或，router 配置里的写法 |
| --- | --- | --- |
| `choice` | `{key: description}`，2 到 255 个选项 | `choices: [{key, description}]` |
| `noul` | `{false: description, true: description}`，两者都可选 | `choices: [{key, description}]`，键为 `false` 和 `true` |
| `score` | `[description, ...]`，2 到 10 级 | `levels: [description, ...]` |
| `set`、`span` | `{label: description}`，1 到 255 个标签 | `labels: [{key, description}]` |

一个问题只用一种形式，且只用其类型的字段。下面两个请求是 Vela 2.0 在 CPU 上答的，数字是四舍五入后的。

```json title="POST /v1/decisions"
{
  "model": "Vela-2.0-0.3B",
  "state": "Write a Python function that merges two sorted lists and explain its running time.",
  "questions": {
    "domain": {
      "type": "choice",
      "instructions": "Which domain does this request belong to?",
      "choices": [
        {"key": "code", "description": "Programming"},
        {"key": "math", "description": "Mathematics"},
        {"key": "other"}
      ]
    },
    "reasoning": {"type": "noul", "instructions": "Does answering this request need multi-step reasoning?"},
    "difficulty": {
      "type": "score",
      "instructions": "How difficult is this request?",
      "levels": ["Trivial", "Moderate", "Hard"]
    }
  }
}
```

```json title="Response"
{
  "model": "Vela-2.0-0.3B",
  "answers": {
    "domain": {
      "type": "choice",
      "choice": "code",
      "confidence": 0.99,
      "probabilities": {"code": 1.0, "math": 0.0, "other": 0.0},
      "abstain_probability": 0.01
    },
    "reasoning": {"type": "noul", "noul": 0.34},
    "difficulty": {
      "type": "score",
      "score": 1.3,
      "confidence": 0.28,
      "legend": {"0": "Trivial", "1": "Moderate", "2": "Hard"},
      "probabilities": {"0": 0.13, "1": 0.43, "2": 0.43}
    }
  }
}
```

Set 把每个标签作为一个 Noul 答在 `<id>.<label>` 下，并在 `sets` 里列出达到其阈值的标签。Span 在 `spans` 里返回它找到的文本，带 Unicode 码点偏移，end 不含，外加一个挂在它 ID 下的 Noul。`thresholds` 给出每个问题实际用的阈值；在问题上设 `threshold` 可以另选一个。

```json title="POST /v1/decisions"
{
  "model": "Vela-2.0-0.3B",
  "state": "My card was charged twice and the parcel never arrived. Write to me at jane.doe@example.com.",
  "questions": {
    "problems": {
      "type": "set",
      "instructions": "Which problems does the customer report?",
      "labels": [
        {"key": "billing", "description": "A payment or charge problem"},
        {"key": "shipping", "description": "A delivery problem"},
        {"key": "account", "description": "A login or account problem"}
      ]
    },
    "contacts": {
      "type": "span",
      "instructions": "Find the personal contact details.",
      "labels": [{"key": "EMAIL_ADDRESS", "description": "An email address"}]
    }
  }
}
```

```json title="Response"
{
  "model": "Vela-2.0-0.3B",
  "answers": {
    "problems.billing": {"type": "noul", "noul": 0.95},
    "problems.shipping": {"type": "noul", "noul": 0.7},
    "problems.account": {"type": "noul", "noul": 0.03},
    "contacts": {"type": "noul", "noul": 1.0}
  },
  "spans": {
    "contacts": [
      {"label": "EMAIL_ADDRESS", "start": 71, "end": 91, "text": "jane.doe@example.com", "probability": 1.0}
    ]
  },
  "sets": {
    "problems": {
      "selected": ["billing", "shipping"],
      "probabilities": {"billing": 0.95, "shipping": 0.7, "account": 0.03}
    }
  },
  "thresholds": {"problems": 0.3, "contacts": 0.65}
}
```

无效的问题答 `{"type", "error": "invalid_question", "message"}`，message 指出字段，比如 `set questions do not take ['colour']`；其他问题照常答。Vela 3.0 只按 instructions 读 Noul，描述了 `false` 或 `true` 的 Noul 答 `invalid_question`。一个问题都无效的请求答 400 `invalid_request`，每个问题的理由在 message 里。

请求要么是标准请求（上面那样的一个 `state` 和它的 `questions`），要么是批量请求：放在 `states` 下的若干具名标准请求，一次调用答完。每一条含一个 `state` 和它自己的 `questions`，也可以带 `images` 和 `videos`；读法和带它们、再加上批量请求的 `model`、`options` 和 `assets` 的标准请求完全一样，在响应的 `states` 里按同名答回。某个条目的图像、视频、资源或 token 数被拒绝时，该条目的每个问题都以 `invalid_input`（原因 `input_malformed`）和拒绝的消息失败，其他条目照常回答；每个条目都被拒绝的批量请求以第一个拒绝作答。两种形式互斥：批量请求没有顶层的 `state`、`questions`、`images` 或 `videos`，混用两种形式的请求答 400 `invalid_request`。一个请求阶段要让一个部署读多段文本时，router 就发批量请求：`relevance` 或 `same_request` 判断要核对的候选（每个一条），或能看历史的越狱或 PII 规则读到的先前消息。关于同一段文本的问题合在同一条里。

```json title="POST /v1/decisions"
{
  "model": "Vela-2.0-0.3B",
  "states": {
    "doc_1": {
      "state": {"request": "How long is the refund window?", "candidate": "Refunds are accepted within 30 days of purchase."},
      "questions": {"relevance": {"type": "noul", "instructions": "Does the candidate passage contain information that helps answer the request?"}}
    },
    "doc_2": {
      "state": {"request": "How long is the refund window?", "candidate": "Our offices close on public holidays."},
      "questions": {"relevance": {"type": "noul", "instructions": "Does the candidate passage contain information that helps answer the request?"}}
    }
  }
}
```

响应是 `{"model", "states": {"doc_1": {"model", "answers", "usage", ...}, "doc_2": {...}}, "usage"}`：每一条的答案和它单独请求时一样，`usage` 计入所有条目。响应自己没有 `answers`；带 `return_meta` 时，它的 `meta` 就是每一条都带的那个。

`src/model-runtime/tools/reference_examples.py --url <runtime>` 把本节的请求发给一个服务 Vela 2.0 0.3B 的 runtime，并核对答案对得上。

## System One 兼容性 {#system-one-compatibility}

`POST /v1/systemone` 和 `POST /v1/systemone/batch` 使用进程的 System One schema，由 `vllm-srun serve --systemone-schema` 设置：

- **`vllm-sr/5.0`**（默认）：System One 标准加上 vllm-sr 扩展。请求被严格解析，其他成员以 400 拒绝并写明名称；没有 `roles` 的对象 state 中没有任何部分有角色；占位符位于 state 前面的资源（依次为图片、视频，然后恰好一个其他元素）在该元素前面读取；`TypeSafe-Danger-Zone-Long-Context: true` 读取长上下文。错误是标准的 `detail`（422 时为列表）加平铺的 `code`、`question`、`state`、`reason` 和 `limit`，过载的进程返回 529 并带 `Retry-After`。见 [System One schema](../../system-one/use/schemas.md)。
- **`vllm-sr/4.0`**：原样接受下文的形式以及所有 5.0 形式；错误在 `error` 旁增加 `detail` 和平铺字段。用于直接服务旧契约客户端的运行时。

在自己的 `instructions` 或 `criteria` 旁写明 `preset` 的类型化问题（5.0 形式），在模型卡列出该 preset 时使用模型训练过的 preset，否则按问题本身回答。

内部形式由 `/v1/decisions`、`/v1/bundle` 和 `vllm-sr/4.0` 进程读取，接受 JSON、压缩 JSON 或 multipart 形式的 System One 请求，并以 System One 字段作答。它也接受上文的扩展（`images`、`videos`、`states` 中的批量请求、Set 和 Span、`choices` / `levels` / `labels`、`options`）以及 `roles`：省略时，对象 state 的键表示其各部分；`null` 时没有任何部分有角色；对象时为每个键的角色。进程不提供的模型会收到一个列出所提供模型的错误。

**请求体。** 请求体可以用一层 `Content-Encoding` 压缩：`gzip`（或 `x-gzip`），或窗口不超过 8 MiB 的 `zstd`；大小限制同时适用于发送时的请求体和解码后的请求体。其他编码或多层编码返回 415。请求体也可以是 `multipart/form-data`：第一部分 `request` 包含 JSON 请求，其余每个部分包含一张图像的字节，并且恰好被一个 asset 引用。

**Assets。** 顶层的 `assets` 对象以引用的方式保存图像，形式为 `{"type": "image", "data_base64": "..."}`，在 multipart 请求体中为 `{"type": "image", "part": "<part name>"}`。每个引用在状态的字符串、问题的 `instructions` 或其选项中出现的地方都会被替换：与引用完全相等的字符串就是那张图像，更长的字符串在每次出现处被拆分，对象的键永远不会被替换。每个 asset 都必须被使用，一个引用为 1 到 1,024 字节（全部引用合计 65,536 字节），每次使用都会再次加入图像：其 token 计在它所在的位置。格式从字节中读取（PNG、JPEG 或 WebP；动图会被拒绝），每张图像最多 16 MiB。只要有一张图像被拒绝，请求就会失败。

```json title="POST /v1/systemone"
{
  "model": "d3-lite",
  "state": {"receipt": "<receipt>", "note": "Uploaded by the customer."},
  "assets": {"<receipt>": {"type": "image", "data_base64": "/9j/4AAQSkZJRgABAQ..."}},
  "questions": {
    "complete": {"type": "noul", "instructions": "Does <receipt> show a total, a date and a merchant?"}
  }
}
```

模型在哪里读取图像由其模型卡的 `image_placements` 决定。每个具有 `image` 模态的模型都读取状态中的图像：在状态中的一次使用，作为状态前面、`images` 之后的又一张图像读取，状态的文本以 `[image k]` 指代它。如果模型卡还列出了 `instructions` 或 `criteria`，问题说明或选项中的图像只属于该问题；在其他模型上，该问题以 `question_images_unsupported` 失败，而不会被移入状态（那样会改变其他问题的答案）。多个问题共用的图像应放在状态中：它在那里只读取一次。

**Box。** `box` 问题在图像中定位其说明所描述的对象：其说明引用的那一张图像，否则为状态中唯一的图像（否则报 `box_needs_image` 和 `box_target_ambiguous`）。带 `at`（秒数）时，它在状态中唯一的视频的那个时刻定位对象；`track` 问题则在每个采样时刻定位，每秒 `rate` 次：1（默认）或 2。答案：

```json title="Box answers"
{"type": "box", "box": {"top": 0.13, "left": 0.25, "bottom": 0.23, "right": 0.35}, "confidence": 0.81}
{"type": "box", "box": {"top": 0.2, "left": 0.1, "bottom": 0.6, "right": 0.5}, "confidence": 0.77, "time": 4.5}
{"type": "track", "track": [{"time": 0.0, "box": {"top": 0.2, "left": 0.1, "bottom": 0.6, "right": 0.5}, "confidence": 0.9, "visible": 0.98}]}
```

边界是相对于图像左上角、占图像高度和宽度的比例，以发送时的图像为准。box 总是会给出，因此当目标可能不存在时，请搭配一个 Noul；`visible` 是目标在那个时刻可见的概率。`confidence` 遵循模型卡的公式，可能为 `null`。模型只有在其模型卡的 `question_types` 列出 `box` 和 `track` 时才回答它们；否则该问题以 `box_unsupported` 或 `track_unsupported` 失败。如果模型卡有 `limits.box_video_max_seconds`，带 `at` 的 `box` 问题和 `track` 问题最多读取该秒数的片段：更长的片段，或无法从其容器读取长度的片段，会让该问题以 `box_video_too_long` 失败。关于视频的其他问题仍使用 `limits.video_max_seconds`。

**错误。** 在 `/v1/systemone` 上，只要请求中有任何问题失败，就像 System One 那样把请求作为整体回答，以保证客户端收到的每个答案都具有其问题的类型：`{"error": {"code": "question_failed", "question_error", "reason", "question", "message"}}`（批量请求中还以 `"state"` 给出该问题所在的条目），`invalid_question` 为 422，`max_length_exceeded`、`scan_budget_exceeded` 和 `invalid_input` 为 400，`invalid_model_output` 为 500，`unavailable` 为 503，`deadline_exceeded` 为 504。由第一个失败的问题决定，消息会列出所有失败的问题。`/v1/decisions` 则就地回答每个失败的问题。

**长上下文。** 模型卡有 `limits.validated_input_tokens` 的模型，会把每个问题（状态加该问题）限制在这个经过验证的长度。带 `"options": {"long_context": true}` 的请求每个问题最多可以有 `limits.max_input_tokens` 个 token，速度更慢、准确率更低；更长的问题以 `max_length_exceeded`（原因 `question_tokens`）失败。没有 `validated_input_tokens` 的模型在两种情况下都最多读取 `max_input_tokens`。只读一次状态的模型（`usage: state_once`）还把每个请求限制在 `limits.max_request_tokens`：状态只计一次，加上每个问题自己的 token，无论是否启用长上下文（视频 Box 或 track 问题读取的帧窗口改由视频限制约束）；更长的请求作为整体以 400 `invalid_request`（原因 `request_tokens`）失败，`limit` 给出该上限。

总结：

| 方面 | vLLM Semantic Router |
| --- | --- |
| 图像格式 | PNG、JPEG、WebP |
| 图像大小 | 最多 16,000,000 像素、长宽比不超过 200，按模型的 `image_max_pixels` 读取（会缩放） |
| 问题和选项中的图像 | `image_placements` 列出它们的模型；否则 422 `question_images_unsupported` |
| Box 和 track | `question_types` 列出它们的模型；否则 422 `box_unsupported` 或 `track_unsupported` |
| 格式错误的问题 | 在 `/v1/systemone` 上整个请求返回 422；在 `/v1/decisions` 上只有该问题失败 |
| 上下文 | 默认每个问题 `validated_input_tokens`，长上下文时 `max_input_tokens`；只读一次状态的模型：每个请求 `max_request_tokens`（状态一次加每个问题） |
| 模型 | 所提供的模型 ID；其他模型会收到列出所提供模型的回答 |
| `GET /v1/models` | `{models}` 以及运行时的模型卡（`data`） |
| Usage | 模型卡标明 `usage: state_once` 的模型状态只计一次；其他模型按每个问题的副本计 |
| 请求大小 | Router 公共 API 上 48 MiB；worker 上为 `--max-request-bytes` |
