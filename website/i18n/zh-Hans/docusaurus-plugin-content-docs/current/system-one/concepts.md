---
title: System One 概念
sidebar_label: 概念
description: 问题类型、状态及其组成部分、预设、训练与零样本回答、span 能力和协议限制。
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/system-one/concepts.md"
  outdated: false
---

# System One 概念 {#system-one-concepts}

一次 System One 调用由两部分组成：**状态（state）**，即问题所针对的上下文；以及**问题（questions）**，每个问题有名称和类型。模型针对每个问题读取一次状态（支持共享状态的模型对所有问题只读一次），并用概率回答每个问题。

## 问题类型 {#question-types}

| 类型 | 问什么 | 需要写 | 答案 | 规则读取 |
| --- | --- | --- | --- | --- |
| `choice` | 这些选项中哪个合适？ | 2 到 255 个 `choices`，每个有 `key` 和 `description` | 选中的选项以及每个选项的概率 | P(选中选项)，或某个标签的 P(选项) |
| `noul` | 这是否成立？ | 只需 `instructions` | P(是) | P(是) |
| `score` | 在有序刻度上程度如何？ | 2 到 10 个 `levels` | 期望等级（从 0 开始计）以及每个等级的概率 | 期望等级 |
| `set` | 这些标签中哪些适用？ | 1 到 255 个 `labels` | 每个标签的概率，以及模型选中的标签 | P(标签) |
| `span` | 每个标签出现在文本的什么位置？ | 1 到 255 个 `labels` | 带标签的字符 span，每个附带概率 | 某个标签最可能的 span |

`noul` 是 System One 的是/否类型。没有原生 `set` 能力的模型，会在同一次调用中把 set 问题当作每个标签一个 `noul` 来回答。有些模型还声明了更多类型：d3-turbo 和 d3-ultra 在 [API](./api) 上回答 `box`（某物在图片中的位置）和 `track`（某物在视频每个时刻的位置），路由规则不会问这两类问题。

像问同事一样写问题说明，并用几个词描述每个选项；[编写自定义问题](./custom-questions)介绍了什么样的写法更适合模型阅读。

## 状态及其组成部分 {#the-state-and-its-parts}

状态是问题所针对的文本。只针对当前用户轮次的问题，状态就是那段文本。当问题需要读取更多内容时，状态是一个由命名部分组成的对象，模型把 `request` 当作用户轮次、`answer` 当作模型的答案，其他部分都当作上下文：

| 部分 | 内容 | 由谁提问 |
| --- | --- | --- |
| `request` | 当前用户轮次：路由文本，即 `global.router.routing_input` 移除指定内容后的文本 | 每个请求阶段的问题 |
| `history` | 之前的用户轮次 | 设置了 `input.history` 的规则 |
| `previous_turns` | 最近 N 个轮次（用户和助手），以对话记录形式给出 | 设置了 `input.previous_turns` 的规则；`turn_relation` 默认读取 4 个 |
| `previous_answer` | 当前轮次所回复的助手答案 | `user_feedbacks`（`feedback` 预设） |
| `tool_results` | 请求中的工具结果 | 设置了 `input.tool_results` 的规则 |
| `candidate` | 评审打分的对象：一篇文档、一个片段、一条缓存请求 | `relevance` 和 `same_request` 评审 |
| `answer` | 模型的答案 | 响应阶段的规则；`answer_check` |
| `context` | 答案的依据上下文：检索到的文档和工具结果 | 设置了 `input.context` 的响应阶段规则；`hallucination` |
| `images`、`videos` | 当前轮次中的图片和视频部分 | 设置了 `input.image` 或 `input.video` 的规则 |

空的部分会被省略，所以在第一轮读取历史的规则，实际只针对请求本身提问。

### 阶段 {#stages}

问题在两个阶段之一提出：

- **request**（默认）：路由之前，针对请求提问。决策、投影和请求插件读取请求阶段的规则。
- **response**：模型回答之后，针对它的答案提问。只有插件读取响应阶段的规则，因为决策在答案产生之前就已完成路由；规则中读取响应阶段规则的决策会在加载时被拒绝。

## 预设 {#presets}

预设是 Router 自有的规范零样本问题：它的问题说明、带描述的选项、读取的阶段和部分、对长输入的覆盖方式，以及是否需要 span。预设是内部概念，你通过提问它们的功能来使用：

- 十一个**应用信号**，每个都把一条规则编译成其预设的一个问题；
- **`match` 评审** `relevance`、`same_request` 和 `tool_rank`，以及 tools 插件的 `tool_need` 门控；
- **`escalate`**，其 `answer_check` 和 `refusal` 检查用来评判答案。

你自己的问题是一条 [`systemone` 规则](../tutorials/signal/systemone)。[预设目录](./reference#preset-catalog)列出了所有预设。

在同一次调用中，就同一状态向同一部署提问同一预设的多条规则共享一个问题，每条规则再把自己的 `labels` 和 `predicate` 应用到共享的答案上。

## 训练回答与零样本回答 {#trained-and-zero-shot-answers}

每个模型都能回答预设的规范形式。当某个模型家族针对某个预设训练过时，Router 改用训练形式提问：

- **训练绑定。** Vela 3.0 的 d3 规格按问题 ID 精确开启训练过的适配器（`domain`、`attack`、`factcheck` 等）；vela3-fast 和 Vela 2.0 按措辞精确匹配，而它们训练过的每个预设的措辞就是规范措辞。训练形式的选项 key 会被映射回来：modality 问题训练时的 `AR`、`DIFFUSION`、`BOTH` 返回为 `text`、`image`、`text_image`。
- **Router 自有 ID。** 没有训练形式的问题使用带 `:` 的 ID 发出，它不会与训练 ID 冲突，因此不会误开任何适配器。
- **校准阈值。** 在 vela3-fast 和 Vela 2.0 上，预设的默认 predicate 是在 router signal suite 上保持 Vela 1.0 专用模型工作点的阈值。其他模型使用 0.5，并在加载时给出指明规则的 `uncalibrated_preset` 警告。在规则上写 `predicate` 即可设置自己的值。

[支持矩阵](./reference#support-matrix)按预设和家族列出答案是训练且校准、训练、零样本还是无法回答。

## Span 能力 {#span-capability}

Span 问题返回带标签的字符偏移。Vela 3.0 和 Vela 2.0 家族能回答；d3 家族（本地或托管）以及 Decision 1.0 和 2.0 不能。以下三者需要 span：

- `pii` 和 `hallucination` 预设；
- 每个问题类型为 `span` 的 `systemone` 规则；
- `masking` 插件，通过它指定的 `pii` 规则。

它们都必须解析到能回答 span 的部署。使用默认决策模型时无需任何额外配置。如果决策模型不支持 span，就把需要 span 的类型绑定到支持的部署，否则配置加载失败并报 `span_required`：

```text
recipes[default].signals.pii[personal_data]: span_required: resolves to deployment "d3-ultra-remote", which cannot answer span questions; bind pii to a span-capable deployment (rule deployment, recipes[].model_bindings.pii or global.model_catalog.bindings.pii)
```

## 媒体输入 {#media-inputs}

规则用 `input.image` 读取当前轮次的图片，用 `input.video` 读取视频。回答问题的部署必须声明自己能读取该媒体，否则配置加载失败并报 `input_modality_unsupported`。d3 家族和 Vela 3.0 的 d3 规格能读取图片。`input.audio` 和 `input.video` 是 schema 的一部分，声明了它们的模型无需改动 schema 就能回答这类规则；目前还没有决策模型能读取音频，所以在出现这样的模型之前，`input.audio` 在所有部署上都会被拒绝。

## 限制 {#limits}

| 限制 | 值 |
| --- | --- |
| `choice` 的选项数 | 2 到 255 |
| `set` 或 `span` 的标签数 | 1 到 255 |
| `score` 的等级数 | 2 到 10 |
| 一次调用中的问题数 | 协议不设限制 |
| choice 评审（`tool_rank`）的候选数 | 最多 255，即选项上限 |
| 逐个候选评审（`relevance`、`same_request`）的候选数 | `judge.max_candidates`，默认 64 |

候选数超过上限的评审会截断，绝不会让请求失败。输入长度取决于模型：vela3-fast 每次决策读取 32,768 个 token，router 问题和 span 问题按 8,192 个 token 的窗口读取；更大的 Vela 3.0 规格每次决策最多读取 262,144 个 token，每个 span 目标 32,768 个。[性能与成本](./performance)介绍了长输入。
