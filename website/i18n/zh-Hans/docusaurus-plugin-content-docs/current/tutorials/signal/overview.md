---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/overview.md"
  outdated: false
---

# 信号 {#signal}

## 概述 {#overview}

信号把请求中的事实变成路由决策可以复用的名称。信号可以识别长提示、大量使用工具的对话、某种语言、某个学科，或可能的提示注入企图；决策再决定信号匹配时做什么。

信号分为三层：

- **`systemone` 原语**用你以自然语言编写的问题询问决策模型。
- **应用信号**是十一个现成的问题，每个都编译成 Router 自有的一个 `systemone` 问题（[预设](../../system-one/concepts#presets)）。
- **启发式信号**是确定性的代码：关键词、语言、请求结构、身份和元数据。

当你需要在决策运行之前把多个信号组合成得分、分区或路由区间时，使用[投影](../projection/overview)。

## 主要优势 {#key-advantages}

- 一个检测器可以被多个决策复用。
- 检测与路由结果相互分离。
- 一条路由可以组合词法、结构、策略和模型判断等多种输入。
- 配置评审更容易：信号名称是稳定的策略构件。

## 解决什么问题？ {#what-problem-does-it-solve}

没有信号层时，每个决策都要内联自己的检测逻辑。这会重复逻辑、让路由策略难以审计，并把“检测到了什么？”和“应该做什么？”混在一起。信号把对请求的理解变成一个命名目录，供路由图的其余部分组合使用。

## 何时使用 {#when-to-use}

在以下情况使用信号：

- 多条路由需要同一个检测器；
- 你想在一棵决策树中混用多种检测方法；
- 你需要在检测、决策、算法和插件之间划出清晰边界；
- 你想在不改写路由结果的情况下调整检测。

## 配置 {#configuration}

信号位于 `recipes[].signals` 下，每种信号类型一个列表：

```yaml
recipes:
  - name: default
    signals:
      keywords:
        - name: urgent_keywords
          operator: OR
          keywords: ["urgent", "asap"]
      domains:
        - name: math
          labels: [math]
      jailbreak:
        - name: prompt_attack
      systemone:
        - name: support_intent
          question:
            type: choice
            instructions: What does the user need?
            choices:
              - key: technical_support
                description: Installation, errors or troubleshooting
              - key: account_management
                description: Billing information or subscription changes
              - key: other
                description: Anything else
          labels: [technical_support]
    decisions:
      - name: urgent-support
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: systemone, name: support_intent}
            - {type: keyword, name: urgent_keywords}
        modelRefs:
          - model: support-model
```

决策的条件通过条件类型和名称指定规则：`{type: domain, name: math}`。按你需要检测的事实类型选择信号。

### System One 原语 {#the-system-one-primitive}

| 信号 | 条件类型 | 用途 |
| ------ | -------------- | --------- |
| [System One](./systemone) | `systemone` | 向决策模型提出你自己的 `choice`、`noul`、`score`、`set` 或 `span` 问题 |

### 应用信号 {#application-signals}

每个应用信号向决策模型提出一个预设问题，除非绑定或规则指定了其他部署。规则采用预设的规范问题，只覆盖其类型允许的字段。

| 信号 | 条件类型 | 用途 |
| ------ | -------------- | --------- |
| [Domain](./application/domain) | `domain` | 按请求所属的学科路由，共 14 个 |
| [Fact Check](./application/fact-check) | `fact_check` | 识别答案依赖需要核查的事实的请求 |
| [User Feedback](./application/user-feedback) | `user_feedback` | 读取一个轮次对上一个答案的反馈 |
| [Jailbreak](./application/jailbreak) | `jailbreak` | 在请求或答案中检测提示注入和越狱企图 |
| [Safety](./application/safety) | `safety` | 检测有害内容，可按指定的危害类别检测 |
| [PII](./application/pii) | `pii` | 定位请求中的个人数据 |
| [Hallucination](./application/hallucination) | `hallucination` | 找出答案中其上下文不支持的陈述 |
| [Refusal](./application/refusal) | `refusal` | 识别拒绝请求的答案 |
| [Turn Relation](./application/turn-relation) | `turn_relation` | 区分追问、重复提问、话题转换和回到旧话题 |
| [Complexity](./application/complexity) | `complexity` | 把请求评为 easy、medium 或 hard |
| [Modality](./application/modality) | `modality` | 读取请求要求的输出：文本、图片、语音、视频…… |

### 启发式信号 {#heuristic-signals}

这些信号使用显式规则、请求结构、身份或有界的元数据，不询问任何模型。

| 信号 | 条件类型 | 用途 |
| ------ | -------------- | --------- |
| [Authz](./heuristic/authz) | `authz` | 按可信的身份、角色或租户策略路由 |
| [Context](./heuristic/context) | `context` | 按请求的估算 token 数路由 |
| [Conversation](./heuristic/conversation) | `conversation` | 识别多轮、大量使用工具或代理式的请求结构 |
| [Input Modality](./heuristic/input-modality) | `input_modality` | 识别请求中出现了哪些输入媒体（文本、图片、音频、视频） |
| [Keyword](./heuristic/keyword) | `keyword` | 匹配词语、短语、BM25 词项或 n-gram，或统计其出现次数 |
| [Language](./heuristic/language) | `language` | 按检测到的请求语言路由 |
| [Metadata](./heuristic/metadata) | `metadata` | 使用调用方提供的有界应用提示 |

请记住以下规则：

- 信号要命名，并且可复用；
- 信号只做检测；路由结果属于决策；
- 分区和派生的路由区间放在 `recipes[].projections` 中，不要放进 `recipes[].signals`；
- 模型选择放在决策的算法中，路由侧的行为放在它的插件中。

## 选择信号读取的用户文本 {#choose-the-user-text-signals-read}

信号读取最后一条含文本的用户消息。有些客户端会在那里放入并非用户问题的文本，`global.router.routing_input` 告诉 Router 把它们排除在外：

```yaml
global:
  router:
    routing_input:
      # A user message whose text matches one of these RE2 patterns is never
      # routing text.
      skip_user_messages_matching:
        - '(?s)^\s*OpenClaw runtime context:.*End OpenClaw runtime context\.\s*$'
      # Spans between these literal markers are cut from the routing text.
      strip_blocks:
        - start: '<relevant-memories>'
          end: '</relevant-memories>'
        - start: '<recalled-memories>'
          end: '</recalled-memories>'
```

- OpenClaw 2026.10.4 及更高版本在每个轮次（包括工具结果轮次）的末尾附加一条只包含运行时元数据的独立用户消息。没有跳过模式时，每个请求都会按这些元数据路由。
- OpenClaw 的记忆插件会在每个轮次把召回的记忆加到用户提示前面：memory-lancedb-pro 用 `<relevant-memories>` 包裹，mem0 的插件用 `<relevant-memories>` 或 `<recalled-memories>` 包裹。剥离这些块可以避免它们淹没问题，也让决策模型不必读取它们。

路由文本是最后一条未被跳过、且剥离后仍有文本的用户消息。在以下场景中，它取代最后一条用户消息：

- 每个路由信号，以及它们读取的对话事实；
- 语义缓存的查询；
- 工具选择；
- RAG 查询；
- 记忆：检索查询、其查询改写读取的历史，以及它存储的用户消息；
- 模型选择算法读取的查询；
- 幻觉检查读取的请求；
- Replay 的提示；
- 路由预览、classify 和 eval API。

被跳过的消息不计为用户轮次，所以工具结果之后的运行时上下文轮次仍使请求保持为工具循环轮次，用户反馈和轮次关系信号也不会在其上触发。发给模型的请求保持不变，上下文长度估算仍会计入全部内容。

越狱、PII 和安全信号会继续读取被跳过的消息和被剥离的块以及路由文本，工具结果检查也不受影响，因此客户端无法把内容藏在伪造的记忆块或伪造的运行时上下文轮次里躲过它们。其他所有信号（包括关键词规则）只读取路由文本。

- 模式使用 Go RE2 语法，除非用 `^` 和 `$` 锚定，否则可在文本任意位置匹配。能匹配空文本的模式会在加载配置时被拒绝。
- 只有内容全部为文本的消息才能被跳过。同时包含图片、音频或工具结果的消息总是参与路由。
- 标记是字面量且区分大小写。块从左到右剥离，不能嵌套：块在其起始标记之后的第一个结束标记处结束。之后没有结束标记的起始标记会留在路由文本中。
- 当所有用户消息都被跳过时，路由按没有用户文本的请求处理。
- 语义缓存比较的是路由文本，但被跳过的消息和被剥离的块也必须一致，才会复用缓存的答案。
- `llm_routing_input_removed_total` 统计被跳过的消息和被剥离的块，Replay 的路由诊断按请求记录这两个数量。被移除的文本从不写入日志。

## 查看匹配结果 {#see-what-matched}

`POST /api/v1/routing/preview` 针对一个请求评估活动配方的每个信号，报告匹配的规则和决策，而不调用模型。在经过路由的响应上，`x-vsr-trace` 的 `signals` 成员以 `type:name` 列出匹配的规则，例如 `domain:math` 或 `pii:contact`，只发给 `global.router.headers.debug` 允许的调用方；`response_guard` 会把它的代码（例如 `hallucination`）加入 `x-vsr-warnings`。见 [VSR 路由头部](../../troubleshooting/vsr-headers)。

## 后续步骤 {#next-steps}

- 阅读 [System One](../../system-one/overview)，了解决策模型如何回答每个由模型判断的信号，以及每个信号由哪个部署回答。
- 需要分区、加权得分聚合或命名路由区间时，阅读[投影](../projection/overview)。
- 从 [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml) 开始了解完整的公开契约。

根据类型不同，信号可以读取请求文本、对话历史、图片、调用方元数据或可信身份。由模型判断的信号会把读取的内容发给回答它们的部署；[哪些数据会离开本机](../../system-one/deployments#what-leaves-the-host)列出了每个信号发送的部分。
