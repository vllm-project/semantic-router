---
title: "发布说明：v0.5，决策模型成为一等模型"
description: v0.5 让决策模型成为回答每个路由判断的唯一模型，在 schema v0.5 下给配置统一的结构，并把响应头精简为一组公开的头。
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/release-notes/decision-model-first-class.md"
  outdated: false
---

# 发布说明：v0.5，决策模型成为一等模型 {#release-note-v05-the-decision-model-is-first-class}

Router 早已用同一个模型家族，即 System One 决策模型，回答大多数路由问题。但它的配置描述的仍是一套更早的系统：每个信号都有自己的小型专用模型，或依赖嵌入相似度技巧。v0.5 让决策模型成为一等模型：

- **信号分为三层：** 用于你自己编写问题的 `systemone` 原语；十一个应用信号，每个都编译成一个 `systemone` 问题；以及七个确定性的启发式信号。
- **处处零样本。** 每个 System One 模型都能零样本回答它支持的每个问题；训练过的适配器和校准阈值是按家族进行的优化。各模型能回答的问题类型不同：Decision 1.0、2.0 和 3.0 不返回 span，vela3-fast 不回答 score 问题。问题解析到不能回答其类型的部署时，配置加载失败（`span_required`、`question_type_unsupported`）。
- **相似度只有一个 `match` 块**，包含可选的嵌入或 BM25 `recall` 阶段，以及可选的 System One `judge` 阶段。每个评审对候选数都有硬上限，溢出只会截断，绝不会报错。
- **嵌入只用于召回，并按需启动。** 没有任何决策路径使用嵌入。
- **调用 System One 只有一种方式：** 标准请求，或一组命名标准请求组成的批量请求。
- **算法分为选择器和执行器**，**插件只负责执行动作**，并通过名称指定它们作用的信号。
- **配置只有一种结构**，schema 为 `v0.5`。
- **响应头精简为一组公开的头**，外加一个受控的 `x-vsr-trace`。

新的 [System One](../system-one/overview) 一章介绍了这一切背后的模型。

## 破坏性变更 {#breaking-changes}

这是一次破坏性版本。运行一次 `vllm-sr config migrate`，然后检查它打印的说明；[迁移指南](../installation/migrate-to-v0-5)列出了每个已移除的形式、它的加载错误和改写方式。

- **配置 schema v0.5。** `version: v0.3` 在加载时以 `invalid_value` 失败，migrate 写入 `version: v0.5`。不再有顶层 `routing`：路由位于扁平的 `recipes` 中，模型事实位于 `providers.models[]`，默认插件位于 `global.plugins`。
- **已移除的形式在加载时以一行错误失败**，并由 migrate 改写。唯一的别名是 `decision`，它按 `systemone` 加载并给出警告。
- **已移除的信号类型：** `classifiers`、`preferences`、`embeddings` 和 `kb`（改为 `systemone` 问题；migrate 会删除永远不会匹配的 `kb` 规则，并给出替代它的问题），`reasks` 和 `topic_continuity`（改为 `turn_relation`），`events` 和 `structure`（改为 `keywords`），以及 `actions`（改为一个 `systemone` 问题）。`global.model_catalog.modules.*` 已移除；模块的阈值变为规则的 `predicate`。
- **已移除的算法：** `prompt`、`router_dc`、`knn`、`kmeans`、`svm` 和 `mlp` 变为 `systemone`；`automix`、`hybrid` 和 `latency_aware` 变为 `multi_factor`；Looper 的 `confidence` 算法和原生 `cascade` 变为 `escalate`。ML 选择器的训练流水线和 Dashboard 的 ML 流水线已移除。
- **已移除的插件：** `tool_selection`（改为 `tools` 插件的 `select`），`rag`、`memory` 和 `context_compression`（改为 `context` 插件），`hallucination` 和 `response_jailbreak`（改为 `response_guard`），`shadow_dispatch`（改为 `router_replay.shadow`），以及各种缓存别名（改为 `response_cache`）。
- **提示压缩改为 `context` 插件的 `compress.signals`。** `global.model_catalog.modules` 已移除：`prompt_compression` 在 `global.plugins` 或配方的插件中（不能在决策的插件中）压缩路由信号读取的文本，只有 `signals` 的 `compress` 块不压缩请求本身。迁移会移动它。
- **Router 的嵌入 API 和 `embeddings.multimodal` 已移除：** `global.services.api.embeddings`、`POST /api/v1/diagnostics/embeddings`、`/api/v1/diagnostics/similarity` 和 `/api/v1/diagnostics/similarity/batch`，以及 `global.model_catalog.embeddings.multimodal`；migrate 会删除这两者。关于图片的决策改为提出读取图片的 `systemone` 问题（`input.image`）。模型运行时的 worker `/v1/embeddings` 保留。
- **`global.model_catalog.embeddings.default` 指定一个部署**，即 `embeddings.default: {deployment: vela-embedding}`，写法和 `system.decision_model` 一样。只写名称会以 `invalid_value` 失败，migrate 会改写它。
- **上一版本的模型资源写法不再能加载：** `external[]` 模型、绑定对象（`contract`、`adapter`、`head`、`mapping_path`、`operating_point`）、针对单条规则的绑定、部署上的 `question_types` 和 `served_name`、`classify` 和 `http` 部署、`candle`、`ort` 和 `openvino` 提供方，以及后端的 `base_url`、`api_key_env`、`auth_header`、`auth_prefix` 和 `extra_headers`。绑定只写部署名称，后端用 `endpoint`、`auth` 和 `headers` 连接；migrate 会改写它们。后端内联的 `api_key` 仍可加载，但不能与 `auth.api_key_env` 同时使用。
- **`vllm-sr config migrate` 需要一个 Router**（二进制或镜像），`--offline` 已移除。采用 config v0.3 之前布局的配置，一次运行即可迁移到 v0.5。
- **Looper 的名称和端点：** `global.integrations.looper.*.model_names` 在加载时会被拒绝；migrate 会把这些名称写成一个入口，指向它写出的配方：默认配方中只保留运行该算法的决策。`global.integrations.looper.endpoint` 会被拒绝，migrate 会删除它：Router 在进程内发起 Looper 的模型调用。Dashboard 去掉了 External Models 卡片，它的 Looper 卡片只编辑超时和请求头；operator 拒绝 `external_models`。
- **模型的输出事实是 `capabilities` 和 `modalities`。** 模型的 `modality` 关键词（`ar`、`diffusion`、`omni`）、基于 `DIFFUSION` 或 `BOTH` 的图片路由、`image_gen` 插件以及音频嵌入规则都已移除；migrate 能改写的会改写，会删除没有决策能执行的路由，其余的会指出来。
- **System One 批量形式只使用 `states`。** 批量请求体在 `states` 下携带命名的标准请求，没有顶层 `state`、`questions`、`images` 或 `videos`；混用两种形式的请求体，或 `states` 为空的请求体，会以 `invalid_request` 被拒绝。批量响应没有顶层 `answers`，其 `usage` 统计所有条目。见 [API](../system-one/api)。
- **模型运行时契约 4.0.0。** 本版本的 Router 拒绝任何其他契约的模型运行时；Router 和它的运行时来自同一个版本。
- **每个请求阶段一次 System One 调用，远程部署也是如此。** Router 不再把一个阶段发往远程 System One API 的问题拆成每次最多八段文本的多次调用。你自己运行的 System One API 必须接受条目数与该阶段所问文本数相同的批量请求，每个评审最多 `judge.max_candidates` 条。
- **Kubernetes operator 的 `SemanticRouter` 资源采用新结构。** `spec.config.routing` 会被拒绝；它的字段改为 `spec.config.recipes`（每个配方的信号、投影、决策、模型绑定和插件），以及 `spec.config.entrypoints`、`spec.config.plugins` 和 `spec.config.models`（`providers.models[]` 的事实）。
- **使用已移除信号类型的 `IntelligentRoute` 资源需要手动改写。** `vllm-sr config migrate` 改写的是配置文件，而不是集群资源：在升级 CRD 之前，把 `events` 和 `structure` 信号改写为 `keywords` 规则，其他已移除类型按迁移指南改写。
- **响应头精简为一组公开的头和一个 trace**，旧名称没有别名（见[响应头](#headers)）。基于 `x-selected-model` 路由的自定义网关必须改为匹配 `x-vsr-route-model`，读取匹配信号或决策响应头的客户端改为读取 `x-vsr-trace` 的成员。

## 预期的行为变化 {#behavior-to-expect}

- **模型没有给出答案的 `jailbreak` 和 `pii` 规则不匹配。** 默认的 `on_unknown` 与以前一样是 `no_match`：决策模型宕机时，防护会放行流量。`on_unknown: match` 是失败关闭的显式选择，migrate 会为早期版本的 `on_error: block` 写入它。模型没能读到的文本仍按攻击或个人数据计入。
- **原来属于其他模型的阈值已重新拟合。** 没有 `predicate` 的规则使用回答模型的校准阈值，否则使用 0.5 并给出 `uncalibrated_preset` 警告。
- **能拒绝或改写回答的 `response_guard` 会扣住流式回答。** 对于守卫会拦截或写入说明的决策，Router 会缓冲它的流，直到守卫读完整个回答，再发送原有事件，或以该协议的流错误形式发送拒绝消息。首 token 时间只在这些决策上增加，被扣住的回答上限为 64 MiB。只做标注或只记录的守卫保持真正的流式传输。
- **需要 span 的规则必须解析到支持 span 的部署**，否则配置以 `span_required` 失败。默认决策模型 vela3-fast 能返回 span。
- **RAG 相关性和工具选择由决策模型评判**，语义缓存命中由它的 `same_request` 问题确认。响应缓存的语义匹配默认使用嵌入召回（`top_k: 3`，`min_similarity: 0.8`），再由 `same_request` 确认（`min_score: 0.8`）；`mode: exact` 不变。
- **每个 `match.judge` 都接受 `max_candidates`。** choice 评审最多读取 255 个候选（这也是默认值）；逐个候选提问的评审默认读取 64 个。候选再多也不会让请求失败：评审按召回排名读取前面的候选，没有召回时读取最新的候选，记录一个 `match.judge.truncated` span 事件，并计入 `vsr_match_judge_truncated_total`。超过 255 个工具的目录会截取前 255 个，或用 `bm25` 召回。
- **RAG 和记忆源可以不经召回直接评审**，基于内存向量库的 RAG 源可以用 `bm25` 召回。上下文压缩针对最新的用户消息评审每个片段，默认评审 `bm25` 排在前面的 64 个。
- **嵌入模型只用于召回，并按需启动。** 预先声明的 `vela-embedding` 只在响应缓存、记忆、向量库或 RAG 源用它召回时才启动。远程的 `openai_embeddings` 部署声明 `capabilities.dimensions` 和 `capabilities.modalities`；不能嵌入文本的召回部署，或与索引维度不一致的维度，会在加载时被 Router 拒绝。
- **提示缓存亲和性移入 `multi_factor`。** `hybrid` 选择器移除后，加权的 `multi_factor` 目标会偏向为会话之前轮次提供服务的候选（后续轮次、携带的历史、`previous_response_id` 或 `prompt_cache_key`），作为有界的平局决胜因素。
- **决策用 `output: {kind, action}` 声明它的输出。** 图片决策在 Responses API 上用生成的图片回答纯文本请求：Router 添加托管的 `image_generation` 操作，并只选择能生成图片的模型。模型没有声明输出所需的能力时，加载以 `output_capability_missing` 失败；本版本无法执行的输出（图片编辑、音频、视频）或无法承载该输出的 API，以 `output_not_executable` 失败。图片决策的 Chat Completions 或 Messages 客户端会以 `unsupported_capability` 被拒绝。
- **Router 有了 Images API 入口。** `POST /v1/images/generations` 按提示词路由，只由输出为图片的决策回答；到达这里的其他请求会在调用任何模型之前得到 `400 image_route_required`。`api_format: images` 可以连接兼容 OpenAI 的图片生成后端。
- **带 `input.image: true` 的 `systemone` 问题按请求图片的内容路由**，由能读取图片的决策模型回答。
- **身份请求头只在配置信任时才被信任。** 独立运行的 Router 会丢弃客户端发送的身份请求头，除非监听器设置了 `identity.trust_headers: true`；读取网关身份的策略（authz 信号、按用户或组的限流、按用户的密钥，以及 `global.router.headers.debug: authenticated`）在没有身份来源的独立模式下会加载失败。调试响应头只发给 `global.router.headers.debug` 允许的调用方。
- **委托凭据是每个上游的一种认证模式。** 上游的 `auth` 可以是静态密钥（`api_key_env`、`api_key_file`，或后端的 `api_key`）、`credential: delegated`，或两者都不设；提供方后端、决策模型和远程 embedding 部署、`escalate` 评审以及 `api_format: systemone` 目标都一样。`global.services.authz.delegated_credential` 指定身份验证器写入用户凭据的标头（默认 `x-vsr-delegated-credential`）。为客户端请求发起的调用携带该请求的凭据，否则该请求在调用之前得到 `401`；Router 代表自己发起的调用从不携带它，委托上游会拒绝这类调用。`backend_refs[].trusted_gateway` 和 `global.services.authz.gateway_credential` 已移除：Router 和 migrate 都会拒绝它们，并给出新的写法。见[安全加固](../installation/security-hardening#delegate-the-users-credential-to-an-upstream)。

## 响应头 {#headers}

路由流水线的每个响应都携带同一组公开的头，无论是上游的回答，还是缓存命中、快速响应、拦截、Looper 回答和路由错误：`x-vsr-schema-version`（现在为 3）、`x-vsr-config-version`、`x-vsr-response-path`、`x-vsr-selected-recipe`、`x-vsr-selected-decision`、`x-vsr-selected-model`、`x-vsr-replay-id`、`x-vsr-warnings`，以及配置的成本（`x-vsr-cost`、`x-vsr-cost-currency`）。Router 报告的其他内容都放在 `x-vsr-trace` 中，它是最多 4 KB 的 RFC 8941 字典，只发给 `global.router.headers.debug` 允许的调用方：超过 4 KB 时，会从最低优先级开始省略成员，并以 `truncated` 结尾。改名后的头不保留别名。

| 旧 | 新 |
| --- | --- |
| `x-vsr-matched-*`、`x-vsr-context-token-count` | trace 的 `signals` 成员 |
| 决策细节、retention、提示缓存、Looper、协议和有效 token 相关的头，缓存相似度 | trace 成员（`ranking`、`confidence`、`algorithm`、`cache_similarity` 等） |
| `x-vsr-selected-modality` | trace 的 `modality` 成员，即选中决策的输出类型 |
| `x-vsr-protocol-warnings`、`x-vsr-response-warnings` | `x-vsr-warnings`，一个 RFC 8941 代码列表 |
| `x-vsr-cache-hit`、`x-vsr-fast-response` | `x-vsr-response-path: cache` 或 `fast_response` |
| `x-selected-model`（Envoy 路由键） | `x-vsr-route-model`（内部）；自定义网关路由必须匹配它 |
| `x-disable-router-memory` | `x-vsr-memory: off` |
| `x-sr-bench-*` 请求头，以及基准测试的回执头 | `x-vsr-bench-*`；回执变为 trace 成员，所以 sr-bench 需要调试权限 |
| 已退役的 Looper 跳转头；未使用的 KV 传输状态 | 已移除 |
| `x-session-id`、`x-claude-code-session-id` | `x-vsr-session-id`；`global.router.headers.session_id_headers` 列出同样接受的客户端名称，默认就是这两个 |

客户端发送的其他 `x-vsr-*` 头，以及提供方返回的每个 `x-vsr-*` 头，都会被 Router 丢弃。Router 从身份验证器读取的是 `x-vsr-delegated-credential`：启用 `global.services.authz.delegated_credential` 时，它和身份请求头一样被信任；任何上游都收不到这个请求头，只有委托上游会收到它的值。

[响应头参考](../api/headers)列出了注册表定义的每个响应头。

## 已移除的测试配置 {#removed-test-profiles}

已移除后端的端到端配置，以及 Vela 1.0 编码器（它们无法回答 System One 问题）的端到端配置都已删除：`local-classifier-backend`、`category-remote-backend`、`complexity-remote-backend` 和 `preference-external-routing`（远程分类器后端）；`ml-model-selection`（训练式选择器）；`multimodal-routing`（嵌入图片锚点）和 `vela-omni`（音频路由）；`vela-halu` 和 `vela-shield`（Vela 1.0 Halu 和 Shield）。基线配置运行在确定性的假 System One 后端上，图片路由改为基于 `systemone` 的 `input.image` 重新测试。

## 测量 {#measurements}

零样本预设背后的测量（危害类别、RAG 和缓存评审、工具选择、轮次关系和复杂度问题、训练 ID 修复，以及输出意图）基于公开数据运行，结果发布后会写入校准表和本说明；它们不是本次发布的门槛。

## 已知限制 {#known-limits}

- **多模态输出处于第一阶段：** 图片生成，以及由一个 omni 模型生成文本加图片。图片编辑、语音和音频（第二阶段）以及视频（第三阶段）在其编解码器落地之前，会在加载时以 `output_not_executable` 失败。
- **目前还没有决策模型能读取音频**，所以 `input.audio` 在所有部署上都会被拒绝；`input.video` 需要声明它的模型。
- **远程决策模型没有校准阈值**；在规则写出自己的 `predicate` 之前，它们的预设使用 0.5。
- **只有 vela3-fast 和 Vela 2.0 有校准阈值**；其他家族使用 0.5 并给出 `uncalibrated_preset` 警告。
