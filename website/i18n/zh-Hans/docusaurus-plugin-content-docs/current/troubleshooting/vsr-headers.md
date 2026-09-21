---
translation:
  source_commit: "bce357c513f391824e8320267d03977794c20f76"
  source_file: "docs/troubleshooting/vsr-headers.md"
  outdated: true
---

# VSR 路由头

Router 使用这些请求头和响应头来保持会话连续性、路由可观测性、回放关联，以及按需调试。

## 默认会出现什么

Router 把头分到两个面：

- **默认面** — 每个非缓存命中响应都包含 `x-vsr-schema-version` 和 `x-vsr-response-path`。成功路由的响应还可以包含最终配方、决策、置信度、算法、模型、路由延迟、成本和回放 id。发生协议转换时会出现协议标记；仅在存在警告时出现协议警告。
- **调试面** — 中间分类细节、匹配信号、工具选择指标和 `x-vsr-retention-*` 指令仅在请求设置 `x-vsr-debug: true` 时内联出现。启用回放时，同样的诊断上下文仍可通过 `x-vsr-replay-id` 获得。

决策和匹配信号头还要求同时满足以下全部条件：

1. 上游响应成功（`2xx`）。
2. 响应不是由响应缓存提供。
3. Router 为该请求评估了路由决策或信号。

缓存命中响应可以发出缓存头，但它们不会重新运行路由，因此不会附加新的匹配信号头。

## 请求头

| 头 | 方向 | 说明 |
| ------ | --------- | ----------- |
| `x-session-id` | request | 客户端提供的稳定会话标识，用于 Chat Completions。路由学习保护会把它与已配置的对话身份一起，用于跨轮次的 stay-vs-switch 决策。 |
| `x-conversation-id` | request | 客户端提供的稳定对话或 Agent 运行标识。当 `scope: conversation` 时，路由学习保护默认使用它。 |
| `x-claude-code-session-id` | request | Claude Code 在 Messages API 请求上提供的对话标识。两者同时存在时，`x-session-id` 优先。 |
| `x-disable-router-memory` | request | 当客户端已经注入记忆、Router 托管记忆会重复时，设为 `true`。 |
| `x-vsr-skip-processing` | request | 在启用 `global.router.skip_processing.enabled` 时，让请求退出 Router 处理。使用值 `true`。 |
| `x-vsr-debug` | request | 让请求进入详细/调试响应头——合约原本省略或降级到回放的头会为该请求内联发出。使用值 `true`。 |

## 协议与回放头

| 头 | 说明 |
| ------ | ----------- |
| `x-vsr-client-protocol` | Router 看到的入站协议形态，例如 `openai` 或 `anthropic`。仅在跨协议处理（客户端协议与上游不同）或设置了 `x-vsr-debug` 时发出。 |
| `x-vsr-upstream-protocol` | 发送到所选上游后端的协议形态。仅在跨协议处理或设置了 `x-vsr-debug` 时发出。 |
| `x-vsr-protocol-warnings` | 逗号分隔的协议转换警告，编码为 `severity;reason;field`。仅在存在警告时发出。 |
| `x-vsr-replay-id` | 不透明的 Router 回放记录标识，用于把响应与回放/Insights 数据关联。 |

## 响应警告

| 头 | 说明 |
| ------ | ----------- |
| `x-vsr-response-warnings` | 本次 completion 的逗号分隔响应质量警告码，固定顺序：`hallucination`、`unverified_factual`、`response_jailbreak`。仅在至少一项适用时发出。 |

每条警告的细节（例如幻觉跨度或越狱置信度）保留在回放记录中，而不是展开到响应头。

## 决策头

最终路由事实使用默认面。中间细节（包括路由学习可观测性）需要 `x-vsr-debug`。

| 头 | 面 | 说明 | 示例 |
| ------ | ------- | ----------- | ------- |
| `x-vsr-selected-recipe` | default | 由入口或 auto/looper 别名选择的路由隔离范围。具体后端直通时省略。 | `support` |
| `x-vsr-selected-decision` | default | 决策引擎选择的最终决策。 | `complex-request` |
| `x-vsr-selected-confidence` | default | 所选决策的模型导出分数。没有分数的结构性或错误策略匹配会缺省。 | `0.9100` |
| `x-vsr-applied-unknown-policy` | default | 未知结果由 `rules.on_unknown` 解析的决策，格式为 `decision=policy` 对。也会出现在 `fail_request` 的 503 上。 | `guarded=no_match` |
| `x-vsr-selected-algorithm` | default | 决策匹配后使用的模型选择算法。 | `static` |
| `x-vsr-selected-model` | default | Router 选择的逻辑模型别名。 | `reasoning-model` |
| `x-vsr-routing-latency-ms` | default | Router 选择模型所花时间，单位毫秒，带亚毫秒精度。 | `0.412` |
| `x-vsr-selected-category` | debug | 运行领域路由时的领域/类别分类器结果。 | `math` |
| `x-vsr-selected-reasoning` | debug | 为请求选择的推理模式。 | `on` |
| `x-vsr-selected-modality` | debug | 模态结果和可选方法。 | `AR;classifier` |
| `x-vsr-session-phase` | debug | 所选路由策略的保护追踪阶段。详细学习动作通过 `x-vsr-learning-*` 头和路由回放暴露。 | `user_turn`、`tool_loop`、`provider_state` |
| `x-vsr-learning-methods` | debug | 本响应汇总的路由学习方法。完整分数/缓存细节在路由回放中。 | `adaptation,protection` |
| `x-vsr-learning-actions` | debug | 按方法键控的紧凑学习动作。 | `adaptation=propose_switch,protection=allow_switch` |
| `x-vsr-learning-scopes` | debug | 学习使用的按方法键控身份范围。 | `protection=conversation` |
| `x-vsr-learning-reasons` | debug | 动作的按方法键控、机器可读原因。 | `adaptation=sampled_win,protection=switch_allowed` |
| `x-vsr-injected-system-prompt` | debug | 系统提示词插件是否向请求注入了文本。 | `true` |

用于 UI 展示时，把 `x-vsr-learning-actions` 翻译成面向用户的短语，例如 `tool/protocol pinned`、`model switched` 或 `learning bypassed`。新对话或会话开始的诊断通常只在调试视图中有用，应显示为中性状态文本，而不是主要路由状态。

## 匹配信号头

匹配信号头包含逗号分隔的规则名。它们需要 `x-vsr-debug`，且在该信号族未匹配时省略。

| 头 | 信号族 |
| ------ | ------------- |
| `x-vsr-matched-keywords` | `keyword` |
| `x-vsr-matched-embeddings` | `embedding` |
| `x-vsr-matched-domains` | `domain` |
| `x-vsr-matched-fact-check` | `fact_check` |
| `x-vsr-matched-user-feedback` | `user_feedback` |
| `x-vsr-matched-reask` | `reask` |
| `x-vsr-matched-preference` | `preference` |
| `x-vsr-matched-language` | `language` |
| `x-vsr-matched-context` | `context` |
| `x-vsr-context-token-count` | `context` 使用的上下文 token 估计 |
| `x-vsr-matched-structure` | `structure` |
| `x-vsr-matched-complexity` | `complexity` |
| `x-vsr-matched-modality` | `modality` |
| `x-vsr-matched-authz` | `authz` |
| `x-vsr-matched-jailbreak` | `jailbreak` |
| `x-vsr-matched-pii` | `pii` |
| `x-vsr-matched-kb` | `kb` |
| `x-vsr-matched-conversation` | `conversation` |
| `x-vsr-matched-event` | `event` |
| `x-vsr-matched-input-modality` | `input_modality` |

## 投影头

| 头 | 说明 |
| ------ | ----------- |
| `x-vsr-matched-projections` | 匹配该请求的逗号分隔投影映射输出。 |

投影分数和完整投影追踪保存在路由回放记录中，而不是展开到响应头。用 `x-vsr-replay-id` 在控制面板或经身份验证的 Router 管理 API 中检查这些细节；公开推理监听器不提供回放记录。

## 留存头

当匹配决策发出留存指令时，调试响应会暴露已设置的字段。这些头帮助运维人员验证策略接线；客户端不应把它们当作命令。

| 头 | 说明 |
| ------ | ----------- |
| `x-vsr-retention-drop` | 响应是否应从响应缓存留存中排除。 |
| `x-vsr-retention-ttl-turns` | 决策级留存寿命，以对话轮次表示。 |
| `x-vsr-retention-keep-current-model` | 策略是否要求后续路由保持当前模型。 |
| `x-vsr-retention-prefer-prefix` | 在运行时支持时，是否优先前缀留存。 |

未设置的字段会省略。缓存命中不发出这些头，因为该响应没有评估决策。

## 成本头

在缓冲（非流式）响应上，Router 用已服务模型的 `pricing` 配置，对模型报告的用量计价。这是配置价格数字，不是提供方账单。流式响应和没有 `pricing` 的模型会省略这两个头。

| 头 | 面 | 说明 | 示例 |
| ------ | ------- | ----------- | ------- |
| `x-vsr-cost` | default | 用量 token 乘以已服务模型的配置价格。 | `0.000054` |
| `x-vsr-cost-currency` | default | `x-vsr-cost` 的货币，来自 `pricing.currency`。 | `USD` |

## 缓存与插件头

`x-vsr-cache-hit` 和 `x-vsr-fast-response` 在默认面上标识立即响应。缓存相似度和工具选择指标需要 `x-vsr-debug`。

| 头 | 面 | 说明 |
| ------ | ------- | ----------- |
| `x-vsr-cache-hit` | default | 响应来自响应缓存。 |
| `x-vsr-fast-response` | default | 响应由 `fast_response` 插件生成，没有上游模型调用。 |
| `x-vsr-cache-similarity` | debug | 响应缓存查找的相似度分数。 |
| `x-vsr-tools-strategy` | debug | 本次请求使用的语义工具选择检索策略。 |
| `x-vsr-tools-confidence` | debug | 工具选择检索器的最高相似度分数。 |
| `x-vsr-tools-latency-ms` | debug | 工具选择检索器延迟，单位毫秒。 |

## 响应示例

默认面 — 关键头、最终路由事实和回放 id 入口：

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-schema-version: 2
x-vsr-response-path: upstream
x-vsr-selected-recipe: default
x-vsr-selected-decision: complex-request
x-vsr-selected-algorithm: static
x-vsr-selected-model: reasoning-model
x-vsr-replay-id: replay_01J...
```

请求带 `x-vsr-debug: true` 时，被降级的中间细节和匹配信号也会内联发出：

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-schema-version: 2
x-vsr-response-path: upstream
x-vsr-selected-recipe: default
x-vsr-selected-decision: complex-request
x-vsr-selected-algorithm: static
x-vsr-selected-model: reasoning-model
x-vsr-session-phase: tool_loop
x-vsr-matched-context: long-context
x-vsr-matched-projections: use-reasoning-model
x-vsr-replay-id: replay_01J...
```

## 兼容性与解读

- 解析可选头之前先看 `x-vsr-schema-version`；当前值为 `2`。
- `x-vsr-matched-projections` 是投影头。单数形式不属于公开合约。
- 配方名限定本地信号、投影、决策、缓存、回放、指标以及学习/会话身份。把响应与 Insights 或指标关联时，把 `x-vsr-selected-recipe` 与本地决策/信号名一起使用。
- `event` 是决策和 DSL 使用的公开信号类型。规范 YAML 把 event 规则存在 `routing.signals.events` 下，与其他复数信号容器一致。
- 路由学习在内部使用 Router 自有的在线状态。用户通过 `global.router.learning.adaptation` 启用在线模型选择学习，通过 `global.router.learning.protection` 启用稳定性保护，传入稳定身份头，并可选设置 `routing.decisions[].adaptations.mode`、组件模式或 `adaptations.adaptation.candidate_set`。`scope: conversation` 保护一个 `x-conversation-id`；`scope: session` 保护更广的 `x-session-id`。旧的 `routing.decisions[].algorithm.session_aware` 形态不属于公开合约。
