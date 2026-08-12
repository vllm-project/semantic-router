---
translation:
  source_commit: "ff987c87"
  source_file: "docs/tutorials/plugin/context-compression.md"
  outdated: false
---

# 上下文压缩

## 概述

`context_compression` 是一个路由级请求插件，可在已选提供商收到请求之前缩减大型工具/函数输出。它与路由器的信号压缩相互独立：路由过程会评估原始请求，随后此插件对上游请求正文执行变更。

该实现采用本地的、抽取式的查询感知方法，并以故障开放方式运行。它使用有界的 BM25 风格排序，保留开头和结尾的上下文，并且绝不会更改系统、用户或助手文本。

## 主要优势

- 减少工具密集型路由的提供商输入 token 数量。
- 保持在原始请求上评估路由和安全信号。
- 按决策应用，而不是更改每个请求。
- 无法解析或重写请求时采用故障开放方式。
- 保留有效的 JSON 结构和非文本多模态块。
- 支持 OpenAI 工具/函数消息和 Anthropic `tool_result` 块。

## 它解决什么问题？

智能体和检索工作负载携带的工具输出通常远大于用户问题。转发每一行低相关性内容会增加延迟和成本，却不能改善回答。

## 何时使用

请将其用于以大型文本工具输出为主的决策。不要在要求工具载荷逐字节完全相同的路由上启用它。

## 配置

```yaml
routing:
  decisions:
    - name: tool-heavy-route
      plugins:
        - type: context_compression
          configuration:
            enabled: true
            mode: auto
            budget:
              trigger_tokens: auto
              target_tokens: auto
              reserve_output_tokens: auto
            targets:
              tool_outputs:
                mode: extractive
                min_tokens: 2000
                target_tokens: 1000
              history:
                mode: preserve
              rag:
                mode: preserve
              memory:
                mode: preserve
            scoring:
              method: bm25
            recovery:
              enabled: false
              ttl_seconds: 900
              max_bytes_per_request: 10485760
              max_total_bytes: 268435456
              max_retrievals: 8
            request_controls:
              enabled: false
              header: x-vsr-compression-control
              allowed: [bypass, target]
              max_target_tokens: 16000
            failure_mode: fail_open
```

`targets.tool_outputs.target_tokens` 必须小于 `min_tokens`。`budget` 适用于完整的已选模型请求；工具输出目标仍保留各自的单项阈值和上限。`auto` 根据已选模型的上下文窗口和所请求的输出预留量推导请求预算。

默认情况下，RAG 和内存证据受类型化溯源信息保护。只有当路由明确接受证据压缩时，才应将对应目标的模式设为 `extractive`。

## 内容处理

- 纯文本会被拆分为有界分块，并根据产生该输出的工具调用意图进行排序；没有该意图时则回退到近期用户文本。
- JSON 对象和数组字符串只会通过字符串叶节点进行压缩。键、数组、对象、数字、布尔值和 null 值均保持其类型。
- OpenAI 数组内容会压缩文本块并保留图像块。
- 支持 Anthropic `tool_result` 的字符串和数组内容；`tool_use_id`、`is_error`、图像和缓存控制元数据会被保留。
- 大型单行、经最小化处理、CJK、表情符号以及无空白字符的载荷使用保守的字节感知 token 估算方法。

如果无法在配置的预算内安全缩减载荷，则在 `fail_open` 下会原样发送该载荷；若明确设置为 `fail_closed`，路由则会失败。

历史记录压缩会保护每条系统消息、当前用户轮次、最新助手轮次以及完整的工具交互。可选的 `recoverable` 目标会将原始内容存入共享 Redis/Valkey 存储，注入预留的 `vsr_context_retrieve` 工具，并使用已配置的 Looper 端点执行非流式后续请求。恢复的作用域限定于请求和受信任用户，并受 TTL、字节数和检索次数限制。流式请求会保留可恢复目标，而不会暴露内部工具。

## 请求控制

除非匹配的路由启用了请求控制，否则这些控制会被忽略。

- `bypass` 跳过压缩。
- `target=N` 覆盖工具输出目标，并受 `max_target_tokens` 限制。

默认请求头为 `x-vsr-compression-control`。绝不接受调用方提供的命名空间、恢复键或无界预算。

`scoring.method` 支持 `bm25`、`embedding` 和 `hybrid`。嵌入工作会批量执行，并存放在有界的记忆化缓存中；配置的嵌入运行时不可用时，混合评分会回退到 BM25。

## 管理与预览

- `GET /api/v1/context-compression/capabilities`
- `GET /api/v1/context-compression/health`
- `GET /api/v1/context-compression/stats`
- `POST /api/v1/context-compression/preview`
- `POST /api/v1/context-compression/recovery/invalidate`

预览仅返回计划、目标索引、token 数量、分数、警告和跳过原因。它绝不会返回源内容或被省略的内容，并且需要 `compression.preview` 权限。限定作用域的恢复失效操作需要 `compression.manage` 权限；它接受受信任的配方、决策、用户和请求坐标，并且绝不会返回派生出的作用域或恢复键。

## 运行时顺序

响应缓存首先检查不可变的规范请求。若未命中，RAG 和内存可以扩充单独的、将发送给提供商的工作正文；随后 `context_compression` 会在提供商请求转换以及提供商提示词缓存标记注入之前运行。最终的 Envoy 正文变更始终使用该工作正文，包括 auto、specified-model、Response API、Anthropic、流式请求和 Looper 路径。

压缩诊断信息会记录到指标和 Router Replay 中：已选模型、策略、请求/条目预算、token 计数器来源、触发原因、压缩前/后/节省的 token 数量、内容格式、已压缩消息数、被省略分块数、恢复次数，以及故障开放或跳过原因。不会记录原始的被省略内容和恢复键。
