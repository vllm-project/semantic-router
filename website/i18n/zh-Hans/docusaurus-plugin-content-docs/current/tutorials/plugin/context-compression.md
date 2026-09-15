---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/plugin/context-compression.md"
  outdated: false
---

# 上下文压缩

## 概览

`context_compression` 是一个路由局部请求插件，在所选提供商收到请求之前缩减大型工具/函数输出。它与路由器信号压缩分开：路由评估原始请求，然后该插件执行上游 body 改写。

压缩是本地的、抽取式的、查询感知的，并且失败开放。它使用有界的 BM25 风格排序，保留开头和结尾上下文，并且从不更改 system、user 或 assistant 文本。

## 主要优势

- 在工具密集的路由上减少提供商输入 token。
- 让路由和安全信号继续基于原始请求。
- 按决策应用，而不是更改每个请求。
- 在无法解析或改写请求时失败开放。
- 保留有效 JSON 结构和非文本多模态块。
- 支持 OpenAI 工具/函数消息和 Anthropic `tool_result` 块。

## 解决什么问题？

Agent 和检索工作负载常常携带远大于用户问题的工具输出。转发每一行低相关性内容会增加延迟和成本，却不改善答案。

## 何时使用

在由大型文本工具输出主导的决策上使用。不要在需要字节级相同工具载荷的路由上启用。

## 配置

在 `routing.decisions[].plugins` 下添加该插件：

```yaml
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

`targets.tool_outputs.target_tokens` 必须低于 `min_tokens`。`budget` 作用于完整的所选模型请求；工具输出目标保留自己的每项阈值和上限。`auto` 根据所选模型上下文窗口和请求的输出预留推导请求预算。

RAG 和 memory 证据默认受类型化来源保护。仅当该路由明确接受证据压缩时，才将对应目标模式设为 `extractive`。

## 内容处理 {#content-handling}

- 纯文本会拆成有界块，并对照发起工具调用的意图排序，回退到最近的用户文本。
- JSON 对象和数组字符串仅通过字符串叶子压缩。键、数组、对象、数字、布尔值和 null 保持类型。
- OpenAI 数组内容压缩文本块并保留图像块。
- 支持 Anthropic `tool_result` 字符串和数组内容；保留 `tool_use_id`、`is_error`、图像和 cache-control 元数据。
- 大型单行、压缩、CJK、emoji 和无空白载荷使用保守的按字节 token 估计。

若载荷无法在配置预算内安全缩减，则在 `fail_open` 下原样发送，或在显式 `fail_closed` 下让路由失败。

历史压缩保护每条 system 消息、当前用户轮次、最近一条 assistant 轮次以及完整的工具交换。可选的 `recoverable` 目标会把原始内容存入共享 Redis/Valkey 存储，注入保留的 `vsr_context_retrieve` 工具，并使用配置的 Looper 端点进行非流式后续请求。恢复按请求和受信任用户限定范围，并受 TTL、字节和检索次数限制。流式请求会保留可恢复目标，而不是暴露内部工具。

## 请求控制 {#request-controls}

除非匹配的路由启用请求控制，否则会忽略它们。

- `bypass` 跳过压缩。
- `target=N` 覆盖工具输出目标，并受 `max_target_tokens` 限制。

默认请求头是 `x-vsr-compression-control`。永不接受调用方提供的命名空间、恢复键和无界预算。

`scoring.method` 支持 `bm25`、`embedding` 和 `hybrid`。嵌入工作会批量进行并保存在有界 memo 缓存中；配置的嵌入运行时不可用时，混合评分回退到 BM25。

## 管理与预览 {#management-and-preview}

- `GET /api/v1/context-compression/capabilities`
- `GET /api/v1/context-compression/health`
- `GET /api/v1/context-compression/stats`
- `POST /api/v1/context-compression/preview`
- `POST /api/v1/context-compression/recovery/invalidate`

预览只返回计划、目标索引、token 计数、分数、警告和跳过原因。它永不返回源内容或被省略内容，并需要 `compression.preview`。限定范围的恢复失效需要 `compression.manage`；它接受受信任的配方、决策、用户和请求坐标，并且永不返回派生范围或恢复键。

## 运行时顺序 {#runtime-order}

响应缓存先检查不可变的规范请求。未命中时，RAG 和 memory 可以丰富一份单独的、绑定提供商的工作 body，然后 `context_compression` 在提供商请求翻译和提供商提示词缓存标记注入之前运行。最终 Envoy body 改写始终使用该工作 body，包括 auto、指定模型、Response API、Anthropic、流式请求和 Looper 路径。

压缩诊断会记录到指标和路由回放：所选模型、策略、请求/项预算、token 计数来源、触发原因、压缩前/后/节省的 token、内容格式、被压缩消息数、省略块数、恢复次数，以及失败开放或跳过原因。不记录被省略的原始内容和恢复键。

完整示例见：
[`config/fragments/plugin/context-compression/tool-output.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/context-compression/tool-output.yaml)。
压缩会更改绑定提供商的上下文，并可能去掉正确答案所需的细节。在该路由具备任务特定质量测试之前，请保持失败开放行为；仅在已认证的共享存储和受信任用户身份下启用可恢复模式。
