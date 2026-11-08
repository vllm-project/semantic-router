---
is_mtpe: false
translation:
  source_commit: "aa7b7e7bc1de4d193342e869a952552a4c15552c"
  source_file: "docs/tutorials/plugin/router-replay.md"
  outdated: true
---

# 路由回放

## 概览

`router_replay` 是一个路由局部插件，用于覆盖单条路由的回放/调试采集。

`global.services.router_replay` 定义共享存储和保留策略；`global.plugins` 中的 `router_replay` 插件定义启用状态和采集默认值。decision 的 `router_replay` 插件只覆盖明确填写的采集字段；省略的字段继承全局默认值。回放不设置配方级默认值。详见[回放 API 和隐私控制](../../api/router#router-replay)。

全部五个内置 MoM 配方（包括 Vault）默认启用 PostgreSQL 回放。`vllm-sr serve` 会管理 PostgreSQL 及持久卷。通用的 `memory` 存储会在配置重载或路由器重启时丢失记录；其他部署方式可在[共享回放服务](../learning/memory-and-replay#configuration)中配置 PostgreSQL 或 Redis。

## 主要优势

- 让一条路由覆盖路由器级回放默认值。
- 支持请求和响应正文控制。
- 明确声明存储限制，而不是隐藏它们。

## 解决什么问题？

回放采集很有用，但有些路由需要不同于路由器级默认值的采集策略。`router_replay` 让一条路由退出或覆盖请求/响应正文采集限制，而不更改全局回放存储设置。

## 何时使用

- 某条路由应覆盖路由器级回放策略
- 采集限制应按路由明确声明
- 应在其他地方保持开启的同时，为特定路由禁用回放

## 配置

### 全局采集默认值

```yaml
global:
  services:
    router_replay:
      store_backend: postgres
  plugins:
    - type: router_replay
      configuration:
        enabled: true
        capture_request_body: true
        capture_response_body: true
        capture_personal_data: false
        max_records: 10000
        max_body_bytes: 4096
        max_tool_trace_bytes: 0
        max_tool_trace_steps: 100
```

没有 `router_replay` 插件时回放保持关闭。`global.plugins` 中的插件提供共享采集默认值，决策的插件会逐字段覆盖它们。内置采集默认值为：请求和响应正文开启、个人数据采集开启、最多 10,000 条记录、每个正文 4,096 字节、工具轨迹单字段字节数不限、最多 100 个工具步骤。存储、保留时间和异步写入属于全局服务设置。

### decision 覆盖

要为某条路由禁用回放，添加：

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: false
```

要为某条路由自定义采集，添加：

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      max_records: 10000
      capture_request_body: true
      capture_response_body: true
      capture_personal_data: false
      max_body_bytes: 4096
      max_tool_trace_steps: 100
```

插件中明确填写的 `false` 会覆盖全局值；`enabled: true` 可以在全局关闭时单独开启该 decision 的回放，`enabled: false` 则关闭它。被拒绝且尚未选中 decision 的请求使用全局默认值，选中后的请求应用该 decision 的覆盖。配置变更不会删除已有记录，也不控制模型提供方的留存策略。

`max_tool_trace_bytes: 0` 和 `max_tool_trace_steps: 0` 表示移除对应上限。`max_body_bytes: 0` 沿用记录器的 4,096 字节回退值；`max_records: 0` 在内存存储中使用 200 条记录的回退值。这些显式零值不会继承全局配置；需要继承时应省略字段。正文和结构化文本按 UTF-8 字节截断并保持字符完整，截断后的原始正文可能不再是完整 JSON，应检查截断标记。

### 不采集个人数据

在全局或 decision 插件中设置 `capture_personal_data: false`。检测到个人数据时，回放保留路由、模型、信号及检测到的 PII 类型等元数据，但不保存请求和响应正文、提示词、工具定义或工具轨迹。提示词在路由之后检查，请求不会等待这一步：除非某个路由 decision 已用 PII 信号读过它，回放记录先不含提示词写入，再由配方已配置的 PII 规则单独读取最新一条用户消息；读取完成且未发现个人数据后，提示词才会补进该记录。

配方没有 `pii` 规则、证据不可用或 `pii` 规则的模型没有给出答案时，也会保守地省略内容；无需为了此设置强制配置 PII 信号。配置 `pii` 规则后，确认没有个人数据的请求可以采集内容，例如：

```yaml
recipes:
  - name: default
    signals:
      pii:
        - name: personal_data
```

## 影子请求 {#shadow}

`shadow` 块把 decision 已批准请求的一份有界、抽样的副本发给另一个模型，replay 记录保存它的结果，主响应既不等待它，也不因它改变。只有在该 decision 记录 replay 时影子请求才会运行。它取代了 `shadow_dispatch` 插件，`vllm-sr config migrate` 会把旧插件改写为这个块。

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
      shadow:
        model: candidate-model
        sample_rate: 0.05
        max_concurrency: 2
        max_queue_depth: 8
        timeout_seconds: 30
        max_response_bytes: 1048576
        max_retries: 0
        capture_response_body: false
        max_capture_bytes: 4096
        tls_skip_verify: false
        forward_headers: []
```

- `model` 必填，必须在 `providers.models` 中有后端。
- `sample_rate` 默认 `1.0`；`0` 保留声明但从不发送。
- `max_concurrency`、`max_queue_depth`、`timeout_seconds`、`max_response_bytes`、`max_retries` 限定资源；超出队列的调用以 `queue_full` 丢弃。
- `capture_response_body` 默认关闭，只保存大小、token 数和 SHA-256；打开后在结果中保存不超过 `max_capture_bytes` 的摘录。
- `forward_headers` 只转发列出的 decision `header_mutation` 头；已知的凭据头不能列出。
- 经 looper 执行的算法（ratings、confidence、fusion、ReMoM、workflows）在加载时拒绝 `shadow`。

影子结果以 `source: shadow_dispatch` 追加到主请求的 replay 记录，指标为 `sr_shadow_dispatch_total`、`sr_shadow_dispatch_latency_seconds` 等。

## Looper 诊断 {#looper-diagnostics}

Confidence Looper 记录包含版本化的 `route_diagnostics.looper` 对象。它包含有界的 attempt 元数据、token 和成本记账、延迟、处置原因码、tracing 启用时的 OpenTelemetry trace ID，以及 `final_attempt_ordinal`。Attempt 详情会从对查看者脱敏的响应中省略，仍可供具有回放-detail 权限的主体使用。

Looper 诊断永不包含提示词、响应、隐藏推理、工具参数、端点 URL、凭据或原始错误。Attempt 数量和编码大小有上限；截断是显式的，被丢弃的 token 用量仍会计入。

请求正文、响应正文和工具 traces 可能包含密钥或个人数据。只采集所需的最少内容，在共享回放服务中设置保留策略，并限制回放读取权限。完整示例见：
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml)。
