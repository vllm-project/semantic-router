---
translation:
  source_commit: "aa7b7e7bc1de4d193342e869a952552a4c15552c"
  source_file: "docs/tutorials/plugin/router-replay.md"
  outdated: false
---

# 路由回放

## 概览

`router_replay` 是一个路由局部插件，用于覆盖单条路由的回放/调试采集。

配方的 `routing.data_policy.replay: false` 优先于全局和路由局部回放设置。该策略禁止采集，包括被拒绝的请求；`router_replay.enabled: true` 也不能覆盖它。Vault 使用此策略，因此其请求不会出现在 Dashboard Insights 中。详见[回放 API 和隐私控制](../../api/router#router-replay)。

默认 `memory` 存储会在配置重载或路由器重启时丢失记录。如果需要在调整配方时保留会话历史，请在[共享回放服务](../learning/memory-and-replay#configuration)中配置 Postgres 或 Redis 等持久化存储。

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
      max_body_bytes: 4096
      max_tool_trace_steps: 100
```

## Looper 诊断 {#looper-diagnostics}

Confidence Looper 记录包含版本化的 `route_diagnostics.looper` 对象。它包含有界的 attempt 元数据、token 和成本记账、延迟、处置原因码、tracing 启用时的 OpenTelemetry trace ID，以及 `final_attempt_ordinal`。Attempt 详情会从对查看者脱敏的响应中省略，仍可供具有回放-detail 权限的主体使用。

Looper 诊断永不包含提示词、响应、隐藏推理、工具参数、端点 URL、凭据或原始错误。Attempt 数量和编码大小有上限；截断是显式的，被丢弃的 token 用量仍会计入。

请求正文、响应正文和工具 traces 可能包含密钥或个人数据。只采集所需的最少内容，在共享回放服务中设置保留策略，并限制回放读取权限。完整示例见：
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml)。
