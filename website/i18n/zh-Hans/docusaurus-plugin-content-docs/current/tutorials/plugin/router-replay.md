---
translation:
  source_commit: "bef43a20a96e7b329d314473083be6aa4bad0f03"
  source_file: "docs/tutorials/plugin/router-replay.md"
  outdated: false
---

# 路由回放

## 概览

`router_replay` 是一个路由局部插件，用于覆盖单条路由的回放/调试采集。

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

Confidence Looper 记录包含版本化的 `route_diagnostics.looper` 对象。它包含有界的 attempt 元数据、token 和成本记账、延迟、处置原因码、tracing 启用时的 OpenTelemetry trace ID，以及 `final_attempt_ordinal`。普通记录还会在 `route_diagnostics` 上包含 `effective_max_output_tokens`、`effective_max_output_tokens_source` 和 `effective_max_output_tokens_fallback`。Looper attempt 会重复这三个字段，以便 Confidence 升级能显示每个模型合成后的上限。Attempt 详情会从对查看者脱敏的响应中省略，仍可供具有回放-detail 权限的主体使用。

Looper 诊断永不包含提示词、响应、隐藏推理、工具参数、端点 URL、凭据或原始错误。Attempt 数量和编码大小有上限；截断是显式的，被丢弃的 token 用量仍会计入。

请求正文、响应正文和工具 traces 可能包含密钥或个人数据。只采集所需的最少内容，在共享回放服务中设置保留策略，并限制回放读取权限。完整示例见：
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml)。
