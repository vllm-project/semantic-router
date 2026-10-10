---
translation:
  source_commit: "aa7b7e7bc1de4d193342e869a952552a4c15552c"
  source_file: "docs/tutorials/plugin/router-replay.md"
  outdated: false
---

# 路由回放

## 概览

`router_replay` 是一个路由局部插件，用于覆盖单条路由的回放/调试采集。

`global.services.router_replay` 定义共享存储、保留策略、启用状态和采集默认值。decision 的 `router_replay` 插件只覆盖明确填写的采集字段；省略的字段继承全局默认值。回放不设置配方级默认值。详见[回放 API 和隐私控制](../../api/router#router-replay)。

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
      enabled: true
      store_backend: postgres
      capture_request_body: true
      capture_response_body: true
      capture_personal_data: false
      max_records: 10000
      max_body_bytes: 4096
      max_tool_trace_bytes: 0
      max_tool_trace_steps: 100
```

未配置服务时回放默认关闭。内置采集默认值为：请求和响应正文开启、个人数据采集开启、最多 10,000 条记录、每个正文 4,096 字节、工具轨迹单字段字节数不限、最多 100 个工具步骤。存储、保留时间和异步写入属于全局服务设置。

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

在全局或 decision 插件中设置 `capture_personal_data: false`。检测到个人数据时，回放保留路由、模型、信号及检测到的 PII 类型等元数据，但不保存请求和响应正文、提示词、工具定义或工具轨迹。该设置会按需评估配方已有的 PII 信号，即使 decision 未引用它们。

没有配置 PII 检测器、证据不可用或分类失败时，也会保守地省略内容；无需为了此设置强制配置 PII 信号。配置检测器后，确认没有个人数据的请求可以采集内容，例如：

```yaml
routing:
  signals:
    pii:
      - name: personal_data
        pii_types_allowed: []
```

## Looper 诊断 {#looper-diagnostics}

Confidence Looper 记录包含版本化的 `route_diagnostics.looper` 对象。它包含有界的 attempt 元数据、token 和成本记账、延迟、处置原因码、tracing 启用时的 OpenTelemetry trace ID，以及 `final_attempt_ordinal`。Attempt 详情会从对查看者脱敏的响应中省略，仍可供具有回放-detail 权限的主体使用。

Looper 诊断永不包含提示词、响应、隐藏推理、工具参数、端点 URL、凭据或原始错误。Attempt 数量和编码大小有上限；截断是显式的，被丢弃的 token 用量仍会计入。

请求正文、响应正文和工具 traces 可能包含密钥或个人数据。只采集所需的最少内容，在共享回放服务中设置保留策略，并限制回放读取权限。完整示例见：
[`config/fragments/plugin/router-replay/debug.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/router-replay/debug.yaml)。
