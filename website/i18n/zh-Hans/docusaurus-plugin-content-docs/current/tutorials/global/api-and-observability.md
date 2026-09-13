---
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/tutorials/global/api-and-observability.md"
  outdated: false
---

# API 与可观测性

## 概览

本页介绍用于暴露接口和遥测的共享运行时块。

这些设置作用于整台路由器，应放在 `global:` 下，而不是路由局部的插件片段中。

## 主要优势

- 让可观测性与接口控制在各路由间保持一致。
- 避免在路由局部配置中重复指标或 API 设置。
- 将回放与 Response API 明确为共享服务。
- 把运维控制集中在一层路由器级配置中。

## 解决什么问题？

如果按路由分别配置 API 和遥测，运维面会碎片化，难以推理。

`global:` 的这一部分把共享接口和监控设置收拢到一处。

## 何时使用

在以下情况使用这些块：

- 路由器应暴露共享 API
- 整台路由器应启用 Response API
- 指标和 tracing 只需配置一次
- 回放采集应作为共享运维服务保留

## 配置

### 路由器配置校验

管理 API 会校验并规范化候选配置，但不会写入：

```http
POST /api/v1/config/validate
Content-Type: application/json

{"yaml":"version: v0.3\n..."}
```

成功响应包含 `valid: true` 以及规范化后的规范 YAML。校验使用与 `PATCH /api/v1/config` 和 `PUT /api/v1/config` 相同的解析器和语义检查，但会原样保留 `${ENV_VAR}` 引用，而不是读取进程密钥。该端点需要 `config.read`；不意味着可以查看明文密钥。

### API

```yaml
global:
  services:
    api:
      batch_classification:
        max_batch_size: 100
```

`max_batch_size` 限制每次 `/api/v1/diagnostics/classify/batch` 请求的 `texts` 数量。超过上限会返回 `400 INVALID_INPUT`。

### Response API

```yaml
global:
  services:
    response_api:
      enabled: true
      store_backend: redis        # default; use "memory" only for local development
      redis:
        address: "redis:6379"
```

`store_backend` 控制响应和对话历史的持久化位置。可用后端：

| 后端 | 持久性 | 适用场景 |
|---------|-----------|----------|
| `redis` | 路由器重启后仍保留，可在副本间共享 | 生产（默认） |
| `memory` | 路由器重启后丢失 | 仅用于本地开发 |

### 可观测性

```yaml
global:
  services:
    observability:
      metrics:
        enabled: true
      tracing:
        enabled: true
        provider: opentelemetry
        exporter:
          type: otlp
          endpoint: jaeger:4317
          insecure: true
        sampling:
          type: probabilistic
          rate: 0.1
```

推荐的 tracing 采样类型是 `probabilistic`。已有配置若使用 `traceidratio` 或 `trace_id_ratio`，仍可作为兼容别名继续工作。

常见 Prometheus 指标族：

| 族 | 示例指标 |
|--------|-----------------|
| 请求 | `llm_model_requests_total`、`llm_request_errors_total` |
| 错误 | `llm_request_errors_total{reason="timeout"}` |
| 延迟 | `llm_model_completion_latency_seconds`、`llm_model_ttft_seconds`、`llm_model_tpot_seconds`、`llm_model_routing_latency_seconds` |
| Token 与成本 | `llm_model_tokens_total`、`llm_model_prompt_tokens_total`、`llm_model_completion_tokens_total`、`llm_model_cost_total` |
| 路由 | `llm_model_routing_modifications_total`、`llm_routing_reason_codes_total` |
| 选择 | `llm_model_selection_total`、`llm_model_selection_duration_seconds`、`llm_model_inflight_requests` |
| Looper | `llm_looper_attempts_total`、`llm_looper_attempt_duration_seconds`、`llm_looper_attempt_first_byte_seconds`、`llm_looper_attempt_tokens_total`、`llm_looper_attempt_cost_total`、`llm_looper_execution_duration_seconds` |
| 缓存 | `llm_cache_plugin_hits_total`、`llm_cache_plugin_misses_total`、`llm_cache_warmth_estimate` |
| RAG | `rag_retrieval_attempts_total`、`rag_retrieval_latency_seconds`、`rag_cache_hits_total`、`rag_cache_misses_total` |
| 会话 | `llm_session_model_transitions_total`、`llm_session_turn_prompt_tokens`、`llm_session_turn_completion_tokens`、`llm_session_turn_cost` |
| 翻译与请求参数策略 | `llm_translation_lossy_total`、`sr_request_params_blocked_total`、`sr_request_params_unknown_field_stripped_total` |
| 信号 | `llm_signal_extraction_total`、`llm_signal_match_total`、`llm_signal_extraction_latency_seconds` |
| 复杂度判定 | `llm_complexity_verdict_total`（按 `rule`、`verdict`、`source`）、`llm_complexity_evaluation_failures_total` |
| 远程分类器后端 | `llm_remote_connector_requests_total`（按 `operation`、`outcome`）、`llm_remote_connector_request_duration_seconds`、`llm_remote_connector_retries_total` |

Looper 指标标签仅限于有界的算法、阶段、状态、原因、token 类型和货币值。请求 ID、trace ID、序号、决策名称、模型名称、分数和阈值应通过 traces 或详细的路由回放查看，而不是作为 Prometheus 标签。当前详细的 attempt 指标覆盖 Confidence 算法。

### 性能分析

Router 可在专用监听器上暴露 Go `pprof` 端点，用于 CPU、堆、goroutine 和执行跟踪调查。

```yaml
global:
  services:
    observability:
      profiling:
        enabled: false        # default; opt in only while investigating
        port: 6060            # default
        bind: 127.0.0.1       # default; loopback only
```

性能分析默认关闭。启用后绑定 `127.0.0.1:6060`，因此 profile 仅可从 Router 容器或主机访问；除非显式更改 `bind`，否则不会发布到可路由接口。

```bash
go tool pprof http://127.0.0.1:6060/debug/pprof/heap
```

说明：

- `bind` 必须是 IP 地址或 `localhost`。空值或主机名会被拒绝，并跳过 profiling 监听器。
- 显式设置 `port: 0` 会请求临时端口；实际地址会写在启动日志行 `profiling_server_starting` 中。
- 端口不得与 ExtProc、指标或管理 API 端口冲突。冲突或无法绑定的监听器会被记录并跳过，不会中止 Router 启动。
- 该开关仅在启动时读取一次。更改后需要重启 Router；配置热重载不会接管 profiling 监听器。

### 跳过处理请求头

`global.router.skip_processing.enabled` 是部署级开关，决定路由器是否尊重 `x-vsr-skip-processing` 请求头。开关打开且上游过滤器将该请求头设为 `true` 时，路由器对该单次请求变为 no-op：每个 Envoy ext_proc 回调都返回 CONTINUE，不进行分类、路由、改写、缓存或检查请求与上游响应。开关关闭时（默认）会完全忽略该请求头。

```yaml
global:
  router:
    skip_processing:
      enabled: false        # default; flip to true to honor the header
```

Helm chart 通过顶层值（`router.skipProcessing.enabled`）暴露同一开关，因此可在安装时启用，而无需编辑嵌入的规范配置：

```bash
helm install vsr ./deploy/helm/semantic-router \
  --set router.skipProcessing.enabled=true
```

仅当由已认证的上游过滤器（Envoy AI Gateway、ext_authz、路由级过滤器等）负责按信任依据设置或剥离该请求头时，才应启用此开关。促成该开关的 AI Gateway 互操作模式背景见 [issue #1808](https://github.com/vllm-project/semantic-router/issues/1808)。

### 路由回放

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres     # explicit durable, SQL-queryable audit storage
      async_writes: true
      postgres:
        host: postgres
        port: 5432
        database: vsr
        user: router
        password: ${ROUTER_REPLAY_POSTGRES_PASSWORD}
```

路由回放默认关闭。将 `global.services.router_replay.enabled` 设为启用后，它对整台路由器生效；启用后，决策会采集回放，除非该决策添加路由局部 `router_replay` 插件并将 `enabled` 设为 `false`。决策也可以显式选择加入。若未配置持久后端，默认内存存储仅存在于进程内，重启后丢失。

`store_backend` 控制路由决策回放记录的持久化位置。可用后端：

| 后端 | 持久性 | 适用场景 |
|---------|-----------|----------|
| `postgres` | 完整 SQL 可查询，长期审计保留 | 生产审计存储 |
| `redis` | 路由器重启后仍保留，可在副本间共享 | 已运行 Redis 的轻量部署 |
| `milvus` | 可向量检索的回放记录 | 语义回放搜索 |
| `qdrant` | 可向量检索的回放记录 | 在 Qdrant 部署中进行语义回放搜索 |
| `memory` | 路由器重启后丢失 | 仅用于本地开发 |

## 数据与安全

- Response API 和路由回放可能持久化提示词、响应、路由结果和工具 traces。启用前请设置 TTL、采集上限、租户/用户范围和读取权限。
- 将管理 API 绑定到私有接口，或在远程暴露前启用基于角色的 token 认证。
- traces 和指标标签应携带有界标识符，而不是原始请求内容或密钥。
- `pprof` 端点会暴露命令行参数、goroutine 栈和堆内容。调查之外请保持关闭，并将 `bind` 留在回环上，除非有意将可访问的监听器置于已认证的访问控制之后。
- 完整服务配置见 [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)。
