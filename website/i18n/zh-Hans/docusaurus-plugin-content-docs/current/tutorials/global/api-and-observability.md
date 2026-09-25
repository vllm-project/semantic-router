---
translation:
  source_commit: "cd975c6129460d700dd9c116ddd3356cdd90e915"
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
      routing_preview:
        request_timeout_seconds: 120
        max_concurrency: 16
      batch_classification:
        max_batch_size: 100
```

`max_batch_size` 限制每次 `/api/v1/diagnostics/classify/batch` 请求的 `texts` 数量。超过上限会返回 `400 INVALID_INPUT`。

`routing_preview` 作用于 `POST /api/v1/routing/preview`。推理时限从请求体解析完成后开始计算，默认 120 秒。`request_timeout_seconds` 可设为 1 至 3600 秒，应根据实际输入长度和部署硬件的测量结果选择。该设置支持配置热更新；其他 HTTP 路由保留现有超时设置。

达到时限后，API 返回 `504 REQUEST_TIMEOUT`，并取消排队中或可取消的推理。已经执行的原生推理可能稍后才结束；在其结束前，模型资源和并发名额都会保留，关闭服务时也不例外。`max_concurrency` 是正整数，默认允许 16 个推理任务并发执行，不提供等待队列；名额用完后，新请求返回 `429 OVERLOADED`。修改此并发上限需要重新部署并重启服务，热更新会拒绝该变更。

响应写入另有 5 秒余量，用于发送结果或超时响应。Dashboard Topology 使用配置的 Preview 时限加上该余量，并传递客户端取消信号。Recipe 探测仍使用 `probes.yaml` 中独立的 `evaluation.request_timeout_seconds` 客户端时限；应按实际测试配置。如果外部 HTTP 客户端或代理需要收到 Router 的超时响应，其时限应至少多留 5 秒。

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

| 指标族 | 示例指标 |
|--------|-----------------|
| 请求终态 | `llm_request_outcomes_total`, `llm_request_duration_seconds` （按有界的 `traffic_kind` 和 `outcome`） |
| 后端派发与错误事件 | `llm_model_requests_total`, `llm_request_errors_total` （不能用作已完成客户端请求的分母） |
| 耗时 | `llm_model_completion_latency_seconds`, `llm_model_first_response_observation_seconds`, `llm_model_response_duration_per_output_token_seconds`, `llm_model_routing_latency_seconds` |
| Token 与成本 | `llm_model_tokens_total`, `llm_model_prompt_tokens_total`, `llm_model_completion_tokens_total`, `llm_model_cost_total` |
| 路由 | `llm_model_routing_modifications_total`, `llm_routing_reason_codes_total` |
| 选择 | `llm_model_selection_total`, `llm_model_selection_duration_seconds`, `llm_model_inflight_requests` |
| Looper | `llm_looper_attempts_total`, `llm_looper_attempt_duration_seconds`, `llm_looper_attempt_first_byte_seconds`, `llm_looper_attempt_tokens_total`, `llm_looper_attempt_cost_total`, `llm_looper_execution_duration_seconds` |
| 缓存 | `llm_cache_plugin_hits_total`, `llm_cache_plugin_misses_total`, `llm_cache_warmth_estimate` |
| RAG | `rag_retrieval_attempts_total`, `rag_retrieval_latency_seconds`, `rag_cache_hits_total`, `rag_cache_misses_total` |
| 会话 | `llm_session_model_transitions_total`, `llm_session_turn_prompt_tokens`, `llm_session_turn_completion_tokens`, `llm_session_turn_cost` |
| 翻译与请求参数策略 | `llm_translation_lossy_total`, `sr_request_params_blocked_total` |
| 信号 | `llm_signal_extraction_total`, `llm_signal_match_total`, `llm_signal_extraction_latency_seconds` |
| 复杂度判定 | `llm_complexity_verdict_total` （按 `rule`、`verdict`、`source`）, `llm_complexity_evaluation_failures_total` |
| 远程分类器后端 | `llm_remote_connector_requests_total` （按 `operation`、`outcome`）, `llm_remote_connector_request_duration_seconds`, `llm_remote_connector_retries_total` |
| Recipe 路由 | `llm_entrypoint_requests_total`, `llm_recipe_selections_total`, `llm_routing_stage_duration_seconds` |
| 投影 | `llm_projection_score` （按配置中的 Recipe 和投影名称） |
| Trace 导出 | `llm_trace_export_spans_total` （按导出批次结果） |

`llm_request_outcomes_total{traffic_kind="inference"}` 在每个公开推理请求到达终态时只计数一次。已认证的内部 Looper 请求使用 `inference_internal`；模型目录、Response 对象和健康检查使用各自的流量类型。结果区分成功、客户端或服务端错误、取消、超时、不完整响应和其他失败。后端选择和错误事件不等于另一个已完成的公开请求。

模型耗时指标按实际观测命名：`llm_model_first_response_observation_seconds` 测量首个流式响应块或非流式响应头；`llm_model_response_duration_per_output_token_seconds` 将完整响应耗时除以已上报的输出 token 数。这两者分别不等同于首 token 延迟和仅解码阶段的逐 token 延迟。可选的窗口指标每个模型最多汇总 10,000 次已观测完成响应，不估算利用率、队列深度或提供方错误率。

`semantic_router.request` span 覆盖整个 ExtProc 请求，包括流式响应和取消。它使用有界的 `traffic.kind` 和路由模板区分推理、目录与健康轮询，不存储 URL 查询参数或资源 ID。信号、决策、算法、插件与实际上游请求是子阶段；`routing.entrypoint`、`routing.recipe`、`decision.name` 和 `routing.algorithm` 记录已解析的路由身份。`routing.backend.resolved` 事件记录选择证据，上游 span 测量提供方响应时间。如果本地响应检查拦截上游 HTTP 200，上游 span 仍记录 200，根 span 记录最终返回给客户端的状态。

上游 span 还带有 OpenTelemetry GenAI 属性，便于支持 GenAI 语义的 trace 后端展示每次提供方调用：`gen_ai.operation.name`（`chat`）、`gen_ai.provider.name`、`gen_ai.request.model`（发往上游的提供方模型 ID），以及提供方上报的 `gen_ai.usage.input_tokens` 和 `gen_ai.usage.output_tokens`。非流式响应还会记录 `gen_ai.response.model`，流式响应暂不记录。提供方未上报的用量保持为空，不做估算；`model.name` 仍为 Router 的逻辑模型名。GenAI 约定在上游仍处于 Development 状态，属性名可能变化。

信号证据事件分别记录有限的实测值和置信度，保留真实零值，缺失数据不填零。投影事件记录实际分数和配置名称，聚合信号阶段不虚构置信度。Trace 不包含原始提示词、信号错误文本或检索异常原文；无法事后补全旧 trace。

本地 `vllm-sr serve` 在 Grafana 中预置 **vLLM Semantic Router**。主面板覆盖公开推理结果、Recipe 流程、后端使用量、插件与响应缓存、遥测健康；折叠的成本和 MoM 面板区分模型上报用量与已支持的 Looper attempt 证据。Recipe 阶段耗时展示实测均值；投影计数展示评估次数，单次分数可在 Insights 查看。模型耗时使用观测均值，避免有限直方图桶把长请求的分位数截断。Prometheus 抓取 Router 和 Jaeger 的内部管理指标。缺少序列表示没有观测，不能当作零流量或健康结果；导出成功只表示 SDK 批次导出完成，不证明所有请求都被采样或持久保存。

本地 Jaeger 使用固定版本的 all-in-one 镜像、非 root Badger 存储及挂载到 `/tmp` 的栈专属命名卷 `<jaeger-container-name>-data`，保留 trace 七天。Grafana 使用镜像的非 root 用户，将数据库与偏好保存在 `/var/lib/grafana` 对应的 `<grafana-container-name>-data` 中。Prometheus 的本地 TSDB 保留十五天。替换容器会保留这些存储，删除遥测存储则开始新的历史；清理操作应与 benchmark、认证、Learning 和配置存储严格区分。

将旧 Jaeger 从内存切换为 Badger 不会迁移原有内存数据。采样仍由 Router 配置控制，搜索上限不是已保留 trace 总数。多节点持久化应使用外部 collector 和存储部署，不要共享单节点 Badger 卷。参见 [Jaeger 1.76 存储文档](https://www.jaegertracing.io/docs/1.76/deployment/#badger---local-storage)。

Looper 指标标签仅包含有界的算法、阶段、状态、原因、token 类型和货币。请求 ID、trace ID、序号、决策名、模型名、分数和阈值通过 trace 或详细 Router Replay 查看，不作为这些 Prometheus 指标的标签。当前详细 attempt 指标覆盖 Confidence 算法；缺少 attempt 证据不能解释为零次调用或完整成本核算。

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
