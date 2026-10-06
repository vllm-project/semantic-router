---
translation:
  source_commit: "e86e1ac69ece8f9921cddbbfa12a4c2d8f50b66b"
  source_file: "docs/api/apiserver.md"
  outdated: true
---

# 路由器管理接口 {#router-management-api}

Router 管理 API 提供配置、路由预览、插件检查、模型诊断、存储和可观测性操作。默认监听端口 `8080`，本地栈将其绑定到 `127.0.0.1`。

模型流量请使用配置的 Envoy 监听器，见 [Router API](./router)。

## 从实时 schema 开始 {#start-with-the-live-schema}

运行中的 Router 根据已注册路由生成端点发现和 OpenAPI 文档。以下页面用于已注册方法、路径和查询参数、请求体字段、访问策略以及响应媒体类型：

| 路径 | 用途 |
| --- | --- |
| `GET /api/v1` | 带权限和敏感度元数据的端点发现 |
| `GET /openapi.json` | 完整 OpenAPI 3.0 文档 |
| `GET /openapi.json?path=...&method=...` | 单个有效路径或操作文档 |
| `GET /docs` | 交互式 Swagger UI |

本页按用户任务分组 API。你所运行版本的字段级事实来源是实时 OpenAPI 文档。

```bash
curl -sS http://localhost:8080/health
curl -sS http://localhost:8080/openapi.json
curl -sS 'http://localhost:8080/openapi.json?path=/api/v1/config&method=PATCH'
```

Agent 应直接调用这些 Router 端点；控制面板不属于发现路径。Website 也会在[可检索的 OpenAPI 参考](./openapi)中渲染生成的契约。

## 资源边界 {#resource-boundaries}

| 能力 | 职责 |
| --- | --- |
| `config` | 规范配置、Recipe、校验、计划及激活 |
| `routing` | 通过已配置的路由流水线评估请求 |
| `plugins` | 发现插件类型，检查 Recipe/Decision 绑定，预览与探测插件行为 |
| `inventory` | 查询已配置及已准备的运行时资源 |
| `diagnostics` | 调用指定的已准备模型或路由分类器 |
| `storage` | 管理持久数据、响应缓存分区和上下文恢复范围 |
| `observability` | 指标、回放、管理审计及结果反馈 |
| `system` | 健康、启动、就绪与 API 发现 |

插件配置仍属于规范文档中的 Recipe Decision。通过
`/api/v1/config/recipes/{name}` 和 `If-Match` 修改，不存在独立的插件配置库或第二套启用状态。
`/config/router` 已不再注册；当前规范配置 API 是 `/api/v1/config`。

## 访问与认证 {#access-and-authentication}

本地 CLI 将管理端口保留在 loopback。对于远程 Router，优先使用私有网络或 SSH 隧道，而不是公开该端口：

```bash
ssh -N -L 8080:127.0.0.1:8080 router-host
```

除非另行配置，否则管理认证默认关闭。若要要求 bearer token，设置 `global.services.management_api.auth.mode: bearer`，并在管理 API 配置中定义角色和 token 来源。然后发送：

```http
Authorization: Bearer <token>
```

`GET /health` 保持公开。启用 bearer 认证后，其他路由会强制执行其分配的权限。配置和回放响应也会脱敏敏感字段，除非主体拥有对应的 detail 权限。

## 健康与发现 {#health-and-discovery}

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/health` | 进程存活 |
| `GET` | `/ready` | 启动是否已完成 |
| `GET` | `/startup-status` | 启动和模型下载进度 |
| `GET` | `/api/v1` | 已注册端点发现 |
| `GET` | `/openapi.json` | 生成的 OpenAPI schema，可按精确的 `path` 和 `method` 收窄 |
| `GET` | `/docs` | Swagger UI |

使用 `/health` 做存活检查，使用 `/ready` 做就绪检查。在模型下载或运行时准备期间，进程可以是健康的，但 `/ready` 仍返回 `503`。

## 不调用推理即可检查信号 {#inspect-signals-without-an-inference-call}

调优信号或诊断决策未匹配的原因时，分类端点很有用。它们不会调用生成后端。

```bash
curl -sS http://localhost:8080/api/v1/diagnostics/classify/intent \
  -H 'Content-Type: application/json' \
  -d '{"text":"Write a Python function that merges two sorted lists."}'
```

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `POST` | `/api/v1/diagnostics/classify/intent` | 评估意图/领域路由 |
| `POST` | `/api/v1/diagnostics/classify/pii` | 检测已配置的 PII 类型 |
| `POST` | `/api/v1/diagnostics/classify/security` | 评估越狱和安全分类 |
| `POST` | `/api/v1/diagnostics/classify/fact-check` | 判断文本是否需要事实核查 |
| `POST` | `/api/v1/diagnostics/classify/user-feedback` | 分类用户反馈 |
| `POST` | `/api/v1/diagnostics/classify/combined` | 运行意图、PII 和安全分类 |
| `POST` | `/api/v1/diagnostics/classify/batch` | 对批次运行选定的分类器 |
| `POST` | `/api/v1/routing/preview` | 评估所有已配置信号 |
| `POST` | `/api/v1/diagnostics/nli` | 评估前提/假设对 |
| `POST` | `/api/v1/diagnostics/embeddings` | 生成已配置的文本或图像嵌入 |
| `POST` | `/api/v1/diagnostics/similarity` | 比较一对文本 |
| `POST` | `/api/v1/diagnostics/similarity/batch` | 运行批量相似度匹配 |

名称、分数和匹配规则取决于当前配方。各端点支持的输入形式见实时 schema。

当命中的决策使用 `fast_response` 时，Preview 返回
`selection_status: not_required` 和 `selection_method: fast_response`，不包含
`selected_model`。这种即时响应不需要模型分配或候选模型的能力、上下文准入检查。
面向客户端的响应模型标识不代表选择或调用了生成后端。

当输入超过配置的推理预算时，Guard 和 PII 会在 `signal_errors` 中报告 `input_limit`。重试前请检查实际生效的模型和部署限制。其他推理失败保留对应的有限错误码；路由结果由配置的未知信号策略决定。

## 诊断已准备的模型绑定 {#diagnose-a-prepared-model-binding}

先请求 `GET /api/v1/diagnostics/models?recipe=<name>`，再使用返回的绑定名称与任务契约。
以下调用均要求显式的 `recipe` 和 `binding`，不会按模型家族名称猜测或加载备用模型。

| `/api/v1/diagnostics/models` 下的操作 | 任务 | Vela 用途 |
| --- | --- | --- |
| `POST /labels` | 标签概率与已配置的滑窗扫描 | Domain、Guard、FactCheck、Feedback、Modality |
| `POST /label-scores` | 独立标签分数及操作点策略 | Safety、Hazard |
| `POST /tokens` | 原始 UTF-8 字节偏移对应的实体 | PII |
| `POST /embeddings` | 绑定的有效嵌入表示 | Embedding |
| `POST /rerank` | 按输入顺序返回查询/文档相关性分数 | Reranker |

```bash
curl -sS 'http://localhost:8080/api/v1/diagnostics/models?recipe=default'
curl -sS http://localhost:8080/api/v1/diagnostics/models/labels \
  -H 'Content-Type: application/json' \
  -d '{"recipe":"default","binding":"<prepared-binding-name>","text":"Explain this Python error."}'
```

响应携带实际 binding、artifact revision、provider、device、precision 和输入限制。
请求在模型调用结束前持有其运行时 generation，热重载不会提前关闭使用中的模型。
未准备或属于其他 Recipe 的绑定明确失败。Guard/PII 保留原有窗口大小和重叠，
Hazard 保留阈值及策略摘要。Vela Encoder 是任务共享的底层 artifact，不需要独立推理端点。
发现接口只列出当前准备的任务，不触发模型加载。

原有 classification、embedding、NLI 和 similarity 便利接口也接受可选 `recipe`：
省略时保持默认 Recipe 行为，指定时必须准确解析。验证具体模型时优先使用上述 typed binding API。Rerank 批量受
`global.services.api.batch_classification.max_batch_size` 限制，未设置时为 100 对。调用复用模型资源的
admission，并设定两分钟请求 deadline；原生推理即使被取消，也在实际结束后才释放租约。

## 检查模型与指标 {#inspect-models-and-metrics}

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/v1/inventory/models` | 已加载模型清单 |
| `GET` | `/api/v1/inventory/classifier` | 分类器配置和状态 |
| `GET` | `/api/v1/inventory/embedding-models` | 已加载的嵌入模型 |
| `GET` | `/v1/models` | OpenAI 兼容模型列表 |
| `GET` | `/api/v1/observability/classification-metrics` | 分类计数器和耗时 |

分类器信息中的密钥会被脱敏，除非调用者拥有 `secret_view`。

## 读取和更改 Router 配置 {#read-and-change-router-configuration}

更改前先读取当前规范文档及其 `ETag`：

```bash
curl -i http://localhost:8080/api/v1/config \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}"
```

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/v1/config` | 读取当前生效的规范配置 |
| `POST` | `/api/v1/config/validate` | 校验并规范化，不写入 |
| `POST` | `/api/v1/config/plan` | 规划精确候选并返回当前/候选 ETag，不写入 |
| `PATCH` | `/api/v1/config` | 合并、校验、持久化并热重载更新 |
| `PUT` | `/api/v1/config` | 替换、校验、持久化并热重载文档 |
| `GET` | `/api/v1/config/versions` | 列出配置备份 |
| `POST` | `/api/v1/config/rollback` | 恢复备份 |
| `GET` | `/api/v1/config/hash` | 比较已持久化、已生成和当前生效的哈希 |

配方操作使用同一份规范文档：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/v1/config/recipes` | 列出默认和命名配方及其入口 |
| `POST` | `/api/v1/config/recipes/validate` | 校验配方变更而不应用 |
| `GET` | `/api/v1/config/recipes/{name}` | 读取单个配方 |
| `PUT` | `/api/v1/config/recipes/{name}` | 创建或替换单个配方 |
| `DELETE` | `/api/v1/config/recipes/{name}` | 删除未被引用的命名配方 |

每一次配置变更（包括回滚和配方 `PUT`/`DELETE`）都要求在 `If-Match` 中提供精确的当前 `ETag`。Router 不接受无保护的写入。变更响应和 `GET /api/v1/config/hash` 使用相同的显式运行时身份字段：`source_config_hash`、`generated_runtime_hash`、`active_runtime_hash` 和 `activation_status`。配置变更会校验、创建备份并触发重载；生效的配置仍不能证明上游模型后端健康。变更后请检查 `/ready` 并发送代表性请求。

配置激活状态为 `active`、`pending`、`failed` 或 `unknown`。`activation` 报告最新候选的
文档哈希、尝试编号、阶段、时间和脱敏错误。激活失败保留旧 generation；配置写入在等待期间
观测到失败时返回 `503` 和 `status: activation_failed`。此时文件已经持久化，应读取最新
ETag 后修正或回滚，不要盲目重试原请求。通过 `/api/v1/config/hash` 轮询时，应在 `failed`
停止等待。该状态保存在当前进程，较新的候选会取代较旧的尝试。

## 管理知识库与已存数据 {#manage-knowledge-bases-and-stored-data}

知识库配置：

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET`、`POST` | `/api/v1/storage/knowledge-bases` | 列出或创建托管知识库 |
| `GET`、`PUT`、`DELETE` | `/api/v1/storage/knowledge-bases/{name}` | 读取、更新或删除单个知识库 |
| `GET` | `/api/v1/storage/knowledge-bases/{name}/map/metadata` | 读取生成的 map 元数据 |
| `GET` | `/api/v1/storage/knowledge-bases/{name}/map/data.ndjson` | 以 NDJSON 流式传输 map 数据 |

在 Router 进程中，创建、更新和删除会持久化候选并返回 `202`，同时带有 `activation_status: pending` 和 `generated_runtime_hash`，直到替换生成准备完成。轮询 `/api/v1/config/hash`，直到 `active_runtime_hash` 匹配该候选。在 pending 期间再次变更 KB 会返回 `409`，错误为 `CONFIG_ACTIVATION_PENDING`，且不会覆盖它。更新使用独立的资产修订路径；删除会移除候选配置条目，同时保留旧生成和回滚生成所需的文件。旧修订在其配置引用退役后需要离线清理。独立 API 服务器报告 `activation_status: unknown` 以及其正常成功状态，因为没有 Router 生成注册表。

Router 管理的存储和记忆：

| 资源 | 基础路径 | 操作 |
| --- | --- | --- |
| 长期记忆 | `/api/v1/storage/memories` | 按范围列出和删除；按 id 读取或删除 |
| 向量存储 | `/api/v1/storage/vector-stores` | 创建、列出、读取、更新、删除和搜索 |
| 向量存储文件 | `/api/v1/storage/vector-stores/{id}/files` | 附加、列出、检查和分离文件 |
| 文件 | `/api/v1/storage/files` | 上传、列出、检查、下载和删除 |

所需服务不可用时，这些路由返回 `503`。文件上传使用 multipart 表单数据，并接受文档（`.txt`、`.md`、`.json`、`.csv`、`.html`）供向量存储摄入；上传图像（`.png`、`.jpg`、`.jpeg`、`.gif`、`.webp`）并设置 `purpose=vision`，即可通过 `file_id` 从 Response API 的 `input_image` 部分引用。限制和字段见实时 schema。它们仅存在于管理监听器。`/v1/files` 和 `/v1/vector_stores` 不是推理监听器别名，也不会由 Router API 注册。

## 发现并检查插件 {#discover-and-inspect-plugins}

`GET /api/v1/plugins` 直接投影规范插件注册表，返回每种插件的 schema、绑定与操作链接。
`GET /api/v1/plugins/{type}` 返回单个描述；
`GET /api/v1/plugins/{type}/bindings?recipe=<name>&decision=<name>` 查询活动配置中的绑定、
启用、可达性与依赖可用性。依赖可用不等于已探测网络健康；`runtime_inspected: false` 表示只有配置信息；绑定为空表示该范围没有配置此插件。

| 插件 | 管理方式 |
| --- | --- |
| `system_prompt`、`request_params`、`header_mutation`、`fast_response` | `POST /api/v1/plugins/{type}/preview`，使用 typed `configuration` 或活动 `binding` |
| `hallucination`、`response_jailbreak` | 预览策略和输入条件；`mode: probe` 调用已配置检测器 |
| `rag` | 用 `supplied_context` 预览注入；`mode: probe` 执行检索并绕过 RAG 结果缓存 |
| `tools`、`tool_selection` | 预览静态工具策略；语义选择必须显式使用 `mode: probe` |
| `context_compression` | 保留已有压缩预览契约 |
| `response_cache`、`memory` | 操作 `/api/v1/storage` 下的资源 |
| `router_replay`、`shadow_dispatch` | 通过 `/api/v1/observability/replays` 查询记录 |

Guard、RAG 和工具预览要求显式 `binding: {"recipe":"<name>","decision":"<name>"}`。
默认 `mode` 为 `preview`，不会自动升级为 probe。Probe 可以调用已配置的本地或远程分类器、
嵌入服务及检索后端，但不会执行选中的工具或生成 provider 回答。RAG 需要 `data.read`，工具策略需要 `config.read`，
Guard 探测需要 `classify.invoke`。响应中的 `mode`、
`persisted`、`backend_calls` 描述实际影响。`persisted: false` 表示不写插件数据或会话状态；
probe 可以更新指标和嵌入记忆缓存，外部检索服务也可能产生其自身的副作用。
Header 预览中的值需要 `secret_view`。

## 操作响应缓存 {#operate-the-response-cache}

响应缓存端点独立于推理时的缓存查找。它们让运维检查后端、测试候选配置，并执行可审计的失效。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/v1/storage/response-cache/capabilities` | 后端能力 |
| `GET` | `/api/v1/storage/response-cache/health` | 后端健康 |
| `GET` | `/api/v1/storage/response-cache/stats` | 脱敏统计 |
| `POST` | `/api/v1/storage/response-cache/test` | 校验并探测候选配置 |
| `POST` | `/api/v1/storage/response-cache/invalidate` | 试运行或使范围内分区失效 |
| `POST` | `/api/v1/storage/response-cache/flush` | 推进范围内或全局缓存 epoch |

破坏性缓存变更前，优先使用范围失效和试运行。Bearer 角色区分读取、失效以及更广的缓存管理权限。

## 检查管理审计 {#inspect-management-audit}

`GET /api/v1/observability/audit` 需要 `audit.read`，覆盖配置、Recipe、数据、缓存和压缩的
受审计管理操作。支持精确 `action`、`limit`（1–1000，默认 100）和 `after_sequence`。
保持过滤条件，用 `next_sequence` 翻页；`has_more` 表示还有匹配条目，`truncated` 表示
游标之前的数据已被淘汰，`oldest_sequence` 给出保留边界。

哈希链环形日志保留当前进程最近 10,000 条，重启会清空日志并重置序列，不是持久审计归档。
条目只包含路由和授权元数据，不包含请求体或插件内容。

## 检查上下文压缩 {#inspect-context-compression}

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/v1/plugins/context_compression/capabilities` | 运行时能力 |
| `GET` | `/api/v1/plugins/context_compression/health` | 运行时健康 |
| `GET` | `/api/v1/observability/plugins/context_compression/stats` | 脱敏统计 |
| `POST` | `/api/v1/plugins/context_compression/preview` | 预览压缩而不持久化 |
| `POST` | `/api/v1/storage/context-recovery/invalidate` | 使受信任的恢复范围失效 |

在重要流量上启用压缩前，使用 `preview` 评估会保留什么。

## 检查回放并提交结果 {#inspect-replay-and-submit-outcomes}

路由回放仅用于管理。其查询端点和脱敏模型见 [Router API](./router#router-replay)。

路由学习可以摄入与其拥有的回放记录关联的结果：

```bash
curl -sS http://localhost:8080/api/v1/observability/outcomes \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}" \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: feedback-123' \
  -d '{
    "replay_id": "replay_...",
    "target": "model",
    "verdict": "good_fit",
    "score": 0.9
  }'
```

`replay_id`、`target` 和 `verdict` 为必填。来源由已认证主体决定，而不是可选的 `source` 请求体字段。客户端可能重试时，使用稳定的 `Idempotency-Key`。摄入还需要活动的路由学习运行时；启用 bearer 认证时还需要 `learning.ingest` 权限。

## API 边界 {#api-boundaries}

- 管理 API 是运维表面，不是公开推理网关。
- 端点可用性可能取决于编译特性和已启用的服务。
- OpenAPI 文档描述形状，而不是某个模型、存储或外部后端的行为。
- 不要把 bearer token 放进 URL 或日志。只给自动化所需的权限。

## 完整端点索引 {#complete-endpoint-index}

以下参考由 Router 已注册路由目录生成。用它扫描每个端点；用上面面向任务的章节获取指引，用运行中的 `/openapi.json` 获取精确 schema。

完整路由、权限、请求及响应契约见[生成的 OpenAPI 参考](./openapi)。英文参考中的端点表直接从路由目录生成，中文页面不维护另一份完整路由清单。

旧 `/api/v1/response-cache/*` 和 `/api/v1/context-compression/*` 已移除。请从 `/api/v1`
或 `/api/v1/plugins` 重新发现操作：缓存迁至 `/api/v1/storage/response-cache/*`，
压缩能力/健康/预览迁至 `/api/v1/plugins/context_compression/*`，统计、恢复失效和审计分别
使用上文的 observability/storage 路径。旧地址没有变更语义重定向或兼容 handler。
