---
translation:
  source_commit: "bce357c513f391824e8320267d03977794c20f76"
  source_file: "docs/api/apiserver.md"
  outdated: false
---

# 路由器管理接口 {#router-management-api}

Router 管理 API 提供健康、分类、配置、存储、缓存、压缩和回放操作。默认监听端口 `8080`，本地栈将其绑定到 `127.0.0.1`。

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

## 操作响应缓存 {#operate-the-response-cache}

响应缓存端点独立于推理时的缓存查找。它们让运维检查后端、测试候选配置，并执行可审计的失效。

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/v1/response-cache/capabilities` | 后端能力 |
| `GET` | `/api/v1/response-cache/health` | 后端健康 |
| `GET` | `/api/v1/response-cache/stats` | 脱敏统计 |
| `GET` | `/api/v1/response-cache/audit` | 脱敏变更审计条目 |
| `POST` | `/api/v1/response-cache/test` | 校验并探测候选配置 |
| `POST` | `/api/v1/response-cache/invalidate` | 试运行或使范围内分区失效 |
| `POST` | `/api/v1/response-cache/flush` | 推进范围内或全局缓存 epoch |

破坏性缓存变更前，优先使用范围失效和试运行。Bearer 角色区分读取、失效以及更广的缓存管理权限。

## 检查上下文压缩 {#inspect-context-compression}

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| `GET` | `/api/v1/context-compression/capabilities` | 运行时能力 |
| `GET` | `/api/v1/context-compression/health` | 运行时健康 |
| `GET` | `/api/v1/context-compression/stats` | 脱敏统计 |
| `POST` | `/api/v1/context-compression/preview` | 预览压缩而不持久化 |
| `POST` | `/api/v1/context-compression/recovery/invalidate` | 使受信任的恢复范围失效 |

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

<!-- BEGIN-GENERATED-ENDPOINT-INDEX -->
### system {#system}

健康、就绪和 API 契约发现。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/health` | 健康检查端点 |
| `GET` | `/ready` | 就绪端点，仅在启动完成后变绿 |
| `GET` | `/startup-status` | 详细的 Router 启动和模型下载状态 |
| `GET` | `/api/v1` | 渐进式 API 能力发现 |
| `GET` | `/openapi.json` | OpenAPI 3.0 规范；可收窄到单个路径或操作 |
| `GET` | `/docs` | 交互式 Swagger UI 文档 |

### config {#config}

校验、检查、应用、版本化和回滚 Router 配置与配方。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/config/recipes` | 列出默认和命名路由配方及其入口 |
| `POST` | `/api/v1/config/recipes/validate` | 校验配方变更，不写入或重载配置 |
| `GET` | `/api/v1/config/recipes/{name}` | 读取单个路由配方及其入口 |
| `PUT` | `/api/v1/config/recipes/{name}` | 原子创建或替换单个路由配方；需要 If-Match |
| `DELETE` | `/api/v1/config/recipes/{name}` | 删除未被引用的命名路由配方；需要 If-Match |
| `GET` | `/api/v1/config/schema` | 渐进发现规范 Router 配置契约，或返回完整 JSON Schema |
| `GET` | `/api/v1/config` | 以 JSON 获取当前 Router 配置（无 secret_view 时密钥脱敏） |
| `POST` | `/api/v1/config/validate` | 校验并规范化 Router 配置，不写入 |
| `POST` | `/api/v1/config/plan` | 规划精确的合并或替换变更（含热重载兼容性），不写入 |
| `PATCH` | `/api/v1/config` | 比较并交换合并 Router 配置更新（校验、备份、写入、触发热重载） |
| `PUT` | `/api/v1/config` | 比较并交换替换 Router 配置（校验、备份、写入、触发热重载） |
| `POST` | `/api/v1/config/rollback` | 比较并交换回滚到先前的 Router 配置版本 |
| `GET` | `/api/v1/config/versions` | 列出可用的 Router 配置备份版本 |
| `GET` | `/api/v1/config/hash` | 比较已持久化源、已生成运行时和当前生效的 Router 配置哈希 |

### routing {#routing}

预览路由行为，不调用生成后端。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/routing/preview` | 预览所有已配置信号及结果路由，不调用生成后端 |

### inventory {#inventory}

检查已配置和已加载的模型与分类器资源。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/inventory/models` | 获取已加载模型的信息 |
| `GET` | `/api/v1/inventory/classifier` | 获取分类器信息和状态（无 secret_view 时密钥脱敏） |
| `GET` | `/api/v1/inventory/embedding-models` | 获取已加载嵌入模型的信息 |
| `GET` | `/v1/models` | OpenAI 兼容的公开模型和入口列表 |

### observability {#observability}

检查路由回放和指标，并提交结果证据。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/observability/classification-metrics` | 获取分类指标和统计 |
| `POST` | `/api/v1/observability/outcomes` | 提交与回放记录关联的路由学习结果反馈 |
| `GET` | `/api/v1/observability/replays` | 列出路由回放记录 |
| `GET` | `/api/v1/observability/replays/aggregate` | 聚合路由回放路由和成本元数据 |
| `GET` | `/api/v1/observability/replays/trajectory` | 构建路由回放会话轨迹 |
| `GET` | `/api/v1/observability/replays/{id}` | 读取单条路由回放记录 |

### storage {#storage}

管理 Router 拥有的知识库、记忆、文件和向量存储。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/storage/knowledge-bases` | 列出已配置的知识库 |
| `POST` | `/api/v1/storage/knowledge-bases` | 创建托管知识库 |
| `GET` | `/api/v1/storage/knowledge-bases/{name}` | 读取知识库 |
| `GET` | `/api/v1/storage/knowledge-bases/{name}/map/metadata` | 读取生成的知识库 map 元数据 |
| `GET` | `/api/v1/storage/knowledge-bases/{name}/map/data.ndjson` | 以 NDJSON 流式传输生成的知识库 map 数据 |
| `PUT` | `/api/v1/storage/knowledge-bases/{name}` | 更新托管知识库 |
| `DELETE` | `/api/v1/storage/knowledge-bases/{name}` | 删除托管知识库 |
| `GET` | `/api/v1/storage/memories` | 列出长期记忆 |
| `DELETE` | `/api/v1/storage/memories` | 按范围删除记忆 |
| `GET` | `/api/v1/storage/memories/{id}` | 读取单条长期记忆 |
| `DELETE` | `/api/v1/storage/memories/{id}` | 删除单条长期记忆 |
| `POST` | `/api/v1/storage/vector-stores` | 创建向量存储 |
| `GET` | `/api/v1/storage/vector-stores` | 列出向量存储 |
| `GET` | `/api/v1/storage/vector-stores/{id}` | 读取向量存储 |
| `POST` | `/api/v1/storage/vector-stores/{id}` | 更新向量存储 |
| `DELETE` | `/api/v1/storage/vector-stores/{id}` | 删除向量存储 |
| `POST` | `/api/v1/storage/vector-stores/{id}/search` | 搜索向量存储 |
| `POST` | `/api/v1/storage/vector-stores/{id}/files` | 将文件附加到向量存储 |
| `GET` | `/api/v1/storage/vector-stores/{id}/files` | 列出附加到向量存储的文件 |
| `DELETE` | `/api/v1/storage/vector-stores/{id}/files/{file_id}` | 从向量存储分离文件 |
| `POST` | `/api/v1/storage/files` | 上传文件 |
| `GET` | `/api/v1/storage/files` | 列出已上传文件 |
| `GET` | `/api/v1/storage/files/{id}` | 读取已上传文件的元数据 |
| `DELETE` | `/api/v1/storage/files/{id}` | 删除已上传文件 |
| `GET` | `/api/v1/storage/files/{id}/content` | 下载已上传文件内容 |

### response-cache {#response-cache}

检查并管理响应缓存服务。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/response-cache/capabilities` | 获取响应缓存后端能力 |
| `GET` | `/api/v1/response-cache/health` | 检查响应缓存后端健康 |
| `GET` | `/api/v1/response-cache/stats` | 获取脱敏的响应缓存统计 |
| `GET` | `/api/v1/response-cache/audit` | 获取脱敏的响应缓存变更审计条目 |
| `POST` | `/api/v1/response-cache/test` | 校验并探测响应缓存候选配置 |
| `POST` | `/api/v1/response-cache/invalidate` | 试运行或使范围内响应缓存分区失效 |
| `POST` | `/api/v1/response-cache/flush` | 推进范围内或全局响应缓存 epoch |

### context-compression {#context-compression}

检查、预览和管理上下文压缩。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/context-compression/capabilities` | 获取上下文压缩能力 |
| `GET` | `/api/v1/context-compression/health` | 检查上下文压缩运行时健康 |
| `GET` | `/api/v1/context-compression/stats` | 获取脱敏的上下文压缩统计 |
| `POST` | `/api/v1/context-compression/preview` | 预览上下文压缩而不持久化 |
| `POST` | `/api/v1/context-compression/recovery/invalidate` | 使受信任的上下文恢复请求范围失效 |

### diagnostics {#diagnostics}

调用底层分类器、嵌入、NLI 和相似度诊断。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/diagnostics/classify/intent` | 将用户查询分类到路由类别 |
| `POST` | `/api/v1/diagnostics/classify/pii` | 检测文本中的个人身份信息 |
| `POST` | `/api/v1/diagnostics/classify/security` | 检测越狱尝试和安全威胁 |
| `POST` | `/api/v1/diagnostics/classify/fact-check` | 判断文本是否需要事实核查 |
| `POST` | `/api/v1/diagnostics/classify/user-feedback` | 分类用户反馈类型（satisfied、need_clarification、wrong_answer、want_different） |
| `POST` | `/api/v1/diagnostics/classify/combined` | 执行组合分类（意图、PII 和安全） |
| `POST` | `/api/v1/diagnostics/classify/batch` | 带可配置 task_type 参数的批量分类 |
| `POST` | `/api/v1/diagnostics/nli` | 对前提和假设对做自然语言推理分类 |
| `POST` | `/api/v1/diagnostics/embeddings` | 生成文本和图像嵌入 |
| `POST` | `/api/v1/diagnostics/similarity` | 计算成对文本相似度 |
| `POST` | `/api/v1/diagnostics/similarity/batch` | 计算批量文本相似度匹配 |
<!-- END-GENERATED-ENDPOINT-INDEX -->
