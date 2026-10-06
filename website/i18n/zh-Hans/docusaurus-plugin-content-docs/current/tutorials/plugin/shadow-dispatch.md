---
translation:
  source_commit: "b2db276cf1b5057c31f2ab2bddbd181e5692dbb6"
  source_file: "docs/tutorials/plugin/shadow-dispatch.md"
  outdated: false
---

# 影子分发

## 概览

`shadow_dispatch` 是一个路由局部插件，将已批准请求的有界采样副本发送到次要模型，并记录结果，而不更改或延迟主响应。

## 主要优势

- 在候选模型服务任何用户之前，用真实流量观察它。
- 让线上响应独立于 shadow 延迟、失败或输出。
- 明确限制采样、并发、队列深度、超时、响应大小和重试。
- 在回放记录上记录主路径与 shadow 的身份、时序、结果和内容哈希，供审计和后续比较。

## 解决什么问题？

把新模型提升进路由配方需要来自生产形态请求的证据。离线评估会错过真实提示词分布，灰度发布又会把未验证模型暴露给用户。Shadow dispatch 填补中间步骤：主模型仍回答每个请求，同时同一份最终请求的副本在后台发送给候选模型。Shadow 结果作为不可变结果存储在该请求的回放记录上，因此运维人员稍后可以关联主路径与 shadow 观察，而无需暴露受保护内容。

Shadow 副本从主分发所基于的同一份已批准中性请求开始，此时所有请求插件都已运行。随后通过主分发使用的同一编码和提供商适配步骤为 shadow 模型渲染，因此推理控制遵循 shadow 模型家族，而不是主模型。寻址方式与主分发对该后端的寻址相同，包括提供商配置的 `chat_path` 和 Azure OpenAI 的 `api_version` 查询。它始终以非流式发送。只传播 W3C `traceparent` 和 `tracestate` 请求头；不传播客户端请求头，这包括 `baggage`，tracing propagator 会从客户端提取它，其中可能携带客户端选择附加的任何内容。决策请求头改写为主后端运行并留在那里，除非 `forward_headers` 点名它们，因此运维人员按请求头选择加入，路由器无法识别的自定义凭据默认永不复制。已知凭据载体，以及主或 shadow 提供商配置解析为其 `auth_header` 的请求头，即使被列出也会丢弃。只使用路由器为 shadow 模型后端准备的静态凭据，因此主路径凭据永不会进入 shadow 后端。Shadow 调用只发送到配置的后端地址：永不跟随重定向应答，因此提示词和 shadow 凭据都不会被转发到配置未点名的源。

## 何时使用

- 应在灰度发布前用线上流量评估候选模型
- 无论 shadow 成功、失败还是超时，主响应都必须保持相同
- 该观察的资源使用必须按路由明确且有界
- 回放或审计工具需要按请求身份关联主路径与 shadow 观察

不要用它影响线上请求、给输出质量打分，或构建训练集。该插件只记录结果。

## 配置

将该插件与 `router_replay` 一起添加到决策，以便结果有记录可附着：

```yaml
plugins:
  - type: router_replay
    configuration:
      enabled: true
  - type: shadow_dispatch
    configuration:
      enabled: true
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

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `enabled` | 必需 | 为该决策打开 shadow。 |
| `model` | 启用时必需 | 接收 shadow 副本的已配置逻辑模型。必须在 `providers.models` 中有后端。 |
| `sample_rate` | `1.0` | 要 shadow 的合格请求比例，范围为 `[0, 1]`。`0` 保持插件已声明但永不分发。 |
| `max_concurrency` | `2` | 该决策的进行中 shadow 调用数。 |
| `max_queue_depth` | `8` | 等待槽位的调用。超出部分会以原因 `queue_full` 丢弃。 |
| `timeout_seconds` | `30` | 队列等待加执行的截止时间，所有重试共享。 |
| `max_response_bytes` | `1048576` | 读取的最大 shadow 响应正文。更大的正文会以 `response_too_large` 失败。 |
| `max_retries` | `0` | 传输错误或可重试状态上的额外尝试。上限为 `3`。 |
| `capture_response_body` | `false` | 在结果中存储 shadow 文本的有界摘录。默认关闭；只保留大小、token 和 SHA-256。 |
| `max_capture_bytes` | `4096` | 开启采集时的摘录上限。 |
| `tls_skip_verify` | `false` | 跳过由内部 CA 签名的 https shadow 后端的证书校验。主路径通过 Envoy 到达后端，Envoy 不校验上游证书。 |
| `forward_headers` | `[]` | Shadow 副本可以携带的决策 `header_mutation` 名称，按不区分大小写匹配。决策为主后端设置的其他内容都不会转发，因此像 `X-Internal-Token` 这样的自定义凭据留在主路径。已知凭据载体（`Authorization`、`Proxy-Authorization`、`Cookie`、`x-api-key`、`api-key`、`x-goog-api-key`、`x-user-*-key` 请求头）即使被列出，也会在配置加载时拒绝并在运行时丢弃。 |

当请求被采样排除，或主分发已经选择了 shadow 模型时，会跳过 shadow，只产生指标而不产生结果。通过 looper 执行的决策（ratings、confidence、fusion、ReMoM、workflows）会在配置加载时拒绝该插件，因为 shadow hook 只在单模型提供商分发上运行。

### 失败开放行为 {#fail-open-behavior}

请求路径做一次非阻塞槽位检查后返回。其余工作在主分发响应构建完成后，于有界 worker 中运行。缓慢、不可用、格式错误或过载的 shadow 端点不能更改主响应或其延迟。每次 shadow 都以恰好一种结果结束：

| 结果 | 原因 |
| --- | --- |
| `completed` | `completed` |
| `failed` | `backend_unresolved`、`credential_unresolved`、`encode_failed`、`timeout`、`transport_error`、`upstream_status`、`redirect_rejected`、`response_too_large`、`malformed_response` |
| `dropped` | `queue_full`、`queue_timeout`、`router_closing`、`same_as_primary`、`internal_request`、`request_unavailable` |
| `sampled_out` | `sampled_out` |

结果和原因导出为 `sr_shadow_dispatch_total{decision,result,reason}`，以及 `sr_shadow_dispatch_latency_seconds`、`sr_shadow_dispatch_inflight` 和 `sr_shadow_dispatch_queued`。由资源边界导致的丢弃通过指标和结构化 `shadow_dispatch_dropped` 事件报告，而不是写入回放，因此过载的 shadow 通道不会放大回放存储负载。

### 解读失败 {#interpreting-failures}

`failed` 结果并不总是说明候选模型本身。请将原因与结果元数据中的 `status_code`、`attempts` 和截断的 `error` 一起阅读：

| 含义 | 原因 | 如何解读 |
| --- | --- | --- |
| 候选拒绝了请求 | 带 4xx `status_code` 的 `upstream_status` | Shadow 模型无法接受已批准请求，例如不支持的参数或上下文窗口太小。计入候选问题。 |
| 候选健康或容量 | 带 5xx `status_code` 的 `upstream_status`、`timeout`、`transport_error` | 后端不可达、过载，或在 `max_retries` 之后仍无法在 `timeout_seconds` 内完成。这衡量部署，而不是回答质量。 |
| 候选尝试重定向 | 带 3xx `status_code` 的 `redirect_rejected` | 后端以重定向应答。路由器永不跟随，因此提示词和 shadow 凭据只到达配置的后端。请将 shadow 模型指向后端的最终地址。 |
| 候选输出问题 | `malformed_response`、`response_too_large` | 后端已应答，但正文对其线格式无效，或超过 `max_response_bytes`。 |
| 路由器侧，与候选无关 | `backend_unresolved`、`credential_unresolved`、`encode_failed` | 路由器无法构建或寻址 shadow 调用。修复配置，并从任何候选比较中排除这些项。 |

`dropped` 和 `sampled_out` 永不进入回放记录。它们表示路由器选择不发送 shadow，因此不携带关于候选模型的信号，只在指标和日志中可见。

### 回放与审计采集 {#replay-and-audit-capture}

已完成和失败的 shadow 会向主请求的回放记录追加一条结果，带有 `source: shadow_dispatch`、`target: model`、`target_ref: <shadow model>`、判定和原因。结果元数据携带主路径与 shadow 的请求身份、主路径与 shadow 的模型和后端、决策和配方、采样率、入队、开始和完成时间戳、队列等待和延迟、尝试次数、状态码、响应大小、停止原因、token 计数，以及 shadow 文本的 SHA-256。结果只追加不改写。

回放脱敏对 shadow 结果的应用方式与记录其余部分相同：没有内容权限的查看者可以看到路由和时序字段，但看不到 `target_ref`、`reason` 或 `metadata`。除非回放存储及其读取者已获准处理提示词级内容，否则请关闭 `capture_response_body`。片段见：
[`config/fragments/plugin/shadow-dispatch/sampled.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/shadow-dispatch/sampled.yaml)。

### 导出对比数据集 {#export-a-comparison-dataset}

采集到的结果可通过 `GET /api/v1/observability/replays/dataset` 导出为对比数据集。该接口读取常规回放过滤条件选中的记录，返回一份带版本的清单，其中包含每条输入的主模型与影子模型分支。

清单以自身摘要作为标识，因此导出请求需要携带拆分方案：`seed` 固定拆分分配，每个可重复的 `split` 写作 `name:weight`，例如 `?seed=2026-q3&split=train:8&split=eval:2`。相同记录在相同 seed 与拆分下会重建出相同的清单，包括每条样本所属的拆分，因此后续新增的观测不会移动已经放置好的样本。

```bash
curl -H "Authorization: Bearer $ROUTER_MANAGEMENT_TOKEN" \
  "$ROUTER_MANAGEMENT_URL/api/v1/observability/replays/dataset?recipe=vault&seed=2026-q3&split=train:8&split=eval:2"
```

清单只携带标识、输出摘要与来源信息，不含提示词或响应文本，因此可以与它支撑的数据一同发布。一条观测要么整条进入，要么完全不进入：失败的请求、未结束的请求、输入被截断的请求，以及从未记录摘要的请求都会被排除并按原因计数，`counts` 会报告保留了什么、丢弃了什么。由于清单描述的是构建它的整个选择集，超过 5000 条记录的选择会被拒绝，而不是按页导出。请缩小过滤条件后重新导出。

某一个 recipe 或某一个 decision 通常会主导线上流量，基于它构建的数据集读起来像是关于整个路由器的结论，实际上只是关于那个 decision 的结论。`balance_by` 与 `balance_max` 限制单个分组最多能贡献多少条，分组方式为 `recipe`、`decision` 或 `primary_model`。上限保留哪些行由 seed 决定而非时间先后，因此均衡后的数据集是对流量的采样，而不是对到达时间的采样；因均衡而丢弃的行与其他排除一样计入 `balance_cap`。两个参数必须同时给出，只给其一会被拒绝。

导出需要 `replay.read` 权限，读取的记录与列表 API 相同。被比较的决策必须开启正文采集，否则未采集到请求的观测会以 `request_body_missing` 被排除。
