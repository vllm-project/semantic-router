---
is_mtpe: false
title: 配置模型
description: 选择目录支持或自定义模型，将其绑定到 provider，并从路由决策中使用它。
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/installation/model-configuration.md"
  outdated: false
---

# 配置模型

已配置的 Model 将稳定的 Router 名称连接到一个模型身份和一个或多个物理后端。在 `providers.models` 下配置 Model，然后从 `providers.defaults.model` 和 `recipes[].decisions[].modelRefs[].model` 引用其 `name`。

## 理解模型标识符

这些字段回答不同的问题，不可互换：

| 字段 | 含义 |
| --- | --- |
| `name` | 默认值、决策和直接模型调用使用的 Router 本地别名。 |
| `catalog` | 来自内置目录的可选 canonical Model Card ID。私有或新发布的模型请省略。 |
| `provider_model_id` | 发送给上游 provider 的可选模型或部署名称。 |
| `backend_refs[].provider` | 稳定的 Provider 契约，提供端点、身份验证、路径、协议和模型映射默认值。 |
| `api_format` | 可选的线协议覆盖：`openai`、`responses` 或 `anthropic`。它从不选择 Provider。 |

例如，`production-reasoner` 可以是你的 Router 别名，`openai/gpt-5.6-sol` 是其目录身份，而 provider 专用的部署名称是其 `provider_model_id`。

## 选择配置路径

| 起点 | 配置 | 推理行为 |
| --- | --- | --- |
| 模型存在于目录中 | 在每个后端上设置 `catalog` 和 Provider。 | 从目录和所选 Provider 映射继承。 |
| 具有已知内置家族的私有或未列出模型 | 省略 `catalog`；设置 `reasoning.family`。 | 复用内置家族契约。 |
| 具有独特控制的私有或未列出模型 | 省略 `catalog`；内联定义 `reasoning`。 | 使用你定义的模型本地契约。 |
| 自定义透传模型 | 同时省略 `catalog` 和 `reasoning`。 | Router 不会合成推理控制。 |

各路径见[目录支持的模型](catalog-backed-models)、[自定义模型](custom-models)和[推理配置](model-reasoning)。[模型配置模式](model-configuration-patterns)展示受支持的组合和无效混合。

## 从一个自定义模型开始

```yaml
version: v0.5
providers:
  defaults:
    model: local-chat
  models:
    - name: local-chat
      provider_model_id: served-model
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: host.docker.internal:8000
          protocol: http
recipes:
  - name: default
```

Router 会为 `local-chat` 创建稀疏的本地 Model Card。仅当路由需要上下文窗口、能力、标签或 LoRA 等元数据时，才添加匹配的 `providers.models` 条目。在顶层 `evaluation.records[]` 下独立添加基准测量。

## 在分发前检查任务能力

所选模型必须同时满足 provider 协议的能力和其声明的任务能力。当 Router 从匹配决策中选择回退时，同样的检查也适用。回退不能重新引入已被请求的上下文窗口检查排除的模型。

Model Card 能力元数据接受 `image_input` 和 `image_generation` 等协议名称，以及以下目录别名：

| 别名 | 含义 |
| --- | --- |
| `vision`、`audio`、`video` | 图像、音频和视频**输入**；它们不授予媒体生成支持 |
| `structured_output` | `structured_json` |
| `tool_use` | `tools` |

`long_context` 或 `coding` 等描述性标签不会抹去与它们并列的已识别能力声明。

没有任何已识别能力声明的模型仅保留协议兼容性。对于已标注的模型，能够编码请求的协议不会覆盖缺失的任务能力。当没有符合条件的决策模型能够服务所请求的任务时，Router 返回 `unsupported_capability`。

## 在控制面板中配置模型

打开 **Build → Routing → Models → Add Model**。然后你可以：

1. 选择 Provider，连接它，并选择已发现或内置的模型 ID。
2. 当 Provider 无法列出模型时，输入模型 ID。
3. 使用 **Advanced settings** 设置名称前缀、内置推理家族、路由元数据、定价和交付设置。
4. 从 Provider 选择器使用 **Manual setup**，以完全控制 `catalog`、内联推理字段、`provider_model_id`、`api_format` 和后端引用。

控制面板以 YAML 形式编辑同一份配置文档。为部署使用一个编写所有者，并在 serve 之前复核导出的 YAML。参见[配置工作流](configuration-workflows)。

## 路由能力 {#routing-capabilities}

作者声明的模型能力在模型选择之前充当资格过滤器：声明的能力无法表达请求的候选会先被过滤掉，然后配置的选择算法才为剩下的候选打分，因此得分更高的兼容模型可以胜过只是排在前面的模型。最终分发只校验选中的模型：迟到的能力不匹配会返回错误，而不会重新选择或替换选中的候选。在 Model Card 上声明能力：

```yaml
recipes:
  - name: default
providers:
  models:
    - name: image-backend
      api_format: images
      capabilities: [image_generation]
      modalities: {input: [text], output: [image]}
```

模型的 `modalities` 指明它读取和写出的媒体；没有声明的模型读写文本。决策的[输出约定](../tutorials/decision/output)会在加载配置时检查它们以及能力。

协议能力词汇由 `llmprotocol` 包定义，一项声明以下列状态之一进入感知能力的分发：

1. **没有已识别的声明**：模型未标注，仅凭线格式的可表达性保持资格。
2. **任务或模态声明**：`image_input`、`image_output`、`image_generation`、`audio_input`、`audio_output`、`video_input`、`video_output`、`file_input` 和 `file_output` 等名称驱动声明任务过滤器：已标注的模型必须声明请求所需的每个任务位，否则分发会以 `unsupported_capability` 被拒绝。
3. **传输或计量声明**：`tools`、`reasoning`、`streaming` 和 `structured_json` 等名称可被识别，但不带任务位。这样的模型属于已标注，因此不符合它从未声明的媒体任务；文本请求不需要任务位，不受影响。
4. **目录别名**：`vision`、`audio` 和 `video` 投影为 `image_input`、`audio_input` 和 `video_input`；`structured_output` 映射到 `structured_json`，`tool_use` 映射到 `tools`。投影在过滤之前进行，因此声明 `[image_input, vision]` 的卡片只按 `image_input` 过滤：它对图像请求保持资格，对它从未声明的音频则被拒绝。
5. **描述性标签**：协议词汇及其别名之外的名称（例如 `long_context`、`coding`）是元数据：它们不提供任务位，也不会使同一声明中的已识别名称失效，因此只含描述性标签的声明按未标注处理。

打分前的过滤和最终校验使用相同的资格判定，因此校验不会接纳过滤会拒绝的模型。

## 调整超时、重试和端点健康 {#tune-timeouts-retries-and-endpoint-health}

提供方模型的 `reliability` 块控制 Router 如何与该模型的后端通信：负载均衡、超时、重试、熔断、异常剔除和主动健康检查。每个字段都是可选的，省略的字段保持下表所示的默认值。

```yaml
providers:
  models:
    - name: production
      reliability:
        lb_policy: least_request
        connect_timeout: 3s
        total_timeout: 120s
        per_try_timeout: 30s
        retry_count: 2
        retry_on: 5xx,retriable-status-codes
        retriable_status_codes: [429]
        retry_after_max: 10s
        consecutive_5xx: 5
        health_check_path: /health
      backend_refs:
        - name: replica-a
          provider: vllm
          endpoint: 10.0.0.1:8000
        - name: replica-b
          provider: vllm
          endpoint: 10.0.0.2:8000
```

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `lb_policy` | `round_robin` | `round_robin` 遵循后端权重；`least_request` 优先选择进行中请求最少的副本。 |
| `connect_timeout` | `10s` | 建立连接的时间，包括 TLS 握手。 |
| `total_timeout` | listener 的 `timeout` | 整个调用的时间：所有尝试以及流式响应。`0s` 关闭它。 |
| `idle_timeout` | listener 的 `timeout` | 等待流式响应后续内容的最长时间。`0s` 关闭它。 |
| `per_try_timeout` | 无 | 每次尝试直到响应开始的时间。 |
| `first_byte_timeout` | 无 | 等待第一个响应字节的时间，这样在开始前就卡住的流仍可重试。仅 standalone 模式。 |
| `retry_count` | `0` | 第一次尝试之后的重试次数，最多 5 次。 |
| `retry_on` | `connect-failure,refused-stream` | 何时重试，使用 Envoy 的条件：`5xx`、`gateway-error`、`reset`、`connect-failure`、`refused-stream`、`retriable-status-codes`、`retriable-4xx`、`reset-before-request`。 |
| `retriable_status_codes` | 无 | 在 `retriable-status-codes` 下重试的状态码，例如 `429`。 |
| `retry_back_off_base`、`retry_back_off_max` | `25ms`，基数的十倍 | 重试之间随机指数等待的上下界。 |
| `retry_after_max` | 无 | 按响应要求的 `Retry-After` 秒数等待，但不超过这个上限。 |
| `retry_budget_percent`、`retry_budget_min_concurrency` | 同时最多三次重试 | 改为把并发重试限制在进行中请求的一定比例（设置任一项后为 20% 和 3）。 |
| `consecutive_5xx` | 无 | 让副本退出轮转的连续失败次数。需要两个或更多副本。 |
| `base_ejection_time`、`max_ejection_percent` | `30s`、`50` | 被剔除的副本退出多久（连续剔除时每次更长），以及同时可退出的比例。 |
| `health_check_path`、`health_check_interval`、`health_check_timeout` | 无、`10s`、`2s` | 主动 `GET` 检查；只有 `200` 算通过。 |

无论哪个网关承载流量，重试和失败的行为都相同：

- 连接失败、重置或单次尝试超时都算完全没有响应，因此 `5xx`、`gateway-error` 和 `reset` 都会重试它。
- 重试优先选择该请求尚未尝试过的副本。
- 响应的任何部分一旦到达客户端，就不再重试。
- 健康检查失败或被剔除的副本会退出轮转。如果健康副本少于一半，Router 会把请求分散到所有副本，而不是压垮剩下的副本。
- 没有任何尝试得到响应时，客户端收到 `503`（超时则为 `504`），即 Envoy 的纯文本本地回复，例如 `upstream connect error or disconnect/reset before headers. reset reason: remote connection failure`。响应阶段不会在它上面运行，因此它不是客户端 API 的错误格式。

Envoy 数据面没有首字节超时，因此设置了 `first_byte_timeout` 的配置会在 Router 渲染其 Envoy 配置时失败。

主动健康检查在承载客户端流量的网关中运行。在 standalone 模式下，由 Router 探测每个副本。在 Envoy 之后，由 Envoy 探测它们，而 Router 自己发起的模型调用（例如执行器的调用）则依靠异常剔除（`consecutive_5xx`）退出轮转。

### 为单个决策覆盖超时和重试 {#override-timeouts-and-retries-for-one-decision}

决策可以为它路由的请求携带自己的 `reliability` 块，例如一个比其提供方模型默认值需要更多时间的长上下文路由：

```yaml
recipes:
  - name: default
    decisions:
      - name: long_context_route
        priority: 100
        rules:
          operator: AND
          conditions:
            - type: context
              name: long_context
        modelRefs:
          - model: production
        reliability:
          total_timeout: 600s
          per_try_timeout: 300s
          retry_count: 1
          retry_on: reset
```

该块接受提供方块中的超时和重试字段，并在所有网关中以相同方式与提供方模型的设置合并：

- `total_timeout`、`per_try_timeout`、`idle_timeout` 和 `first_byte_timeout` 替换提供方模型的值；`0s` 关闭对应项。
- `retry_count` 替换提供方模型的值。`retry_on` 和 `retriable_status_codes` 追加到提供方模型的条件中。提供方模型没有重试条件时，除非决策写出自己的条件，重试使用 `connect-failure,refused-stream`。
- `retry_back_off_base`、`retry_back_off_max` 和 `retry_after_max` 替换提供方模型的值。
- 连接超时、负载均衡、重试预算、剔除和健康检查留在提供方模型上，因为它们属于其后端。

在 Envoy 之后，Router 把覆盖值作为每请求的请求头传给 Envoy 的 router：`x-envoy-upstream-rq-timeout-ms`、`x-envoy-upstream-rq-per-try-timeout-ms`、`x-envoy-max-retries`、`x-envoy-retry-on` 和 `x-envoy-retriable-status-codes`。项目附带的每个 Envoy 配置都允许 Router 设置且只设置这些请求头。如果你在自己的 Envoy 之后运行 Router，请在 ext_proc 过滤器的 `mutation_rules` 中允许它们：

```yaml
mutation_rules:
  allow_expression:
    regex: "^x-envoy-(upstream-rq-timeout-ms|upstream-rq-per-try-timeout-ms|max-retries|retry-on|retriable-status-codes)$"
```

reliability 块是设置这些请求头的唯一方式：Router 在加载配置时会从决策的 `header_mutation` 插件中丢弃它们，并记录一条警告。

Envoy 无法对单个请求应用空闲超时、首字节超时、退避或 `Retry-After` 上限。设置了 `idle_timeout`、`first_byte_timeout`、`retry_back_off_base`、`retry_back_off_max` 或 `retry_after_max` 的决策会校验失败，除非 Router 以 `--gateway standalone` 提供服务。

## 回退到另一个模型 {#fall-back-to-another-model}

模型失败时，Router 可以把请求改发给决策中排名下一位的候选。这种跨模型回退默认关闭。可以全局开启，也可以为某个配方或某个决策开启：

```yaml
global:
  router:
    fallback:
      enabled: true
      max_attempts: 3
      total_timeout: 30s
      per_attempt_timeout: 10s
      retryable_status_codes: [502, 503, 504]
      circuit_breaker:
        consecutive_failures: 3
        cooldown_period: 30s
        half_open_probes: 1
recipes:
  - name: default
    decisions:
      - name: escalate
        priority: 100
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: small
          - model: large
        fallback:
          max_attempts: 2
          retryable_status_codes: [429, 503]
```

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `enabled` | `false` | 失败的尝试是否转到下一个候选。 |
| `max_attempts` | `3` | 跨候选的尝试次数，包括第一次。 |
| `total_timeout` | `30s` | 从请求开始算起、仍会尝试其他候选的时间。每个候选最多得到剩余的时间，用完后不再开始新的候选。 |
| `per_attempt_timeout` | `10s` | 第一次之后每个候选的时间。不能超过 `total_timeout`。 |
| `retryable_status_codes` | `[502, 503, 504]` | 会转到下一个候选的状态码。 |
| `circuit_breaker` | 3 次失败、`30s`、1 次探测 | 在冷却期内跳过持续失败的后端。仅用于全局和配方策略。 |

请求所用的策略逐字段合并，最具体的一层优先：

1. 请求图步骤的 `fallback`，用于请求图发起的模型调用；
2. 决策的 `fallback`；
3. 配方的 `recipes[].fallback`；
4. `global.router.fallback`。

- **各层。**某一层省略的字段保持下一层的值，`enabled` 只在你写出它的地方设置。
- **决策。**决策的块接受除 `circuit_breaker` 以外的所有字段，`circuit_breaker` 属于后端。
- **候选**始终是决策中对该请求仍有资格的排名模型。没有单独的回退列表，提供方模型也没有自己的回退：同一模型的其他副本改由其重试、剔除和健康检查来尝试。

两种网关模式运行相同的策略，只有一处有意的差异：standalone 模式在完全没有后端应答时（例如连接被拒绝或超时）也会回退，而在 Envoy 之后，客户端会收到 Envoy 的回复。这样的失败计为 `503` 或 `504`，因此要在这类失败上回退，请在 `retryable_status_codes` 中保留这两个状态码。响应的任何部分一旦到达客户端，就不再回退。

两个回退超时都不会缩短第一次尝试，它沿用提供方模型和决策的 `reliability` 超时。要在某个模型始终不应答时回退，请在 standalone 模式下为该提供方模型设置 `reliability.per_try_timeout`：

- 超时计为 `504`，随后尝试下一个候选。
- 该候选最多得到 `per_attempt_timeout` 和 `total_timeout` 剩余的时间，因此 `total_timeout` 要长于该 `per_try_timeout`。
- 决策自己的 `per_try_timeout` 会限制它尝试的每个候选，因此较慢的候选也会以同样方式失败。

在 Kubernetes 上，决策使用同样的 `reliability` 和 `fallback` 块，字段名也相同：位于 `IntelligentRoute` 的 `spec.decisions[]` 和 `SemanticRouter` 的 `spec.config.decisions[]` 中。应用资源时，CRD 会检查时长、重试次数和状态码。

## 校验结果

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

校验会在启动前解析目录条目、Provider 映射、推理控制、Model Card 身份和后端池兼容性。

## 配置 Router 任务使用的模型

对于 Router 自己使用的决策模型和嵌入模型，从[模型运行时](model-runtime/get-started/overview.md)开始。它介绍本地和远程模型、它们的配置以及运维。
