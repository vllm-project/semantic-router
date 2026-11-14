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
| `retry_on` | `connect-failure,refused-stream` | 何时重试，使用 Envoy 的条件名称。两种网关都会处理 `5xx`、`gateway-error`、`reset`、`connect-failure`、`refused-stream`、`retriable-status-codes`、`retriable-4xx`、`reset-before-request` 和 `envoy-ratelimited`。Envoy 的其他名称（`retriable-headers`、`http3-post-connect-failure`，以及 gRPC 的 `cancelled`、`deadline-exceeded`、`internal`、`resource-exhausted`、`unavailable`）可以加载，但 standalone 网关会忽略它们。Envoy 不认识的名称会导致配置加载失败。 |
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

## 对照上下文窗口检查提示 {#check-prompts-against-the-context-window}

Router 会两次对照模型的 `context_window_size` 检查请求：从请求到达起，对照请求能到达的每个模型；在决策之后，对照该决策的候选。它在信号运行的同时，用每个模型自己的分词器为提示计数，不请任何后端计数：

```yaml
providers:
  models:
    - name: deepseek-v4.1-flash
      context_window_size: 1048576
      tokenizer: deepseek-ai/DeepSeek-V4.1-Flash   # 可选 @revision
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: deepseek.example:8000
          protocol: http
global:
  router:
    context_window:
      margin: 0.03
      proxy_margin: 0.30
      proxy_tokenizer: o200k_base
```

| 字段 | 默认值 | 作用 |
| --- | --- | --- |
| `providers.models[].tokenizer` | 模型的 Hugging Face 仓库；没有时由代理分词器计数 | 模型自己的分词器：Hugging Face 仓库 id，即 `org/name` 或 `org/name@revision`；内置编码（`o200k_base`、`cl100k_base`）；或本地的 `tokenizer.json` 或包含它的目录（绝对路径，或以 `./`、`../` 开头的路径）。 |
| `global.router.context_window.margin` | `0.03` | 用模型自己的分词器计数时，窗口周围的区间，以窗口的比例表示。 |
| `global.router.context_window.proxy_margin` | `0.30` | 用代理分词器计数时的区间。 |
| `global.router.context_window.proxy_tokenizer` | `o200k_base` | 为没有分词器、或其分词器尚未加载的模型的提示计数。 |

- **分词器。**大多数模型无需设置 `tokenizer`。模型的目录卡片若在 Hugging Face 上发布开放权重（`distribution.source`），或由 vLLM、SGLang 等自托管引擎以仓库 id 提供服务（`provider_model_id: Qwen/Qwen3-8B`），就用该仓库的分词器计数。只有其他模型，或需要固定版本时，才需指定。Router 只下载仓库中的 `tokenizer.json`，使用与其他模型下载相同的 Hugging Face CLI 和设置（`HF_ENDPOINT`、受限仓库使用的 `HF_TOKEN`、`HF_HOME`、版本），存到同一个模型目录：`models/tokenizers/<org>/<name>`，或 `models/tokenizers/<org>/<name>@<revision>`。下载在 Router 启动时于后台进行，启动和请求都不等待；设置了 `HF_HUB_OFFLINE=1` 时不下载。Router 内置 `o200k_base` 和 `cl100k_base`，并能读取字节级 BPE 的 `tokenizer.json`，即 DeepSeek、Qwen、GLM、MiniMax 和 gpt-oss 发布的那类文件。
- **分词器加载之前。**模型的分词器下载期间，或无法加载时（无法访问 Hugging Face、受限仓库没有 `HF_TOKEN`、SentencePiece 分词器），由代理分词器为它计数，并使用更宽的区间。日志记为 `tokenizer_pending`、`tokenizer_download_failed` 或 `tokenizer_unavailable`，该分词器的 `llm_context_window_tokenizer_loaded` 为 `0`；分词器加载完成后为 `1`，日志记为 `tokenizer_loaded`。
- **字节上限。**字节级分词器产生的 token 数从不超过提示的 UTF-8 字节数。字节数加上聊天模板和消息框架的预留后，放得下请求能到达的最宽窗口的请求，不会被计数。
- **计数。**更长的请求从到达起就开始计数，与包括守卫在内的信号同时进行，两者互不等待。每个分词器计数一次，分成多段并行，这个计数也用于决策之后的检查。计数覆盖文本、工具调用与结果、工具定义和输出 schema，并为聊天模板给每条消息加几个 token。4 MB 的提示在单核上约需 50 ms。带有图片、音频、视频或文件的请求不计数。
- **区间。**用模型自己的分词器计数时，计数不超过窗口减去 `margin` 时放得下，达到窗口加上 `margin` 时放不下。代理计数可能与模型自己的计数相差较多：`o200k_base` 对中文文本的计数比某些模型的分词器高出多达四分之一，对英文和代码则略低。因此代理计数使用 `proxy_margin`，只有达到窗口加上 30% 时才拒绝。若计数排除了请求能到达的所有模型，Router 会取消仍在运行的信号，使守卫停止读取请求，并立即以 `400` 和 `context_length_exceeded` 应答，消息给出最宽的窗口和估算的计数；见[路由错误](../api/router#routing-errors)。计数介于两者之间时，请求继续发往后端，由后端决定；后端的拒绝以 `context_length_exceeded` 到达客户端，回退只会继续到窗口更宽的候选。若配方中有决策压缩上下文，拒绝留给决策之后的检查；若有决策替换客户端的系统提示或丢弃其工具（`none` 工具策略），只有去掉它们后仍超出区间的请求才会被拒绝。
- **由什么决定。**每个模型的检查在每个请求中记录一次日志 `context_window_checked`，带上计数的种类（`exact` 或 `proxy`）、所用区间和结论（`fits`、`in_band` 或 `exceeds`），并计入 `llm_context_window_checks_total`；拒绝计入 `llm_context_window_refusals_total`，`llm_context_window_margin` 给出两个区间。`context_window_counted` 记录计数用时，以及请求在信号之后等待计数的时间，后者也记入 `llm_context_window_wait_seconds`：约为零。
- **没有计数时。**带媒体的请求从不因长度被拒绝。它的 token 估算仍会让它避开窗口看起来太小的候选；当估算排除了所有候选时，最宽的候选保留。

token 估算会根据后端报告的提示用量学习每个 token 的字节数比率。这个比率只用于把请求放进配方的[上下文区间](../tutorials/signal/heuristic/context)：它从不拒绝请求。

### 无法访问 Hugging Face 时的分词器 {#tokenizers-without-hugging-face-access}

无法访问 Hugging Face 的 Router 从模型目录读取分词器，与其他模型相同。在能访问的环境中填好这个目录，再交付：

1. 把每个分词器下载到 Router 查找的位置，或用 `--download-only` 运行一次 Router，它会连同其他模型一起下载分词器：

   ```bash
   hf download deepseek-ai/DeepSeek-V4.1-Flash --include tokenizer.json \
     --local-dir models/tokenizers/deepseek-ai/DeepSeek-V4.1-Flash
   hf download Qwen/Qwen3-8B --include tokenizer.json --revision <commit> \
     --local-dir models/tokenizers/Qwen/Qwen3-8B@<commit>
   ```

2. 把 `models/` 作为 Router 的模型卷交付（镜像中为 `/app/models`），并设置 `HF_HUB_OFFLINE=1`。
3. 配置保持不变：指定的或推断出的仓库 id 会找到这些文件。存放在别处的分词器也可以用路径指定，如 `tokenizer: /opt/tokenizers/glm/tokenizer.json`。
4. 在 Router 启动时检查日志：`tokenizer_loaded` 给出每个分词器及其词表大小，`tokenizer_pending` 给出文件缺失的分词器，其模型随后由代理分词器计数，并使用更宽的区间。

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
| `forward_request_headers` | 无 | Router 自己发送的回退尝试从客户端请求中复制的请求头；参见[在回退尝试中携带客户端请求头](#carry-client-headers-on-a-fallback-attempt)。仅用于全局和配方策略。 |

请求所用的策略逐字段合并，最具体的一层优先：

1. 请求图步骤的 `fallback`，用于请求图发起的模型调用；
2. 决策的 `fallback`；
3. 配方的 `recipes[].fallback`；
4. `global.router.fallback`。

- **各层。**某一层省略的字段保持下一层的值，`enabled` 只在你写出它的地方设置。
- **决策。**决策的块接受除 `circuit_breaker` 以外的所有字段，`circuit_breaker` 属于后端。
- **候选**始终是决策中对该请求仍有资格的排名模型。没有单独的回退列表，提供方模型也没有自己的回退：同一模型的其他副本改由其重试、剔除和健康检查来尝试。
- **推理。**候选按它自己在 `modelRefs` 中的推理设置（`use_reasoning`、`reasoning_effort`、`reasoning_mode`）发送，并像第一次尝试那样写成其后端的格式。Router 无法准备的候选（例如被其限流拒绝的候选）会被跳过，改试下一个，并以 `fallback_candidate_skipped` 记录日志，注明原因。
- **流式。**流式请求同样会回退，无论它带有什么 `stream_options`。在 Envoy 之后，Router 以非流式方式调用候选，再按客户端自己的 `stream_options` 把回答以流式发给客户端，因此设置了 `include_usage` 的客户端仍会最后收到用量块。在 standalone 模式下，候选以流式发送，并像第一次尝试那样向 Chat Completions 后端请求用量。
- **应答。**没有成功就结束的回退链以最后一次尝试应答。候选的错误保留其状态码，并以客户端的协议到达客户端，因此候选的 `400` 仍是 `400`，而对超出窗口的提示的拒绝会变成 `context_length_exceeded`。最后一次尝试没有任何后端应答时，客户端收到带 `upstream_unavailable` 的 `503`，或带 `upstream_timeout` 的 `504`，不包含传输层的任何细节。
- **长提示。**Router 的[计数](#check-prompts-against-the-context-window)超出其窗口区间的候选从不尝试。后端因长度拒绝提示后，回退链只会继续到窗口更宽的候选。两者都以 `fallback_candidate_skipped` 记录日志。在 Envoy 之后看不到第一个模型的错误响应体，因此当计数落在该模型窗口的区间内时，它的 `400` 或 `413` 视为因长度的拒绝。

两种网关模式运行相同的策略，只有一处有意的差异：standalone 模式在完全没有后端应答时（例如连接被拒绝或超时）也会回退，而在 Envoy 之后，客户端会收到 Envoy 的回复。这样的失败计为 `503` 或 `504`，因此要在这类失败上回退，请在 `retryable_status_codes` 中保留这两个状态码。响应的任何部分一旦到达客户端，就不再回退。

两个回退超时都不会缩短第一次尝试，它沿用提供方模型和决策的 `reliability` 超时。要在某个模型始终不应答时回退，请在 standalone 模式下为该提供方模型设置 `reliability.per_try_timeout`：

- 超时计为 `504`，随后尝试下一个候选。
- 该候选最多得到 `per_attempt_timeout` 和 `total_timeout` 剩余的时间，因此 `total_timeout` 要长于该 `per_try_timeout`。
- 决策自己的 `per_try_timeout` 会限制它尝试的每个候选，因此较慢的候选也会以同样方式失败。

### 在回退尝试中携带客户端请求头 {#carry-client-headers-on-a-fallback-attempt}

在 Envoy 之后，第一次尝试经由 Envoy 发出，带着客户端的请求头。回退尝试是 Router 自己发起的调用，只携带后端的 `headers` 和凭据。要让某个客户端请求头也透传过去，例如上游网关据以归属用量的 `x-user-id`，请把它列在 `forward_request_headers` 中：

```yaml
global:
  router:
    fallback:
      enabled: true
      forward_request_headers: [x-user-id]
```

- **匹配。**名称匹配不区分大小写，只有客户端带值发送了某个请求头，尝试才会复制它。
- **优先级。**客户端的值替换后端 `headers` 中同名的值；客户端未发送时，后端的值作为默认值保留。凭据所用的请求头始终是 Router 自己的。
- **网关模式。**standalone 模式下，每次尝试（包括回退尝试）都从客户端请求开始构建，已经带着客户端的请求头，因此该列表在那里不起作用。从客户端请求开始构建的尝试（Envoy 之后的第一次尝试也是如此）发送的是后端 `headers` 中的值，而不是客户端的值。
- **各层。**只有全局策略和配方策略接受该列表，与 `circuit_breaker` 相同。配方的列表替换全局的列表。
- **日志。**回退记录中的每次尝试列出它复制的请求头名称（`forwarded_headers`），从不记录其值。

列表中出现 Router 构建的调用不得从客户端复制的请求头时，配置加载失败，并指明该请求头：

- 伪请求头，例如 `:authority`；
- 逐跳或帧请求头：`connection`、`keep-alive`、`te`、`trailer`、`transfer-encoding`、`upgrade`、任何 `proxy-*` 请求头、`host`、`content-length`，以及 Router 为自己的请求设置的正文请求头（`content-type`、`content-encoding`、`accept`、`accept-encoding`、`expect`）；
- 凭据：`authorization`、`proxy-authorization`、`cookie`、`x-api-key`、`api-key`、`ocp-apim-subscription-key`、每用户密钥请求头（`x-user-*-key`），以及任何后端的认证请求头；
- Router 自己的 `x-vsr-*` 请求头、Envoy 的 `x-envoy-*` 控制请求头，以及身份请求头（`x-authz-*` 和 `global.services.authz.identity` 设置的请求头名），Router 不会把它们发给任何上游。

远程部署对 Router 发往它的调用也接受同样的设置；参见[转发客户端请求头](model-catalog.md#forward-client-headers)。

在 Kubernetes 上，决策使用同样的 `reliability` 和 `fallback` 块，字段名也相同：位于 `IntelligentRoute` 的 `spec.decisions[]` 和 `SemanticRouter` 的 `spec.config.decisions[]` 中。应用资源时，CRD 会检查时长、重试次数和状态码。

## 校验结果

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

校验会在启动前解析目录条目、Provider 映射、推理控制、Model Card 身份和后端池兼容性。

## 配置 Router 任务使用的模型

对于 Router 自己使用的决策模型和嵌入模型，从[模型运行时](model-runtime/get-started/overview.md)开始。它介绍本地和远程模型、它们的配置以及运维。
