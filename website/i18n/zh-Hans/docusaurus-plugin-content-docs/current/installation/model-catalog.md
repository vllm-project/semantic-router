---
is_mtpe: false
title: 模型目录参考
sidebar_label: 模型目录参考
description: global.model_catalog.deployments 的每个字段、各 provider 及其能回答的内容、绑定、embedding、解析顺序、阈值和准入。
translation:
  source_commit: "3e9e4262e5c4e286ede3fb75cdac83edb985b6aa"
  source_file: "docs/installation/model-catalog.md"
  outdated: false
---

# 模型目录参考

`global.model_catalog` 声明 Router 询问的每个模型：回答路由信号的模型和 embedding 模型。请求路由的目标，即回答用户的模型，仍在 [`providers.models`](model-configuration.md) 中配置。[为信号选择模型](choose-a-model-for-a-signal.md)说明每个信号由哪个部署回答。

## 部署 {#deployments}

部署是 Router 可以询问的、有名称的模型。`provider` 指明 Router 与它通信所用的 API。

| `provider` | Router 调用的 API | 能回答 |
| --- | --- | --- |
| `model_runtime` | 内置模型运行时：托管（由 Router 启动）或挂接（设置了 `endpoint`） | 其模型卡提供的任何内容 |
| `systemone` | System One API 的 `POST <endpoint>` 和 `GET <endpoint>/models` | 每个问题类信号和选择器 |
| `classify` | `POST <endpoint>/classify` | masking、响应 jailbreak 和幻觉检测插件的模型（`pii`、`jailbreak` 和 `hallucination` 绑定） |
| `openai_chat` | `POST <endpoint>/chat/completions` | 响应 jailbreak 插件的防护模型和幻觉检测插件的检测模型 |
| `openai_embeddings` | `POST <endpoint>/embeddings` | Embedding |

部署被分配了它无法回答的信号时，Router 在加载时拒绝该部署。

有三个部署始终已声明，只有当某处解析到它们时才会启动：

| 名称 | 模型 | 默认用于 |
| --- | --- | --- |
| `primary` | CPU 上的 `vllm-sr/vela3-fast` | `system.decision_model` |
| `vela-embedding` | Vela 1.0 Embedding | `embeddings.default` |
| `omni-nano` | Vela 1.0 Omni Nano | `embeddings.multimodal` |

## 字段参考 {#field-reference}

```yaml title="field-reference"
global:
  model_catalog:
    deployments:
      # Managed model_runtime: no endpoint; the Router starts and supervises it.
      managed-example:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Hazard  # Hub repository or absolute package path
        revision: 5dd25f2cc3c98f338e6a79b667662d60f936a28d  # 40-character commit
        device: cpu                      # cpu | auto | cuda | rocm, or cuda:N / rocm:N
        input: {max_tokens: 32768, overflow: reject}  # reject | truncate | window
        public_name: hazard-public       # its name on the inference API
        operating_point:                 # pins the head's packaged operating point
          path: operating_point.json
          sha256: e79a78f48bf45eb38e3f5402de3b3b18eeaa822e00b42b3640bf471276290de5
      replicated-example:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        revision: cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764
        replicas: [{device: cpu}, {endpoint: http://worker-2.example.com:8100, model: kai}]
      # Connection block: an attached model_runtime and every remote provider.
      remote-example:
        provider: systemone              # or classify, openai_chat, openai_embeddings
        endpoint: https://api.example.com/v1/systemone  # full URL; unix:///abs.sock for an attached runtime
        model: vllm-sr/d3-ultra          # the remote model; an attached runtime's served model
        auth: {header: Authorization, prefix: Bearer, api_key_env: EXAMPLE_KEY}  # or api_key_file: /abs/path
        headers: {X-Org: acme}           # non-secret headers only
        timeout_ms: 1500                 # per attempt
        max_retries: 1                   # transport retries within the caller's deadline
        tls: {ca_file: /etc/ssl/ca.pem, server_name: api.example.com, insecure_skip_verify: false}
        limits: {max_request_bytes: 262144, max_response_bytes: 4194304}
        capabilities: {question_types: [choice, noul, score], presets: [pii, halu]}
    system:
      decision_model: {deployment: primary}
    bindings: {pii: primary}
    embeddings: {default: vela-embedding, multimodal: omni-nano}
```

### 托管 `model_runtime` 字段 {#managed-model_runtime-fields}

| 字段 | 含义 |
| --- | --- |
| `artifact` | Hugging Face 仓库 ID，或模型包的绝对路径 |
| `revision` | 仓库 artifact 的 40 位提交哈希 |
| `device` | `cpu`、`auto`（主机上最好的设备）、`cuda`、`rocm`，或单个设备，例如 `cuda:0` |
| `input` | 部署的输入预算：`max_tokens`，以及 `overflow`（`reject`、`truncate` 或 `window`） |
| `replicas` | 同一逻辑部署的多个 worker：每个是一个 `device`，或在别处启动的 worker 的 `endpoint` 和 `model` |
| `public_name` | 部署在推理 API 上的名称 |
| `operating_point` | 该头的模型包所提供工作点的 `path` 和 `sha256`。由此部署回答的分类器规则按工作点的阈值读取相互独立的标签分数 |
| `max_num_seqs`、`max_num_batched_tokens`、`batch_window_ms`、`dtype`、`enable_shared_context`、`max_model_len` | 引擎参数；参见[引擎参数](../model-runtime/engine-args.md) |

### 连接块 {#connection-block}

同一组字段既用于连接挂接的模型运行时（设置了 `endpoint` 的 `model_runtime` 部署），也用于每个远程 provider。

| 字段 | 含义 | 限制 |
| --- | --- | --- |
| `endpoint` | API 的完整 URL，包括路径前缀。Router 在其后追加操作路径（`/classify`、`/chat/completions`、`/embeddings`）；systemone 的 endpoint 就是提交问题的 URL。挂接的运行时可以使用 `unix:///abs/path.sock` | `http` 或 `https`；不得包含凭据或片段标识（fragment） |
| `model` | 远程 API 用来回答的模型；在挂接的运行时上是它提供服务的模型 | `systemone`、`openai_chat`、`openai_embeddings` 必填 |
| `auth` | `header`（默认 `Authorization`）、`prefix`（默认 `Bearer`；`""` 表示发送不带前缀的密钥），以及 `api_key_env` 和 `api_key_file`（绝对路径，每次调用时读取，因此轮换后的密钥会生效）二者之一 | 密钥从不写在配置中 |
| `headers` | 其他非机密请求头 | 拒绝 `Authorization`、`Proxy-Authorization`、`X-Api-Key`、`Api-Key`、`Ocp-Apim-Subscription-Key`、`Cookie` 以及 `auth` 所用的请求头 |
| `timeout_ms` | 单次尝试的时限 | systemone ≤ 60000；其他 ≤ 300000 |
| `max_retries` | 首次尝试之后的传输层重试次数，不超出调用方的截止时间 | 0–10；挂接的运行时不支持 |
| `tls` | `ca_file`、`server_name`、`cert_file` 和 `key_file`（客户端证书）、`insecure_skip_verify` | 仅限 `https`；文件为绝对路径 |
| `limits` | `max_request_bytes`、`max_response_bytes` | 挂接的运行时不支持 |
| `capabilities` | 远程模型的 API 未列出任何内容时，它能回答什么：`question_types` 和 `presets`（systemone），`dimensions`（openai_embeddings） | |
| `generation` | `openai_chat` 部署生成补全时的 `max_tokens`、`temperature` 和 `reasoning`（`{family, use_reasoning, reasoning_effort}`，`family` 为内置推理系列） | `max_tokens` ≤ 32768；`temperature` 0–2；`reasoning` 需要 `family` 和 `use_reasoning` |

重定向会被拒绝，因此凭据永远不会随重定向发往其他主机。

## 绑定和规则部署 {#bindings-and-rule-deployments}

```yaml
global:
  model_catalog:
    deployments:
      local-vela: {provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu}
    bindings:            # every recipe
      pii: local-vela
routing:
  model_bindings:        # this routing profile; recipes[].routing.model_bindings for one recipe
    domain: d3-remote
```

绑定键是应用信号类型：`domain`、`fact_check`、`user_feedback`、`jailbreak`、`safety`、`pii`、`hallucination`、`refusal`、`turn_relation`、`complexity` 和 `modality`。绑定回答该类型中未指定部署的规则；规则写 `deployment:` 可改用其他部署。`routing.signals.systemone[]` 规则和决策的 `algorithm.systemone` 也写 `deployment:`。`jailbreak`、`pii` 和 `hallucination` 绑定同时指定该类型插件模块的模型。

绑定也可以写成对象，用于选择多头 `model_runtime` 部署的某个头：`pii: {deployment: local-vela, head: pii}`。工作点通过 `operating_point` 固定在部署上。

每个信号的解析顺序：规则的 `deployment`，然后是配方的绑定，然后是全局绑定，问题类信号最后使用 `system.decision_model`。Embedding 通过 `embeddings.default` 解析，用自己模型做 embedding 的路由配置则通过它的 `model_bindings.embedding` 解析。

## 阈值 {#thresholds}

规则未写出的 predicate 来自回答的模型：`vela3-fast` 或 Vela 2.0 各尺寸校准的阈值；其他模型（本地或 `systemone`）没有校准值，使用 0.5（加载时警告指明该规则的 `uncalibrated_preset`）。你写出的 `predicate` 按原值使用。插件模块以同样方式保留各自的模块阈值，远程分类器沿用 Vela 1.0 的默认值。

## 准入 {#admission}

`global.model_catalog.admission` 限制并发调用数。键可以是部署名称，表示该部署的所有使用者共享的闸门；也可以是保留自有模型的插件模块的信号类型（`jailbreak`、`pii`、`hallucination`），表示该模块的闸门。

```yaml
global:
  model_catalog:
    admission:
      jailbreak: {max_concurrency: 4, max_queue: 64, queue_timeout_ms: 250, on_overflow: shed}
      d3-remote: {max_concurrency: 16}
```

## 示例 {#examples}

### R1. 远程 embedding 模型 {#r1-remote-embedding-model}

```yaml title="r1-remote-embedding"
global:
  model_catalog:
    deployments:
      remote-embed:
        provider: openai_embeddings
        endpoint: https://embedding.example.com/v1
        model: BAAI/bge-m3
        auth: {api_key_env: EMBEDDING_API_KEY}
        timeout_ms: 5000
        max_retries: 2
        capabilities: {dimensions: 1024}
    embeddings:
      default: remote-embed        # multimodal: <name> for multimodal embeddings
```

Router 向 `https://embedding.example.com/v1/embeddings` 发送请求：操作路径追加在 endpoint 的路径之后，从不由主机名和端口拼出。

### R3. 越狱与幻觉规则使用远程决策模型 {#r3-remote-decision-model-for-jailbreak-and-hallucination}

```yaml title="r3-remote-judge"
global:
  model_catalog:
    deployments:
      d3-judge:
        provider: systemone
        endpoint: https://judge.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: JUDGE_KEY}
        timeout_ms: 5000
routing:
  model_bindings:
    jailbreak: d3-judge                  # request- and response-stage rules
    hallucination: d3-judge
```

该作用域中的每条 `jailbreak` 和 `hallucination` 规则都向 `d3-judge` 提问；其他信号仍由 `primary` 回答。绑定只写部署名：信号决定问题，provider 决定 Router 如何提问。幻觉规则提出 span 问题，因此其部署必须能回答 span；列出 `capabilities.question_types` 的 `systemone` 部署必须包含 `span`。

### R4. 网关后的 System One，使用网关自己的认证请求头 {#r4-system-one-behind-a-gateway-with-its-own-auth-header}

```yaml title="r4-systemone-gateway"
global:
  model_catalog:
    deployments:
      d3-gateway:
        provider: systemone
        endpoint: https://gateway.example.com/Unified/v1/systemone
        model: vllm-sr/d3-ultra
        auth:
          header: Ocp-Apim-Subscription-Key
          prefix: ""                     # no Bearer prefix
          api_key_file: /run/secrets/systemone-key
        timeout_ms: 3000
        capabilities: {question_types: [choice, noul, score], presets: [pii, halu]}
    system:
      decision_model: {deployment: d3-gateway}
```

### R5. 挂接的模型运行时 {#r5-attached-model-runtime}

设置了 `endpoint` 时，Router 连接一个已经在运行的模型运行时；不设置时，Router 自己启动并监管运行时。

```yaml title="r5-attached-runtime"
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        endpoint: https://decision-worker.example.com:8100
        model: decision-kai                      # the served model
        auth: {api_key_env: WORKER_KEY}          # optional
        tls: {ca_file: /etc/ssl/internal-ca.pem}
        timeout_ms: 1000
    system:
      decision_model: {deployment: decision-kai}
```

### R6. 请求后端使用同样的连接形式 {#r6-request-backends-use-the-same-connection-shape}

```yaml title="r6-request-backends"
providers:
  defaults: {model: claude}
  models:
    - name: claude
      backend_refs:
        - provider: anthropic
          endpoint: https://gateway.example.com/anthropic
          auth: {header: x-api-key, prefix: "", api_key_env: ANTHROPIC_KEY}
          headers: {X-Org: acme}
    - name: native-d3              # a native System One alias: no second endpoint or key
      api_format: systemone
      deployment: d3-remote
global:
  model_catalog:
    deployments:
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}
```

`providers.models` 仍是一个独立的概念：回答用户、由路由在其中做选择的模型。共享的只有连接形式（`endpoint`、`auth`、`headers`）。后端从 `auth.api_key_env` 读取密钥。部署（包括决策模型）永远不是路由目标，也从不出现在 `/v1/models` 中。原生 System One API 只在部署的 `public_name` 下，或通过指向它的原生别名（如 `native-d3`）为它作答。
