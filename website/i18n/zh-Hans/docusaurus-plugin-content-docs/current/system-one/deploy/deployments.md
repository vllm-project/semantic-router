---
title: 选择部署
sidebar_label: 选择部署
sidebar_position: 1
description: 在本地运行决策模型、附加你自己运行的运行时，或询问远程 System One API，以及 Router 在加载时检查什么。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/system-one/deploy/deployments.md"
  outdated: false
---

# 选择部署 {#choose-a-deployment}

每个 System One 问题都由一个**部署**回答：`global.model_catalog.deployments` 中的一个命名条目，说明由哪个模型回答以及它在哪里运行。部署只需声明一次，然后在使用它的地方引用其名称。

| 你想要 | 使用 | 章节 |
| --- | --- | --- |
| 由 Router 为你运行模型，在它的 CPU 或 GPU 上 | 本地 `model_runtime` 部署 | [本地模型](#local-models) |
| 由多个 Router 共享一台 GPU 服务器 | 带 `endpoint` 的 `model_runtime` 部署 | [附加你自己运行的运行时](#attach-a-runtime-you-run) |
| 托管的 Decision API 或另一个 vLLM-SR 引擎 | `systemone` 部署 | [远程 System One API](#remote-system-one-apis) |

## 问题的部署从何而来 {#where-a-questions-deployment-comes-from}

最具体的层级优先：

| 层级 | 写在哪里 | 适用于 |
| --- | --- | --- |
| 规则 | 规则、选择器（`algorithm.systemone`）、评审（`match.judge`）或检查（`escalate.accept.systemone`）上的 `deployment` | 该规则或功能 |
| Recipe | `recipes[].model_bindings.<type>` | 某个 recipe 中某一应用信号类型的规则 |
| 全局 | `global.model_catalog.bindings.<type>` | 每个 recipe 中某一应用信号类型的规则 |
| 默认 | `global.model_catalog.system.decision_model` | 上述层级未指定的每个问题 |

绑定键是应用信号类型：`domain`、`fact_check`、`user_feedback`、`jailbreak`、`safety`、`pii`、`hallucination`、`refusal`、`turn_relation`、`complexity` 和 `modality`。[为信号选择模型](../../installation/choose-a-model-for-a-signal.md)用 Router 测试会加载的示例逐一讲解每个层级。

## 本地模型 {#local-models}

`model_runtime` 部署运行在内置[模型运行时](../../model-runtime/get-started/overview.md)中，由 Router 启动和监管。预先声明的 `primary` 部署是在 CPU 上运行的 vela3-fast；重新声明 `primary` 可以更换其模型或设备：

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-lite
        device: rocm:0
```

| 你想要 | 模型 | 运行于 |
| --- | --- | --- |
| 默认：每个内置问题都经过训练，并支持 span | `vllm-sr/vela3-fast` | CPU；加载后的部署约占 7 GB |
| 在小型 GPU 上更低的延迟，多语言 | `vllm-sr/vela3-lite`（0.8B） | GPU |
| 更高准确率，内置问题仍经过训练 | `vllm-sr/vela3-mini`（4B）、`-flash`（9B）、`-turbo`（27B） | GPU |
| 关于图像和视频的问题 | Decision 3.0 各规格，`vllm-sr/d3-edge` 到 `vllm-sr/d3` | GPU；不支持 span |
| 在 CPU 上运行你自己的问题 | `vllm-sr/Decision-2.0-Kai-0.6B` | CPU 或 GPU；不支持 span |

每个内置制品都固定到确切的 Hugging Face 修订版本，没有任何内容解析到的部署永远不会启动。[支持的模型](../../model-runtime/supported-models/overview.md)为每个模型提供一页，列出其规模、显存、问题类型和部署片段。

### 附加你自己运行的运行时 {#attach-a-runtime-you-run}

带 `endpoint` 的 `model_runtime` 部署会附加到一个不由 Router 管理的引擎，例如由多个 Router 共享的 GPU 服务器：

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      gpu-decider:
        provider: model_runtime
        artifact: vllm-sr/vela3-flash
        endpoint: http://gpu-runtime.example.com:8100
        model: vela3-flash
    system:
      decision_model: {deployment: gpu-decider}
```

`model` 是运行时在 `GET /v1/models` 中列出的 ID。Router 在运行时启动后读取其模型卡，并在那时据此检查规则。[把 Router 连接到模型运行时](../../model-runtime/deploy/connect-the-router.md#attach-to-a-runtime-you-run)介绍运行时一侧，[在 Kubernetes 上运行](../../model-runtime/deploy/kubernetes.md)介绍集群中的共享运行时。

## 远程 System One API {#remote-system-one-apis}

`systemone` 部署询问一个不由 Router 运行的 System One API：托管的 Decision API、另一个 vLLM-SR 引擎，或其前面的网关。它只回答 System One 问题；它不是请求要路由到的 `providers.models` 之一。

```yaml title="config.yaml"
global:
  model_catalog:
    deployments:
      d3-remote:
        provider: systemone
        endpoint: https://systemone.example.com/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: SYSTEMONE_API_KEY}
        timeout_ms: 1500
    system:
      decision_model: {deployment: d3-remote}
    bindings:
      pii: primary
      hallucination: primary
```

- `endpoint` 是 API 接收 `POST` 请求的 URL，包括任何路径前缀；`model` 是它用来回答的模型。
- API 必须接受两种[请求形式](../use/api.md)：Router 把每个阶段的问题放在一次调用中发送，是一个条目数与该阶段所问对象数相同的批量请求（每个评审最多 `judge.max_candidates` 个），并且从不拆分。
- 问题类型、预设和输入模态在 API 的 `GET <endpoint>/models` 列出该模型时取自那里；否则取自部署的 `capabilities`（`question_types`、`presets`、`modalities`）；再否则为针对文本的 `choice`、`noul` 和 `score`。
- 没有自己 `timeout_ms` 的问题使用部署的值，否则为 1000 ms。缓慢、失败或不可达的 API 会让其问题成为未知；它本身绝不会让请求失败，除非是请求没有携带凭证的委托调用。

:::warning
d3 家族不回答 span，所以这个示例把 `pii` 和 `hallucination` 留在本地 `primary` 部署上。没有这两个绑定，`pii` 规则会以 `span_required` 加载失败。
:::

### 认证 {#authentication}

- 密钥来自 `auth.api_key_env` 或 `auth.api_key_file`，绝不来自配置本身，并以 `Authorization: Bearer <key>` 发送。在自己的头中接收密钥的网关设置 `auth.header`，并用 `auth.prefix: ""` 发送不带前缀的密钥。
- 按用户授权的 API 用 `auth: {credential: delegated}` 代替密钥：每次调用都携带认证器为发起请求的用户签发的凭证（`global.services.authz.delegated_credential`），使用相同的头和前缀。没有凭证的请求在调用前就以 `401` 回答，Router 自己读取列表的请求会被拒绝（此时使用 `capabilities`），并且不会为其他请求保留任何答案。参见[安全加固](../../installation/security-hardening.md#delegate-the-users-credential-to-an-upstream)。

## Router 在加载时检查什么 {#what-the-router-checks-at-load}

`vllm-sr config validate --config config.yaml` 运行同样的检查。不会有任何静默回退：

| 检查 | 错误或警告 |
| --- | --- |
| 规则、绑定或功能引用的每个部署都已声明 | 该名称，以及已声明的名称 |
| 部署能回答规则的问题类型 | 规则、部署及其能回答的类型 |
| 需要 span 的规则或插件解析到支持 span 的部署，或其 `capabilities.presets` 列出相应 `pii` 或 `halu` 预设的部署 | `span_required` |
| 规则的 `input.image`、`input.video` 或 `input.audio` 能被其部署读取 | `input_modality_unsupported` |
| 预设在没有校准阈值的情况下被回答 | `uncalibrated_preset`（警告；规则使用 0.5） |
| 内联写入的密钥（`auth.api_key`、`Authorization` 头） | 密钥应来自 `api_key_env` 或 `api_key_file` |

模型只有在运行后才能确定的附加运行时，会在准备问题时以相同的消息进行检查。

## 哪些内容会离开主机 {#what-leaves-the-host}

发往远程部署的问题会发送它所读取的请求部分。选择绑定时请考虑这一点：

| 询问方 | 发送 |
| --- | --- |
| `domains`、`fact_check`、`complexity`、`modality` | 当前用户轮次 |
| `user_feedbacks` | 当前用户轮次及其所回复的助手回答 |
| `jailbreak`、`safety` | 当前用户轮次，以及设置 `input.history` 时之前的用户轮次；在响应阶段为模型的回答 |
| `pii` | 当前用户轮次，以及它被要求读取的之前用户轮次和工具结果 |
| `turn_relation` | 当前用户轮次和对话的最近几轮 |
| `hallucination` | 模型的回答及其依据上下文 |
| `refusal`、`answer_check` | 请求和模型的回答 |
| `relevance`、`same_request` | 请求以及每个候选文档、片段或缓存请求 |
| `tool_need`、`tool_rank` | 请求以及候选工具的名称和描述 |
| `systemone` 规则 | 其 `input` 指定的部分，以及设置 `input.image` 时的图像 |

:::note
远程模型没有校准阈值，因此在规则写出自己的 `predicate` 之前，其预设使用 0.5 并给出 `uncalibrated_preset` 警告。嵌入部署与此完全无关：远程嵌入提供方会收到它要嵌入的文本，而且只有召回功能使用它。
:::
