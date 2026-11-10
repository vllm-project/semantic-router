---
title: 选择部署
sidebar_label: 选择部署
description: 在本地运行决策模型、挂接你自己运行的运行时，或向远程 System One API 提问，以及 Router 在加载时检查什么。
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/system-one/deployments.md"
  outdated: false
---

# 选择部署 {#choose-a-deployment}

每个问题都由一个**部署**回答：它是 `global.model_catalog.deployments` 下的一个命名条目，说明由哪个模型回答、在哪里运行。部署只声明一次，在用到它的地方写它的名称即可。

## 问题的部署从何而来 {#where-a-questions-deployment-comes-from}

最具体的层级优先：

| 层级 | 写在哪里 | 适用于 |
| --- | --- | --- |
| 规则 | 规则、选择器（`algorithm.systemone`）、评审（`match.judge`）或检查（`escalate.accept.systemone`）上的 `deployment` | 该规则或功能 |
| 配方 | `recipes[].model_bindings.<type>` | 一个配方中某个应用信号类型的规则 |
| 全局 | `global.model_catalog.bindings.<type>` | 所有配方中某个应用信号类型的规则 |
| 默认 | `global.model_catalog.system.decision_model` | 上述层级未覆盖的每个问题 |

绑定的键是应用信号类型：`domain`、`fact_check`、`user_feedback`、`jailbreak`、`safety`、`pii`、`hallucination`、`refusal`、`turn_relation`、`complexity` 和 `modality`。[为信号选择模型](../installation/choose-a-model-for-a-signal)用 Router 测试会加载的示例逐一演示了每个层级。

## 本地模型 {#local-models}

`model_runtime` 部署运行在内置[模型运行时](../model-runtime/overview)中，由 Router 启动和监管。预先声明的 `primary` 部署是运行在 CPU 上的 vela3-fast；重新声明 `primary` 即可更换它的模型或设备：

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-lite
        device: rocm:0
```

每个内置 artifact 都固定到一个确切的 Hugging Face 修订版。没有任何内容解析到的部署永远不会启动。

| 需求 | 模型 | 运行于 |
| --- | --- | --- |
| 默认：每个内置问题都经过训练，并支持 span | `vllm-sr/vela3-fast` | CPU；加载后的部署约占 7 GB |
| 在小 GPU 上降低延迟、多语言 | `vllm-sr/vela3-lite`（0.8B） | GPU |
| 更高准确率，内置问题仍经过训练 | `vllm-sr/vela3-mini`（4B）、`-flash`（9B）、`-turbo`（27B） | GPU |
| 针对图片和视频的问题 | Decision 3.0 各规格，`vllm-sr/d3-edge` 到 `vllm-sr/d3` | GPU；不支持 span |
| 在 CPU 上回答你自己的问题 | `vllm-sr/Decision-2.0-Kai-0.6B` | CPU 或 GPU；不支持 span |

[选择模型、规格和硬件](../model-runtime/choose-a-model)列出了每个规格的内存、精度和实测延迟。

### 挂接你自己运行的运行时 {#attach-a-runtime-you-run}

带 `endpoint` 的 `model_runtime` 部署会挂接到一个不由 Router 管理的引擎，例如多个 Router 共享的一台 GPU 服务器：

```yaml
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

`model` 是运行时在 `GET /v1/models` 中列出的 ID。Router 在运行时就绪后读取它的模型卡，并在那时检查规则。[挂接你自己运行的运行时](../model-runtime/deploy#attach-to-a-runtime-you-run)介绍了运行时一侧、副本和 Kubernetes。

## 远程 System One API {#remote-system-one-apis}

`systemone` 部署向一个不由 Router 运行的 System One API 提问：托管的 Decision API、另一个 vLLM-SR 引擎，或其前面的网关。它只回答 System One 问题，不属于请求会被路由到的 `providers.models`。

```yaml
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

- `endpoint` 是该 API 接收 `POST` 请求的 URL（含路径前缀）；`model` 是它用来回答的模型。
- 该 API 必须接受两种[请求形式](./api)：Router 把每个阶段的问题放在一次调用中发送，即条目数与该阶段所问内容相同的批量请求（每个评审最多 `judge.max_candidates` 条），并且从不拆分。
- 密钥来自 `auth.api_key_env` 或 `auth.api_key_file`，绝不来自配置本身，并以 `Authorization: Bearer <key>` 发送。如果网关在自己的请求头中接收密钥，设置 `auth.header`；设置 `auth.prefix: ""` 则发送不带前缀的密钥。
- 按用户授权的 API 用 `auth: {credential: delegated}` 代替密钥：每次调用都在同一请求头和前缀中携带身份验证器为发起请求的用户签发的凭据（`global.services.authz.delegated_credential`）。没有凭据的请求会在调用之前得到 `401`，Router 自己读取模型列表的调用会被拒绝（此时使用 `capabilities`），答案也不会为其他请求保留。见[安全加固](../installation/security-hardening#delegate-the-users-credential-to-an-upstream)。
- 问题类型、预设和输入模态来自该 API 的 `GET <endpoint>/models`（如果它列出了该模型）；否则来自部署的 `capabilities`（`question_types`、`presets`、`modalities`）；再否则为基于文本的 `choice`、`noul` 和 `score`。
- 没有自己 `timeout_ms` 的问题使用部署的 `timeout_ms`，否则为 1000 毫秒。缓慢、失败或不可达的 API 会让它的问题变为未知，但本身绝不会让请求失败；请求没有携带凭据的委托 API 除外。
- d3 家族不支持 span，所以这个示例把 `pii` 和 `hallucination` 留在本地的 `primary` 部署上。如果没有这两个绑定，`pii` 规则会以 `span_required` 加载失败。

## Router 在加载时检查什么 {#what-the-router-checks-at-load}

`vllm-sr config validate --config config.yaml` 执行相同的检查。没有任何静默回退：

| 检查 | 错误或警告 |
| --- | --- |
| 规则、绑定或功能指定的每个部署都已声明 | 该名称，以及已声明的名称 |
| 部署能回答规则的问题类型 | 该规则、该部署以及它能回答的类型 |
| 需要 span 的规则或插件解析到支持 span 的部署 | `span_required` |
| 规则的 `input.image`、`input.video` 或 `input.audio` 能被其部署读取 | `input_modality_unsupported` |
| 某个预设没有校准阈值 | `uncalibrated_preset`（警告；该规则使用 0.5） |
| 内联写入的密钥（`auth.api_key`、`Authorization` 请求头） | 密钥必须来自 `api_key_env` 或 `api_key_file` |

模型要等运行后才能确定的挂接运行时，会在准备问题时以相同的消息进行检查。

## 哪些数据会离开本机 {#what-leaves-the-host}

发往远程部署的问题会发送它读取的请求部分。选择绑定时请考虑这一点：

| 提问方 | 发送 |
| --- | --- |
| `domains`、`fact_check`、`complexity`、`modality` | 当前用户轮次 |
| `user_feedbacks` | 当前用户轮次，以及它所回复的助手答案 |
| `jailbreak`、`safety` | 当前用户轮次，设置 `input.history` 时还有之前的用户轮次；响应阶段发送模型的答案 |
| `pii` | 当前用户轮次，以及规则要求读取的之前用户轮次和工具结果 |
| `turn_relation` | 当前用户轮次和最近几轮对话 |
| `hallucination` | 模型的答案及其依据上下文 |
| `refusal`、`answer_check` | 请求和模型的答案 |
| `relevance`、`same_request` | 请求以及每个候选文档、片段或缓存请求 |
| `tool_need`、`tool_rank` | 请求以及候选工具的名称和描述 |
| `systemone` 规则 | 其 `input` 指定的部分，设置 `input.image` 时还有图片 |

远程模型没有校准阈值，所以在规则写出自己的 `predicate` 之前，它的预设使用 0.5 并给出 `uncalibrated_preset` 警告。嵌入部署与这些完全分开：远程嵌入提供方会收到它要嵌入的文本，而且只有召回功能使用它。
