---
is_mtpe: false
title: 为信号选择模型
sidebar_label: 为信号选择模型
description: 每个信号由哪个部署回答，如何在全局、按配方或按规则更改，以及 Router 加载配置时检查什么。
translation:
  source_commit: "3e9e4262e5c4e286ede3fb75cdac83edb985b6aa"
  source_file: "docs/installation/choose-a-model-for-a-signal.md"
  outdated: false
---

# 为信号选择模型

Router 询问的每个模型都在 `global.model_catalog.deployments` 下按名称声明一次。用到模型的地方都只写这个名称。本页说明这个名称写在哪里。[模型目录参考](model-catalog.md)列出了部署的全部字段。

## 规则 {#the-rule}

由哪个部署回答一个信号，在三个层级之一确定。最具体的层级优先。

| 层级 | 部署名称写在哪里 | 适用于 |
| --- | --- | --- |
| 规则 | 规则上的 `deployment:` | 每条应用规则（`domains`、`fact_check`、`user_feedbacks`、`jailbreak`、`safety`、`pii`、`hallucination`、`refusal`、`turn_relation`、`complexity`、`modality`）、每条 `systemone` 规则，以及 System One 选择器 `algorithm.systemone` |
| 信号类型 | `global.model_catalog.bindings.<type>`，或配方中的 `routing.model_bindings.<type>` | 未指定部署的应用信号类型规则：`domain`、`fact_check`、`user_feedback`、`jailbreak`、`safety`、`pii`、`hallucination`、`refusal`、`turn_relation`、`complexity`、`modality` |
| 默认 | `global.model_catalog.system.decision_model` | 上面两个层级未覆盖的所有问题 |

`recipes[].routing.model_bindings` 中的绑定只作用于该配方，并优先于同一信号类型的全局绑定。

Embedding 和重排序不提问。它们在使用的地方指定：

| 资源 | 在哪里指定 | 默认值 |
| --- | --- | --- |
| Embedding | `global.model_catalog.embeddings.default` | `vela-embedding` |
| 多模态 embedding | `global.model_catalog.embeddings.multimodal` | `omni-nano` |
| 重排序 | `rag` 插件中的 `rerank.deployment` | 无 |

有三个部署始终已声明，只有当某处解析到其中一个时 Router 才会启动它：`primary`（CPU 上的 `vllm-sr/vela3-fast`，默认决策模型）、`vela-embedding`（Vela 1.0 Embedding）和 `omni-nano`（Vela 1.0 Omni Nano）。要更改其中一个，声明一个同名部署即可。

## S1. 全部默认：什么都不写 {#s1-all-defaults-nothing-written}

```yaml title="s1-defaults"
global:
  model_catalog:
    system:
      decision_model: {deployment: primary}
```

`decision_model: {deployment: primary}` 就是默认值，所以不写它也不会有任何变化。

| 对象 | 解析到 |
| --- | --- |
| 每个问题类信号和选择器 | `primary`（按名称） |
| Embedding | `vela-embedding` |
| 多模态 embedding | `omni-nano` |

## S2. 全部交给远程 System One 模型 {#s2-everything-on-a-remote-system-one-model}

```yaml title="s2-remote-systemone"
global:
  model_catalog:
    deployments:
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}      # keys come from an env var or a file, never inline
        timeout_ms: 1500
        capabilities: {presets: [pii, halu]}  # used if the API's /models lists none
    system:
      decision_model: {deployment: d3-remote}
```

- 每个内置问题类信号、自定义 `systemone` 信号和选择器都询问 `d3-remote`。Router 在加载时检查该部署能回答配置所提的每种问题类型。
- 在一个请求阶段内，发给同一部署的所有问题合并为一次调用。API 报告失败的问题会单独再问一次。
- 规则的默认 predicate 来自回答的模型：它的模型卡校准的阈值，或规则写的 `predicate`。没有校准阈值的模型（本地或远程）使用 0.5，加载时报告指明该规则的 `uncalibrated_preset` 警告。
- PII 和幻觉检测规则需要片段。`pii` 或 `hallucination` 规则解析到的部署，若其声明的 `capabilities` 既不回答 span 问题、也没有 `pii`/`halu` 预设，则加载失败（`span_required`）。masking 插件同样需要 PII 片段，因此也会拒绝这样的部署。
- Embedding 和重排序仍使用各自的本地部署。

## S3. 混合：PII 在本地，其余在远程 {#s3-mixed-pii-local-everything-else-remote}

```yaml title="s3-mixed"
global:
  model_catalog:
    deployments:
      local-vela: {provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu}
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}
    system:
      decision_model: {deployment: d3-remote}
    bindings:
      pii: local-vela
```

`pii` 解析到 `local-vela`；其他所有信号解析到 `d3-remote`。绑定的键是信号类型，与 `routing.signals` 中的命名一致；值是部署名称。

## S4. systemone 信号和 domain 信号共用默认值，再各改一个 {#s4-a-systemone-signal-and-the-domain-signal-share-the-default-then-one-of-each-is-changed}

```yaml title="s4-type-and-rule-overrides"
global:
  model_catalog:
    deployments:
      kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        revision: cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764
        device: auto
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}
    system:
      decision_model: {deployment: kai}
routing:                           # or recipes[].routing, which affects only that recipe
  model_bindings:
    domain: d3-remote              # signal type: every domain rule shares one answer
  signals:
    domains: [{name: math, description: Mathematics, labels: [math]}]
    systemone:
      - name: needs_reasoning      # no deployment: asks the default, kai
        question: {type: noul, instructions: "Does answering need multi-step reasoning?"}
      - name: support_team
        deployment: d3-remote      # rule: only this question goes remote
        question:
          type: choice
          instructions: Which team?
          choices: [{key: billing}, {key: engineering}]
```

`domain` 解析到 `d3-remote`，`needs_reasoning` 解析到 `kai`，`support_team` 解析到 `d3-remote`。同样的内容写在 `recipes[].routing` 下时，只改变该配方：

```yaml title="s4-recipe-scope"
global:
  model_catalog:
    deployments:
      kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        revision: cd49ea3813fd8ba0928a9a23ef6c9a0f2f0cd764
        device: auto
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}
    system:
      decision_model: {deployment: kai}
entrypoints:
  - {model_names: [vllm-sr/support], recipe: support}
recipes:
  - name: support
    routing:
      model_bindings:
        domain: d3-remote          # only the support recipe asks d3-remote about domains
      signals:
        domains: [{name: math, description: Mathematics, labels: [math]}]
```

在 `support` 配方中，`domain` 解析到 `d3-remote`；在其他地方解析到 `kai`。

## S5. 自己指定部署的规则 {#s5-rules-that-name-their-own-deployment}

```yaml title="s5-rule-level"
routing:
  signals:
    domains:
      - {name: math, labels: [math]}
      - {name: law, labels: [law], deployment: kai}
    safety:
      - name: harmful
        deployment: d3-remote
        predicate: {gte: 0.5}
  decisions:
    - name: pick-model
      modelRefs: [{model: general}, {model: small}]
      algorithm: {type: systemone, systemone: {deployment: kai, instructions: Pick the cheapest capable model.}}
```

`d3-remote` 和 `kai` 是已声明的部署，这里分别是一个 System One API 和一个 Decision 模型。`harmful` 询问 `d3-remote`，`law` 和 `pick-model` 选择器询问 `kai`，`math` 询问默认部署 `primary`。同一类型、同一部署上的规则仍共享一个问题。

## 远程模型和挂接的模型 {#remote-and-attached-models}

连接块（`endpoint`、`model`、`auth`、`headers`、`timeout_ms`、`max_retries`、`tls`、`limits`）对每个远程服务和 Router 挂接的模型运行时都相同。[模型目录参考](model-catalog.md#examples)给出了远程 embedding 模型、远程重排序模型、远程 jailbreak 防护、使用自有认证请求头的 System One 网关、挂接的运行时以及请求后端的示例。

## Router 在加载时检查什么 {#what-the-router-checks-at-load}

Router 在加载配置时检查每个名称和每一处对应关系；不会静默回退。`vllm-sr config validate --config config.yaml` 运行相同的检查。

| 错误 | Router 报告的内容 |
| --- | --- |
| 绑定或规则指向未声明的部署 | 该绑定或规则、未知名称以及已声明的名称 |
| 部署无法回答分配给它的信号，例如 `bindings.pii: remote-embed`，或不提供 embedding 的 `embeddings.default` | 该信号、该部署以及它能回答什么 |
| 绑定键不是信号类型，例如 `bindings.piii` | 该键以及最接近的信号类型 |
| 部署无法回答规则提出的问题，例如 `domains` 规则指向一个 `classify` 部署 | 该规则、该部署以及能回答它的 provider |
| 在配置中直接写密钥，例如 `auth.api_key` 或 `Authorization` 请求头 | 密钥来自 `api_key_env` 或 `api_key_file` |
| `pii` 或 `hallucination` 规则所在的部署不提供片段 | `span_required`；需要这些片段的 masking 插件同样被拒绝 |

用早期形式写的配置仍可加载，每个字段报告一条 `deprecated_model_field` 警告。[迁移模型资源](model-resource-migration.md)将每种早期形式与本页的形式并列展示。
