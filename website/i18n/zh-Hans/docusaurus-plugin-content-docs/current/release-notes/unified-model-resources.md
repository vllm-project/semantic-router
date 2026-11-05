---
is_mtpe: false
translation:
  source_commit: "3e9e4262e5c4e286ede3fb75cdac83edb985b6aa"
  source_file: "docs/release-notes/unified-model-resources.md"
  outdated: false
---

# 发布说明：Router 询问的模型统一在一个目录中 {#release-note-one-catalog-for-the-models-the-router-asks}

Router 询问的每个模型（其信号、embedding 和重排序背后的模型）现在都在 `global.model_catalog.deployments` 下按名称声明一次。用到模型的地方都只写这个名称。`provider` 指明 Router 与每个部署通信所用的 API：`model_runtime`、`systemone`、`classify`、`openai_chat`、`openai_embeddings` 或 `rerank`。

```yaml
global:
  model_catalog:
    deployments:
      local-vela: {provider: model_runtime, artifact: vllm-sr/vela3-fast, device: cpu}
      d3-remote:
        provider: systemone
        endpoint: https://app.vllm-sr.ai/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: VSR_APP_KEY}
        timeout_ms: 1500
    system:
      decision_model: {deployment: d3-remote}
    bindings:
      pii: local-vela
```

这里 PII 在本地的 vela3-fast 上运行，其他每个问题类信号都询问远程 System One 模型。

## 变化 {#what-changes}

- **一条规则，三个层级。** 自己提问的规则（`decision`、`classifiers`、`safety` 及其 `hazard` 部分、`complexity`，以及决策选择器）写自己的 `deployment`。规则共享同一个回答的信号类型（`domain`、`jailbreak`、`pii`、`fact_check`、`user_feedback`、`modality`、`hallucination`、`preference`、`reask`）用 `global.model_catalog.bindings.<type>` 绑定，在配方中则用 `routing.model_bindings.<type>`。其余一切询问决策模型 `global.model_catalog.system.decision_model`。Embedding、重排序和 hazard 头在使用处指定。见[为信号选择模型](../installation/choose-a-model-for-a-signal.md)。
- **一个连接块。** 挂接的模型运行时和每个远程服务共用 `endpoint`（完整 URL）、`model`、`auth`（`api_key_env` 或 `api_key_file`；密钥从不写在配置中）、`headers`、`timeout_ms`、`max_retries`、`tls` 和 `limits`。`providers.models` 中的请求后端使用相同的 `endpoint`、`auth` 和 `headers`，原生 System One 别名可以指向一个 `systemone` 部署。见[模型目录参考](../installation/model-catalog.md)。
- **加载时检查。** 未声明的名称、无法回答所分配信号的部署、不是信号类型的绑定键，以及写错层级的部署，都会在加载配置时被拒绝，并给出原因。不会静默回退。
- **阈值跟随回答的模型。** 配置未写出的模块阈值来自回答的部署：内置模型的校准值；`systemone` 模型不发布阈值，使用 0.5；其他情况使用 Vela 1.0 的默认值。

## 行为变化 {#behavior-changes}

- `systemone` 决策模型现在也回答内置信号。以前这些信号在本地的 `vela3-fast` 上运行。除非配置写出阈值，它们的阈值为 0.5，加载时警告 `uncalibrated_preset`。
- 安全规则的 hazard 部分没有自己的模型时，在任何决策模型下都使用预先声明的 `hazard` 部署（Vela 1.0 Hazard）。以前，含有这种 hazard 部分的配置在 `systemone` 决策模型下无法加载。

其他一切都解析到与之前相同的模型。

## 早期配置 {#earlier-configurations}

用早期形式写的配置仍可加载，并使用相同的模型。Router 为每个早期字段报告一条 `deprecated_model_field` 警告，`vllm-sr config validate --config config.yaml` 会列出它们，`vllm-sr config migrate --config config.yaml` 会把它们改写为新形式。[迁移模型资源](../installation/model-resource-migration.md)将每种早期形式与取代它的形式并列展示。
