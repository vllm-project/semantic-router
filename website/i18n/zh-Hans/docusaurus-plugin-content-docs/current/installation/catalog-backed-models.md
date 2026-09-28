---
title: 目录支持的模型
description: 将内置 Model Card 绑定到受支持的 Provider，而无需重复模型元数据或推理规则。
translation:
  source_commit: "6a4e51ad570a95c2a493ce4b1494bcedb525fa08"
  source_file: "docs/installation/catalog-backed-models.md"
  outdated: false
---

# 目录支持的模型

当内置目录已经描述了该模型时，使用目录支持的 Model。目录拥有 canonical Model Card 和推理家族。Provider 映射补充上游模型 ID、支持的协议、推理传输、定价，以及该 Provider 已知的任何限制。

## 配置最小绑定

```yaml
version: v0.3

providers:
  defaults:
    model: production
  models:
    - name: production
      catalog: openai/gpt-5.6-sol
      backend_refs:
        - name: openai-primary
          provider: openai
          api_key_env: OPENAI_API_KEY

routing: {}
```

`production` 是本地别名。Router 会从 `catalog: openai/gpt-5.6-sol` 解析 canonical card 和 OpenAI 专用映射；不要在 YAML 中重复其推理定义。

在 **Model Hub** 中浏览内置身份及其 Provider 支持，或在 **Build → Models → Add Model** 中选择 Provider。控制面板将目录选项标记为 **Built-in**，并同时保存本地别名和 canonical 身份。

## 在需要时提供部署名称

有些 Provider 会将目录模型映射到固定的上游 ID。另一些（例如托管部署服务）则要求你在部署模型时指定的名称。此时设置 `provider_model_id`：

```yaml
providers:
  models:
    - name: team-reasoner
      catalog: microsoft/mai-thinking-1
      provider_model_id: my-production-deployment
      backend_refs:
        - provider: microsoft-foundry
          base_url: https://example.services.ai.azure.com
          api_key_env: FOUNDRY_API_KEY
```

仅当所选映射需要时，控制面板才会显示 **Deployment name** 或 **Provider model ID**。它是上游标识符，而不是第二个目录身份。

## 仅覆盖运维人员拥有的元数据

你可以添加本地标签、描述、能力或评估，而无需分叉内置 card。覆盖必须使用 canonical `catalog` ID，而不是本地别名：

```yaml
routing:
  modelCards:
    - name: openai/gpt-5.6-sol
      tags: [production, approved]
      description: Approved for production reasoning traffic.
```

不要在目录支持的 Model 上设置 `providers.models[].reasoning`。该组合会被拒绝，以免别名悄悄更改仓库拥有的推理契约。

## 仅在必要时覆盖协议

所选 Provider 映射通常会提供正确的协议和请求路径。仅当后端有意暴露另一种兼容线格式时，才设置 `api_format: openai`、`responses` 或 `anthropic`。该字段不会选择 Provider，也不会替代 `backend_refs[].provider`。

Provider 专用的推理模式和 effort 限制会对照有效的 Model 与 Provider 映射进行校验。在添加决策级控制之前，请先阅读[推理配置](model-reasoning)。
