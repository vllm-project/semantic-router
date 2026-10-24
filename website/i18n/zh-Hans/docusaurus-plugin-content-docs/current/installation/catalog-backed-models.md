---
is_mtpe: false
title: 目录支持的模型
description: 将内置 Model Card 绑定到受支持的 Provider，而无需重复模型元数据或推理规则。
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
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

在 **Model Hub** 中浏览内置身份及其 Provider 支持，或在 **Build → Routing → Models → Add Model** 中选择 Provider。控制面板将目录选项标记为 **Built-in**，并同时保存本地别名和 canonical 身份。

## MiniMax M3.1 Flash

`minimax/minimax-m3.1-flash` 对应采用 MiniMax Community License 的[官方公开权重](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)。MiniMax API 与实验性的 vLLM 映射共用该模型卡，两条映射目前均声明支持 Chat Completions。

模型卡根据[固定版本的权重配置](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/resolve/76faee1b4ac0f412e547781172e94548a6526a88/config.json)记录 1,048,576 tokens 上下文窗口；MiniMax API 映射单独记录该服务的 1,000,000 tokens 限制。

```yaml
version: v0.3
providers:
  models:
    - name: flash
      catalog: minimax/minimax-m3.1-flash
      backend_refs:
        - name: primary
          provider: minimax
          api_key_env: MINIMAX_API_KEY
routing:
  decisions:
    - name: flash_default
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: flash
          use_reasoning: true
          reasoning_effort: high
```

模型采用 adaptive thinking，支持 `low`、`medium`、`high`、`xhigh`、`max` 五档，目录默认档位为 `max`。关闭推理、选择 `reasoning_mode: enabled` 或设置 `reasoning_effort: none` 会被拒绝。Router 向 MiniMax 发送顶层 effort 字段。[官方 API](https://platform.minimax.io/docs/api-reference/text-openai-api) 支持 1,000,000 tokens 上下文，以及包含思考内容在内的最多 524,288 output tokens。API 忽略 `temperature` 和 `top_p` 的采样调节，固定使用 1 和 0.95；不支持 `service_tier`。工具调用对话回传 assistant 历史时，应保留 `reasoning_content` 和 tool-call ID。

已有自部署服务时，将上述 backend 替换为：

```yaml
        - name: primary
          provider: vllm
          base_url: http://127.0.0.1:8000/v1
```

该映射解析到 `MiniMaxAI/MiniMax-M3.1-Flash`，将 effort 放入 `chat_template_kwargs`。请使用 [vLLM 配方](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash)：本次核对的版本要求 vLLM 0.33.0 nightly、专用 `minimax-m31-flash` 镜像、`--block-size 128` 和 `minimax_m3` reasoning/tool parser，普通稳定版 pip 安装不足以部署。Hopper 示例将上下文限制为 262,144 并关闭 video；应在 canonical 模型卡 override 中声明实际提供的窗口和能力。映射为实验性，不代表任何具体部署已通过验证。

五条已发布评测记录来自固定版本的官方模型卡，推理档位为 `unspecified`，属于厂商声明而非 vLLM-SR 复测。Terminal-Bench 4.0 与 2.1 分开记录，各可选 effort 档位尚无完整 Intelligence 1.0 证据。

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
