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
version: v0.5
providers:
  defaults:
    model: production
  models:
    - name: production
      catalog: openai/gpt-5.6-sol
      backend_refs:
        - name: openai-primary
          provider: openai
          auth: {api_key_env: OPENAI_API_KEY}
recipes:
  - name: default
```

`production` 是本地别名。Router 会从 `catalog: openai/gpt-5.6-sol` 解析 canonical card 和 OpenAI 专用映射；不要在 YAML 中重复其推理定义。

在 **Models → Model Hub** 中浏览内置身份及其 Provider 支持，或在 **Models → Backend models → Connect model** 中选择 Provider。控制面板将目录选项标记为 **Built-in**，并同时保存本地别名和 canonical 身份。

## Gemini 4

`gemini` Provider 通过 [Chat Completions](https://aistudio.google.com/docs/openai#thinking)
将以下规范模型卡映射到 Google 的精确 API ID：

| Catalog ID | API 模型 ID | 推理档位 | 默认值 | 已收录 AA 成绩的档位 |
| --- | --- | --- | --- | --- |
| `google/gemini-4-argon` | `gemini-4-argon` | `low`、`medium`、`high` | `medium` | `high` |
| `google/gemini-4-carbon` | `gemini-4-carbon` | `low`、`medium`、`high` | `medium` | `high` |
| `google/gemini-4-hydrogen` | `gemini-4-hydrogen` | `low`、`medium`、`high`、`xhigh` | `high` | `xhigh` |
| `google/gemini-4-neon` | `gemini-4-neon` | `low`、`medium`、`high`、`xhigh` | `high` | `xhigh` |

```yaml
version: v0.5
providers:
  models:
    - name: neon
      catalog: google/gemini-4-neon
      backend_refs:
        - name: primary
          provider: gemini
          auth: {api_key_env: GEMINI_API_KEY}
recipes:
  - name: default
    decisions:
      - name: neon_deep
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: neon
            use_reasoning: true
            reasoning_effort: xhigh
```

Router 将选定档位作为顶层 `reasoning_effort` 发送。
Gemini 4 [不支持关闭 thinking](https://aistudio.google.com/docs/thinking#thinking-levels)。
目录仅暴露原生档位；兼容 API 中映射到 `low` 的 `minimal` 别名不是独立可选档位。
模型卡记录统一的 1,048,576 token 上下文和 65,536 token 输出上限。
Argon/Carbon 的单次前台非流式调用上限为 32,768 tokens；流式调用及 Hydrogen/Neon
支持 65,536 tokens。详见 Google 的[上下文与输出限制](https://aistudio.google.com/docs/thinking#token-limits)。
long decode 是单独的 API 功能。

每张卡在上表指定档位收录五项 [Artificial Analysis](https://artificialanalysis.ai/methodology/intelligence-benchmarking)
公开评测：GDP.pdf All-pass、Terminal-Bench 4.0 pass@1、AA-Briefcase v1.1 Elo
（含 95% 区间）、AA-Omniscience v1.1 Accuracy、Net Score 与 Hallucination Rate，以及完整 1,000 题
（含 272 道图像题）、无工具的 HLE-Diamond pass@1。记录保留原始指标、harness 和查阅日期。
这些独立版本不替换 Intelligence 1.0 的组成项，其他档位继续保留证据缺失状态；
公开成绩属于导入的第三方评测，不是 Router 本地复测。

## MiniMax M3.1 Flash

`minimax/minimax-m3.1-flash` 对应采用 MiniMax Community License 的[官方公开权重](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/tree/262ae6830934f3512211081b8a3f16425b93a239)。MiniMax API 与实验性的 vLLM 映射共用该模型卡，两条映射目前均声明支持 Chat Completions。

模型卡根据[固定版本的权重配置](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/resolve/262ae6830934f3512211081b8a3f16425b93a239/config.json)记录 1,048,576 tokens 上下文窗口；MiniMax API 映射单独记录该服务的 1,000,000 tokens 限制。

```yaml
version: v0.5
providers:
  models:
    - name: flash
      catalog: minimax/minimax-m3.1-flash
      backend_refs:
        - name: primary
          provider: minimax
          auth: {api_key_env: MINIMAX_API_KEY}
recipes:
  - name: default
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
          endpoint: http://127.0.0.1:8000/v1
```

该映射解析到 `MiniMaxAI/MiniMax-M3.1-Flash`，将 effort 放入 `chat_template_kwargs`。请使用 [vLLM 配方](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash)：本次核对的版本要求 vLLM 0.33.0 nightly、专用 `minimax-m31-flash` 镜像、`--block-size 128` 和 `minimax_m3` reasoning/tool parser，普通稳定版 pip 安装不足以部署。Hopper 示例将上下文限制为 262,144 并关闭 video；应按下文所述，在别名上声明实际提供的窗口和能力。映射为实验性，不代表任何具体部署已通过验证。

五条已发布评测记录来自[固定版本的官方模型卡](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/262ae6830934f3512211081b8a3f16425b93a239/README.md)，推理档位为 `unspecified`，属于厂商声明而非 vLLM-SR 复测。Terminal-Bench 4.0 与 2.1 分开记录，各可选 effort 档位尚无完整 Intelligence 1.0 证据。

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
          endpoint: https://example.services.ai.azure.com
          auth: {api_key_env: FOUNDRY_API_KEY}
```

仅当所选映射需要时，控制面板才会显示 **Deployment name** 或 **Provider model ID**。它是上游标识符，而不是第二个目录身份，因此映射中的定价、推理传输和 effort 限制仍然适用。自部署服务以其他名称提供该模型时（例如 vLLM 的 `--served-model-name`），也以同样方式设置它。

## 仅覆盖运维人员拥有的元数据

你可以设置本地标签、描述、更窄的能力列表或实际提供的上限，而无需分叉内置 card。请在目录支持的别名上设置它们；每个值只替换该别名对应的 card 值：

```yaml
version: v0.5
providers:
  models:
    - name: production
      catalog: openai/gpt-5.6-sol
      tags: [production, approved]
      description: Approved for production reasoning traffic.
      backend_refs:
        - name: openai-primary
          provider: openai
          auth: {api_key_env: OPENAI_API_KEY}
recipes:
  - name: default
```

以目录 ID 命名的单独 `providers.models` 条目（例如 `name: openai/gpt-5.6-sol`）不会作用于该别名：与绑定该 card 的别名同时存在时会因歧义被拒绝，单独存在时则声明了一个没有后端的独立 Model。

不要在目录支持的 Model 上设置 `providers.models[].reasoning`。该组合会被拒绝，以免别名悄悄更改仓库拥有的推理契约。

## 仅在必要时覆盖协议

所选 Provider 映射通常会提供正确的协议和请求路径。仅当后端有意暴露另一种兼容线格式时，才设置 `api_format: openai`、`responses` 或 `anthropic`。该字段不会选择 Provider，也不会替代 `backend_refs[].provider`。

Provider 专用的推理模式和 effort 限制会对照有效的 Model 与 Provider 映射进行校验。在添加决策级控制之前，请先阅读[推理配置](model-reasoning)。
