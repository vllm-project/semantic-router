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

## GPT-6.2 Astra、GPT-6.2 Sol 和 GPT-6.1 Terra

将 `openai/gpt-6.2-astra`、`openai/gpt-6.2-sol` 或 `openai/gpt-6.1-terra` 绑定到 `openai` Provider。三者在 Chat Completions 和 Responses 上均支持 `low`、`medium`、`high`、`xhigh`、`max`。只有 Terra 支持关闭推理：设置 `use_reasoning: false` 会发送 `none`。Terra 默认使用 `medium`。工具调用需要 Responses（`api_format: responses`）。

目录为每张卡收录五条已发布的评测。Astra 和 Sol 的记录来自 `xhigh`，Terra 的记录来自 `high`；分数不会复制到其他档位。只有同一评测条件下可比模型数量足够时，结果才会进入 Model Hub 的公开对比。

## Qwen4 Max

`dashscope` 将 `qwen/qwen4-max` 映射到原生模型 ID `qwen4-max`，协议为 [Chat Completions](https://www.alibabacloud.com/help/en/model-studio/compatibility-of-openai-with-dashscope)。该绑定复用 `top_level_effort_boolean_switch`：开启推理的决策在请求顶层发送 `reasoning_effort` 和 `enable_thinking: true`。Model Studio 接受 `low`、`high` 和 `xhigh`，声明的默认值为 `high`。`minimal`、`medium` 和 `max` 是这些档位的提供方别名，不是目录中的独立档位。关闭推理的决策发送 `enable_thinking: false`，请求中不带 `reasoning_effort`，与阿里云的 [Chat Completions 示例](https://www.alibabacloud.com/help/en/model-studio/multi-round-conversation) 一致。

Provider 的默认主机仍是 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`，供已经映射到那里的模型使用。调用 Qwen4-Max 时，应将 `endpoint` 设为签发 API key 的地域所对应的工作空间主机。把 `{WorkspaceId}` 换成 Model Studio 控制台中的业务空间 ID：

| 地域 | `endpoint` |
| --- | --- |
| 中国（北京） | `https://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1` |
| 新加坡 | `https://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1` |
| 美国（弗吉尼亚） | `https://{WorkspaceId}.us-east-1.maas.aliyuncs.com/compatible-mode/v1` |
| 中国（香港） | `https://{WorkspaceId}.cn-hongkong.maas.aliyuncs.com/compatible-mode/v1` |
| 日本（东京） | `https://{WorkspaceId}.ap-northeast-1.maas.aliyuncs.com/compatible-mode/v1` |

```yaml
version: v0.5
providers:
  models:
    - name: qwen4
      catalog: qwen/qwen4-max
      backend_refs:
        - name: primary
          provider: dashscope
          endpoint: https://ws-example.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1
          auth: {api_key_env: DASHSCOPE_API_KEY}
recipes:
  - name: default
    decisions:
      - name: qwen4_xhigh
        priority: 1
        rules:
          operator: AND
          conditions: []
        modelRefs:
          - model: qwen4
            use_reasoning: true
            reasoning_effort: xhigh
```

该卡片使用已公布的 1,000,000 token 上下文窗口。思考模式接受 983,616 个输入 token 和 262,144 个输出 token，后者即卡片上的输出上限。五条导入的 [Artificial Analysis](https://artificialanalysis.ai/models/qwen4-max) 记录都是 `xhigh`：GDP.pdf All-pass、Terminal-Bench 4.0 pass@1、带 95% 区间的 AA-Briefcase v1.1 Elo、AA-Omniscience v1.1 的 Accuracy、Net Score 与 Hallucination Rate，以及覆盖全部 1,000 题的 HLE-Diamond pass@1。`low` 和 `high` 仍无测量。这些带版本的基准不替代 Intelligence 1.0，分数也不是 Router 本地复现的结果。

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
