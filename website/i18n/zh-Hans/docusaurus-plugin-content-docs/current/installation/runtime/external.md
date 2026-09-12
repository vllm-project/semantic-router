---
title: 外部服务
description: 连接独立部署的分类、防护或嵌入服务。
translation:
  source_commit: "dc7f402642a8b8ecec8218e2086a4c6f186ea406"
  source_file: "docs/installation/runtime/external.md"
  outdated: false
---

模型在 Router 之外运行时，使用外部服务。Router 向其 API 发送输入，再将结果用于配置的路由策略。服务本身管理模型和硬件。

## 连接防护服务 {#connect-a-guard-service}

本例要求 HTTPS 服务接受 `POST /classify` 和 `{"inputs":"text"}`，并返回防护模型所配置标签的分数。将以下片段合入现有 `config.yaml`，并替换服务地址：

```yaml
global:
  model_catalog:
    external:
      - name: guard-service
        model_role: guardrail
        llm_endpoint:
          address: guard.example.com
          port: 443
          protocol: https
        llm_timeout_seconds: 5
        max_response_bytes: 1048576
    deployments:
      guard-http:
        provider: http
        external_model: guard-service
    modules:
      prompt_guard:
        enabled: true
        threshold: 0.7
        positive_labels: [INJECTION]
routing:
  model_bindings:
    prompt_guard:
      deployment: guard-http
      contract: label_distribution.v1
      adapter: http_classify
```

`guard-service` 命名 API 连接；`guard-http` 让配方的提示词防护使用该连接。要根据结果采取行动，还需按[安全模型](safety.md)配置越狱信号和决策。

运行 `vllm-sr config validate --config config.yaml`，然后重启或重载 Router。通过服务支持的凭据配置方式设置认证信息。服务会接收到待检查的文本。

## 选择服务类型 {#choose-a-service-type}

| 服务 | 配置 | 用途 |
| --- | --- | --- |
| Classify API | `adapter: http_classify` | 领域或自定义分类、提示词防护、PII、复杂度 |
| Chat API | `adapter: http_chat` | 提示词防护、幻觉检测、基于 LLM 的分类 |
| Embedding API | `backend: openai_compatible` | [远程文本嵌入](embeddings.md#remote-embeddings) |
| MCP 工具 | `modules.classifier.mcp` | 通过现有 MCP 服务器分类 |

示例使用返回分数的分类器。对于基于聊天模型的提示词防护，使用 `contract: label_decision.v1`，将 adapter 改为 `http_chat`，并在外部服务条目中设置 `llm_model_name`。服务必须返回支持的防护判定格式。通用分类器的响应格式见[分类器信号](/zh-Hans/docs/tutorials/signal/learned/classifier)。

Classify 请求只包含输入文本，不发送模型名称。不同的 classify 模型应使用不同的服务地址。Chat 和 embedding 请求会携带配置的模型名称。

## 服务要求 {#service-requirements}

- 分类结果应包含每个已配置标签及有效分数。缺失、重复或未知标签会导致推理错误。
- PII 应返回带分数的实体和有效文本偏移。幻觉检测则接收上下文、问题和回答，返回相对于回答的文本片段位置。
- 为服务设置请求超时和响应大小限制。HTTP 分类器应省略本地分词器的 `input` 配置，由外部服务限制 token 数。
- 目前没有用于事实核查、反馈、输出模态分类和 NLI 的外部适配器。

MCP 的传输方式、工具名称和超时在 `global.model_catalog.modules.classifier.mcp` 中配置，详见[配置参考](/zh-Hans/docs/api/configuration-schema)。

使用旧 `prompt_guard.protocol` 配置时，先运行 `vllm-sr config migrate --config config.yaml`；如果配置了多个外部服务，请明确选择目标服务。
