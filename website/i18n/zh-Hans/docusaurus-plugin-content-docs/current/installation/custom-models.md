---
title: 自定义模型
description: 为私有、自托管或新发布的模型配置可选元数据和同构后端副本。
translation:
  source_commit: "2b7519a84aec96963b02a3534e82908beba33f76"
  source_file: "docs/installation/custom-models.md"
  outdated: false
---

# 自定义模型

自定义 Model 是指 canonical 身份由你的部署拥有、而不是由内置目录拥有的模型。这包括私有 checkpoint、微调模型、新发布的模型，以及自托管别名。省略 `catalog` 即可保持 Model 为自定义。

## 绑定自定义模型

```yaml
version: v0.3

providers:
  defaults:
    model: private-chat
  models:
    - name: private-chat
      provider_model_id: private-chat-awq
      api_format: openai
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: model-gateway.example:8000
          protocol: http

routing:
  modelCards:
    - name: private-chat
      display_name: Private Chat
      context_window_size: 131072
      capabilities: [chat, tools]
      tags: [private, production]
```

对于自定义 Model，`routing.modelCards[].name` 必须等于 Model 别名。card 是可选的：当 Router 只需要后端绑定时可以省略。如果省略 `provider_model_id`，上游 ID 默认等于别名；当已服务的 checkpoint 使用另一个名称时，请显式设置。

Model Card 元数据还可以包含发布者、展示、分发和 LoRA。它不得包含凭据、价格或评估记录。价格放在 `providers.models[].pricing`，记录放在顶层 `evaluation.records[]`，凭据放在后端绑定中，最好通过 `api_key_env`。

## 添加评估证据

将内置基准的结果关联到自定义 Model Card 身份：

```yaml
evaluation:
  records:
    - model: private-chat
      benchmark: livecodebench/livecodebench@6.0.0
      benchmark_profile: independent-code-generation
      reasoning_effort: high
      metrics: {pass_at_1: 0.61}
      measured_at: 2026-09-09
      source: https://benchmarks.example/runs/private-chat-lcb6
```

结果会以精确的 reasoning effort 进入每个兼容的内置索引。对于组织专用基准，先在 `evaluation` 下声明其指标和任何索引。参见[自定义评估](../benchmarking/custom-evaluations)。

## 添加自定义推理

推理是可选的，支持两种形式：

```yaml
# 复用一个内置家族。
reasoning:
  family: qwen3
```

```yaml
# 当没有匹配的内置家族时，定义模型本地家族。
reasoning:
  type: reasoning_effort
  parameter: reasoning_effort
  levels: [low, medium, high]
  default: medium
  modes: [enabled, disabled]
  default_mode: enabled
  disabled: none
```

恰好选择一种形式。此前的全局自定义家族注册表不属于 canonical v0.3；内联定义位于使用它们的自定义 Model 上。字段语义和决策控制见[推理配置](model-reasoning)。

## 仅对相同上游契约使用副本

同一 Model 下的多个 `backend_refs` 会形成负载均衡的副本池：

```yaml
providers:
  models:
    - name: private-chat
      provider_model_id: private-chat-awq
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: model-a.example:8000
          protocol: http
          weight: 80
        - name: secondary
          provider: vllm
          endpoint: model-b.example:8000
          protocol: http
          weight: 20
```

这些副本必须保持同一个 Provider、线协议、原生模型 ID、凭据、标头、有效请求路径和 TLS 行为。在受支持的传输规则内，主机、端口和权重可以不同。当 Provider 或请求语义不同时，创建单独的 Model 别名，然后让路由决策在这些别名之间选择。

## 使用控制面板

打开 **Build → Models → Add Model**：

- 选择 Provider，并在连接流程中输入自定义模型 ID；或
- 选择 **Manual setup**，编辑完整的身份、推理、Model Card、定价、可靠性和后端引用字段。

保持 **Built-in Catalog Model** 为空。**Reasoning Family** 仅选择内置家族；当模型需要新契约时，使用 **Inline Reasoning** 字段。
