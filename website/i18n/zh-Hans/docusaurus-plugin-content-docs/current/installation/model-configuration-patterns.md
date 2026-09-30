---
title: 模型配置模式
description: 在编写 YAML 之前，比较受支持的目录、自定义、推理、provider 和副本组合。
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/installation/model-configuration-patterns.md"
  outdated: false
---

# 模型配置模式

使用此矩阵选择最小的有效 Model 声明。所有示例都是 `providers.models` 下的条目。

## 受支持的组合

| 目标 | `catalog` | `reasoning` | `provider_model_id` | 后端 |
| --- | --- | --- | --- | --- |
| 通过列出该模型的 Provider 使用目录模型 | Canonical Model Card ID | 省略；继承 | 通常省略；由 Provider 支持提供 | 设置 Provider ID。 |
| 通过运维人员命名的部署使用目录模型 | Canonical Model Card ID | 省略；继承 | 设置部署名称 | 设置部署 Provider。 |
| 使用不带 Router 生成推理控制的私有模型 | 省略 | 省略 | 与 `name` 不同时设置 | 设置 Provider 和端点。 |
| 使用已知内置家族的私有模型 | 省略 | `family: <built-in-id>` | 可选 | 设置兼容的 Provider 和端点。 |
| 使用具有独特推理控制的私有模型 | 省略 | 完整内联定义 | 可选 | 设置兼容的 Provider 和端点。 |
| 为私有模型添加路由元数据 | 省略 | 可选 | 可选 | 在别名下添加 `routing.modelCards`。 |
| 为目录模型添加本地元数据 | Canonical Model Card ID | 省略；继承 | 取决于 Provider | 在目录 ID 下添加 `routing.modelCards`。 |
| 对相同副本做负载均衡 | 任一 | 由模型身份决定 | 一个有效 ID | 将同构目标放入同一个 `backend_refs` 列表。 |
| 跨不同 Provider 或协议路由 | 每个契约配置一个 Model | 按 Model 决定 | 按 Provider 决定 | 在同一决策中使用单独别名。 |

## 带自动映射的目录模型

```yaml
- name: production
  catalog: openai/gpt-5.6-sol
  backend_refs:
    - provider: openai
      api_key_env: OPENAI_API_KEY
```

## 带运维部署名称的目录模型

```yaml
- name: team-reasoner
  catalog: microsoft/mai-thinking-1
  provider_model_id: my-production-deployment
  backend_refs:
    - provider: microsoft-foundry
      base_url: https://example.services.ai.azure.com
      api_key_env: FOUNDRY_API_KEY
```

## 自定义透传模型

```yaml
- name: local-chat
  provider_model_id: served-chat
  api_format: openai
  backend_refs:
    - provider: vllm
      endpoint: host.docker.internal:8000
      protocol: http
```

## 带内置家族的自定义模型

```yaml
- name: local-qwen
  provider_model_id: served-qwen
  reasoning:
    family: qwen3
  backend_refs:
    - provider: vllm
      endpoint: host.docker.internal:8000
      protocol: http
```

## 带内联推理的自定义模型

```yaml
- name: private-reasoner
  reasoning:
    type: reasoning_effort
    parameter: reasoning_effort
    levels: [low, medium, high]
    default: medium
    modes: [enabled, disabled]
    default_mode: enabled
    disabled: none
  backend_refs:
    - provider: vllm
      endpoint: host.docker.internal:8000
      protocol: http
```

## 校验会拒绝的组合

- `catalog` 与任何 `providers.models[].reasoning` 块同时出现；
- `reasoning.family` 与内联推理字段同时出现；
- 内联定义缺少 `type` 或 `parameter`；
- Model Card 覆盖以目录支持别名命名，而不是其 canonical `catalog` ID；
- 自定义 Model Card 的名称与自定义 Model 别名不同；
- 决策的 effort 或 mode 不被 Model 家族或 Provider 映射支持；以及
- 同一副本池内存在异构的 Provider、协议、凭据、原生 ID、路径或 TLS 语义。

组合模式后运行 `vllm-sr config validate --config config.yaml`。身份模型见[配置模型](model-configuration)，完整推理字段见[推理配置](model-reasoning)。
