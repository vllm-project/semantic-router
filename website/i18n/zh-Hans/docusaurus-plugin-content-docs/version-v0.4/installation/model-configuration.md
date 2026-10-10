---
title: 配置模型
description: 选择目录支持或自定义模型，将其绑定到 provider，并从路由决策中使用它。
translation:
  source_commit: "f8c1197a9ed47f7a265cbab83bf2d84eb5fa505e"
  source_file: "docs/installation/model-configuration.md"
  outdated: false
---

# 配置模型

已配置的 Model 将稳定的 Router 名称连接到一个模型身份和一个或多个物理后端。在 `providers.models` 下配置 Model，然后从 `providers.defaults.model` 和 `routing.decisions[].modelRefs[].model` 引用其 `name`。

## 理解模型标识符

这些字段回答不同的问题，不可互换：

| 字段 | 含义 |
| --- | --- |
| `name` | 默认值、决策和直接模型调用使用的 Router 本地别名。 |
| `catalog` | 来自内置目录的可选 canonical Model Card ID。私有或新发布的模型请省略。 |
| `provider_model_id` | 发送给上游 provider 的可选模型或部署名称。 |
| `backend_refs[].provider` | 稳定的 Provider 契约，提供端点、身份验证、路径、协议和模型映射默认值。 |
| `api_format` | 可选的线协议覆盖：`openai`、`responses` 或 `anthropic`。它从不选择 Provider。 |

例如，`production-reasoner` 可以是你的 Router 别名，`openai/gpt-5.6-sol` 是其目录身份，而 provider 专用的部署名称是其 `provider_model_id`。

## 选择配置路径

| 起点 | 配置 | 推理行为 |
| --- | --- | --- |
| 模型存在于目录中 | 在每个后端上设置 `catalog` 和 Provider。 | 从目录和所选 Provider 映射继承。 |
| 具有已知内置家族的私有或未列出模型 | 省略 `catalog`；设置 `reasoning.family`。 | 复用内置家族契约。 |
| 具有独特控制的私有或未列出模型 | 省略 `catalog`；内联定义 `reasoning`。 | 使用你定义的模型本地契约。 |
| 自定义透传模型 | 同时省略 `catalog` 和 `reasoning`。 | Router 不会合成推理控制。 |

各路径见[目录支持的模型](catalog-backed-models)、[自定义模型](custom-models)和[推理配置](model-reasoning)。[模型配置模式](model-configuration-patterns)展示受支持的组合和无效混合。

## 从一个自定义模型开始

```yaml
version: v0.3

providers:
  defaults:
    model: local-chat
  models:
    - name: local-chat
      provider_model_id: served-model
      backend_refs:
        - name: primary
          provider: vllm
          endpoint: host.docker.internal:8000
          protocol: http

routing: {}
```

Router 会为 `local-chat` 创建稀疏的本地 Model Card。仅当路由需要上下文窗口、能力、标签或 LoRA 等元数据时，才添加匹配的 `routing.modelCards` 条目。在顶层 `evaluation.records[]` 下独立添加基准测量。

## 在分发前检查任务能力

所选模型必须同时满足 provider 协议的能力和其声明的任务能力。当 Router 从匹配决策中选择回退时，同样的检查也适用。回退不能重新引入已被请求的上下文窗口检查排除的模型。

Model Card 能力元数据接受 `image_input` 和 `image_generation` 等协议名称。目录别名 `vision`、`audio` 和 `video` 分别描述图像、音频和视频**输入**；它们不授予媒体生成支持。`structured_output` 映射到 `structured_json`，`tool_use` 映射到 `tools`。`long_context` 或 `coding` 等描述性标签不会抹去与它们并列的已识别能力声明。

没有任何已识别能力声明的模型仅保留协议兼容性。对于已标注的模型，能够编码请求的协议不会覆盖缺失的任务能力。当没有符合条件的决策模型能够服务所请求的任务时，Router 返回 `unsupported_capability`。

## 在控制面板中配置模型

打开 **Build → Models → Add Model**。然后你可以：

1. 选择 Provider，连接它，并选择已发现或内置的模型 ID。
2. 当 Provider 无法列出模型时，输入模型 ID。
3. 使用 **Advanced settings** 设置名称前缀、内置推理家族、路由元数据、定价和交付设置。
4. 从 Provider 选择器使用 **Manual setup**，以完全控制 `catalog`、内联推理字段、`provider_model_id`、`api_format` 和后端引用。

控制面板编辑的是与 YAML 相同的 v0.3 文档。为部署使用一个编写所有者，并在 serve 之前复核导出的 YAML。参见[配置工作流](configuration-workflows)。

## 校验结果

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

校验会在启动前解析目录条目、Provider 映射、推理控制、Model Card 身份和后端池兼容性。

## 配置 Router 任务使用的模型

对于 Router 内部使用的分类器、安全检查和嵌入，从 [Router Runtime](native-backends) 开始。它覆盖进程内和外部模型、其配置以及运维。
