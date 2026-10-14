---
title: 模型、入口与服务
description: 连接推理后端，组合 Mixture-of-Model，并暴露稳定的 OpenAI 兼容模型名。
translation:
  source_commit: "6a4e51ad570a95c2a493ce4b1494bcedb525fa08"
  source_file: "docs/tutorials/global/models-entrypoints-serving.md"
  outdated: false
---

# 模型、入口与服务

## 概览

Semantic Router 为应用提供稳定的模型名，同时运维人员可以更改其背后的物理模型和路由策略。控制面板是到达可用拓扑的最快路径；YAML 仍可用于经过评审、受版本控制的部署。

## 解决什么问题？

应用应调用稳定的模型名，而不把自己耦合到某个提供商、端点或 checkpoint。入口保持该公开契约稳定，同时配方和已连接的模型可以独立演进。

该拓扑有四个面向用户的对象：

| 对象 | 表示什么 |
| --- | --- |
| 模型 | 一个逻辑模型，连接到一个或多个推理端点。 |
| 配方 | 可复用的信号、投影、决策、算法和插件。 |
| Mixture-of-Model | 已为其决策分配已连接模型的配方。 |
| 入口 | 解析到该 Mixture-of-Model 的一个或多个公开模型名。 |

```text
client model -> entrypoint -> recipe decision -> selected model -> inference endpoint
```

公开模型名是客户端契约。它不标识 checkpoint，也绝不会作为所选后端模型 ID 转发。

## 何时使用

当一个公开模型应在多个已连接模型之间路由，或必须在不更新每个客户端的情况下更改路由策略时，使用此工作流。单后端测试时，直接使用模型更简单。

## 构建模型路径 {#build-a-model-path}

### 1. 连接模型 {#1-connect-models}

启动堆栈并打开控制面板：

```bash
vllm-sr serve
vllm-sr dashboard
```

打开 **Build → Models**，选择提供商并输入端点凭据。对于兼容提供商，控制面板会发现可用的模型 ID，因此可以一步导入多个。仅在需要覆盖元数据、定价、连接行为，或选择内置推理家族时，才使用 **Advanced settings**。自定义内联推理契约时使用 **Manual setup**。将其分配给配方之前，请验证每个连接。YAML 与控制面板路径见[配置模型](../../installation/model-configuration)，支持的组合见[模型配置模式](../../installation/model-configuration-patterns)。

### 2. 选择配方 {#2-choose-a-recipe}

打开 **Build → Mixture-of-Models → Recipes**。配方描述路由逻辑，而不嵌入提供商 URL 或凭据。检查其决策和 probe，或从已维护的信号、投影和决策创建自定义配方。

### 3. 发布 Mixture-of-Model {#3-publish-a-mixture-of-model}

在 **Models** 中创建 Mixture-of-Model，选择配方，并为每个决策分配合格的已连接模型。当算法支持多个候选时，一个决策可以使用一个模型或有序列表。添加简洁的公开别名，并仅在拓扑和 probe 完成后发布。

### 4. 在 Playground 中测试 {#4-test-in-playground}

在 **Playground** 中选择新的公开模型并发送代表性请求。响应元数据会显示决策、算法、所选模型、延迟、TTFT 和 TPOT，而不会中断对话。更深入的路由 trace 和成本比较请使用 **Insights**。

### 5. 调用 OpenAI 兼容 API {#5-call-the-openai-compatible-api}

列出运行中堆栈暴露的公开模型名：

```bash
curl -sS http://localhost:8899/v1/models
```

然后通过标准 `model` 字段使用入口：

```bash
curl http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "vllm-sr/mom-v1-flash",
    "messages": [
      {"role": "user", "content": "Summarize the release notes."}
    ]
  }'
```

Router 解析入口，只评估其配方，选出合格 Model，并将上游请求改写为面向提供商的模型 ID。

## 配置

控制面板写入的模型、配方和入口契约，与 Router 从 YAML 读取的相同。交互式编写使用控制面板，经过评审的部署使用检入的 YAML；避免在两者之间拆分所有权。

## 运维堆栈 {#operate-the-stack}

```bash
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy -f
vllm-sr dashboard
vllm-sr stop
```

`vllm-sr serve --config my-models.yaml` 仍是使用经过评审的用户自有配置的显式路径。Kubernetes 部署通过 Helm 或 Operator 使用同一配置：

```bash
vllm-sr serve --target k8s --config my-models.yaml --namespace semantic-router
```

将提供商凭据放在环境绑定或 Kubernetes Secrets 中，而不是配方资产、ConfigMaps、shell 历史或已提交的 Helm values 中。

## 迁移自定义配方 {#move-a-custom-recipe}

打包经过评审的配方目录以便传输：

```bash
vllm-sr recipe pack path/to/custom-recipe
```

归档包含路由策略，不包含物理模型凭据或运行时依赖。在目标主机上按名称授权每个所需环境变量，并服务完整配置：

```bash
export PROVIDER_API_KEY=...
vllm-sr serve --config path/to/recipe/config.yaml \
  --recipe-env PROVIDER_API_KEY
```

对于较旧的配置，请在服务前显式迁移：

```bash
vllm-sr config migrate --config old-config.yaml
```

## 下一步 {#next}

- [配置模型](../../installation/model-configuration)：模型身份、Provider 绑定、自定义模型和推理。
- [虚拟模型](entrypoints-and-recipes)：请求解析与隔离。
- [入口](entrypoints)：命名与校验规则。
- [配方](recipes)：生命周期行为与限制。
- [Mixture of Models](../../overview/mom-model-family)：MoM 架构。
- [配置工作流](../../installation/configuration-workflows)：YAML、控制面板、Helm、Operator 和 DSL 所有权。
