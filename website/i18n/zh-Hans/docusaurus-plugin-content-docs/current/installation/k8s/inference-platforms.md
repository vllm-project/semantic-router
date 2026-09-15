---
title: 推理平台
description: 将语义模型选择放在负责模型部署和副本调度的平台之上。
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/installation/k8s/inference-platforms.md"
  outdated: false
---

# 推理平台

推理平台和 Semantic Router 解决不同的路由问题。Semantic Router 根据请求含义和策略选择模型或模型池。推理平台部署该模型，并根据容量、局部性和健康状况选择副本。

## 选择集成

| 平台 | 从此开始 | 典型所有权 |
| --- | --- | --- |
| vLLM Production Stack | [Production Stack](production-stack) | vLLM 模型服务、发现和副本路由。 |
| AIBrix | [AIBrix](aibrix) | 模型部署、自动扩缩容和副本级流量管理。 |
| llm-d | [llm-d](llm-d) | `InferencePool` 端点选择和分布式推理模式。 |
| NVIDIA Dynamo | [Dynamo](dynamo) | Dynamo 图、worker 和前端生命周期。 |

使用基础设施团队已经支持的平台。这些指南不替代平台特定版本的安装、容量规划或升级文档。

## 保持两层对齐

对每个模型池，对齐：

- Semantic Router 提供商名称和 `provider_model_id`；
- 平台所服务的模型标识；
- 稳定的 Service、Gateway 或前端地址；以及
- 路由策略声明的模态、上下文、工具和协议能力。

使用 Service DNS 或托管网关地址，而不是 Kubernetes `ClusterIP`。将副本调度排除在语义策略之外，也将提示词含义排除在副本调度器之外。

## 验证完整路径

1. 验证每个模型池的直接生成。
2. 校验规范 Router 配置。
3. 通过每个公开虚拟模型验证请求。
4. 同时确认语义选择和正在服务的副本。
5. 明确演练模型不可用的行为；除非你已配置并测试跨模型回退，不要假定 Router 或平台会提供该能力。

周围的部署选择见[选择部署方式](../deployment-options)。
