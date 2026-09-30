---
title: 选择部署方式
description: 为部署 vLLM Semantic Router 选择 Docker、Kubernetes 或硬件专用路径。
translation:
  source_commit: "a565be11ad49666c149840c91134ad1e4678e49b"
  source_file: "docs/installation/deployment-options.md"
  outdated: false
---

# 选择部署方式

独立选择两件事：

1. Semantic Router 在哪里运行；以及
2. 模型后端在哪里运行。

Router 不会加载或预配自定义配置所引用的模型权重。它将请求发送到可到达的模型端点，这些端点可以运行在同一主机、集群中，或托管 API 之后。

## 选择 Router 拓扑

| 需求 | 推荐路径 | 从这里开始 |
| --- | --- | --- |
| 在本地评估或在单主机上运行 | 由 CLI 管理的 Docker 栈 | [使用 Docker 部署](docker) |
| 通过 GitOps 部署完整的 canonical 配置 | Helm | [CLI 与 Helm 工作流](configuration-workflows#helm) |
| 让 Kubernetes 协调 Router 资源和发现 | Kubernetes Operator | [Kubernetes Operator](k8s/operator) |
| 将路由策略附加到现有网关 | 网关集成 | [网关](k8s/gateways) |
| 让另一个平台拥有模型副本和调度 | 推理平台集成 | [推理平台](k8s/inference-platforms) |

网关和推理平台集成不会替代 Router 策略。它们将语义模型选择连接到拥有流量或模型生命周期的基础设施。

在确定路径之前，请在[部署支持](support-matrix)中检查其项目维护状态、周期性测试证据以及外部所有权边界。

## 选择模型后端

| 后端情况 | 从这里开始 |
| --- | --- |
| 已有可到达的模型或 provider 端点 | [协议兼容性](protocol-compatibility)，然后是[后端目标兼容性](backend-target-compatibility) |
| 需要一个用于评估的小型本地模型 | [使用 Ollama 的本地模型](ollama) |
| 希望在 AMD Instinct 上服务模型 | [AMD ROCm](amd-rocm) |
| 希望在 NVIDIA 上服务模型，或加速 Router 侧模型 | [NVIDIA CUDA](nvidia-cuda) |
| Kubernetes 平台拥有模型部署和副本 | [推理平台](k8s/inference-platforms) |

硬件是叠加层，而不是单独的 Router 拓扑。GPU 支持的模型服务器可以连接到 Docker 或 Kubernetes 的 Router 部署。除非测量表明本地嵌入或分类器能从 GPU 加速中受益，否则将 Router 保留在 CPU 上。

在通过 Router 发送流量之前，先直接测试模型端点。已配置的 URL 并不能证明后端实现了所选线协议，或支持配方的上下文、模态和工具要求。

## 投产之前

在对外暴露部署之前：

1. 将 Router、模型服务器、模型和集成版本一起固定；
2. 通过实际数据平面校验缓冲、流式、失败和回滚行为；
3. 将凭据移入密钥管理器；以及
4. 复核[配置](configuration)、[安全加固](security-hardening)、[数据与存储](storage-overview)和[升级与回滚](upgrade-rollback)。
