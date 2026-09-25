---
title: 部署支持
description: 查看 Semantic Router 项目维护哪些部署路径、集成、示例和硬件配置。
translation:
  source_commit: "d8e75b89b7290df941743270c69a111f80dde50a"
  source_file: "docs/installation/support-matrix.md"
  outdated: false
---

# 部署支持

使用本页检查项目维护哪些 Router 部署路径和集成。它并不认证每一种平台、模型服务器、模型或加速器组合。

如果仍在选择拓扑，从[选择部署方式](deployment-options)开始。对于线格式和端点配置，使用[协议兼容性](protocol-compatibility)和[后端目标兼容性](backend-target-compatibility)。

## 每种状态的含义

| 状态 | 含义 |
| --- | --- |
| Maintained reference stack | 项目拥有并测试安装或生命周期契约。 |
| Supported integration | 项目测试 Router 侧连接；外部平台拥有其生命周期。 |
| Experimental example | 这些文件演示你必须自行限定的功能或测试拓扑。 |
| Deprecated | 该选项暂时保留，用于已记录的迁移。 |

证据标签显示最强的周期性检查：**PR CI** 运行端到端配置，**Contract** 在没有外部平台的情况下校验静态契约，**Manual** 需要可选环境。

## 使用一套经过测试的版本

对于每个链接的选项：

1. 使用其指南中命名的外部平台版本；如果没有命名，在你测试之前将你的选择视为未限定；
2. 从同一发行版获取你使用的每个 Semantic Router 产物——CLI、chart、CRD、控制器和镜像；以及
3. 在升级任何一个组件之前，先在你的环境中测试该精确集合。

支持覆盖的是该经过测试的版本集合，而不是外部项目支持的每个版本。

## 维护中的参考栈

| 选项 | 分类 | 项目覆盖 |
| --- | --- | --- |
| [Helm chart](configuration-workflows#helm) | Maintained reference stack | **Contract。** Router、可选控制面板、ingress、自动扩缩、持久化和可观测性资源。网关和存储仍是外部的。 |
| [本地部署](docker) | Maintained reference stack | **PR CI。** CLI 管理 Router、Envoy、控制面板和支持服务。你提供自定义模型端点，并加固本地默认值。 |
| [Kubernetes Operator](k8s/operator) | Maintained reference stack | **PR CI + Contract。** 项目拥有 CRD、协调、Router 工作负载、Service 和路由 API。Kubernetes 调度工作负载；你的网关承载流量。 |

## 受支持的集成

| 集成 | 分类 | 项目覆盖 |
| --- | --- | --- |
| [agentgateway](k8s/agentgateway) | Supported integration | **PR CI。** Router 提供 ExtProc 策略；agentgateway 拥有数据平面。将请求正文设置为 `FullDuplexStreamed`。 |
| [Envoy AI Gateway](k8s/ai-gateway) | Supported integration | **PR CI。** Router 提供路由策略；网关拥有 provider 流量。单独验证你的 provider。 |
| [AIBrix](k8s/aibrix) | Supported integration | **PR CI。** Router 选择模型或池；AIBrix 拥有部署、自动扩缩和副本。使用 AIBrix 的硬件支持矩阵。 |
| [NVIDIA Dynamo](k8s/dynamo) | Supported integration | **Manual。** Router 选择目标；Dynamo 拥有图、worker 和 frontend。使用指南版本；较旧的 fixture 仅用于测试。 |
| [Istio Gateway](k8s/istio) | Supported integration | **PR CI。** Router 提供 ExtProc 策略；Istio 承载请求。附带的 GPU 工作负载仅为示例。 |
| [llm-d](k8s/llm-d) | Supported integration | **PR CI + Contract。** Router 选择模型或池；llm-d 拥有发现和副本路由。不要添加竞争的直接 Service 路由。 |
| [使用 Envoy AI Gateway 的流式](k8s/streamed-extproc) | Supported integration | **PR CI。** 网关流式传输；Router 使用配置的 ExtProc 正文模式。显式测试该模式。 |
| [Valkey 智能体记忆](valkey-memory) | Supported integration | **Contract + Manual。** Router 拥有记忆行为；Valkey 拥有持久化和 Search。你拥有安全、保留和备份。 |
| [使用 Redis 的 Responses API 状态](../tutorials/global/api-and-observability#response-api) | Supported integration | **Manual。** Router 拥有 Responses 行为；Redis 存储状态。你拥有 Redis 安全、持久化和驱逐。 |
| [响应缓存](../tutorials/plugin/response-cache) | Supported integration | **Contract + Manual。** Router 拥有缓存行为；你的后端拥有存储和可用性。将缓存数据视为敏感。 |
| [Valkey 向量存储](storage-overview) | Supported integration | **Manual。** Router 拥有存储引用；Valkey 拥有索引和持久性。将 Valkey、Search 和嵌入模型一起固定。 |

## 实验性示例

| 示例 | 分类 | 适用于 / 不适用于 |
| --- | --- | --- |
| KServe 示例 | Experimental example | KServe 集成冒烟测试；不是经过限定的 KServe 或模型服务部署。 |
| OpenShift 示例 | Experimental example | 将资源适配到 Route 和安全约束；不是加固的 OpenShift 配置。 |
| 幻觉策略演示 | Experimental example | 事实核查策略行为；不是经过限定的护栏或模型。 |
| 越狱错误处理演示 | Experimental example | 分类器失败路径；不是安全的生产策略。 |
| Provider mocker 与可选 tiny-model smoke | Experimental example | 确定性协议测试数据；需要真实推理时，使用上游 llama.cpp server 和固定版本的 Qwen3-0.6B。 |
| PII 远程后端演示 | Experimental example | 远程 token_spans.v1 PII 后端及其 on_error 策略；不是经过限定的 PII 模型或脱敏策略。 |
| 可观测性演示 | Experimental example | Prometheus、Grafana、告警和控制面板接线；替换所有示例安全和保留设置。 |
| 响应越狱演示 | Experimental example | 响应分类器窗口行为；不是生产护栏模型。 |
| Responses API Kubernetes 演示 | Experimental example | Redis 持久化和重启行为；不是加固的 Redis 部署。 |
| 路由动作演示 | Experimental example | 聚焦的路由动作行为；在生产前将其组合到维护中的栈。 |
| Router 回放恢复演示 | Experimental example | 回放和重启恢复；不是受支持的部署平台。 |
| 路由策略演示 | Experimental example | 聚焦的策略示例；用有代表性的流量限定路由质量和后端容量。 |
| 本地工具数据库 | Experimental example | 用于示例和测试的本地工具定义；在生产中使用经过认证的持久注册表。 |

当前没有随附选项处于 **Deprecated** 状态。迁移指导见[升级与回滚](upgrade-rollback)。

## 硬件叠加层

硬件支持适用于部署栈；它不是单独的 Router 拓扑。Router 加速和后端模型服务也是分开的选择。

| 硬件配置 | 状态 | 覆盖内容 |
| --- | --- | --- |
| Linux x86-64 CPU | Maintained | 标准 Router 镜像、CLI、Helm chart 和 Operator 获得最广泛的周期性覆盖。模型服务器要求仍然分开。 |
| Linux Arm64 CPU | Build-qualified | 发行工作流在声明处发布多架构镜像。这并不限定 Arm64 上的每个集成或可选原生依赖。 |
| Linux x86-64 上的 NVIDIA CUDA | Supported integration | 遵循 [NVIDIA CUDA](nvidia-cuda) 了解受支持的 Router 侧模型，或将 Router 保留在 CPU 上，并对照 vLLM 的支持矩阵限定单独的 NVIDIA 后端。 |
| Linux x86-64 上的 AMD ROCm | Supported integration | 遵循 [AMD ROCm](amd-rocm) 了解受支持的 Router 侧模型，并将 Router、vLLM、ROCm 和模型 revision 作为一套进行限定。 |
| AMD AI PC/NPU | Experimental, not qualified | 尚无维护中的部署契约；由 [issue #2373](https://github.com/vllm-project/semantic-router/issues/2373) 跟踪。 |
| NVIDIA DGX Spark Arm64 | Experimental, not qualified | Arm64 镜像并不限定此平台上的 CUDA 或端到端推理；由 [issue #2374](https://github.com/vllm-project/semantic-router/issues/2374) 跟踪。 |
| 其他加速器和操作系统 | Not qualified | 没有维护中的配置。用可复现的硬件、软件、镜像和测试证据开一个限定 issue。 |

## 投产之前

先直接测试模型端点，然后通过真实数据平面演练缓冲、流式、失败、升级和回滚路径。复核[安全加固](security-hardening)、[数据与存储](storage-overview)和[升级与回滚](upgrade-rollback)，了解此矩阵之外的控制项。
