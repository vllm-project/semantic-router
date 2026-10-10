---
title: Kubernetes 网关
description: 选择 Kubernetes 网关如何通过 Envoy ExtProc 调用 Semantic Router。
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/installation/k8s/gateways.md"
  outdated: false
---

# Kubernetes 网关

Semantic Router 可以运行在多种基于 Envoy 的 Kubernetes 网关后面。路由策略保持不变；网关特定资源决定 ExtProc 何时运行、所选模型如何到达后端，以及哪一组件负责认证或流量策略。

## 选择集成

| 现有数据面 | 从此开始 | 它负责什么 |
| --- | --- | --- |
| Envoy AI Gateway | [Envoy AI Gateway](ai-gateway) | 提供商转换、提供商凭证、速率限制和 Gateway API 流量策略。 |
| agentgateway | [agentgateway](agentgateway) | Gateway API 代理、后端资源和 ExtProc 阶段策略。 |
| Istio | [Istio Gateway](istio) | Ingress、`HTTPRoute` 处理，以及调用 Semantic Router 的 Envoy 过滤器。 |
| Gateway API Inference Extension | [GIE](gateway-api-inference-extension) | Semantic Router 选择模型池之后的 `InferencePool` 端点选择。 |

使用平台已经运维的网关。不要仅为获得语义路由而安装第二个网关，除非你已经比较过所有权、安全策略和升级要求。

## 共享约定

所有集成必须在以下方面一致：

1. 客户端请求的公开模型或入口；
2. Semantic Router 写入的模型名称；
3. 该模型对应的 Gateway API 匹配或提供商后端；以及
4. 推理端点接受的已服务模型标识。

除非部署明确将控制权分配到别处，网关负责客户端认证和传输策略。PII 或越狱检测等语义信号本身不会拦截流量；必须由决策或插件执行预期动作。

## 请求缓冲与流式

从所选集成要求的处理模式开始。仅在请求大小或立即流式响应需要时才更改。body 缓冲、模式覆盖和流式约束见 [Streamed ExtProc](streamed-extproc)。

## 生产前验证

- 向每个后端发送直接请求；
- 通过网关发送同一请求；
- 检查所选模型和决策头；
- 确认网关解析了所选后端；以及
- 测试凭证、请求限制、流式和失败行为。

[测试 Kubernetes Gateway 部署](gateway-testing) 提供通用清单，不假定集群分配的地址。
