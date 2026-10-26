---
title: 与 AIBrix 集成
description: 将 Semantic Router 的模型池选择与 AIBrix 的模型部署和副本路由结合。
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/installation/k8s/aibrix.md"
  outdated: false
---

# 与 AIBrix 集成

当 AIBrix 负责模型部署、扩缩容和副本级路由，而 Semantic Router 应根据请求含义和策略选择模型池时，使用此拓扑。

AIBrix 与 Semantic Router 独立发布。请按[当前 AIBrix 安装指南](https://aibrix.readthedocs.io/latest/getting_started/installation/installation.html)安装，并使用其维护的 [Semantic Router 示例](https://github.com/vllm-project/aibrix/tree/main/samples/semantic-router) 获取特定版本的清单。本页说明两层控制如何配合。

## 职责划分

| 组件 | 负责 |
| --- | --- |
| Semantic Router | 信号、决策、模型池选择，以及按配方作用的插件。 |
| AIBrix | 模型工作负载、自动扩缩容、服务发现和副本级路由。 |
| Gateway API | 客户端流量，以及对 Semantic Router 的 ExtProc 调用。 |

Semantic Router 选择逻辑模型。随后 AIBrix 选择服务该模型的健康副本。将自动扩缩容和副本负载策略留在 AIBrix；不要在语义决策中重复实现。

## 开始之前

需要：

- 所选 AIBrix 版本支持的 Kubernetes 集群；
- Gateway API 以及该版本所需的网关；
- `kubectl`、Helm、模型凭证和足够的推理容量；以及
- 若要观察语义模型选择，至少两个模型名称。

固定 AIBrix 版本，并在应用清单前阅读其发行说明。不要从本页复制历史版本 URL，也不要假定开发标签适合生产。

## 1. 部署并验证 AIBrix

按上游安装指南部署模型服务。在加入 Semantic Router 之前，通过 AIBrix 网关验证每个已服务模型：

```bash
kubectl get pods -A
kubectl get services -A
```

对计划暴露的每个模型名称发送直接的 Chat Completions 请求。这可以在独立于语义策略的情况下确认模型访问、容量和网关路由。

## 2. 对齐模型标识

对每个 AIBrix 模型，创建 Semantic Router 提供商模型，其 `provider_model_id` 为 AIBrix 端点接受的名称。将后端绑定到稳定的 Gateway 或 Service DNS 名称，不要绑定到 `ClusterIP`。

然后从模型卡和决策中引用这些提供商名称。同一标识必须在四处一致：

1. 面向请求的虚拟模型或入口；
2. Semantic Router 提供商模型；
3. AIBrix/Gateway 路由资源；以及
4. 模型服务器所服务的模型名称。

部署前校验完整的 Router 文档：

```bash
vllm-sr config validate --config config.yaml
```

## 3. 部署 Semantic Router 和 Gateway 策略

上游示例为其当前 AIBrix 版本提供完整集成。若改编而非原样使用该示例：

- 按[配置工作流](../configuration-workflows#helm) 用 `configOverride` 部署 Router 配置；
- 将提供商凭证放在 Kubernetes Secrets 中；
- 保留网关期望的 ExtProc 处理模式；以及
- 同时更新 Router 提供商和 Gateway 后端。

对于大型请求体或立即流式响应，更改处理模式前请先阅读 [Streamed ExtProc](streamed-extproc)。

## 4. 验证集成

1. 向每个 AIBrix 模型发送直接请求。
2. 通过 Semantic Router 虚拟模型发送请求。
3. 检查 Router 选择头。
4. 确认所选的 AIBrix 模型和正在服务的副本。

通用检查见[测试 Kubernetes Gateway 部署](gateway-testing)。正确的路由决策与成功的生成是分开的信号；两者都要测。

## 清理

用你安装时的 release 名称移除 Router 和网关资源。按 AIBrix 对应版本的卸载流程移除 AIBrix。模型卷和缓存可能在 Deployment 删除后仍存在，删除持久数据前请先检查。
