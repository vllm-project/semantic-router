---
title: 与 vLLM Production Stack 集成
description: 将 Semantic Router 的模型池选择连接到由 vLLM Production Stack 管理的模型服务。
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/installation/k8s/production-stack.md"
  outdated: false
---

# 与 vLLM Production Stack 集成

当 vLLM Production Stack 已经负责模型部署、服务发现和副本调度，而 Semantic Router 应根据请求含义和策略选择模型池时，使用此拓扑。

本页描述集成约定。Production Stack 发行版和 Helm values 独立变化，因此请按当前 [Production Stack 文档](https://github.com/vllm-project/production-stack) 安装，不要从本指南复制冻结的 chart 配置。

## 职责划分

| 组件 | 负责 |
| --- | --- |
| Semantic Router | 信号、决策、模型池选择，以及按配方作用的插件。 |
| vLLM Production Stack | 模型服务器、服务发现、副本调度和推理生命周期。 |
| Gateway | 客户端流量，以及对 Semantic Router 的 ExtProc 连接。 |

Semantic Router 选择合格模型。随后 Production Stack 选择服务该模型的副本。仅当决策和插件执行策略时，PII、越狱或其他信号才会影响流量；仅检测不会拦截请求。

## 开始之前

需要：

- 可用的 Production Stack 部署，且至少有一个 OpenAI 兼容模型端点；
- 这些端点的稳定 Kubernetes 服务名称；
- 能通过 ExtProc 调用 Semantic Router 的 Gateway；以及
- `kubectl`、Helm、模型凭证和足够的推理容量。

不要将 Router 提供商绑定到 Service `ClusterIP`。使用 Kubernetes DNS，以便配置在 Service 重建后仍然有效。

## 1. 验证模型服务

按上游安装指南操作，然后记录模型名称、命名空间、Service 名称和端口：

```bash
kubectl get services -A
kubectl get pods -A
```

加入 Semantic Router 之前，向 Production Stack 端点发送直接的 Chat Completions 请求。这样可以把后端或调度器故障与语义路由故障分开。

## 2. 在规范配置中绑定模型

为策略要选择的每个模型池创建一个 Semantic Router 提供商模型。Kubernetes 后端引用使用此形状：

```yaml
providers:
  defaults:
    model: production/qwen3
  models:
    - name: production/qwen3
      provider_model_id: Qwen/Qwen3-8B
      backend_refs:
        - name: production-stack
          endpoint: vllm-router-service.default.svc.cluster.local:80
          protocol: http
          provider: vllm
          weight: 100
```

用部署中的值替换端点和模型标识。若 Production Stack 为每个模型暴露不同服务，则为每个服务创建 binding。若它暴露一个多模型服务，则保持不同的提供商模型，并使用后端所服务的模型标识。

添加引用这些提供商名称的模型卡、决策和入口，然后校验完整文档：

```bash
vllm-sr config validate --config config.yaml
```

## 3. 部署 Semantic Router

使用[配置工作流](../configuration-workflows#helm) 通过 `configOverride` 部署已校验的配置，然后接入受支持的 [Kubernetes 网关](ai-gateway) 之一。生产环境固定 chart 和镜像版本；开发用的 `0.0.0-latest` chart 用于测试当前 main。

上游 [Semantic Router 集成教程](https://github.com/vllm-project/production-stack/blob/main/tutorials/24-semantic-router-integration.md) 可以提供额外背景，但应用前请对照当前 Production Stack 发行版审阅其镜像标签和 values。

## 4. 验证两层路由

1. 向每个模型服务发送直接请求。
2. 通过 Gateway 使用已配置的虚拟模型发送同一请求。
3. 检查 Semantic Router 选择头。
4. 确认 Production Stack 将请求发送到所选模型池的副本。

通用 Gateway 检查见[测试 Kubernetes Gateway 部署](gateway-testing)。成功的语义决策并不能证明所选模型已就绪，因此同时保留直接和已路由的生成测试。

## 清理

按各自指南中的命令移除 Semantic Router 和 Gateway 资源。用安装时选择的 release 名称和命名空间移除 Production Stack；不要复制为不同发行版编写的清理命令。
