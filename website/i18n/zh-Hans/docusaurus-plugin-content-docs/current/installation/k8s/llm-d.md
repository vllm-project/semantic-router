---
translation:
  source_commit: "eb18d86"
  source_file: "docs/installation/k8s/llm-d.md"
  outdated: false
is_mtpe: true
sidebar_position: 2
---

# 使用 LLM-D 安装

本指南提供了将 vLLM Semantic Router (vsr) 与 [LLM-D](https://github.com/llm-d/llm-d) 结合部署的分步说明。这也将说明一个关键设计模式，即使用 vsr 作为 model selector，结合使用 LLM-D 作为 endpoint selector。

Model selector 提供将 LLM 查询路由到多个完全不同的 LLM 模型之一的能力，而 endpoint selector 则选择多个 endpoint 之一，每个 endpoint 服务一个等效模型（通常是完全相同的基础模型）。因此，此部署展示了 vLLM Semantic Router 作为 model selector 如何与 LLM-D 等 endpoint selector 解决方案完美互补。

由于 LLM-D 有多种部署配置，其中一些需要更大的硬件设置，我们将演示 LLM-D 与 vsr 配合工作的基线版本，以介绍核心概念。当使用更复杂的 LLM-D 配置和生产级良好路径时，这些核心概念同样适用，如 LLM-D 仓库中[此链接](https://github.com/llm-d/llm-d/tree/main/guides)所述。

此外，我们将使用 LLM-D 与 Istio 作为 Inference Gateway，以构建在本仓库中记录的 [Istio 部署示例](istio)的步骤和硬件设置之上。无论是否使用 vsr，Istio 也常用作 LLM-D 的默认网关。

## 架构概览

部署包含以下组件：

- **vLLM Semantic Router**：为基于 Envoy 的 Gateway 提供智能请求路由和处理决策
- **LLM-D**：用于大规模 LLM 推理的分布式推理平台，具有 SOTA 性能。
- **Istio Gateway**：Istio 的 Kubernetes Gateway API 实现，底层使用 Envoy 代理
- **Gateway API Inference Extension**：通过 ExtProc 服务器扩展 Gateway API 用于推理的附加 API
- **两个 vLLM 实例各服务一个模型**：此拓扑中用于演示 Semantic Router 的示例后端 LLM

## 前置条件

开始之前，请确保已安装以下工具：

- [Docker](https://docs.docker.com/get-docker/) - 容器运行时
- [minikube](https://minikube.sigs.k8s.io/docs/start/) - 本地 Kubernetes
- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [Helm](https://helm.sh/docs/intro/install/) - Kubernetes 包管理器
- [istioctl](https://istio.io/latest/docs/ops/diagnostic-tools/istioctl/) - Istio CLI

我们在下面的描述中使用 minikube。如上所述，本指南构建在本仓库的 vsr + Istio [部署指南](istio)之上，因此将参考该指南的公共部分并在此添加增量步骤。

与 Istio 指南一样，您需要一台至少有 2 个 GPU 支持的机器来运行本练习，以便我们可以部署并测试使用 vsr 在两个不同的 LLM 基础模型之间进行模型路由。

## 步骤 1：Istio 指南的公共步骤

首先，按照 [Istio 指南](istio)中的步骤创建本地 minikube 集群。

## 步骤 2：安装 Istio Gateway、Gateway API、Inference Extension CRDs

完全按照 [Istio 指南](istio)中的描述安装 Kubernetes Gateway API、Gateway API Inference Extension、Istio 控制平面和 Istio Gateway 实例的 CRDs。使用该指南中记录的相同版本的 Istio。如果您按照 LLM-D 的良好路径，这部分将由 LLM-D 仓库中的 Gateway 提供商 Helm charts 完成。在本指南中，我们手动设置这些以保持与本仓库 Istio 指南的通用性和可重用性。这也将帮助读者理解基于 GIE/EPP 的部署和基于 LLM-D 的部署之间的共同点，以及 vsr 如何在两种情况下使用。

如果正确安装，您应该使用以下命令看到 gateway api 和 inference extension 的 api CRDs 以及 Istio gateway 和 Istiod 的运行 pods。

```bash
kubectl get crds | grep gateway
```

```bash
kubectl get crds | grep inference
```

```bash
kubectl get pods | grep istio
```

```bash
kubectl get pods -n istio-system
```

## 步骤 3：部署 LLM 模型

现在与 [Istio 指南](istio)文档类似部署两个 LLM 模型。LLM-D 部署文档中此步骤的对应部分是 LLM-D Model Service 的设置。为简单起见，本指南不需要 LLM-D Model service。

```bash
kubectl create secret generic hf-token-secret --from-literal=token=$HF_TOKEN
```

```bash
# 创建运行 llama3-8b 的 vLLM 服务
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/vLlama3.yaml
```

第一次运行时可能需要几分钟（10+）来下载模型，直到运行此模型的 vLLM pod 处于 READY 状态。同样地部署第二个 LLM (phi4-mini) 并等待几分钟直到 pod 处于 READY 状态。

```bash
# 创建运行 phi4-mini 的 vLLM 服务
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/vPhi4.yaml
```

完成后，您应该能够使用以下命令看到两个 vLLM pod 都处于 READY 状态并正在服务这些 LLM。您还应该看到 Kubernetes 服务暴露了这些模型服务的 IP/端口。在下面的示例中，llama3-8b 模型通过服务 IP 为 10.108.250.109 和端口 80 的 kubernetes 服务提供服务。

```bash
# 验证运行两个 LLM 的 vLLM pod 处于 READY 状态并正在服务

kubectl get pods
NAME                                           READY   STATUS    RESTARTS     AGE
llama-8b-57b95475bd-ph7s4                      1/1     Running   0            9d
phi4-mini-887476b56-74twv                      1/1     Running   0            9d

# 查看这些模型服务的 Kubernetes 服务 IP/端口

kubectl get service
NAME                                  TYPE           CLUSTER-IP       EXTERNAL-IP      PORT(S)                        AGE
kubernetes                            ClusterIP      10.96.0.1        <none>           443/TCP                        36d
llama-8b                              ClusterIP      10.108.250.109   <none>           80/TCP                         18d
phi4-mini                             ClusterIP      10.97.252.33     <none>           80/TCP                         9d
```

## 步骤 4：部署 InferencePools 和 LLM-D 调度器

LLM-D（和 Kubernetes IGW）使用一个名为 InferencePool 的 API 资源，以及一个调度器（称为 LLM-D 推理调度器，有时也等效地称为 EndPoint Picker/EPP）。

部署提供的清单以创建与本练习中使用的 2 个基础模型对应的 InferencePool 和 LLM-D 推理调度器。

为了展示模型选择和端点选择的完整组合，通常需要至少 2 个 inferencepools，每个池至少有 2 个端点。由于这需要 4 个 vllm 服务 pods 实例和 4 个 GPU，这需要更复杂的硬件设置。本指南在两个 InferencePools 中各部署 1 个模型端点，以展示 vsr 的模型选择与 LLM-D 调度器的端点选择配合工作并相互补充的核心设计。

```bash
# 为 Llama3-8b 模型创建 LLM-D 调度器和 InferencePool
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/llmd-base/inferencepool-llama.yaml
```

```bash
# 为 phi4-mini 模型创建 LLM-D 调度器和 InferencePool
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/llmd-base/inferencepool-phi4.yaml
```

## 步骤 5：LLM-D 连接的额外 Istio 配置

添加 DestinationRule 以允许每个 EPP/LLM-D 调度器使用无 TLS 的 ExtProc（当前 Istio 限制）。

```bash
# Llama3-8b 池调度器的 Istio destinationrule
kubectl apply -f deploy/kubernetes/llmd-base/dest-rule-epp-llama.yaml
```

```bash
# phi4-mini 池调度器的 Istio destinationrule
kubectl apply -f deploy/kubernetes/llmd-base/dest-rule-epp-phi4.yaml
```

## 步骤 6：更新 vsr 配置

由于本指南基于使用与 [Istio 指南](istio)相同的后端模型，我们将重用该指南中的相同 vsr 配置，因此您不需要更新文件 deploy/kubernetes/istio/config.yaml。如果您在 LLM-D 部署中使用不同的后端模型，则需要更新此文件。

## 步骤 7：部署 vLLM Semantic Router

部署包含所有必需组件的 Semantic Router 服务：

```bash
# 使用 Kustomize 部署 Semantic Router 
kubectl apply -k deploy/kubernetes/istio/

# 等待部署就绪（模型下载可能需要几分钟）
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# 验证部署状态
kubectl get pods -n vllm-semantic-router-system
```

## 步骤 6：VSR 连接的额外 Istio 配置

安装 Istio gateway 使用 ExtProc 接口与 vLLM Semantic Router 所需的 destinationrule 和 envoy filter。

```bash
kubectl apply -f deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f deploy/kubernetes/istio/envoyfilter.yaml
```

## 步骤 7：安装 gateway 路由

在 Istio gateway 中安装 HTTPRoutes。注意这里与之前的 vsr + istio 指南中使用的 http 路由有一个区别，这里基于路由匹配的 backendRefs 指向 InferencePools，而 InferencePools 又指向这些池的 LLM-D 调度器，而不是 backendRefs 指向模型的 vllm 服务端点，如[不带 llm-d 的 istio 指南](istio)中所做的那样。

```bash
kubectl apply -f deploy/kubernetes/llmd-base/httproute-llama-pool.yaml
kubectl apply -f deploy/kubernetes/llmd-base/httproute-phi4-pool.yaml
```

## 步骤 8：测试部署

要将 Istio gateway 监听集群外客户端请求的 IP 暴露出来，您可以选择任何标准的 kubernetes 外部负载均衡选项。我们通过[部署和配置 metallb](https://metallb.universe.tf/installation/) 到集群中作为 LoadBalancer 提供商来测试我们的功能。如有需要，请参阅 metallb 文档了解安装过程。最后，对于 minikube 情况，我们如下获取外部 url。

```bash
minikube service inference-gateway-istio --url
http://192.168.49.2:32293
```

现在我们可以通过 curl 向 http://192.168.49.2:32293 发送 LLM prompt，访问 Istio gateway，然后使用 vLLM Semantic Router 的信息动态路由到我们在本例中作为后端使用的两个 LLM 之一。使用您从 "minikube service" 命令输出中获得的端口号在下面的 curl 示例中。

### 发送测试请求

尝试以下有和没有模型 "auto" 选择的情况，以确认 Istio + vsr 能够将查询路由到适当的模型。查询响应将包含用于服务该请求的模型信息。

示例查询包括以下

```bash
# 显式提供模型名称 llama3-8b，不更改模型，发送到 llama EPP 进行端点选择
curl http://192.168.49.2:32293/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "llama3-8b",
        "messages": [
          {"role": "user", "content": "Linux is said to be an open source kernel because "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

```bash
# 模型名称设置为 "auto"，应分类为 "computer science" 并路由到 llama3-8b
curl http://192.168.49.2:32293/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "auto",
        "messages": [
          {"role": "user", "content": "Linux is said to be an open source kernel because "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

```bash
# 显式提供模型名称 phi4-mini，不更改模型，发送到 phi4-mini EPP 进行端点选择
curl http://192.168.49.2:32293/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "phi4-mini",
        "messages": [
          {"role": "user", "content": "2+2 is  "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

```bash
# 模型名称设置为 "auto"，应分类为 "math" 并路由到 phi4-mini
curl http://192.168.49.2:32293/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "auto",
        "messages": [
          {"role": "user", "content": "2+2 is  "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

## 故障排除

### 基本 Pod 验证

如果您遵循了上述步骤，您应该看到类似下面的 pods 处于 READY 状态作为快速初始验证。这些包括 LLM 模型 pods、Istio gateway pod、LLM-D/EPP 调度器 pods、vsr pod 和 istiod controller pod，如下所示。您还应该看到 InferencePools 和 HTTPRoute 实例，如下所示，状态显示路由处于 resolved 状态。

```bash
$ kubectl get pods -n default
NAME                                           READY   STATUS    RESTARTS   AGE
inference-gateway-istio-6fc8864bfb-gbcz8       1/1     Running   0          14h
llama-8b-6558848cc8-wkkxn                      1/1     Running   0          3h26m
phi4-mini-7b94bc69db-rnpkj                     1/1     Running   0          17h
vllm-llama3-8b-instruct-epp-7f7ff88677-j7lst   1/1     Running   0          134m
vllm-phi4-mini-epp-6f5dd6bbb9-8pv27            1/1     Running   0          14h
```

```bash
$ kubectl get pods -n vllm-semantic-router-system
NAME                              READY   STATUS    RESTARTS   AGE
semantic-router-bf6cdd5b9-t5hpg   1/1     Running   0          5d23h
```

```bash
$ kubectl get pods -n istio-system
NAME                     READY   STATUS    RESTARTS   AGE
istiod-6f5ccc65c-vnbg5   1/1     Running   0          15h
```

```bash
$ kubectl get inferencepools
NAME                      AGE
vllm-llama3-8b-instruct   139m
vllm-phi4-mini            15h
```

```bash
$ kubectl get httproutes
NAME            HOSTNAMES   AGE
vsr-llama8b                 13h
vsr-phi4-mini               13h
```

```bash
$ kubectl get httproute vsr-llama8b -o yaml | grep -A 1 "reason: ResolvedRefs"
      reason: ResolvedRefs
      status: "True"
```

### 常见问题

**Gateway/前端无法工作：**

```bash
# 检查 istio gateway 状态
kubectl get gateway

# 检查 istio gw 服务状态
kubectl get svc inference-gateway-istio

# 检查 Istio 的 Envoy 日志
kubectl logs deploy/inference-gateway-istio -c istio-proxy
```

** Semantic Router 无响应或路由不正确：**

```bash
# 检查 Semantic Router  pod
kubectl get pods -n vllm-semantic-router-system

# 检查 Semantic Router 服务
kubectl get svc -n vllm-semantic-router-system

# 检查 Semantic Router 日志
kubectl logs -n vllm-semantic-router-system deployment/semantic-router
```

## 清理

```bash

# 删除 Semantic Router 
kubectl delete -k deploy/kubernetes/istio/

# 删除 Istio
istioctl uninstall --purge

# 删除 LLMs
kubectl delete -f deploy/kubernetes/istio/vLlama3.yaml
kubectl delete -f deploy/kubernetes/istio/vPhi4.yaml

# 停止 minikube 集群
minikube stop

# 删除 minikube 集群
minikube delete
```

## 后续步骤

- 测试/尝试 vLLM Semantic Router 的不同功能
- 测试/尝试更复杂的 LLM-D 配置和良好路径
- 设置监控和可观测性
- 实施身份验证和授权
- 为生产工作负载扩展 Semantic Router 部署
