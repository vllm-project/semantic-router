---
translation:
  source_commit: "7e2f073"
  source_file: "docs/installation/k8s/istio.md"
  outdated: false
is_mtpe: true
sidebar_position: 1
---

# 使用 Istio Gateway 安装

本指南提供了在 Kubernetes 上使用 Istio Gateway 部署 vLLM Semantic Router (vsr) 的分步说明。Istio Gateway 底层使用 Envoy，因此可以与 vsr 配合使用。但是，不同的基于 Envoy 的 Gateway 处理 ExtProc 协议的方式有所不同，因此这里描述的部署与其他基于 Envoy 的 Gateway 部署有所不同。将 Istio Gateway 与 vsr 结合有多种架构选项。本文档描述其中一种选项。

## 架构概览

部署包含以下组件：

- **vLLM Semantic Router**：为基于 Envoy 的 Gateway 提供智能请求路由和处理决策
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

minikube 或 kind 都可以用于部署本练习所需的本地 kubernetes 集群，因此您只需要其中一个。我们在下面的描述中使用 minikube，但相同的步骤在创建集群后也适用于 Kind 集群。

我们还将在本练习中部署两个不同的 LLM 以更清楚地说明 Semantic Router 和 model routing 功能，因此理想情况下您应该在支持 GPU 的机器上运行本练习，以运行两个模型并具有足够的内存和存储空间。您也可以在较小的服务器上使用等效步骤，在基于 CPU 的服务器上运行较小的 LLM，无需 GPU。

## 步骤 1：创建 Minikube 集群

通过 minikube（或等效地通过 Kind）创建本地 Kubernetes 集群。

```bash
# 创建集群
$ minikube start \
    --driver docker \
    --container-runtime docker \
    --gpus all \
    --memory no-limit \
    --cpus no-limit

# 验证集群就绪
$ kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

## 步骤 2：部署 LLM 模型

在本练习中，我们部署两个 LLM，即 llama3-8b 模型 (meta-llama/Llama-3.1-8B-Instruct) 和 phi4-mini 模型 (microsoft/Phi-4-mini-instruct)。我们使用两个独立的 [vLLM 推理服务器](https://docs.vllm.ai/en/latest/) 实例在 kubernetes 集群的 default 命名空间中服务这些模型。您可以选择任何其他推理引擎，只要它们暴露 OpenAI API 端点。首先为之前存储在环境变量 HF_TOKEN 中的 HuggingFace 令牌安装密钥，然后如下所示部署模型。请注意，本指南中示例 kubectl 命令中使用的文件路径名称预期从此仓库的顶层文件夹执行。

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

## 步骤 3：安装 Istio Gateway、Gateway API、Inference Extension CRDs

我们将在本练习中使用较新版本的 Istio，以便也可以选择使用 Gateway API Inference Extension CRDs 和 EPP 功能的 v1.0.0 GA 版本。

按照 Gateway API [Inference Extensions 文档](https://gateway-api-inference-extension.sigs.k8s.io/guides/)中描述的过程部署 1.28（或更新）版本的 Istio 控制平面、Istio Gateway、Kubernetes Gateway API CRDs 和 Gateway API Inference Extension v1.0.0。但是，不要从该指南安装任何 HTTPRoute 资源或 EndPointPicker，只使用它来部署 Istio gateway 和 CRDs。如果正确安装，您应该使用以下命令看到 gateway api 和 inference extension 的 api CRDs 以及 Istio gateway 和 Istiod 的运行 pods。

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

## 步骤 4：更新 vsr 配置（可选）

semantic router 配置通过 Helm values 文件提供。如果您需要自定义配置（例如匹配不同的模型名称或端点），请下载 values 文件并进行修改：

```bash
# 下载 values 文件以进行自定义
curl -O https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/istio/semantic-router-values/values.yaml
```

确保配置文件中的模型与您使用的模型匹配。通常最好从 vsr 的基本功能开始，如提示词分类和模型路由，然后再尝试其他功能如 PromptGuard 或 ToolCalling。

## 步骤 5：部署 vLLM Semantic Router

使用 Helm 部署包含所有必需组件的 Semantic Router 服务：

```bash
# 使用 Kustomize 部署 Semantic Router 
kubectl apply -k deploy/kubernetes/istio/

# 等待部署就绪（模型下载可能需要几分钟）
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# 验证部署状态
kubectl get pods -n vllm-semantic-router-system
```

## 步骤 6：安装额外的 Istio 配置

安装 Istio gateway 使用 ExtProc 接口与 vLLM Semantic Router 所需的 destinationrule 和 envoy filter

```bash
kubectl apply -f deploy/kubernetes/istio/destinationrule.yaml
kubectl apply -f deploy/kubernetes/istio/envoyfilter.yaml
```

## 步骤 7：安装 gateway 路由

在 Istio gateway 中安装 HTTPRoutes。

```bash
kubectl apply -f deploy/kubernetes/istio/httproute-llama3-8b.yaml
kubectl apply -f deploy/kubernetes/istio/httproute-phi4-mini.yaml
```

## 步骤 8：测试部署

要将 Istio gateway 监听集群外客户端请求的 IP 暴露出来，您可以选择任何标准的 kubernetes 外部负载均衡选项。我们通过[部署和配置 metallb](https://metallb.universe.tf/installation/) 到集群中作为 LoadBalancer 提供商来测试我们的功能。如有需要，请参阅 metallb 文档了解安装过程。最后，对于 minikube 情况，我们如下获取外部 url。

```bash
minikube service inference-gateway-istio --url
http://192.168.49.2:30913
```

现在我们可以通过 curl 向 http://192.168.49.2:30913 发送 LLM prompt，访问 Istio gateway，然后使用 vLLM Semantic Router 的信息动态路由到我们在本例中作为后端使用的两个 LLM 之一。

### 发送测试请求

尝试以下有和没有模型 "auto" 选择的情况，以确认 Istio + vsr 能够将查询路由到适当的模型。查询响应将包含用于服务该请求的模型信息。

示例查询包括以下

```bash
# 显式提供模型名称 llama3-8b，应路由到此后端
curl http://192.168.49.2:30913/v1/chat/completions   -H "Content-Type: application/json"   -d '{
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
curl http://192.168.49.2:30913/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "auto",
        "messages": [
          {"role": "user", "content": "Linux is said to be an open source kernel because "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

```bash
# 显式提供模型名称 phi4-mini，应路由到此后端
curl http://192.168.49.2:30913/v1/chat/completions   -H "Content-Type: application/json"   -d '{
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
curl http://192.168.49.2:30913/v1/chat/completions   -H "Content-Type: application/json"   -d '{
        "model": "auto",
        "messages": [
          {"role": "user", "content": "2+2 is  "}
         ],
        "max_tokens": 100,
        "temperature": 0
      }'
```

## 故障排除

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

** Semantic Router 无响应：**

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
- Istio Gateway 的其他用例/拓扑（包括与 EPP 和 LLM-D）
- 设置监控和可观测性
- 实施身份验证和授权
- 为生产工作负载扩展 Semantic Router 部署
