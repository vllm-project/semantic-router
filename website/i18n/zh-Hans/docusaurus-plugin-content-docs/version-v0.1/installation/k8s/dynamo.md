---
source_commit: b71b08b23825d2b4ec6de1fc3ad1ad4673f8802b
outdated: false
---

# 使用 NVIDIA Dynamo 安装

本指南提供了将 vLLM Semantic Router 与 NVIDIA Dynamo 集成的分步说明。

## 关于 NVIDIA Dynamo

[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) 是一个专为大语言模型服务设计的高性能分布式推理平台。Dynamo 通过智能路由和缓存机制，提供优化 GPU 利用率和降低推理延迟的高级功能。

### 主要特性

- **分离式服务**：将 Prefill 和 Decode 工作节点分离，实现最优 GPU 利用率
- **KV 感知路由**：将请求路由到具有相关 KV 缓存的工作节点，优化前缀缓存
- **动态扩缩容**：Planner 组件根据工作负载处理自动扩缩容
- **多层 KV 缓存**：GPU HBM → 系统内存 → NVMe，实现高效缓存管理
- **工作节点协调**：使用 etcd 和 NATS 进行分布式工作节点注册和消息队列
- **后端无关**：支持 vLLM、SGLang 和 TensorRT-LLM 后端

### 集成优势

将 vLLM Semantic Router 与 NVIDIA Dynamo 集成可提供以下优势：

1. **双层智能**：Semantic Router 提供请求级智能（模型选择、分类），而 Dynamo 优化基础设施级效率（工作节点选择、KV 缓存复用）

2. **智能模型选择**：Semantic Router 分析传入请求，根据内容理解将其路由到最合适的模型，同时 Dynamo 的 KV 感知路由器高效选择最优工作节点

3. **双层缓存**：语义缓存（请求级，Milvus 支持）结合 KV 缓存（token 级，Dynamo 管理），最大程度降低延迟

4. **增强安全性**：PII 检测和越狱防护在请求到达推理工作节点之前进行过滤

5. **分离式架构**：分离 Prefill 和 Decode 工作节点，配合 KV 感知路由，降低延迟并提高吞吐量

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                          CLIENT                                  │
│  curl -X POST http://localhost:8080/v1/chat/completions         │
│       -d '{"model": "MoM", "messages": [...]}'                  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ENVOY GATEWAY                                  │
│  • Routes traffic, applies ExtProc filter                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              SEMANTIC ROUTER (ExtProc Filter)                    │
│  • Classifies query → selects category (e.g., "math")           │
│  • Selects model → rewrites request                             │
│  • Injects domain-specific system prompt                        │
│  • PII/Jailbreak detection                                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              DYNAMO FRONTEND (KV-Aware Routing)                  │
│  • Receives enriched request with selected model                │
│  • Routes to optimal worker based on KV cache state             │
│  • Coordinates workers via etcd/NATS                            │
└─────────────────────────────────────────────────────────────────┘
                     │                          │
                     ▼                          ▼
     ┌───────────────────────────┐  ┌───────────────────────────┐
     │   PREFILL WORKER (GPU 1)  │  │   DECODE WORKER (GPU 2)   │
     │   Processes input tokens  │──▶  Generates output tokens  │
     │   --is-prefill-worker     │  │                           │
     └───────────────────────────┘  └───────────────────────────┘
```

## 前提条件

### GPU 要求

**此部署需要至少 3 个 GPU 的机器：**

| 组件 | GPU | 描述 |
|-----------|-----|-------------|
| Frontend | GPU 0 | 带 KV 感知路由的 Dynamo Frontend (`--router-mode kv`) |
| VLLMPrefillWorker | GPU 1 | 处理推理的 Prefill 阶段 (`--is-prefill-worker`) |
| VLLMDecodeWorker | GPU 2 | 处理推理的 Decode 阶段 |

### 必需工具

在开始之前，请确保已安装以下工具：

- [kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation) - Kubernetes in Docker
- [kubectl](https://kubernetes.io/docs/tasks/tools/) - Kubernetes CLI
- [Helm](https://helm.sh/docs/intro/install/) - Kubernetes 包管理器

### NVIDIA 运行时配置（一次性设置）

配置 Docker 使用 NVIDIA 运行时作为默认值：

```bash
# 将 NVIDIA 运行时配置为默认
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default

# 重启 Docker
sudo systemctl restart docker

# 验证配置
docker info | grep -i "default runtime"
# 预期输出：Default Runtime: nvidia
```

## 步骤 1：创建支持 GPU 的 Kind 集群

创建支持 GPU 的本地 Kubernetes 集群。选择以下选项之一：

### 选项 1：快速设置（外部文档）

如需快速设置，请参阅官方 Kind GPU 文档：

```bash
kind create cluster --name semantic-router-dynamo

# 验证集群就绪
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

有关 GPU 支持，请参阅 [Kind GPU 文档](https://kind.sigs.k8s.io/docs/user/configuration/#extra-mounts) 了解配置额外挂载和部署 NVIDIA 设备插件的详细信息。

### 选项 2：完整 GPU 设置（E2E 流程）

这是我们 E2E 测试中使用的流程，包含在 Kind 中设置 GPU 支持所需的所有步骤。

#### 2.1 创建带 GPU 配置的 Kind 集群

创建带 GPU 挂载支持的 Kind 配置文件：

```bash
# 创建 GPU 支持的 Kind 配置
cat > kind-gpu-config.yaml << 'EOF'
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
name: semantic-router-dynamo
nodes:
  - role: control-plane
    extraMounts:
      - hostPath: /mnt
        containerPath: /mnt
  - role: worker
    extraMounts:
      - hostPath: /mnt
        containerPath: /mnt
      - hostPath: /dev/null
        containerPath: /var/run/nvidia-container-devices/all
EOF

# 使用 GPU 配置创建集群
kind create cluster --name semantic-router-dynamo --config kind-gpu-config.yaml --wait 5m

# 验证集群就绪
kubectl wait --for=condition=Ready nodes --all --timeout=300s
```

#### 2.2 在 Kind 工作节点中设置 NVIDIA 库

将 NVIDIA 库从主机复制到 Kind 工作节点：

```bash
# 设置工作节点名称
WORKER_NAME="semantic-router-dynamo-worker"

# 检测 NVIDIA 驱动版本
DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "Detected NVIDIA driver version: $DRIVER_VERSION"

# 验证 Kind 工作节点中存在 GPU 设备
docker exec $WORKER_NAME ls /dev/nvidia0
echo "✅ GPU devices found in Kind worker"

# 创建 NVIDIA 库目录
docker exec $WORKER_NAME mkdir -p /nvidia-driver-libs

# 复制 nvidia-smi 二进制文件
tar -cf - -C /usr/bin nvidia-smi | docker exec -i $WORKER_NAME tar -xf - -C /nvidia-driver-libs/

# 从主机复制 NVIDIA 库
tar -cf - -C /usr/lib64 libnvidia-ml.so.$DRIVER_VERSION libcuda.so.$DRIVER_VERSION | \
  docker exec -i $WORKER_NAME tar -xf - -C /nvidia-driver-libs/

# 创建符号链接
docker exec $WORKER_NAME bash -c "cd /nvidia-driver-libs && \
  ln -sf libnvidia-ml.so.$DRIVER_VERSION libnvidia-ml.so.1 && \
  ln -sf libcuda.so.$DRIVER_VERSION libcuda.so.1 && \
  chmod +x nvidia-smi"

# 验证 nvidia-smi 在 Kind 工作节点中可用
docker exec $WORKER_NAME bash -c "LD_LIBRARY_PATH=/nvidia-driver-libs /nvidia-driver-libs/nvidia-smi"
echo "✅ nvidia-smi verified in Kind worker"
```

#### 2.3 部署 NVIDIA 设备插件

部署 NVIDIA 设备插件以使 GPU 在 Kubernetes 中可分配：

```bash
# 创建设备插件清单
cat > nvidia-device-plugin.yaml << 'EOF'
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      containers:
      - image: nvcr.io/nvidia/k8s-device-plugin:v0.14.1
        name: nvidia-device-plugin-ctr
        env:
        - name: LD_LIBRARY_PATH
          value: "/nvidia-driver-libs"
        securityContext:
          privileged: true
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
        - name: dev
          mountPath: /dev
        - name: nvidia-driver-libs
          mountPath: /nvidia-driver-libs
          readOnly: true
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
      - name: dev
        hostPath:
          path: /dev
      - name: nvidia-driver-libs
        hostPath:
          path: /nvidia-driver-libs
EOF

# 应用设备插件
kubectl apply -f nvidia-device-plugin.yaml

# 等待设备插件就绪
sleep 20

# 验证 GPU 可分配
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.allocatable.nvidia\\.com/gpu
echo "✅ GPU setup complete"
```

:::tip[E2E 测试]
Semantic Router 项目包含自动化 E2E 测试，可自动处理所有 GPU 设置。您可以运行：

```bash
make e2e-test E2E_PROFILE=dynamo E2E_VERBOSE=true
```

这将创建支持 GPU 的 Kind 集群、部署所有组件并运行测试套件。
:::

## 步骤 2：安装 Dynamo 平台

部署 Dynamo 平台组件（etcd、NATS、Dynamo Operator）：

```bash
# 添加 Dynamo Helm 仓库
helm repo add dynamo https://nvidia.github.io/dynamo
helm repo update

# 安装 Dynamo CRDs
helm install dynamo-crds dynamo/dynamo-crds \
  --namespace dynamo-system \
  --create-namespace

# 安装 Dynamo 平台（etcd、NATS、Operator）
helm install dynamo-platform dynamo/dynamo-platform \
  --namespace dynamo-system \
  --wait

# 等待平台组件就绪
kubectl wait --for=condition=Available deployment -l app.kubernetes.io/instance=dynamo-platform -n dynamo-system --timeout=300s
```

## 步骤 3：安装 Envoy Gateway

部署启用 ExtensionAPIs 的 Envoy Gateway 以支持 Semantic Router 集成：

```bash
# 使用自定义值安装 Envoy Gateway
helm install envoy-gateway oci://docker.io/envoyproxy/gateway-helm \
  --version v1.3.0 \
  --namespace envoy-gateway-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/envoy-gateway-values.yaml

# 等待 Envoy Gateway 就绪
kubectl wait --for=condition=Available deployment/envoy-gateway -n envoy-gateway-system --timeout=300s
```

**重要：** 该 values 文件启用了 `extensionApis.enableEnvoyPatchPolicy: true`，这是 Semantic Router ExtProc 集成所必需的。

## 步骤 4：部署 vLLM Semantic Router

使用 Dynamo 特定配置部署 Semantic Router：

```bash
# 从 GHCR OCI 仓库安装 Semantic Router
helm install semantic-router oci://ghcr.io/vllm-project/charts/semantic-router \
  --version v0.0.0-latest \
  --namespace vllm-semantic-router-system \
  --create-namespace \
  -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/semantic-router-values/values.yaml

# 等待部署就绪
kubectl wait --for=condition=Available deployment/semantic-router -n vllm-semantic-router-system --timeout=600s

# 验证部署状态
kubectl get pods -n vllm-semantic-router-system
```

**注意：** 该 values 文件将 Semantic Router 配置为路由到由 Dynamo 工作节点服务的 TinyLlama 模型。

## 步骤 5：部署 RBAC 资源

应用 RBAC 权限以允许 Semantic Router 访问 Dynamo CRDs：

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/rbac.yaml
```

## 步骤 6：部署 DynamoGraphDeployment

使用 DynamoGraphDeployment CRD 部署 Dynamo 工作节点：

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/dynamo-graph-deployment.yaml

# 等待 Dynamo Operator 创建部署
kubectl wait --for=condition=Available deployment/vllm-frontend -n dynamo-system --timeout=600s
kubectl wait --for=condition=Available deployment/vllm-vllmprefillworker -n dynamo-system --timeout=600s
kubectl wait --for=condition=Available deployment/vllm-vllmdecodeworker -n dynamo-system --timeout=600s
```

DynamoGraphDeployment 创建：

- **Frontend**：带 KV 感知路由的 HTTP API 服务器
- **VLLMPrefillWorker**：专门用于 Prefill 阶段的工作节点
- **VLLMDecodeWorker**：专门用于 Decode 阶段的工作节点

## 步骤 7：创建 Gateway API 资源

部署 Gateway API 资源以连接所有组件：

```bash
kubectl apply -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/gwapi-resources.yaml

# 验证 EnvoyPatchPolicy 已被接受
kubectl get envoypatchpolicy -n default
```

**重要：** EnvoyPatchPolicy 状态必须显示 `Accepted: True`。如果显示 `Accepted: False`，请验证 Envoy Gateway 是否使用正确的 values 文件安装。

## 测试部署

### 方法 1：端口转发（推荐用于本地测试）

设置端口转发以在本地访问网关：

```bash
# 获取 Envoy 服务名称
export ENVOY_SERVICE=$(kubectl get svc -n envoy-gateway-system \
  --selector=gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router \
  -o jsonpath='{.items[0].metadata.name}')

kubectl port-forward -n envoy-gateway-system svc/$ENVOY_SERVICE 8080:80
```

### 发送测试请求

使用数学查询测试推理端点：

```bash
curl -i -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ]
  }'
```

### 预期响应

```bash
HTTP/1.1 200 OK
server: fasthttp
date: Thu, 06 Nov 2025 06:38:08 GMT
content-type: application/json
x-vsr-selected-category: math
x-vsr-selected-reasoning: on
x-vsr-selected-model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
x-vsr-injected-system-prompt: true
transfer-encoding: chunked

{"id":"chatcmpl-...","model":"TinyLlama/TinyLlama-1.1B-Chat-v1.0","choices":[{"message":{"role":"assistant","content":"..."}}],"usage":{"prompt_tokens":15,"completion_tokens":54,"total_tokens":69}}
```

**成功指标：**

- ✅ 请求使用 `model="MoM"` 发送
- ✅ 响应显示 `model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"`（由 Semantic Router 重写）
- ✅ 请求头显示类别分类和系统提示词注入

### 检查 Semantic Router 日志

```bash
# 查看分类和路由决策
kubectl logs -n vllm-semantic-router-system deployment/semantic-router -f | grep -E "category|routing_decision"
```

预期输出：

```text
Classified as category: math (confidence=0.933)
Selected model TinyLlama/TinyLlama-1.1B-Chat-v1.0 for category math with score 1.0000
routing_decision: original_model="MoM", selected_model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```

### 验证 EnvoyPatchPolicy 状态

```bash
kubectl get envoypatchpolicy -n default -o yaml | grep -A 5 "status:"
```

预期状态：

```yaml
status:
  conditions:
  - type: Accepted
    status: "True"
  - type: Programmed
    status: "True"
```

## 清理

要删除整个部署：

```bash
# 删除 Gateway API 资源
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/gwapi-resources.yaml

# 删除 DynamoGraphDeployment
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/dynamo-graph-deployment.yaml

# 删除 RBAC
kubectl delete -f https://raw.githubusercontent.com/vllm-project/semantic-router/refs/heads/main/deploy/kubernetes/dynamo/dynamo-resources/rbac.yaml

# 删除 Semantic Router
helm uninstall semantic-router -n vllm-semantic-router-system

# 删除 Envoy Gateway
helm uninstall envoy-gateway -n envoy-gateway-system

# 删除 Dynamo 平台
helm uninstall dynamo-platform -n dynamo-system
helm uninstall dynamo-crds -n dynamo-system

# 删除命名空间
kubectl delete namespace vllm-semantic-router-system
kubectl delete namespace envoy-gateway-system
kubectl delete namespace dynamo-system

# 删除 Kind 集群（可选）
kind delete cluster --name semantic-router-dynamo
```

## 生产配置

对于使用更大模型的生产部署，修改 DynamoGraphDeployment：

```yaml
# 示例：使用 Llama-3-8B 替代 TinyLlama
args:
  - "python3 -m dynamo.vllm --model meta-llama/Llama-3-8b-hf --tensor-parallel-size 2 --enforce-eager"
resources:
  requests:
    nvidia.com/gpu: 2  # 增加以支持张量并行
```

**生产环境注意事项：**

- 使用适合您用例的更大模型
- 配置张量并行以支持多 GPU 推理
- 为多节点部署启用分布式 KV 缓存
- 设置监控和可观测性
- 根据 GPU 利用率配置自动扩缩容

## 后续步骤

- 查看 [NVIDIA Dynamo 集成提案](../../proposals/nvidia-dynamo-integration.md) 了解详细架构
- 设置 [监控和可观测性](../../tutorials/observability/metrics.md)
- 为生产环境配置 [Milvus 语义缓存](../../tutorials/semantic-cache/milvus-cache.md)
- 扩展部署以适应生产工作负载

## 参考资料

- [NVIDIA Dynamo GitHub](https://github.com/ai-dynamo/dynamo)
- [Dynamo 文档](https://docs.nvidia.com/dynamo/latest/)
- [演示视频：Semantic Router + Dynamo E2E](https://www.youtube.com/watch?v=rRULSR9gTds&list=PLmrddZ45wYcuPrXisC-yl7bMI39PLo4LO&index=2)
