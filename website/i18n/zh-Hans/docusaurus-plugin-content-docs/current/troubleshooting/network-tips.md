---
title: 网络技巧
sidebar_label: 网络技巧
translation:
  source_commit: "820f5a7"
  source_file: "docs/troubleshooting/network-tips.md"
  outdated: false
---

本指南展示如何在受限或慢速网络环境中，通过仓库原生的 `make` 和 `vllm-sr` 工作流完成构建与运行。规范 Dockerfile 已支持通过构建参数设置镜像，无需维护分叉的 Dockerfile。

本文将解决：

- Hugging Face 模型下载被阻止/缓慢
- Docker 构建期间 Go 模块获取被阻止
- 容器镜像拉取被阻止或速度缓慢

## TL;DR：选择您的路径

- 最快且最可靠：将本地模型放在 `config/config.yaml` 同级的 `config/models` 中，完全跳过 HF 网络。
- 否则：在运行 `vllm-sr serve` 前设置 Hugging Face 镜像环境变量。
- 源码构建：将 `GOPROXY` 和 `GOSUMDB` 传给 `make vllm-sr-dev`。
- 保持 Go 校验和数据库启用；模块代理不能替代完整性校验。

您可以根据情况混合使用这些方法。

## 1. Hugging Face 模型

除非您在本地提供模型，否则路由将在首次运行时下载嵌入模型。如果可能，优先选择方案 A。

### 方案 A — 使用本地模型（无外部网络）

1) 使用任何可达的方法（VPN/离线）将所需模型下载到仓库维护的 `config/config.yaml` 同级目录 `config/models` 中。示例布局：

   - `config/models/all-MiniLM-L12-v2/`
   - `config/models/category_classifier_modernbert-base_model`

2) 在 `config/config.yaml` 中，指向本地路径。示例：

   ```yaml
   global:
     model_catalog:
       embeddings:
         semantic:
           # 指向 /app/models 下的本地文件夹（由本地运行时挂载）
           bert_model_path: /app/models/all-MiniLM-L12-v2
   ```

3) 无需额外环境变量。本地 `vllm-sr` 运行时会把配置目录中的 `models` 文件夹挂载到 `/app/models`。

### 方案 B — 使用 HF 镜像

在启动规范本地运行时前设置区域端点（以下示例使用中国镜像）。`vllm-sr serve` 会将 `HF_ENDPOINT` 传入路由容器：

```bash
export HF_ENDPOINT=https://hf-mirror.com
make vllm-sr-dev
vllm-sr serve --config config/config.yaml --image-pull-policy never
```

## 2. 使用 Go 镜像构建

仓库中的路由和 ExtProc Dockerfile 支持 `GOPROXY` 与 `GOSUMDB` 构建参数。请通过规范 Make 目标传入：

```bash
# CPU 路由镜像、本地 CLI 和配套镜像
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev

# AMD/ROCm 路由镜像
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev VLLM_SR_PLATFORM=amd

# NVIDIA/CUDA 路由镜像
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev VLLM_SR_PLATFORM=nvidia

# 需要独立 ExtProc 产物时
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make docker-build-extproc
```

请使用可信的 HTTPS 代理和校验和数据库。不要设置 `GOSUMDB=off`，否则会关闭 Go 的公共模块完整性校验。

## 3. 构建和运行

从仓库构建，然后通过受支持的 CLI 路径启动刚刚构建的本地镜像：

```bash
# CPU
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev
vllm-sr serve --config config/config.yaml --image-pull-policy never

# AMD/ROCm
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev VLLM_SR_PLATFORM=amd
vllm-sr serve --config config/config.yaml --image-pull-policy never --platform amd

# NVIDIA/CUDA
GOPROXY=https://goproxy.cn,direct \
GOSUMDB=sum.golang.google.cn \
make vllm-sr-dev VLLM_SR_PLATFORM=nvidia
vllm-sr serve --config config/config.yaml --image-pull-policy never --platform nvidia
```

请像上面示例一样成对使用 GPU 构建和启动参数，避免 CLI 回退到 CPU 镜像。NVIDIA 前置条件与执行提供程序验证请参阅仓库中的 `docs/agent/nvidia-local.md` 操作手册。

## 4. 出口受限的 Kubernetes 集群

Kubernetes 节点上的容器运行时不会自动复用宿主机 Docker 守护进程的设置。当镜像仓库缓慢或被阻止时，Pod 可能停留在 `ImagePullBackOff`。选择以下一种或组合多种缓解措施：

### 4.1 配置 containerd 或 CRI 镜像

- 对于由 containerd 支持的集群（Kind、k3s、kubeadm），编辑 `/etc/containerd/config.toml` 或使用 Kind 的 `containerdConfigPatches` 为 `docker.io`、`ghcr.io` 或 `quay.io` 等仓库添加区域镜像端点。
- 更改后重启 containerd 和 kubelet 以使新镜像生效。
- 避免将镜像指向回环代理，除非每个节点都能访问该代理地址。

### 4.2 预加载或侧载镜像

- 在本地构建所需镜像，然后推送到集群运行时。对于 Kind，运行 `kind load docker-image --name <cluster> <image:tag>`；对于其他集群，在每个节点上使用 `crictl pull` 或 `ctr -n k8s.io images import`。
- 当您知道镜像已存在于节点上时，修补部署以设置 `imagePullPolicy: IfNotPresent`。

### 4.3 发布到可访问的镜像仓库

- 标记并推送镜像到集群可达的仓库（云提供商仓库、私有托管的 Harbor 等）。
- 使用新镜像名称更新您的 `kustomization.yaml` 或 Helm values，如果仓库需要身份验证则配置 `imagePullSecrets`。

### 4.4 运行本地透传缓存

- 在同一网络内启动仓库代理（`registry:2` 或供应商特定缓存），在 containerd 中将其配置为镜像，并定期用您需要的镜像预热它。

### 4.5 调整后验证

- 使用 `kubectl describe pod <name>` 或 `kubectl get events` 确认拉取错误消失。
- 检查 `semantic-router-metrics` 等服务现在是否暴露端点并通过端口转发响应（`kubectl port-forward svc/<service> <local-port>:<service-port>`）。

## 5. 故障排除

- Go 模块仍然超时：
  - 验证 go-builder 阶段日志中是否存在 `GOPROXY` 和 `GOSUMDB`。
  - 重试前运行 `make -n vllm-sr-build GOPROXY=<proxy> GOSUMDB=<sumdb>`，确认构建参数已经传入。
  - 如果怀疑层缓存过期，请复制该 dry-run 命令，在所选容器运行时的 `build` 后添加 `--no-cache`，并直接运行；不要清理共享的 builder 缓存。

- HF 模型下载仍然缓慢：
  - 优先选择方案 A（本地模型）。
  - 在运行 `vllm-sr serve` 的同一个 shell 中导出 `HF_ENDPOINT`。
