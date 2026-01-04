---
title: 网络技巧
sidebar_label: 网络技巧
translation:
  source_commit: "aa7b7cb"
  source_file: "docs/troubleshooting/network-tips.md"
  outdated: false
---

本指南展示如何在受限或慢速网络环境中构建和运行，而无需修改仓库文件。您将使用小型本地覆盖文件和 compose 覆盖，以保持代码库整洁。

本文将解决：

- Hugging Face 模型下载被阻止/缓慢
- Docker 构建期间 Go 模块获取被阻止
- mock-vLLM 测试镜像的 PyPI 访问

## TL;DR：选择您的路径

- 最快且最可靠：使用 `./models` 中的本地模型，完全跳过 HF 网络。
- 否则：通过 compose 覆盖挂载 HF 缓存 + 设置镜像环境变量。
- 构建时：使用覆盖 Dockerfile 设置 Go 镜像（提供示例）。
- mock-vllm：使用覆盖 Dockerfile 设置 pip 镜像（提供示例）。

您可以根据情况混合使用这些方法。

## 1. Hugging Face 模型

除非您在本地提供模型，否则路由将在首次运行时下载嵌入模型。如果可能，优先选择方案 A。

### 方案 A — 使用本地模型（无外部网络）

1) 使用任何可达的方法（VPN/离线）将所需模型下载到仓库的 `./models` 文件夹中。示例布局：

   - `models/all-MiniLM-L12-v2/`
   - `models/category_classifier_modernbert-base_model`

2) 在 `config/config.yaml` 中，指向本地路径。示例：

   ```yaml
   bert_model:
     # 指向 /app/models 下的本地文件夹（已由 compose 挂载）
     model_id: /app/models/all-MiniLM-L12-v2
   ```

3) 无需额外环境变量。`deploy/docker-compose/docker-compose.yml` 已挂载 `./models:/app/models:ro`。

### 方案 B — 使用 HF 缓存 + 镜像

创建 compose 覆盖以持久化缓存并使用区域镜像（以下示例使用中国镜像）。保存为仓库根目录的 `docker-compose.override.yml`（当您同时指定两者时，Compose 将自动与 `deploy/docker-compose/docker-compose.yml` 合并）：

```yaml
services:
  semantic-router:
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    environment:
      - HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface
      - HF_HUB_ENABLE_HF_TRANSFER=1
      - HF_ENDPOINT=https://hf-mirror.com  # 示例镜像端点（中国）
```

可选：在宿主机上预热缓存（仅当您安装了 `huggingface_hub` 时）：

```bash
python -m pip install -U huggingface_hub
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir="~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2")
PY
```

## 2. 使用 Go 镜像构建（Dockerfile 覆盖）

构建 `tools/docker/Dockerfile.extproc` 时，Go 阶段可能在 `proxy.golang.org` 上卡住。创建覆盖 Dockerfile 以启用镜像，而无需修改原始文件。

1) 创建 `tools/docker/Dockerfile.extproc.cn`，内容如下：

```Dockerfile
# syntax=docker/dockerfile:1

FROM rust:1.90 AS rust-builder
RUN apt-get update && apt-get install -y make build-essential pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY tools/make/ tools/make/
COPY Makefile ./
COPY candle-binding/Cargo.toml candle-binding/
COPY candle-binding/src/ candle-binding/src/
RUN make rust

FROM golang:1.24 AS go-builder
WORKDIR /app

# Go 模块镜像（示例：goproxy.cn）
ENV GOPROXY=https://goproxy.cn,direct
ENV GOSUMDB=sum.golang.google.cn

RUN mkdir -p src/semantic-router
COPY src/semantic-router/go.mod src/semantic-router/go.sum src/semantic-router/
COPY candle-binding/go.mod candle-binding/semantic-router.go candle-binding/

# 预下载模块，如果镜像不可达则快速失败
RUN cd src/semantic-router && go mod download && \
    cd /app/candle-binding && go mod download

COPY src/semantic-router/ src/semantic-router/
COPY --from=rust-builder /app/candle-binding/target/release/libcandle_semantic_router.so /app/candle-binding/target/release/

ENV CGO_ENABLED=1
ENV LD_LIBRARY_PATH=/app/candle-binding/target/release
RUN mkdir -p bin && cd src/semantic-router && go build -o ../../bin/router cmd/main.go

FROM quay.io/centos/centos:stream10
WORKDIR /app
COPY --from=go-builder /app/bin/router /app/extproc-server
COPY --from=go-builder /app/candle-binding/target/release/libcandle_semantic_router.so /app/lib/
COPY config/config.yaml /app/config/
ENV LD_LIBRARY_PATH=/app/lib
EXPOSE 50051
COPY scripts/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
```

1) 通过扩展 `docker-compose.override.yml` 将 compose 指向覆盖 Dockerfile：

```yaml
services:
  semantic-router:
    build:
      dockerfile: tools/docker/Dockerfile.extproc.cn
```

## 3. Mock vLLM（通过 Dockerfile 覆盖配置 PyPI 镜像）

对于可选的测试配置文件，创建覆盖 Dockerfile 以配置 pip 镜像。

1) 创建 `tools/mock-vllm/Dockerfile.cn`：

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# Pip 镜像（示例：中国清华镜像）
RUN python -m pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -m pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

1) 扩展 `docker-compose.override.yml` 以对 `mock-vllm` 使用覆盖 Dockerfile：

```yaml
services:
  mock-vllm:
    build:
      dockerfile: Dockerfile.cn
```

## 4. 构建和运行

覆盖就绪后，正常构建和运行（Compose 将自动合并）：

```bash
# 使用覆盖构建所有镜像（显式引用重定位后的 compose 文件）
docker compose -f deploy/docker-compose/docker-compose.yml -f docker-compose.override.yml build

# 运行路由 + envoy
docker compose -f deploy/docker-compose/docker-compose.yml -f docker-compose.override.yml up -d

# 如果需要测试配置文件（mock-vllm）
docker compose -f deploy/docker-compose/docker-compose.yml -f docker-compose.override.yml --profile testing up -d
```

## 5. 出口受限的 Kubernetes 集群

Kubernetes 节点上的容器运行时不会自动复用宿主机 Docker 守护进程的设置。当镜像仓库缓慢或被阻止时，Pod 可能停留在 `ImagePullBackOff`。选择以下一种或组合多种缓解措施：

### 5.1 配置 containerd 或 CRI 镜像

- 对于由 containerd 支持的集群（Kind、k3s、kubeadm），编辑 `/etc/containerd/config.toml` 或使用 Kind 的 `containerdConfigPatches` 为 `docker.io`、`ghcr.io` 或 `quay.io` 等仓库添加区域镜像端点。
- 更改后重启 containerd 和 kubelet 以使新镜像生效。
- 避免将镜像指向回环代理，除非每个节点都能访问该代理地址。

### 5.2 预加载或侧载镜像

- 在本地构建所需镜像，然后推送到集群运行时。对于 Kind，运行 `kind load docker-image --name <cluster> <image:tag>`；对于其他集群，在每个节点上使用 `crictl pull` 或 `ctr -n k8s.io images import`。
- 当您知道镜像已存在于节点上时，修补部署以设置 `imagePullPolicy: IfNotPresent`。

### 5.3 发布到可访问的镜像仓库

- 标记并推送镜像到集群可达的仓库（云提供商仓库、私有托管的 Harbor 等）。
- 使用新镜像名称更新您的 `kustomization.yaml` 或 Helm values，如果仓库需要身份验证则配置 `imagePullSecrets`。

### 5.4 运行本地透传缓存

- 在同一网络内启动仓库代理（`registry:2` 或供应商特定缓存），在 containerd 中将其配置为镜像，并定期用您需要的镜像预热它。

### 5.5 调整后验证

- 使用 `kubectl describe pod <name>` 或 `kubectl get events` 确认拉取错误消失。
- 检查 `semantic-router-metrics` 等服务现在是否暴露端点并通过端口转发响应（`kubectl port-forward svc/<service> <local-port>:<service-port>`）。

## 6. 故障排除

- Go 模块仍然超时：
  - 验证 go-builder 阶段日志中是否存在 `GOPROXY` 和 `GOSUMDB`。
  - 尝试干净构建：`docker compose build --no-cache`。

- HF 模型下载仍然缓慢：
  - 优先选择方案 A（本地模型）。
  - 确保缓存卷已挂载且 `HF_ENDPOINT`/`HF_HUB_ENABLE_HF_TRANSFER` 已设置。

- mock-vllm 的 PyPI 缓慢：
  - 确认该服务正在使用 CN Dockerfile。
