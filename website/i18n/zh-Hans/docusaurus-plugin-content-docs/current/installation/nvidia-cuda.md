---
title: NVIDIA CUDA 部署
description: 在 NVIDIA GPU 上运行 vLLM 后端，并可选择用 CUDA 加速 Semantic Router 的本地信号模型。
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/installation/nvidia-cuda.md"
  outdated: false
---

# 使用 NVIDIA CUDA 部署

模型服务器和 Semantic Router 是独立服务。常见部署将 Router 保留在 CPU 上，并把 NVIDIA GPU 交给 vLLM。当本地嵌入或分类器也需要 GPU 加速时，使用 Router 的 CUDA 镜像。

`--platform nvidia` 仅影响本地 Router 栈。它选择 CUDA Router 镜像，将 NVIDIA GPU 传入 Router 容器，并更改其生成的运行时配置，使受支持的本地信号模型优先使用 CUDA。它**不会**下载语言模型或启动 vLLM 服务器。

## 前置条件

- Linux，以及你计划运行的 vLLM 发行版所支持的 NVIDIA GPU；
- 使用当前 Semantic Router CUDA 镜像时需要 x86-64 主机；
- 与所选容器镜像兼容的 NVIDIA 驱动；
- Docker 和 NVIDIA Container Toolkit；
- 有足够的 GPU 内存用于 vLLM 模型、KV cache 以及任何 Router 侧模型；以及
- 一份完整的 Semantic Router 配置，并带有可到达的模型端点。

使用当前的 [vLLM NVIDIA 要求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)了解受支持的硬件。按照 [NVIDIA Container Toolkit 指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 安装并配置运行时，然后同时验证主机驱动和容器访问：

```bash
nvidia-smi
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

在容器命令能够看到预期 GPU 之前不要继续。第二条命令是 NVIDIA 的[示例工作负载](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html)。

## 启动并验证 vLLM 后端

以下示例遵循官方 [vLLM Docker 部署](https://docs.vllm.ai/en/latest/deployment/docker/)。它在端口 `8000` 上发布 OpenAI 兼容端点，并将下载的模型文件保存在命名卷中：

```bash
docker volume create vllm-huggingface-cache

docker run -d \
  --name vllm-nvidia \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  -p 8000:8000 \
  -v vllm-huggingface-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-0.6B
```

选择适合可用 GPU 的模型和 vLLM 参数。当模型需要身份验证时，将 `HF_TOKEN` 作为环境变量传入；不要把 token 放入镜像、命令历史或 Router 配置。对于受控部署，固定 vLLM 镜像和模型 revision，而不是依赖 `latest`。

等待模型加载完成，然后在添加 Router 之前测试 vLLM：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models

curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Reply with: ready"}],
    "max_tokens": 16
  }'
```

Router 校验无法证明后端能够加载模型或生成响应，因此在继续之前先修复任何直接的 vLLM 错误。

## 连接后端

在 canonical 配置中绑定已服务的模型。对于本地 Docker 栈，Router 可以通过 `host.docker.internal` 到达主机发布的端口：

```yaml
providers:
  defaults:
    model: local/qwen
  models:
    - name: local/qwen
      provider_model_id: Qwen/Qwen3-0.6B
      api_format: openai
      backend_refs:
        - name: nvidia-vllm
          endpoint: host.docker.internal:8000
          protocol: http
          provider: vllm
          weight: 1
```

这是 provider 片段，而不是完整的 Router 配置。将匹配的 model card 和路由添加到现有配方，或在控制面板中配置端点。`provider_model_id` 必须匹配 vLLM `/v1/models` 端点返回的模型。完整的最小文档见[配置](configuration)，后端绑定和路由策略见[模型、入口点与服务](../tutorials/global/models-entrypoints-serving)。

该示例在主机上发布端口 `8000` 以便直接测试。用主机网络控制限制该端口，或在生产部署中使用私有服务发现。不要将未认证的 vLLM 端点暴露给不受信任的网络。

## 在 NVIDIA 上运行 Router

如果 vLLM 应拥有全部 GPU 内存，将 Router 保留在 CPU 上：

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

要在 CUDA 上运行受支持的 Router 侧 ONNX 嵌入和分类器，使用 `--platform nvidia`。CLI 默认选择并拉取已发布的 `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest` 镜像：

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --platform nvidia --config config.yaml
```

对于源码检出，先构建维护中的 CUDA 镜像。`ifnotpresent` 策略会保留该本地构建，同时仍允许 CLI 获取缺失的配套镜像：

```bash
VLLM_SR_PLATFORM=nvidia make vllm-sr-build
vllm-sr serve \
  --platform nvidia \
  --config config.yaml \
  --image-pull-policy ifnotpresent
```

在生产中固定发行标签或 digest。如果 Router 与 vLLM 共享 GPU，请在有代表性的并发下测量内存和延迟；将小型、batch-one 的信号模型移到 CUDA 并不总是能改善端到端延迟。

## 验证已路由路径

检查本地栈和 Router 日志：

```bash
vllm-sr status
vllm-sr logs router | grep 'Using CUDA execution provider'
nvidia-smi
```

仅当活动配方加载受支持的本地 ONNX 模型时，才会出现 CUDA 日志。然后通过该配方暴露的入口点发送请求。如果你的配置使用另一个公共模型名称，请替换 `vllm-sr/auto`：

```bash
curl --fail --include http://127.0.0.1:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Explain prefix caching briefly."}],
    "max_tokens": 64
  }'
```

成功的直接 vLLM 请求证明模型服务器可用。成功的已路由请求证明 Router、配方和后端绑定可以一起工作。

## 故障排查

### Docker 拒绝 `--gpus all`

用 `nvidia-ctk` 配置 Docker，重启 Docker，并重复 NVIDIA 的示例容器命令。在调试 vLLM 或 Semantic Router 之前，先调试容器运行时。

### Router 使用 CPU

确认 `--platform nvidia` 选择了 `vllm-sr-cuda` 镜像，并且未启用 `VLLM_SR_NVIDIA_PRESERVE_CPU`。检查生成的运行时配置和启动日志，而不仅仅是源配方。没有本地 ONNX 信号模型的配方没有什么可以移到 CUDA。

### vLLM 或 Router 耗尽 GPU 内存

vLLM 模型、KV cache 和 Router 侧模型争夺同一设备内存。将 Router 留在 CPU 上，降低 vLLM 内存或并发设置，或将服务放在不同 GPU 上。不要假设静默的 CPU 回退能满足相同的延迟目标。

### 后端直接可用，但已路由请求失败

检查 Router 容器能否到达后端端点，并且 `provider_model_id` 与 `/v1/models` 完全匹配。在 Router 容器内，`localhost:8000` 指向 Router 自身；使用 `host.docker.internal:8000`、共享网络上的容器 DNS，或可到达的服务地址。

### Kubernetes 不调度 GPU

`--platform nvidia` 是本地容器快捷方式。对于 Kubernetes，通过 Helm values 或 Operator 选择 CUDA 镜像，并配置 GPU 资源、NVIDIA 设备插件和节点放置。部署边界见[配置工作流](configuration-workflows#helm)。
