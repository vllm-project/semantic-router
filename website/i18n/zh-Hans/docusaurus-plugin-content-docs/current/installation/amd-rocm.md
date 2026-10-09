---
title: AMD ROCm 部署
description: 连接 AMD vLLM 后端，并在 AMD GPU 上运行 Vela 路由模型。
translation:
  source_commit: "96399a94b9030d66f46c5d45f9a838defc091153"
  source_file: "docs/installation/amd-rocm.md"
  outdated: true
---

# 使用 AMD ROCm 部署

Semantic Router 可以在 CPU 上运行，同时由 vLLM 在 AMD Instinct GPU 上服务所选模型。本指南先启动一个 ROCm 后端，直接验证它，然后再将其连接到本地 Router 栈。若还需要在 AMD 上运行全部十个 Vela 路由任务模型，使用下文的 [Vela AMD 配方](#run-vela-routing-models-on-amd)。

该示例用一个 checkpoint 对应多个已服务模型别名，以便维护中的 `balance` 配方可以演练其路由通道。这对功能评估有用，但并不会把一个 checkpoint 变成多个模型。在生产环境中，将每个逻辑 provider 绑定到具备配方所声明能力、容量和运行成本的后端。

## 前置条件

- 主机和 GPU 受所选 vLLM 镜像中的 ROCm 版本支持；
- Docker 能够访问 `/dev/kfd` 和 `/dev/dri`；
- 有足够的 GPU 内存用于模型、上下文上限和并发设置；
- 持久的 Hugging Face 缓存目录；以及
- 能够下载模型的网络访问，除非模型已经缓存。

在开始大规模下载之前，确认设备可见：

```bash
rocminfo | head
docker run --rm \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  rocm/dev-ubuntu-24.04:latest rocminfo | head
```

在受控环境中固定镜像 digest 和模型 revision。下面的标签是便于阅读的示例，并不保证不可变。

## 启动 vLLM 后端

创建本地 Router 栈使用的网络，并选择缓存目录：

```bash
docker network inspect vllm-sr-network >/dev/null 2>&1 || \
  docker network create vllm-sr-network

export VLLM_HF_CACHE=/mnt/data/huggingface-cache
mkdir -p "$VLLM_HF_CACHE"
```

启动参考后端：

```bash
docker run -d \
  --name vllm \
  --network vllm-sr-network \
  --restart unless-stopped \
  -p 8000:8000 \
  -v "$VLLM_HF_CACHE:/root/.cache/huggingface" \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --ipc=host \
  --shm-size=32g \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_USE_AITER_UNIFIED_ATTENTION=1 \
  -e VLLM_ROCM_USE_AITER_MHA=0 \
  --entrypoint python3 \
  vllm/vllm-openai-rocm:v0.17.0 \
  -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-122B-A10B-FP8 \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name \
      qwen/qwen3.5-rocm \
      google/gemini-2.5-flash-lite \
      google/gemini-3.1-pro \
      openai/gpt5.4 \
      anthropic/claude-opus-4.6 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --max-model-len 262144 \
    --language-model-only \
    --max-num-seqs 128 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.85
```

主机端口只用于你自己检查后端；Router 通过 `vllm-sr-network` 上的 `vllm:8000` 访问它。不要使用 8090：本地栈的 sr-bench 服务占用该端口，否则 `vllm-sr serve` 会因“sr-bench port 8090 is already in use”而停止。设置 `VLLM_ROCM_USE_AITER=1` 时，首次启动会编译 AITER 内核，在空闲 CPU 核心较少的主机上可能需要几十分钟。

该命令只挂载模型缓存。不要将整个家目录挂载到模型服务容器中。该示例也省略了 `SYS_PTRACE`、未受限的 seccomp 配置文件和 `--trust-remote-code`；仅当经过审核且已固定的工作负载明确需要时，才添加更广泛的权限或远程模型代码。

根据可用硬件调整 `--max-model-len`、`--max-num-seqs`、张量并行和 GPU 内存利用率。以较小限制启动成功的模型，在复制这些参考值后可能失败或驱逐有用的缓存。

## 先验证后端

等待模型加载完成，然后独立于 Router 验证后端：

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/v1/models

curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen/qwen3.5-rocm",
    "messages": [{"role": "user", "content": "Reply with: ready"}],
    "max_tokens": 16
  }'
```

在直接生成请求成功之前不要继续。Router 校验检查的是路由配置；它并不能证明 provider 能够生成。

## 安装并配置 Semantic Router

按[快速开始](installation/installation.md#安装)安装 CLI。只安装 CLI、不启动栈：

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --mode cli --runtime skip --no-launch
```

对于简单的单模型部署，启动栈。`vllm-sr serve` 自动检测执行后端；`--platform rocm` 显式选择 ROCm 镜像与设备访问，显式配置的模型放置保持不变（见[在 AMD 上运行 Vela 路由模型](#run-vela-routing-models-on-amd)）：

```bash
vllm-sr serve
```

然后打开 `http://localhost:8700` 的控制面板，以 vLLM 为提供方、填入 served model name 和地址 `vllm:8000` 接入模型，并激活生成的配置。

若要评估维护中的 balance 配方，请将其下载到当前工作区，而不是依赖仓库相对路径：

```bash
curl --fail --location \
  --output balance.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/balance/config.yaml

vllm-sr config validate --config balance.yaml
vllm-sr serve --config balance.yaml
```

balance 配方期望示例后端暴露的五个别名。阅读其 [Model Card](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/README.md)，了解预期用途、路由行为、数据处理和限制。在替换别名、阈值、价格或 provider 角色之前，先分叉配置。

## 验证已路由路径

通过 Router 的监听器使用自动入口点发送请求：

```bash
curl --fail --include http://127.0.0.1:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "vllm-sr/auto",
    "messages": [{"role": "user", "content": "Explain prefix caching briefly."}],
    "max_tokens": 64
  }'
```

确认响应成功，并检查路由标头中的所选决策和 provider 模型。使用配方维护的探针进行更广泛的路由评估；使用有代表性的应用请求，衡量实际部署上的回答质量和运行行为。

## 在 AMD 上运行 Vela 路由模型 {#run-vela-routing-models-on-amd}

Router 自身的模型（Vela 分类器、embedding、reranker 和决策模型）运行在[模型运行时](model-runtime/overview.md)中。在 AMD Instinct MI300X 和 MI325X GPU 上，运行时通过 ROCm 版 PyTorch 执行这些模型，该路径已经验证。`--platform rocm` 选择 AMD 镜像，其中包含运行时和经过验证的软件栈（ROCm 7.2 版 PyTorch 2.12、FLA 0.5.2，以及为 ROCm 构建的 `causal-conv1d` 1.7.0），并把 GPU 传给 Router。每个模型加载时都会用在该栈上核验过的参考答案自检；见[选择模型](model-runtime/choose-a-model.md#hardware)。

[Vela AMD 模型卡片](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)及完整配置把全部十个任务模型放在 `rocm:0` 上。连接已有的 OpenAI 兼容后端，使用 `--served-model-name vela-default`。配置预期地址是 `http://vllm:8000`：将后端接入 `vllm-sr-network` 并设置网络别名 `vllm`，或修改 endpoint。先用这个名称验证直连请求。为 Router 和生成后端保留足够内存与算力；选择 Router GPU 时使用 `VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES`，deployment 中索引 `0` 指向可见 GPU。

```bash
curl --fail --location --output vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform rocm --config vela-amd.yaml
```

平台标志选择镜像和设备访问；每个 deployment 的 `device` 决定其模型在哪里运行，显式的 CPU 选择仍然保留。首次启动会下载模型；CLI 默认等待 1,800 秒，可用 `--startup-timeout SECONDS` 设置更长的有界等待。超时后所属容器仍保留，可继续查看日志与就绪状态。

`/ready` 成功后，查看真实信号与时延：

```bash
curl --fail http://localhost:8080/ready
curl --fail 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vela-auto","text":"Debug this Python program and fix its error."}' \
  | jq '{decision_result, signal_confidences, signal_values, signal_errors, metrics, eval_trace}'
```

Preview 不执行检索或生成。按配方和[神经重排](../tutorials/plugin/rag.md#neural-reranking)指南将文档入库，再使用 `vela-auto` 发送真实聊天请求验证 RAG。

### 更长的输入 {#longer-inputs}

在 ROCm 上，每个 Vela 任务模型最多读取 32,768 tokens。用 `input.max_tokens` 设置 deployment 接受的最长输入；短请求不会被填充到固定长度，因此仍然很快：

```yaml
global:
  model_catalog:
    deployments:
      domain-amd:
        provider: model_runtime
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        device: rocm:0
        input:
          max_tokens: 32768
          overflow: truncate
```

`truncate` 对前 32,768 个 token（含特殊 token）分类并报告已截断；`reject` 则对更长的输入让该信号保持未知。此限制只作用于分类器，不会缩短聊天请求，也不改变生成模型的上下文窗口。Guard 和 PII 用重叠窗口扫描长输入（`overflow: window`），见[提示词攻击与不安全内容](model-runtime/guides/safety.md)和[检测 PII](model-runtime/guides/pii.md)。

早期版本在这里选择固定的 ONNX Runtime 计算图（`head: onnx/model_rocm_32k.onnx`）和 MIGraphX 编译缓存。`vllm-sr config migrate` 会移除这些设置，见[从原生绑定迁移](model-runtime/migrate.md)。

## 生产检查清单

- 固定 Router、vLLM 镜像和模型 revision。
- 只给容器所需的设备、文件和网络访问。
- 当策略依赖于真实能力或成本差异时，使用不同的 provider 端点。
- 保护后端端口，避免不受信任的网络访问。
- 根据实测内存使用来确定上下文、并发和并行度。
- 监控后端健康、排队、GPU 内存和已路由生成，而不仅仅是 Router 配置校验。
