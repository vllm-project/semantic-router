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
  -p 8090:8000 \
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

该命令只挂载模型缓存。不要将整个家目录挂载到模型服务容器中。该示例也省略了 `SYS_PTRACE`、未受限的 seccomp 配置文件和 `--trust-remote-code`；仅当经过审核且已固定的工作负载明确需要时，才添加更广泛的权限或远程模型代码。

根据可用硬件调整 `--max-model-len`、`--max-num-seqs`、张量并行和 GPU 内存利用率。以较小限制启动成功的模型，在复制这些参考值后可能失败或驱逐有用的缓存。

## 先验证后端

等待模型加载完成，然后独立于 Router 验证后端：

```bash
curl --fail http://127.0.0.1:8090/health
curl --fail http://127.0.0.1:8090/v1/models

curl --fail http://127.0.0.1:8090/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "qwen/qwen3.5-rocm",
    "messages": [{"role": "user", "content": "Reply with: ready"}],
    "max_tokens": 16
  }'
```

在直接生成请求成功之前不要继续。Router 校验检查的是路由配置；它并不能证明 provider 能够生成。

## 安装并配置 Semantic Router

安装 CLI：

```bash
curl -fsSL https://vllm-sr.ai/install.sh | \
  bash -s -- --channel stable --mode cli --runtime skip --no-launch
```

对于简单的单模型部署，打开 `http://localhost:8700` 的控制面板，添加位于 `vllm:8000` 的 OpenAI 兼容后端，并激活生成的配置。

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

通过 Envoy 使用自动入口点发送请求：

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

[Vela AMD 模型卡片](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)及完整配置显式选择十个任务模型的 GPU 执行。Embedding 和 Reranker 通过 ROCm 使用固定的 CK FlashAttention 图，保留 native 精度并拒绝 CPU fallback；分类器使用 MIGraphX。完整信号流水线在 8K 完成测量，独立 Embedding/Reranker 执行已验证至 32K。Hazard 保留已验证的 2,048-token 窗口及 32K 逻辑预算。这些结果不代表每个分类器都完成了 AMD 32K 验证。

连接已有的 OpenAI 兼容后端，使用 `--served-model-name vela-default`。配置预期地址是 `http://vllm:8000`：将后端接入 `vllm-sr-network` 并设置网络别名 `vllm`，或修改 endpoint。前面的多别名示例默认不提供 `vela-default`，需要添加该服务名。先用这个名称验证直连请求。为 Router 和生成后端保留足够内存与算力；选择 Router GPU 时使用 `VLLM_SR_AMD_ROUTER_VISIBLE_DEVICES`，deployment 中索引 `0` 指向可见 GPU。

```bash
curl --fail --location --output vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform amd --config vela-amd.yaml
```

平台标志选择镜像和设备访问，具名 deployment 选择实际 provider 与计算图；显式 CPU 选择仍然保留。MIGraphX 冷编译可能比缓存启动更慢。CLI 默认等待 1,800 秒；若实测需要更长时间，可用 `--startup-timeout SECONDS` 设置有界等待。超时后所属容器仍保留，可继续查看日志与就绪状态。

`/ready` 成功后，查看真实信号与时延：

```bash
curl --fail http://localhost:8080/ready
curl --fail 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vela-auto","text":"Debug this Python program and fix its error."}' \
  | jq '{decision_result, signal_confidences, signal_values, signal_errors, metrics, eval_trace}'
```

使用默认配方时，全部信号的 Preview 输入应保持在 8K 分类器预算内。Preview 不执行检索或生成。按配方和[神经重排](../tutorials/plugin/rag.md#neural-reranking)指南将文档入库，再使用 `vela-auto` 发送真实聊天请求验证 RAG。

### 可选的 Domain 和 FactCheck 32K ROCm 部署 {#optional-32k-domain-and-factcheck-on-rocm}

[Vela Domain](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-Domain) 和 [Vela FactCheck](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M-FactCheck) 提供固定 32K FP32 图 `onnx/model_rocm_32k.onnx`。以下配置已锁定包含该图的模型发布版本。

若需启用，在 `vela-amd.yaml` 中用以下片段替换这两个 binding 条目及两个完整的 deployment 条目。下面的 ROCm 条目会替换原 MIGraphX 条目，包括移除其 `compilation_cache_dir` 设置；保留配方的其余配置。

```yaml
routing:
  model_bindings:
    domain_classifier:
      deployment: domain-amd
      contract: label_distribution.v1
      adapter: modernbert
      head: onnx/model_rocm_32k.onnx
      mapping_path: models/Vela-1.0-Encoder-307M-Domain/category_mapping.json
    fact_check_classifier:
      deployment: factcheck-amd
      contract: label_distribution.v1
      adapter: modernbert
      head: onnx/model_rocm_32k.onnx
global:
  model_catalog:
    deployments:
      domain-amd:
        artifact: models/Vela-1.0-Encoder-307M-Domain
        revision: f6354f54adcf38770f635ad903be2b00577f6c11
        provider: ort
        device: rocm:0
        precision: native
        input:
          max_tokens: 32768
          overflow: reject
      factcheck-amd:
        artifact: models/Vela-1.0-Encoder-307M-FactCheck
        revision: 99ede1aba1563e59e416f744d25b3f6b7e9d8274
        provider: ort
        device: rocm:0
        precision: native
        input:
          max_tokens: 32768
          overflow: reject
```

用上面的相同命令校验并启动修改后的配置。每个 binding 始终使用选定的图，不会按请求长度自动切换。固定图会将短输入也填充到 32,768 tokens，增加时延和显存开销；若工作负载能容纳在 8K 预算内，可继续使用默认的 8K MIGraphX 部署。

容量规划可参考：一条使用这两个分类器的 27,001-token Preview 请求实测耗时 15.84 秒。独立的单分类器验证达到 38.70 GiB GPU 显存占用，还需为其他常驻模型及并发请求预留容量。这些测量仅覆盖 Domain/FactCheck 选项；其他启用的分类器仍保留各自输入上限，修改这两个 deployment 不代表全部十个模型的流水线支持 32K。

## 生产检查清单

- 固定 Router、vLLM 镜像和模型 revision。
- 只给容器所需的设备、文件和网络访问。
- 当策略依赖于真实能力或成本差异时，使用不同的 provider 端点。
- 保护后端端口，避免不受信任的网络访问。
- 根据实测内存使用来确定上下文、并发和并行度。
- 监控后端健康、排队、GPU 内存和已路由生成，而不仅仅是 Router 配置校验。
