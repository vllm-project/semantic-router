---
title: 快速开始
description: 安装模型运行时，运行一个模型，向它发送请求，再让路由器使用它。
translation:
  source_commit: "e201c95209820af2d88e900b7b14c8a982c0d244"
  source_file: "docs/model-runtime/quickstart.md"
  outdated: false
---

# 快速开始

大约十分钟内，你将安装模型运行时，在 CPU 上运行一个模型并向它提问，
然后让路由器用同一个模型做路由。

你需要 Linux 或 macOS、Python 3.10 或更新版本，以及约 3 GB 可用磁盘空间用于下载模型。
不需要 GPU。

## 1. 安装 {#1-install}

运行时是 Python 包 `vllm-srun`，与 `vllm-sr` CLI 一起发布，版本号相同。先从
[与你的硬件匹配的索引](https://pytorch.org/get-started/locally/)（CPU、CUDA 或 ROCm）
安装 PyTorch，再安装带 `runtime` extra 的 CLI。本指南在虚拟环境中使用 CPU 版：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "vllm-sr[runtime]"
```

在 ROCm 上，只有路由器镜像保证答案与已发布的模型包逐字节一致，因为镜像携带发布时所用的
PyTorch 构建。用官方 PyTorch wheel 经 `pip` 安装同样能运行这些模型，但答案可能略有差异
（见[选择模型](./choose-a-model.md#hardware)）。只有开发运行时才需要仓库 checkout
（`make model-runtime-install`）。

确认运行时能看到它的内置模型：

```bash
vllm-srun models
```

## 2. 运行模型 {#2-serve-a-model}

启动最小的决策模型 Decision 2.0 Kai。决策模型回答你用自然语言写下的问题。

```bash
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
```

首次启动会把模型（约 1.5 GB）下载到 Hugging Face 缓存，并对照固定的哈希校验每个文件。
健康检查通过后模型即就绪。在第二个终端中：

```bash
curl -s localhost:8100/health
```

模型加载完成并通过自检后，它返回 `{"status": "ready", ...}`；在此之前返回 HTTP 503 和当前阶段。

## 3. 发送请求 {#3-send-a-request}

对同一个请求同时提两个问题：它属于哪类工作（一个**选择题**，choice），
以及回答它是否需要多步推理（一个**是/否问题**，称为 **noul**）。

```bash
curl -s localhost:8100/v1/decisions -H 'content-type: application/json' -d '{
  "state": "Write a Python function that merges two sorted lists.",
  "questions": {
    "kind": {"type": "choice", "instructions": "What kind of work is this?",
             "criteria": {"code": "Writing or fixing code", "math": "Mathematics", "chat": "Anything else"}},
    "reasoning": {"type": "noul", "instructions": "Does answering this need multi-step reasoning?"}
  }
}'
```

`kind` 的答案给出选中的选项以及每个选项的概率。`reasoning` 的答案是回答为“是”的概率：

```json title="Response"
{
  "model": "Decision-2.0-Kai-0.6B",
  "answers": {
    "kind": {"type": "choice", "choice": "code", "probabilities": {"code": 0.504, "math": 0.133, "chat": 0.362}, "confidence": 0.106},
    "reasoning": {"type": "noul", "noul": 0.519}
  },
  "usage": {"input_tokens": 168, "output_tokens": 0}
}
```

在请求中加上 `"options": {"return_meta": true}`，响应还会带上 `meta`：作答的 revision、profile、设备以及耗时。在 16 个 CPU 核上，
这个请求约需 0.2 秒。`GET /v1/models` 显示已加载的模型、运行位置以及是否通过自检。

同一个命令也能运行分类器。用 Ctrl-C 停止服务，改为运行 Vela Domain 分类器：

```bash
vllm-sr serve vllm-sr/Vela-1.0-Encoder-307M-Domain --device cpu --port 8100
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["What is the derivative of x squared?"]}'
```

结果列出最可能的领域（`label`）以及全部 14 个领域各自的概率。

## 4. 在路由器中使用 {#4-use-it-from-the-router}

路由器会替你运行模型。把模型声明为 `provider: model_runtime` 的 **deployment**，
然后在 `decision` 信号中向它提问。把下面的内容保存为 `config.yaml`，并把 `vllm:8000`
换成为你的用户提供回答的 OpenAI 兼容后端：

```yaml
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: answer-model
  models:
    - name: answer-model
      backend_refs:
        - name: answer
          endpoint: vllm:8000
          protocol: http
routing:
  modelCards:
    - name: answer-model
  signals:
    decision:
      - name: needs_reasoning
        deployment: decision-kai
        question:
          type: noul
          instructions: Does answering this request need multi-step reasoning?
        predicate:
          gte: 0.7
  decisions:
    - name: think-first
      priority: 100
      rules:
        operator: AND
        on_unknown: no_match
        conditions:
          - type: decision
            name: needs_reasoning
      modelRefs:
        - model: answer-model
          use_reasoning: true
    - name: default-route
      priority: 1
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: answer-model
          use_reasoning: false
global:
  model_catalog:
    deployments:
      decision-kai:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: cpu
```

校验配置文件并启动路由器：

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
```

路由器会为 `decision-kai` 启动自己的运行时。首次启动时，它把模型的一份副本下载到
`config.yaml` 旁边的 `models/` 目录，供以后启动复用。通过路由器发送一个请求，看看它选择了哪条路由：

```bash
curl -s -D - -o /dev/null localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' -H 'x-vsr-debug: true' \
  -d '{"model": "auto", "messages": [{"role": "user", "content": "Plan a three-step proof that there are infinitely many primes."}]}' \
  | grep -i '^x-vsr-'
```

`x-vsr-selected-decision` 给出路由名称，`x-vsr-matched-decision-model` 列出匹配的决策信号。
模型仍在加载时，该信号为未知，`on_unknown: no_match` 会把请求送到 `default-route`。

如果想复用第 2 步启动的服务，而不是再运行一份模型，把 `artifact` 和 `device` 换成它的地址，
例如 `endpoint: http://host.docker.internal:8100`，并用 `--host 0.0.0.0` 启动那个服务，
让路由器容器能够访问它。

## 下一步 {#next-steps}

- [选择模型、规模和硬件](model-runtime/choose-a-model.md)
- 开启一个内置功能：[请求分类](model-runtime/guides/classify.md)、
  [检测 PII](model-runtime/guides/pii.md)、[拦截提示词攻击](model-runtime/guides/safety.md)、
  [使用 embeddings](model-runtime/guides/embeddings.md)
- [与路由器一起运行](model-runtime/deploy.md)：GPU、Kubernetes、共享运行时
- [故障排查](model-runtime/troubleshooting.md)
