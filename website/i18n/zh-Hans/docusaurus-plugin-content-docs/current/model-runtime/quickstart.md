---
title: 快速开始
description: 安装模型运行时，运行一个模型，向它发送请求，再让路由器使用它。
translation:
  source_commit: "2e7e0775e88b0fc9426daf7c4fa0fe6e0ec20654"
  source_file: "docs/model-runtime/quickstart.md"
  outdated: false
---

# 快速开始

大约十分钟内，你将安装 `vllm-sr` CLI，在 CPU 上运行一个模型并向它提问，
然后让路由器用同一个模型做路由。

你需要装有 Docker 或 Podman 的 Linux、macOS 或 WSL2、Python 3.10 或更新版本，以及几 GB 可用磁盘空间，
用于路由器镜像和模型下载。不需要 GPU。engine 模式（`vllm-sr serve ARTIFACT --engine`）晚于 0.4.0 版本；
见[发布渠道说明](../installation/installation.md)。

## 1. 安装 {#1-install}

把 CLI 装进一个虚拟环境：

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install vllm-sr
```

模型运行时是 Python 包 `vllm-srun`，只随路由器镜像发布。`vllm-sr serve ARTIFACT --engine` 在 `vllm-sr`
镜像中启动同一实例前端及受管理的模型worker，CLI 首次使用时拉取该镜像，所以你的机器上不用再装别的东西。在 GPU 主机上，
`--platform rocm` 或 `--platform cuda` 选用 `vllm-sr-rocm` 或 `vllm-sr-cuda` 镜像并把 GPU
透传进容器；在 macOS 上运行时只用 CPU，因为那里的容器拿不到 GPU。镜像携带发布时所用的 PyTorch
构建，所以答案就是模型发布时的答案（见[选择模型](./choose-a-model.md#hardware)）。

不加 `--engine`（简写 `-e`）的每次启动都使用 Router 模式，包括已保存的 Engine
配置。MODEL 只修改默认判断模型的 artifact，保留已有副本位置和 profile；全新配置
默认使用 Vela 2.0 0.3B。`--platform auto` 检查实际部署目标，可显式选择 `cpu`、`cuda`
或 `rocm`；`--device-ids` 使用 Docker 主机 GPU 编号并遵守已有可见设备掩码。

## 2. 运行模型 {#2-serve-a-model}

启动最小的决策模型 Decision 2.0 Kai。决策模型回答你用自然语言写下的问题。

```bash
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine --platform cpu
```

在 AMD GPU 上，用下面的命令在第一块 GPU 上运行同一个模型：

```bash
vllm-sr serve vllm-sr/Decision-2.0-Kai-0.6B --engine --platform rocm --device-ids 0
```

首次使用时 `vllm-sr-rocm` 镜像约需下载 6.5 GB。`--device-ids N` 选择另一块主机 GPU；canonical YAML 使用映射后的 `rocm:N` 运行时序号。

CLI 启动持久运行的前端、Dashboard 和模型 worker。首次启动下载模型到实例的
模型缓存，后续启动复用缓存。`vllm-sr status` 查看状态，`vllm-sr stop` 停止实例。

首次 Engine 配置会在默认 listener 发布选中的模型。已有配置保留自己的
`listeners[].systemone.models` 和 API Key；模式或模型切换不会扩展授权范围。
在第二个终端查看原生模型列表：

```bash
curl -s localhost:8899/v1/systemone/models
```

## 3. 发送请求 {#3-send-a-request}

对同一个请求同时提两个问题：它属于哪类工作（一个**选择题**，choice），
以及回答它是否需要多步推理（一个**是/否问题**，称为 **noul**）。

```bash
curl -s localhost:8899/v1/systemone -H 'content-type: application/json' -d '{
  "model": "vllm-sr/Decision-2.0-Kai-0.6B",
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
  "model": "vllm-sr/Decision-2.0-Kai-0.6B",
  "answers": {
    "kind": {"type": "choice", "choice": "code", "probabilities": {"code": 0.504, "math": 0.133, "chat": 0.362}, "confidence": 0.106},
    "reasoning": {"type": "noul", "noul": 0.519}
  },
  "usage": {"input_tokens": 168, "output_tokens": 0}
}
```

在请求中加上 `"options": {"return_meta": true}` 可查看作答 revision、profile、设备和耗时。
记录的 worker benchmark 在 16 个 CPU 核上约需 0.2 秒。`POST /v1/decisions` 是
`/v1/systemone` 的别名，都要求明确的公开模型 ID。原生模型发现使用
`/v1/systemone/models`，`/v1/models` 仍属于 Chat。classify、embeddings、rerank
和 bundle 属于独立的 worker API，参见[任务指南](model-runtime/guides/classify.md)。

## 4. 在路由器中使用 {#4-use-it-from-the-router}

路由器会替你运行模型。把模型声明为 `provider: model_runtime` 的 **deployment**，
然后在 `decision` 信号中向它提问。把下面的内容保存为 `config.yaml`，并把 `host.docker.internal:8000`
换成为你的用户提供回答的 OpenAI 兼容后端：

```yaml
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
    systemone:
      models: [vllm-sr/Decision-2.0-Kai-0.6B]
providers:
  defaults:
    model: answer-model
  models:
    - name: answer-model
      backend_refs:
        - name: answer
          endpoint: host.docker.internal:8000
          protocol: http
routing:
  modelCards:
    - name: answer-model
  signals:
    decision:
      - name: needs_reasoning
        deployment: primary
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
      primary:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Kai-0.6B
        device: cpu
```

校验配置文件，然后使用这份配置重启到 Router 模式：

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml --replace-active-config
```

`--replace-active-config` 用刚写入的文件替换此前保存的 Engine 配置。
后续重启可以省略它，以保留 Dashboard 中的修改。`primary` 部署同时回答路由问题
和直接 System One 请求；listener 显式发布它的原生模型名。
然后发送 Chat 请求查看选择的路由：

```bash
curl -s -D - -o /dev/null localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' -H 'x-vsr-debug: true' \
  -d '{"model": "vllm-sr/auto", "messages": [{"role": "user", "content": "Plan a three-step proof that there are infinitely many primes."}]}' \
  | grep -i '^x-vsr-'
```

`x-vsr-selected-decision` 给出路由名称，`x-vsr-matched-decision-model` 列出匹配的决策信号。
如果运行时之后重启，在模型恢复之前该信号为未知，`on_unknown: no_match` 会在此期间把请求送到 `default-route`。

Engine 与 Router 共享同一个受管理实例。若要接入独立运营的 worker，明确配置
`endpoint` 和 `served_name`；[部署指南](model-runtime/deploy.md) 说明其独立 API 和生命周期。

## 下一步 {#next-steps}

- [选择模型、规模和硬件](model-runtime/choose-a-model.md)
- 开启一个内置功能：[请求分类](model-runtime/guides/classify.md)、
  [检测 PII](model-runtime/guides/pii.md)、[拦截提示词攻击](model-runtime/guides/safety.md)、
  [使用 embeddings](model-runtime/guides/embeddings.md)
- [与路由器一起运行](model-runtime/deploy.md)：GPU、Kubernetes、共享运行时
- [故障排查](model-runtime/troubleshooting.md)
