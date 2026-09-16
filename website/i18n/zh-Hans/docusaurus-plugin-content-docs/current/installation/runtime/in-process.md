---
title: 进程内模型
description: 在本地运行 Vela，选择 CPU 或 GPU 推理，并查看实际路由信号。
translation:
  source_commit: "96399a94b9030d66f46c5d45f9a838defc091153"
  source_file: "docs/installation/runtime/in-process.md"
  outdated: false
---

在 Router 内运行 [Vela 模型](../../tutorials/global/vela-models.md)，完成请求分类、嵌入和文档重排。启用相应功能后，CLI 会下载已注册的模型。生成聊天回答仍需要独立的后端。

## 选择硬件 {#choose-your-hardware}

按照[安装指南](../installation.md)安装 CLI 和匹配的 Router 镜像。

| 硬件 | 运行时 | Vela 模型格式 | 配置入口 |
| --- | --- | --- | --- |
| CPU | Candle | 原生权重 | 下方示例 |
| CPU | ONNX Runtime | ONNX | `provider: ort`、`device: cpu` |
| AMD GPU | ONNX Runtime ROCm 或 MIGraphX | ONNX | [Vela AMD 配方](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md) |
| NVIDIA GPU | Candle CUDA 构建 | 原生权重 | `provider: candle`、`device: cuda:0`；在目标 GPU 上验证 |
| Apple GPU | Candle Metal 构建 | 兼容的原生权重 | `provider: candle`、`device: metal:0`；确认模型兼容性 |

Candle 支持 GPU 索引 `0`；BERT 和 BERT LoRA 模型不支持 Metal。CPU 和 AMD 路径已有运行验证，NVIDIA 和 Apple 部署需要在目标硬件上验证。独立部署的模型见[外部服务](external.md)。

## 在 CPU 上运行 Vela 分类器 {#run-a-vela-classifier-on-cpu}

本例使用 Vela Domain 识别编程请求，并由现有后端回答。将 `vllm:8000` 替换为 Router 容器可访问的地址，保存为 `config.yaml`：

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
  model_bindings:
    domain_classifier:
      deployment: vela-domain
      contract: label_distribution.v1
      adapter: modernbert
      mapping_path: models/Vela-1.0-Encoder-307M-Domain/category_mapping.json
  signals:
    domains:
      - name: computer science
        description: Programming and computer science requests.
        mmlu_categories: [computer science]
  decisions:
    - name: programming
      priority: 100
      rules:
        operator: AND
        on_unknown: fail_request
        conditions:
          - type: domain
            name: computer science
      modelRefs:
        - model: answer-model
global:
  model_catalog:
    deployments:
      vela-domain:
        artifact: models/Vela-1.0-Encoder-307M-Domain
        provider: candle
        device: cpu
        precision: fp32
        input:
          max_tokens: 512
          overflow: reject
```

**Deployment** 选择模型、引擎、设备和输入预算；**binding** 将其连接到配方中的功能。本例的 `domain_classifier` 使用 `vela-domain`，匹配与未匹配请求都发送到 `answer-model`。修改决策的后端或插件即可应用你的路由策略。

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
curl -fsS http://localhost:8080/ready
curl -fsS 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","text":"Help me debug this Python program."}' \
  | jq '{decision_result, signal_confidences, signal_errors, metrics}'
```

Preview 实际运行分类器，返回决策、信号分数、错误和耗时，但不调用回答后端。通过以下请求测试完整链路：

```bash
curl -fsS http://localhost:8899/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Explain Python tuples briefly."}],"max_tokens":128}'
```

## 在 AMD 上运行 Vela {#run-vela-on-amd}

[Vela AMD 配方](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)提供全部十个任务模型，包括嵌入和 RAG 重排，并配置好模型版本、ONNX 图、设备及 Preview 示例。

```bash
curl -fL -o vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform amd --config vela-amd.yaml
```

发送聊天请求前，连接配方中的 `vela-default` 后端。`--platform amd` 选择镜像和 GPU 访问方式；各模型的位置仍由显式 deployment 决定。首次启动可能需要数分钟编译 MIGraphX 模型，见[启动排查](lifecycle-diagnostics.md#check-startup)。

| AMD 配方组件 | 输入上限 |
| --- | --- |
| 完整路由信号链路 | 8,192 tokens |
| 独立 Embedding 和 Reranker | 32,768 tokens |
| Hazard | 在 32,768-token 请求中使用 2,048-token 窗口 |

所有上限均包含特殊 token。完整 AMD 链路目前为 8K，即使其中的检索模型单独支持 32K。

## 选择输入预算 {#choose-an-input-budget}

本地分类器默认使用 512 tokens。对于支持更长输入的权重和 ONNX 图，设置 deployment 的 `input.max_tokens` 即可增加预算。例如，原生 Vela CPU 路径可设为 `32768`。`overflow: reject` 会拒绝超长输入。

长输入会显著增加 CPU 推理耗时。选择覆盖实际负载的最小预算，并测量质量与时延。扫描长请求中的局部风险见[安全模型输入策略](safety.md#native-classifier-context)；嵌入信号另有[完整上下文设置](embeddings.md#input-policy)。

## 添加其他模型或任务 {#add-another-model-or-task}

- [嵌入模型](embeddings.md)：语义匹配、记忆、缓存和检索。
- [安全模型](safety.md)：Guard、Safety、Hazard 和 PII。
- [神经重排](../../tutorials/plugin/rag.md#neural-reranking)：绑定 `rag.reranker`，启用 RAG 插件的 `rerank`。
- [分类器信号](../../tutorials/signal/learned/classifier.md)：自定义标签及独立类别分数。

自定义模型可直接使用本地目录，无需注册。目录应包含完整权重、tokenizer、配置、任务标签及 ONNX 外部张量文件。LoRA 部署需要合并后的权重。`modernbert` 等名称选择推理 adapter，模型仍需具备适合该任务的分类头。

ONNX 分类器使用 `provider: ort`，选择设备，并在 binding 中指定 `head`，例如 `onnx/model.onnx`。GPU 专用图应采用模型提供的运行配置。Router 会拒绝不可用的 GPU provider 和 CPU 回退。

Binding 属于配方；放在相应配方的 `routing` 下即可独立更换模型。完整字段见[配置参考](../../api/configuration-schema.mdx)，更新流程见[运行中模型更新](lifecycle-diagnostics.md#update-a-running-model)。

## MIGraphX 高级设置 {#advanced-migraphx-settings}

设置 `compilation_cache_dir` 可在重启后复用已编译的模型。此可选设置要求 `provider: ort`、`migraphx:N` 设备，以及模型目录之外可持久保存且可写的绝对路径目录，默认关闭。

模型、GPU、精度或编译器发生变化时，可能需要重新编译。与部署配置冲突的设置见 [AMD 故障排查](lifecycle-diagnostics.md#amd-startup-problems)。
