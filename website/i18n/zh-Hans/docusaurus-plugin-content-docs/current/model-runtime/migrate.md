---
is_mtpe: false
title: 从原生绑定迁移
sidebar_label: 从原生绑定迁移
description: 更新使用了 candle、ONNX Runtime 或 OpenVINO 后端、旧模型名、NLI 解释器或模型运行时旧档位（profile）的配置。
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/model-runtime/migrate.md"
  outdated: true
---

# 从原生绑定迁移

:::note
`legacy.yaml` 示例来自早期版本，保留了当时的模型字段，例如模块的 `model_id` 以及绑定的 `contract` 和 `adapter`。`vllm-sr config migrate` 也会改写这些字段；[迁移模型资源](../installation/model-resource-migration.md)将每个早期字段与取代它的“部署加绑定”形式并列展示。
:::

早期版本在路由器内部用三种后端运行模型：candle、ONNX Runtime（`ort`）和 OpenVINO。
这些本地绑定已经由[模型运行时](model-runtime/overview.md)替代；显式配置的外部服务仍单独调用。
设备由 deployment 和启动时的平台选择决定。该版本的全部破坏性变更见[发布说明](release-notes/built-in-model-runtime.md)。

**升级后请重新验证路由结果。** 未指定模型的判断任务现在使用默认 Vela 2.0 部署；
这不保证与旧 Vela 1.0 专用模型得出相同结果。检查[默认模型与任务绑定](choose-a-model.md)，
用自己的请求验证阈值和准确度。

如果你的配置包含以下任何一项，就需要迁移：

- `provider: candle`、`provider: ort` 或 `provider: openvino`；
- deployment 上的 `precision`、`custom_ops_profile` 或 `compilation_cache_dir`；
- 模块上的 `variant`、`model_type`、`use_modernbert` 或 `use_mmbert_32k`；
- 指向 ONNX 或 OpenVINO 计算图文件的 binding `head`；
- 旧模型名，例如 `models/mom-domain-classifier`、
  `models/mmbert-embed-32k-2d-matryoshka` 或 `lettucedect`；
- NLI 解释器（`hallucination_explainer`、`nli_model`、`enable_nli_filtering`、`use_nli`）
  或响应缓存的 `polarity_guard` 设置；
- 使用 `backend: endpoint` 的幻觉检测器；
- 模型运行时 deployment 上的 `profile`（见[引擎参数](#engine-arguments)）。

路由器在启动时会拒绝这些设置，并提示你运行迁移命令。

## 1. 运行迁移 {#1-run-the-migration}

```bash
vllm-sr config migrate --config legacy.yaml --output legacy.migrated.yaml
```

该命令写出迁移后的文件，并打印它做的每一处修改，例如：

```text
Changes to review
  global.model_catalog.deployments.vela-domain
      provider candle -> model_runtime, device cuda:0
  global.model_catalog.deployments.vela-domain.precision
      removed fp16: the model runs in its published precision (dtype: auto); dtype: float16 runs it in half precision on a GPU
```

对于仍需你亲自处理的事项，它会打印 **Warning**：无法安全改写的、指向你自有模型的相对路径
（迁移后的文件保留该值，在你修正之前路由器会拒绝它），或因 embedding 模型改变而需要重新向量化的已存储向量
（见[更换 embedding 模型时重新向量化](#re-embed-when-the-embedding-model-changes)）。

## 2. 检查修改 {#2-review-the-changes}

| 之前 | 之后 |
| --- | --- |
| `provider: candle`、`ort` 或 `openvino` | `provider: model_runtime` |
| `device: cpu` | `device: cpu` |
| `device: cuda:N` | `device: cuda:N` |
| `device: rocm:N` 或 `migraphx:N` | `device: rocm:N` |
| `device: metal:0` | `device: mps` |
| OpenVINO 设备（`CPU`、`GPU`、`NPU` 等） | `device: cpu`；Intel GPU 可使用 `device: xpu:0`（尚未验证） |
| `precision: fp16` | 移除：模型以发布精度运行；`dtype: float16` 在 GPU 上以半精度运行（见[引擎参数](model-runtime/engine-args.md#precision)） |
| `precision: native` 或 `fp32` | 移除：默认的 `dtype: auto` 本就以发布精度运行，这些模型是 FP32 |
| `custom_ops_profile`、`compilation_cache_dir` | 移除：运行时自行选择内核 |
| 计算图 `head`，例如 `onnx/model_fa.onnx` | 移除：运行时自行选择模型的计算图 |
| `artifact: models/Vela-1.0-Encoder-307M-...` | `artifact: vllm-sr/Vela-1.0-Encoder-307M-...`，即 Hub 仓库 |
| `embeddings.semantic`，连同其中的 `embedding_config.backend: candle` 或 `openvino`，以及 `gemma_model_path` 或 `bert_model_path` | 移除：运行时没有 EmbeddingGemma 或 MiniLM 系列；`embeddings.default` 指定 embedding 部署，不改就是 Vela Embedding（见 [Embeddings](./guides/embeddings.md)） |
| 响应缓存、记忆或向量存储上的 `embedding_model: bert` 或 `gemma` | 移除：存储改用 `embeddings.default` 向量化；重新向量化已存储的向量 |
| 用 Vela Embedding 的存储上设置了它不提供的向量维度（例如 MiniLM 的 384） | 保留：记忆改为 256，响应缓存和向量存储改为 768，并按该维度重新创建集合或索引 |
| `global.router.model_selection.ml` 以及训练得到的 `knn`、`kmeans`、`svm` 和 `mlp` 选择器 | 移除：由不需要训练模型的 `systemone` 选择器替代 |
| 模块上的 `variant`、`model_type`、`use_modernbert`、`use_mmbert_32k` | 移除：运行时从模型读取架构 |
| 旧模型目录中的标签映射，例如 `category_mapping_path: models/mom-domain-classifier/category_mapping.json` | 移除：路由器读取所运行模型的标签 |
| fusion 算法的 `grounding.nli_contradiction_penalty` | `grounding.contradiction_penalty`：grounding 现在读取幻觉检测器 |
| 使用 `backend: endpoint`、`endpoint` 和 `model_id` 的幻觉检测器 | 该聊天服务的 `openai_chat` 部署，以及指向它的 `hallucination` 绑定；路径不是 `/v1` 的 endpoint 需要手动编写 |

### 更名的模型 {#model-names-that-changed}

较旧的任务模型由同一任务的 Vela 1.0 模型替代。替代模型使用相同的标签名，
因此你的信号规则和路由条件仍匹配原来匹配的内容。

| 旧模型 | 新模型 | 变化 |
| --- | --- | --- |
| `mom-domain-classifier`、`mmbert32k-intent-*` 及其别名 | `Vela-1.0-Encoder-307M-Domain` | 相同的 14 个领域 |
| `mom-pii-classifier`、`mom-mmbert-pii-detector`、`mmbert32k-pii-*` | `Vela-1.0-Encoder-307M-PII` | 相同的 17 种 PII 类型 |
| `mom-jailbreak-classifier`、`mmbert32k-jailbreak-*` | `Vela-1.0-Encoder-307M-Guard` | 相同的 `benign` / `jailbreak` 标签 |
| `mom-halugate-sentinel`、`mmbert32k-factcheck-*` | `Vela-1.0-Encoder-307M-FactCheck` | 相同的标签 |
| `mom-feedback-detector`、`mmbert32k-feedback-*` | `Vela-1.0-Encoder-307M-Feedback` | 新增 `NO_FEEDBACK`，用于不含反馈的消息 |
| `mmbert32k-modality-router-merged` | `Vela-1.0-Encoder-307M-Modality` | 相同的 `AR` / `DIFFUSION` / `BOTH` 标签 |
| `mom-halugate-detector`、LettuceDetect v1 和 v2 | `Vela-1.0-Encoder-307M-Halu` | 相同的输入；仍返回回答中的片段 |
| `mmbert-embed-32k-2d-matryoshka` | `Vela-1.0-Encoder-307M-Embedding` | **重新向量化**已存储的向量 |
| EmbeddingGemma（`mom-embedding-flash`）、MiniLM（`mom-embedding-light`） | `Vela-1.0-Encoder-307M-Embedding`，或 OpenAI 兼容的 embedding 端点 | **重新向量化**已存储的向量 |
| `multi-modal-embed-small` / `-large` | `Vela-1.0-Omni-Nano` / `-Mini` | **重新向量化**已存储的向量 |
| Qwen3-Embedding-0.6B（`mom-embedding-pro`） | 不变 | |

### 已退役的功能 {#features-that-were-retired}

- **NLI 解释器**（`hallucination_explainer`、`enable_nli_filtering`、本地检测器上的
  `include_explanation`、`use_nli`）。幻觉检查仍会标出回答中无依据的片段，只是不再为每个片段附加 NLI 判定。
- **响应缓存的 `polarity_guard` 设置。** 它的 NLI 层已移除，因此没有可选的内容了：
  能识别否定和反义词的词面防护始终运行。迁移会删除这个块。
- **OpenVINO。** Intel CPU 使用 `cpu` 运行模型；Intel GPU 可以使用 `xpu` 设备。
- **ONNX Runtime 的 MIGraphX 和 CK flash-attention 路径。** AMD GPU 通过 ROCm 版 PyTorch
  运行模型（`device: rocm:N`）。

## 3. 校验并启动 {#3-validate-and-start}

```bash
vllm-sr config validate --config legacy.migrated.yaml
vllm-sr serve --config legacy.migrated.yaml
```

对于已有的本地 Docker 栈，CLI 可能提示已保留保存的活动配置。先检查并将需要保留的
Dashboard 修改合并到 `legacy.migrated.yaml`，再加上 `--replace-active-config` 重新运行，
明确应用这份文件。如果已有活动的 Recipe 包，请先通过 Recipe 工作流更换或停用它。

首次启动时，运行时会下载尚未拥有的模型。[与路由器一起运行](model-runtime/deploy.md#check-what-is-running)
介绍如何查看每个 deployment 何时就绪。

## 完整示例 {#a-complete-example}

下面的配置在 AMD GPU 上用 ONNX Runtime 运行 embedding，在 NVIDIA GPU 上用 candle 运行领域分类器，
并使用响应缓存的 NLI polarity guard：

```yaml title="legacy.yaml"
version: v0.5
listeners: []
providers:
  defaults:
    model: answer-model
  models:
    - name: answer-model
      backend_refs:
        - name: answer
          endpoint: vllm:8000
          provider: vllm
recipes:
  - name: default
    model_bindings:
      embedding:
        deployment: vela-embedding
        contract: embedding.v1
        adapter: mmbert
        head: onnx/model_fa.onnx
global:
  model_catalog:
    deployments:
      vela-embedding:
        artifact: models/Vela-1.0-Encoder-307M-Embedding
        provider: ort
        device: rocm:0
        precision: native
        custom_ops_profile: ck_flash_attention
      vela-domain:
        artifact: models/Vela-1.0-Encoder-307M-Domain
        provider: candle
        device: cuda:0
        precision: fp16
  stores:
    response_cache:
      backend_type: memory
      polarity_guard:
        mode: lexical+nli
  plugins:
    - type: response_cache
      configuration:
        enabled: true
        scope: global
```

`vllm-sr config migrate --config legacy.yaml` 写出：

```yaml title="legacy.migrated.yaml"
version: v0.5
listeners: []
providers:
  defaults:
    model: answer-model
  models:
    - name: answer-model
      backend_refs:
        - name: answer
          endpoint: vllm:8000
          provider: vllm
recipes:
  - name: default
global:
  model_catalog:
    deployments:
      vela-domain:
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Domain
        provider: model_runtime
        device: cuda:0
      vela-embedding:
        artifact: vllm-sr/Vela-1.0-Encoder-307M-Embedding
        provider: model_runtime
        device: rocm:0
  stores:
    response_cache:
      backend_type: memory
  plugins:
    - type: response_cache
      configuration:
        enabled: true
        scope: global
```

响应缓存原本使用此前的默认模型 MiniLM；现在它用 `embeddings.default` 向量化，即文件在 AMD GPU 上声明的 `vela-embedding` 部署，它也取代了 embedding 绑定。
原先由模块 `model_id` 指定的防护模型和幻觉检测器，现在是以各自模型命名的部署，并通过 `bindings.jailbreak` 和 `bindings.hallucination` 绑定。
对迁移后的文件再次运行该命令不会做任何修改。

## 引擎参数 {#engine-arguments}

模型运行时以前用档位（profile）选择批处理和数值行为：`exact`（默认）、`batching`、`shared_context` 或 `max_speed`。现在它像 vLLM 一样默认对并发请求做批处理，改用 vLLM 风格的[引擎参数](model-runtime/engine-args.md)。路由器会拒绝仍设置 `profile` 的 deployment，`vllm-sr config migrate` 会改写它：

| 之前 | 之后 |
| --- | --- |
| `profile: exact` | 移除。并发请求现在共用前向，答案可能因舍入而变动；没有引擎参数能恢复每次前向只跑一个请求 |
| `profile: batching` | 移除：批处理就是默认行为 |
| `profile: shared_context` | `enable_shared_context: true`；其他请求的行现在也会共用前向 |
| `profile: max_speed` | `enable_shared_context: true`；它的批处理和更快的内核已是默认 |
| 其他任何 `profile` | 移除，并给出警告：设置它所代表的引擎参数 |

deployment 已设置的引擎参数会保留。命令行参数和请求选项也随之变化；使用已移除的参数会报错，并给出替代参数：

| 之前 | 之后 |
| --- | --- |
| `vllm-sr serve --runtime-profile P` | 引擎参数，例如 `--max-num-seqs` 和 `--dtype` |
| `vllm-srun serve --profile P` | 同样的引擎参数 |
| `vllm-srun serve --max-batch-tokens N` | `--max-num-batched-tokens N` |
| 请求上的 `"options": {"profile": "exact"}` | 拒绝（`invalid_request`）：请求按其 deployment 的引擎参数运行 |
| `GET /v1/models` 和响应 `meta` 中的 `profile` | `engine_args`、`dtype` 和 `numerics`（`exact` 或 `approximate`） |

## 更换 embedding 模型时重新向量化 {#re-embed-when-the-embedding-model-changes}

更换 embedding 模型会改变向量空间。路由器会把不同模型的向量隔离开，所以不会出错，
但已存储的数据不会沿用：

- **语义缓存：** 旧模型生成的条目不再命中；缓存会随新流量重新填充。
- **记忆：** 用旧模型存储的记忆不会返回。请重新添加需要保留的记忆。
- **向量存储与 RAG：** 用新模型重新索引你的文档。Milvus、Qdrant 和混合 RAG 后端在原先使用
  MiniLM（384 维）的地方改用 Vela Embedding（768 维）对查询向量化，因此请用 Vela Embedding
  重新向量化这些集合，或在 RAG 源的 `backend_config.embedding` 中指定为集合建索引的部署。
  请在你自己的流量上检查相似度阈值；不同模型的分数不可直接比较。

## Kubernetes 与 Docker {#kubernetes-and-docker}

路由器镜像不再包含原生库。它们包含 CPU 运行时，因此托管模型开箱即用。
请删除仅为 candle、ONNX Runtime 或 OpenVINO 存在的环境变量、init 容器或卷。
需要 GPU 时，运行 GPU 运行时并让路由器连接它；见[与路由器一起运行](model-runtime/deploy.md#on-kubernetes)。

如果使用 operator 部署，请在升级前从 `SemanticRouter` 资源中删除 `embedding_models`。
operator 会拒绝仍设置它的资源；其路由器用 `embeddings.default` 向量化，除非资源的配置指定了别的部署，否则就是 Vela Embedding。
请重新向量化由 EmbeddingGemma 存储的向量（见[更换 embedding 模型时重新向量化](#re-embed-when-the-embedding-model-changes)）。
