# Vela Router 模型 {#vela-router-models}

## 概览 {#overview}

[Vela 1.0](https://huggingface.co/collections/llm-semantic-router/vela-10)
是包含十四个已发布 checkpoint 的智能路由模型家族。当前 Router 模型注册表包含 Vela 307M Encoder 基座与十个任务模型，覆盖路由、提示词保护、内容安全、检索和重排，并将每个版本固定到不可变的 revision。

| 模型 | 作用 |
| --- | --- |
| Encoder | 适配新路由任务的共享基座 |
| Domain | 识别 14 类请求主题 |
| Guard | 检测提示注入和越狱攻击 |
| Safety | 检测不安全内容 |
| Hazard | 识别 12 类独立内容风险 |
| PII | 提取 17 类个人信息实体 |
| FactCheck | 判断请求是否需要事实核查 |
| Feedback | 四类反馈及中性 `NO_FEEDBACK` 类别 |
| Modality | 从文本请求判断文本、图像或混合输出意图 |
| Embedding | 多语言检索和语义匹配 |
| Reranker | 为查询与文档对计算相关性 |

完整模型名由 `Vela-1.0-Encoder-307M` 和任务后缀组成。Modality 从文本请求判断所需的输出模态。FactCheck 判断是否需要核查，并不验证回答的事实真伪。

公开模型集合还包括检查回答证据支持情况的 **Halu**，以及生成文本、图像和音频嵌入的 **Omni Nano / Omni Mini**。这三个 checkpoint 可直接使用并开展集成工作，尚未成为 Router 默认的幻觉检测或多模态组件。完整模型家族见 [Vela 1.0 发布公告](/blog/introduce-vela)。

## 解决什么问题 {#what-problem-does-it-solve}

共享模型家族提供任务专用信号与检索组件，并明确模型身份、输入预算和服务策略。

## 何时使用 {#when-to-use}

使用 Vela 执行内置路由任务，或从共享 Encoder 适配新任务。根据实际工作负载的质量和时延要求选择输入长度与表示大小。

## Omni checkpoints {#omni-checkpoints}

9 月 19 日发布的版本通过独立的文本、图像和音频编码器生成共享空间中的嵌入。这两个 checkpoint 仍供直接使用，尚未成为 Router 默认的多模态组件。

| Checkpoint | 总参数量 | 输出维度 | 文本上限 | 文本基座与读出 |
| --- | ---: | ---: | ---: | --- |
| [Omni Nano](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/0496b39a51c8199592e58cbff81c250f056bd94b/README.md) | 163.8M（163,771,288） | 384 | 512 tokens | 冻结的 GIST-small；CLS 读出 |
| [Omni Mini](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/README.md) | 1.36B（1,361,475,288） | 768 | 32,768 tokens | Qwen3-Embedding-0.6B；末 token Matryoshka 读出 |

参数量包含全部模态分支，包括新增的 CLAP 音频分支；整数和运行统计量 buffer 不计入参数。Nano 保留 CLS 读出及恒等投影；Mini 对最后一个非 padding token 的完整 1024 维状态归一化，取前 768 维后再次归一化。

Mini 默认使用适合跨模态比较的共享文本模式。纯文本任务可向 `encode_text` 传入 `task=` 或自定义 `instruction=`，二者互斥。检索预设要求指定 `role="query"` 或 `role="document"`，文档不添加前缀。32,768-token 预算包含指令前缀和特殊 token；超限默认报错，只有显式设置 `truncate=True` 才截断。Nano 保留 512-token 上限并拒绝超长输入。带指令的 English 评测使用固定的官方任务指令，不是搜索通用预设得到的结果。

两者都接收不超过 30 秒、保留原始采样率的 PCM，可为单声道或 channels-first 数组。传入真实采样率后，两个分支分别从原始波形生成 Whisper 的 16 kHz 输入与 CLAP 的 48 kHz 输入。存在更高采样率 PCM 时，不应先降采样到 16 kHz。CLAP 读取按端点分布、每段最多十秒的窗口，聚合归一化向量，经冻结的 TRAIN 统计量标准化及已训练的残差映射后，加到保留的未归一化 Whisper 仿射输出上，最后做 L2 归一化。文本/图像路径保留，音频路径经过新的训练与评测。

最新模型卡将路由与跨模态检索能力分别与[原始 small](https://huggingface.co/llm-semantic-router/multi-modal-embed-small/tree/fdf8e01b7b0f3a69ac1ac8e2a64dcb1ede177ba4)和[原始 large](https://huggingface.co/llm-semantic-router/multi-modal-embed-large/tree/e21cde3ccc414c56f504b322662f42c603a939ee)模型对比。分数范围为 0–100，每格表示原始模型 → 当前模型：

| 指标 | 原始 small → Nano | 原始 large → Mini |
| --- | ---: | ---: |
| Banking77，accuracy | 70.42 → 87.99 | 75.78 → 86.56 |
| MASSIVE English，accuracy | 65.95 → 81.97 | 72.31 → 80.96 |
| COCO，图像 → 文本，R@1 | 40.83 → 60.87 | 42.53 → 67.44 |
| COCO，文本 → 图像，R@1 | 30.18 → 55.82 | 35.04 → 61.60 |
| LibriSpeech，音频 → 文本，R@1 | 4.21 → 16.12 | 56.99 → 86.14 |
| LibriSpeech，文本 → 音频，R@1 | 9.58 → 20.34 | 78.58 → 94.90 |

双方使用相同的留出评测集，这些已知测试集在不同版本间复用，文本上限统一为 128 tokens。分类使用有标签 TRAIN 数据构建的原型；检索使用完整候选池和全部匹配正例。该协议不同于官方 MTEB 分类探针，完整指标和不确定性见下方链接的评测文档。

完整 MTEB 2.21.0 面板以 **Mean(TaskType)** 为主要指标，Mean(Task) 为补充指标。原始 small/large 模型尚未按这些完整面板协议评测。

| 面板与模式 | Mean(TaskType) | 全局排名 | 不大于该尺寸的排名 | 距该尺寸内最佳分数 | Mean(Task) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nano · English v2，41 个任务 · 默认文本 | 60.78 | 66/188 | 5/75 | 0.61 pp | 64.88 |
| Mini · English v2，41 个任务 · 官方指令文本 | 64.68 | 38/188 | 10/134 | 3.78 pp | 70.38 |
| Nano · MAEB audio-only，19 个任务 · 默认音频 | 52.34 | 19/64 | 6/27 | 3.51 pp | 43.59 |
| Mini · MAEB audio-only，19 个任务 · 默认音频 | 54.87 | 12/64 | 5/50 | 2.85 pp | 47.77 |

排名汇总 9 月 17 日注册表快照与当前 Vela 模型，包括单模态专用模型，已报告的评测协议存在差异。尺寸上限按全模型总参数量计算。**两个 Vela 模型都不在这两个完整面板的观测前沿上**，两种聚合指标下均如此。单任务优势包括 Nano 的 IMDb **91.95 accuracy**、NMSQA **62.90 max AP**，以及 Mini 的 Mridingham **68.14 accuracy**、SIBFLEURS **39.07 accuracy**；这些任务级前沿不能证明整体榜单领先。

Nano 的 English 分数通过未改变的冻结文本路径保留。Mini 的带指令 English 评测使用固定的官方指令，并重新运行了匹配的原始文本对照：Mean(TaskType) 从 **58.79 提升到 64.68**，38 个任务提升、3 个下降。该指令模式不能替代跨图像/音频比较中的默认共享模式。两个新音频路径均重新评测；虽然音频聚合分数提升，语音与文本检索仍有退步。实际测量身份、全部任务和退步项见固定版本的 [Nano 评测](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Nano/blob/0496b39a51c8199592e58cbff81c250f056bd94b/benchmarks/EVALUATION.md)、[Mini 评测](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/benchmarks/EVALUATION.md)及[指令模式匹配对照](https://huggingface.co/llm-semantic-router/Vela-1.0-Omni-Mini/blob/f7fafd36abf49adf88b1b2ec0186c68b008eeb07/benchmarks/instruction-mode.md#matched-raw-comparison)。这些面板不代表完整多语言或图像覆盖，也不证明时延或内存表现。

## 默认值与输入预算 {#defaults-and-input-budgets}

内置 Domain、Guard、Safety、PII、FactCheck、Feedback 和语义 Embedding 默认使用 Vela。参考配置还选择了 Vela Modality、Hazard 和 Reranker。只有 recipe 实际需要的模型才会加载；Encoder 基座用于训练，不作为额外路由信号加载。

默认阈值为 Guard **0.5**、FactCheck **0.95**、Feedback **0.7**。`NO_FEEDBACK` 不产生反馈匹配。Safety 独立于 Guard，有害内容不必同时被判断为提示词攻击。Hazard 使用与模型产物绑定的逐标签阈值，单一阈值不能代表它的发布决策策略。

输入预算由部署选择。模块的 `max_sequence_length: 0` 保留保守的 512-token 策略；Embedding 默认采用 22 层、768 维和 `full_context: false`。显式模型绑定可设置最多 **32,768 tokens**（含特殊 token）及 `overflow: reject`，超限输入会被拒绝，不会静默缩短。

## 配置 {#configuration}

以下片段将 Domain 绑定到 32K 输入预算的 CPU 部署。将它合入已有的完整配置，保留 providers、signals 和 decisions。

```yaml
routing:
  model_bindings:
    domain_classifier:
      deployment: vela-domain
      contract: label_distribution.v1
      adapter: modernbert
      mapping_path: models/Vela-1.0-Encoder-307M-Domain/category_mapping.json

global:
  model_catalog:
    deployments:
      vela-domain:
        artifact: models/Vela-1.0-Encoder-307M-Domain
        provider: candle
        device: cpu
        precision: fp32
        input:
          max_tokens: 32768
          overflow: reject
```

其他分类器使用相同的 deployment 与 consumer binding 结构。PII 返回 `token_spans.v1`；Embedding 使用 `mmbert` adapter 和 `embedding.v1`；Reranker 使用 `vela_reranker` 和 `relevance_scores.v1`。Adapter 名表示推理架构，与发布名称独立。完整契约见[进程内推理](/docs/installation/runtime/in-process)。

Hazard 使用独立分类契约 `label_scores.v1`，并通过 SHA-256 固定 `operating_point.json`。该策略绑定权重、tokenizer、执行设置、重叠窗口和十二个阈值。Decision 选择标签，不覆盖这些阈值；参考配置包含完整示例。

PII 的重叠扫描、Hazard 的窗口策略与整段文本分类不同。应按任务选择输入策略，并使用应用中的实际输入长度测量时延。

## 推理引擎与硬件 {#inference-engines-and-hardware}

原生 Vela 产物通过 Candle 运行。CPU 路径已验证，包括 32K 输入。Candle 仍提供 CUDA 后端，NVIDIA 性能需要在目标硬件上测量。

ONNX 是 ORT provider 使用的可移植推理格式。[Vela AMD 配方](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)显式将十个任务模型绑定到 AMD GPU：Embedding 和 Reranker 使用 ROCm 下的 CK FlashAttention，分类器使用 MIGraphX。配方固定每个产物的 revision，并保留发布的运行策略。`--platform amd` 选择 AMD 镜像及设备访问，不会让所有模型自动使用 GPU，也不会覆盖显式 CPU 部署。

完整信号流水线以 **8K** 输入预算完成测量。独立 Embedding 和 Reranker 执行已验证至 **32K**；Hazard 在 32K 逻辑预算内使用已验证的 2,048-token 窗口。这些结果不代表所有分类器都完成了 AMD 32K 验证。首次 GPU 编译时间与预热后的请求时延应分别记录。

Domain 和 FactCheck 的更长请求可选择[32K ROCm 部署](../../installation/amd-rocm.md#optional-32k-domain-and-factcheck-on-rocm)。此选项需要更多显存，也会增加短输入的时延。

Embedding 和 Reranker 仓库包含共享外部权重的 FP32 ONNX 计算图。完整表示使用 `onnx/model.onnx`；缩减表示需要匹配已训练层数或层数与维度的计算图。下载器根据模型的编码器配置识别完整表示，因此显式选择完整层数和维度也可以使用主计算图。

CK 版本使用 `onnx/model_fa.onnx` 表示完整的 22 层、768 维表示。选择这个精确的 `head`，并设置 `device: rocm:0`、`custom_ops_profile: ck_flash_attention` 和 `precision: native`。Reranker 的 `pair_scorer` 必须与图匹配：缩减出口使用 `onnx/model_fa_layer_N_dim_D.onnx`，层数和维度也设为对应值。Embedding 使用匹配的 `onnx/model_fa_layer_N.onnx` 配套图。这些图共享已发布的外部权重，同时保留可移植导出。

更新原生权重时，必须重新生成对应的 ONNX 产物再发布。

各模型卡片列出支持的输入长度、用法及可比评测结果。公开对比以此前的 mmBERT 家族为基线，使用匹配数据；质量分数、最大可接受输入长度和推理性能衡量的是不同属性。

## 验证真实路由与重排 {#verify-live-routing-and-reranking}

使用 Vela AMD 配方时，下载并验证完整配置，再选择 AMD 镜像启动。按配方模型卡片说明连接已有的 vLLM 后端：服务名为 `vela-default`，地址为 `vllm:8000`。

```bash
curl --fail --location --output vela-amd.yaml \
  https://raw.githubusercontent.com/vllm-project/semantic-router/main/config/recipes/vela-amd/config.yaml
vllm-sr config validate --config vela-amd.yaml
vllm-sr serve --platform amd --config vela-amd.yaml
```

Route Preview 返回实际信号值、决策和逐信号时延。输入应保持在配方的 8K 分类器预算内：

```bash
curl --fail 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"vela-auto","text":"Help me debug this Python program."}' \
  | jq '{decision_result, signal_confidences, signal_values, signal_errors, metrics, eval_trace}'
```

使用 entrypoint 声明的公共模型名。请求前检查 `/ready`，并查看响应中的信号值与求值轨迹。

重排发生在路由后的 vectorstore RAG 插件中。按[神经重排](/docs/tutorials/plugin/rag#neural-reranking)绑定 `rag.reranker` 并启用 `rerank`，通过已入库文档和真实聊天请求验证。请求 trace 记录候选数、reranker 身份、相关性分数和重排时延；routing Preview 不执行 RAG 插件。

## 保留旧部署 {#preserve-an-earlier-deployment}

显式指定的旧模型路径和别名仍指向原仓库。复现旧分类器时，应同时选择匹配的 mapping、阈值和输入策略。升级 Vela 不会重写显式模型选择。

Embedding 权重或表示发生变化时，向量空间也随之变化。响应缓存和 memory 按表示身份隔离数据；已有 vector store 需要兼容的 embedding 或重新索引，不会静默接管或删除旧数据。详见[存储与工具](./stores-and-tools.md)。
