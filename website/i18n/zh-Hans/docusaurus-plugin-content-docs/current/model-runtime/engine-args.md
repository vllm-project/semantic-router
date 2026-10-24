---
title: 引擎参数
description: 用 vLLM 风格的引擎参数调整批处理、精度和共享上下文。
translation:
  source_commit: "1c55d8a0e21995a99d46eb7f6f36fa2ac779ebe7"
  source_file: "docs/model-runtime/engine-args.md"
  outdated: true
is_mtpe: true
---

# 引擎参数 {#engine-arguments}

模型 runtime 默认就做批处理，和 vLLM 一样：并发请求的行共用前向，所以忙碌的部署每秒能答更多请求。引擎参数调整这套批处理、模型计算用的精度和共享上下文。按部署设置；没设的参数一律用 runtime 的默认值。

| 参数 | 部署字段 | 默认值 | 含义 |
| --- | --- | --- | --- |
| `--max-num-seqs` | `max_num_seqs` | `8192` | 一次前向最多装多少行（跨请求计）。`1` 表示每行单独跑。 |
| `--max-num-batched-tokens` | `max_num_batched_tokens` | `65536` | 一次前向最多装多少个补齐后的 token：行数乘以最长那一行。 |
| `--batch-window-ms` | `batch_window_ms` | `2.0` | 第一个请求到达后，调度器再等多久收集更多请求。 |
| `--dtype` | `dtype` | `auto` | 模型计算用的精度：`auto`、`float32`、`bfloat16` 或 `float16`。 |
| `--enable-shared-context`、`--no-enable-shared-context` | `enable_shared_context` | 由模型家族决定 | 多问题决策请求的共享状态只算一次。 |

## 批处理 {#batching}

队列里所有请求的所有行放在一起规划，最长的先排：决策请求每个问题占一行（10 个问题的请求占 10 行），classify、embeddings 或 rerank 请求每个输入或窗口占一行。一次前向最多装 `--max-num-seqs` 行、`--max-num-batched-tokens` 个补齐后的 token；比 token 预算还长的行单独跑。这就是 vLLM 的批处理，只是“行”的含义不同：runtime 的模型一次前向答完一行，而 vLLM 的序列要一步接一步地解码。

- **`--max-num-seqs`** 数的是行，就像 vLLM 的 `--max-num-seqs` 数的是一步里的序列。默认值在默认 token 预算下永远不会成为限制：每行至少占 8 个补齐后的 token，所以 65,536 个 token 的前向本来就最多装 8,192 行。想让前向小一些就调低它；`1` 让每行各占一次前向。
- **`--max-num-batched-tokens`** 数的是补齐后的 token：前向的行数乘以最长那一行，向上取整到 8 的倍数。对把行首尾相接、不做补齐的模型（native 引擎上的 Vela 1.0 任务模型）也这样数，所以对它们会高估一次前向的实际计算量。有些家族自己会把前向压得更短：Decision 3.0 按实测的每 token 开销，把合并后的前向控制在约 0.3 秒设备时间；在 CPU 上，Vela 2.0 0.3B 和 Decision 1.0 的编码器尺寸分别不超过 2,048 和 1,024 个补齐后的 token。
- **`--batch-window-ms`** 是第一个请求到达后，调度器等多久再规划队列。vLLM 没有这个设置：生成请求一步接一步地解码，每一步都是新请求加入运行中批次的时机。这里一个请求就是一次前向，没有窗口的话，一波请求里的第一个会单独跑，其余的排在它后面。单独到来的请求要付出这段窗口时间；`0` 表示立刻规划，前向运行期间排进来的请求照样会合并。

如果一个模型的每一行单独跑和在任意批次里跑答案都一样，比如 CPU 上用 oneDNN 的 Vela 1.0 任务模型或 embedder，它会直接合并已在排队的请求，不等窗口，并把这些前向控制在 512 个补齐后的 token 以内，这样短请求永远不会排在长请求后面。它的答案从不变化。

对其他模型，一行旁边的其他行会改变它的补齐方式和 GEMM 形状，所以答案可能因舍入而变动，接近平局的结果可能翻转。记录拿这一点和模型自身批次本来就有的噪声作比较：

- ROCm 上的 Decision 2.0：合并的行改变的决策数，和把同样的问题拆开单独问而不是一起问时一样多（例如 Kai-0.6B 在 2,000 个里是 16 对 18），Decision Index 差值的 95% 区间包含 0，不过 Eos-0.8B 的估计偏负（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/rocm-mi325x-batching.md)）。
- Decision 3.0：d3-edge 的决策有 99.69% 与每次前向只跑一个请求时一致，金标准准确率从 75.17% 变为 75.18%；把一个请求的问题逐个发送而不是一起发送，改变的决策一样多（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-batching.md)）。

请求了 `meta`（`"options": {"return_meta": true}`）的响应带有 `numerics`：答案从不受批处理影响的模型为 `exact`，否则为 `approximate`。`GET /v1/models` 报告每个模型的 `engine_args` 和 `numerics`。

## 精度 {#precision}

`--dtype auto` 用模型发布时的精度运行，也就是它的就绪检查记录时所用的精度：

| 模型 | GPU 上的 `auto` | CPU 上的 `auto` |
| --- | --- | --- |
| Decision 3.0 | BF16 | BF16 |
| Decision 2.0；基于 Qwen3.5 的 Decision 1.0 和 Vela 2.0 | BF16 | FP32 |
| Vela 1.0 任务模型、embedder 和 reranker，Vela 1.0 Omni，Vela 2.0 0.3B，Decision 1.0 Kai、Lex 和 Route | FP32 | FP32 |

设成其他值时，骨干网络的全部权重都以该 dtype 存放并以它计算，使用参考 kernel。取值等于发布精度时与 `auto` 相同。runtime 启动时照样检查模型对已记录的请求两次回答一致，但记录的参考答案只适用于 `auto`，所以模型的 `golden.status` 是 `unverified`。`float16` 只能在 GPU 上跑，ONNX Runtime 引擎只支持 `auto`。

## 共享上下文 {#shared-context}

一个决策请求针对同一个输入问很多问题时，共享上下文把输入只跑一次，所有问题都从它出发，作为一棵打包的树来算：在 Vega-27B 上，对一个输入问 128 个问题时最多快 3.4 倍。只有当省下的 token 抵得过它在该骨干上多出的 kernel 启动时，请求才会共享，所以只有一两个短问题的请求照常运行。

和 vLLM 的 `--enable-prefix-caching` 一样，它在不影响精度的地方默认开启，其他地方是一个开关：

- **Decision 2.0：开启。** 这些树的答案与已发布 runtime 自带的共享上下文开关完全一致，而那个开关在每个评测面板上的准确率变化都在 0 的 95% 区间以内（[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/rocm-mi325x-batching.md)）。
- **Decision 1.0 和 Vela 2.0 的解码器尺寸：关闭。** 它们的记录显示决策和片段集合会变化，所以在那里 `--enable-shared-context` 是显式选择。
- 编码器整段读入请求，没有共享上下文路径。

## 发布精度下的更快 kernel {#faster-kernels-at-the-published-precision}

只要模型的记录显示没有决策改变，它就默认以发布精度运行更快的 kernel：

- **Ampere 及更新的 NVIDIA GPU，装有 Triton：** Decision 2.0 运行融合的逐元素 kernel，快 11% 到 48%，其[记录](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-fused-approximate.md)中没有决策改变。Vela 2.0 保留参考 kernel，因为融合 kernel 会改变它的部分片段答案。
- **带 oneDNN 的 x86 CPU：** Vela 2.0 0.3B 和 Decision 1.0 Kai、Lex、Route 在线性层的预打包 FP32 副本上运行合并的前向，快 1.5–1.7 倍，没有决策改变。副本放在就绪检查所用的权重旁边：一个 Decision 1.0 编码器的三个层栈共 1.32 GB。

## 设置方法 {#set-them}

router 托管的模型，在部署上设这些字段：

```yaml
global:
  model_catalog:
    deployments:
      decision-lux:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Lux-9B
        device: rocm:0
        max_num_seqs: 256
        batch_window_ms: 1.0
```

`vllm-sr serve` 把它们设在默认部署上，`vllm-srun serve` 则把它们作为参数，作用于它服务的每个模型（[models 文件](./reference.md#models-file)里的条目可以各自设置）：

```bash
vllm-sr serve vllm-sr/Decision-2.0-Lux-9B --engine --platform rocm --device-ids 0 --max-num-seqs 256
vllm-srun serve vllm-sr/Vela-2.0-0.3B --device cpu --batch-window-ms 0
```

仍然写着已移除设置的配置会被拒绝，并给出改写它的命令；见[迁移](./migrate.md#engine-arguments)。

## 复现已发布的结果 {#reproduce-published-results}

runtime 为吞吐而设计。复现模型发布的数字是模型自己仓库的事，那里附带了测出这些数字的参考代码。`--max-num-seqs 1` 能消除其他请求的影响，但它让每行单独跑，这并不是已发布 runtime 批量处理一个请求所有问题的方式。一个可选的确定性模式（让每行在任意批次里都得到相同答案）是未来可能的工作。

## 不改变答案的性能手段 {#performance-without-changing-answers}

下面这些从不拿正确性换速度：

- 同一模型、同一路由阶段的兼容工作可以共用一次打包调用；一个包仍可能需要多次前向；
- 硬件容量允许时，独立的 worker 可以并行执行；把多个 worker 放在同一块 GPU 上不会增加算力；
- 可缓存的重复工作在 worker 有界的结果缓存（`--result-cache-entries`）里还在时，可以复用近期结果；各副本的 worker 缓存相互独立；
- Decision 3.0 模型对重复的图片或视频只解码和预处理一次；
- 只有在你要求窗口化时，长输入才会被切成窗口。
