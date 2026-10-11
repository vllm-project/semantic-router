---
title: 运维与故障排查
description: 检查就绪状态、查看实际路由信号并管理模型更新。
translation:
  source_commit: "915ddf56e0335e2046c38aa17c4aec6233908039"
  source_file: "docs/installation/runtime/lifecycle-diagnostics.md"
  outdated: false
---

配置好[本地模型](in-process.md)或[外部服务](external.md)后，使用本页检查运行状态。

## 检查启动状态 {#check-startup}

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
curl -fsS http://localhost:8080/ready
curl -fsS http://localhost:8080/startup-status
```

验证命令检查配置。启动时加载所需模型、检查能力并预热，完成后才进入就绪状态。AMD 首次编译 MIGraphX 模型可能需要数分钟。

CLI 默认等待 1,800 秒。如果实测冷启动需要更长时间，可以增加预算：

```bash
vllm-sr serve --config config.yaml --startup-timeout 7200
```

超时后容器继续运行，便于检查状态和日志：

```bash
vllm-sr status
vllm-sr logs router
```

`--startup-timeout` 控制就绪等待，不改变推理请求的超时设置。

## 查看实际执行路径 {#inspect-the-executed-path}

Route Preview 执行配置的信号，不生成答案。原生输出选模可能调用后端的 render 端点检查容量：

```bash
curl -fsS 'http://localhost:8080/api/v1/routing/preview?trace=true' \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","text":"Help me debug this Python program."}' \
  | jq '{decision_result, signal_confidences, signal_values, signal_errors, metrics, eval_trace}'
```

将 `auto` 替换为你的公开入口名，Vela AMD 配方使用 `vela-auto`。重点查看：

| 字段 | 含义 |
| --- | --- |
| `decision_result` | 选中的路由与模型 |
| `signal_confidences`、`signal_values` | 实际分类分数与提取值 |
| `signal_errors` | 失败或不可用的信号评估 |
| `metrics` | 请求及各信号的评估耗时 |
| `eval_trace` | 条件如何产生路由决策 |

Preview 不执行 RAG 检索和重排。索引文档后，发送真实的 `/v1/chat/completions` 请求测试这些功能。Trace 包含 `rag.rerank_candidates`、`rag.rerank_latency_seconds`、`rag.reranker_identity` 和相关性分数。复用缓存上下文时可能不会再次调用重排模型，见[神经重排](../../tutorials/plugin/rag.md#neural-reranking)。

应分别比较预热后推理、启动、检索和后端生成的耗时。Preview 耗时不等于完整聊天响应耗时。

## 常见问题 {#common-problems}

| 现象 | 检查内容 |
| --- | --- |
| 模型无法加载 | 完整权重或 ONNX 文件、tokenizer、标签、挂载路径 |
| 引擎或设备不可用 | 镜像、宿主机依赖和 GPU 访问方式是否匹配 deployment |
| 标签不匹配 | 权重标签顺序、规则标签和映射文件 |
| 输入被拒绝 | 输入预算和特殊 token，见[输入策略](in-process.md#choose-an-input-budget) |
| 缺少嵌入层或维度 | 模型导出及缓存、记忆、已存向量的要求 |
| 远程推理失败 | 地址、凭据、超时、响应格式与大小上限 |
| Preview 返回 429 或 504 | `global.services.api.routing_preview` 的并发饱和或超时 |
| 置信度为 `null` | 没有可用的模型分数，见[失败策略](safety.md#handle-failures-and-missing-scores) |

已超时的原生 Preview worker 会保留槽位直到推理结束。增加超时前，优先降低输入预算或请求并发。

## AMD 启动排查 {#amd-startup-problems}

`rocm:N` 选择 ROCm provider，`migraphx:N` 选择 MIGraphX。CK 图还需要 `custom_ops_profile: ck_flash_attention` 和镜像中的配套库。使用 [Vela AMD 配方](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)提供的图与 provider 组合。请求 GPU 推理时不允许回退到 CPU。

| 现象 | 处理方式 |
| --- | --- |
| 图算子不支持 | 使用兼容所选 provider 的导出 |
| 缺少 GPU 嵌入预算 | 将 deployment 的 `input.max_tokens` 设为正数 |
| 冷启动慢 | 留出编译及预热时间，考虑 [MIGraphX 编译缓存](in-process.md#advanced-migraphx-settings) |
| MIGraphX 环境冲突 | 移除下列覆盖变量，在 deployment 中配置精度与缓存 |

维护的 MIGraphX 镜像保留 `MIGRAPHX_MLIR_USE_SPECIFIC_OPS=~attention`。以下变量应保持未设置状态，包括值为 `0` 的情况：

```text
ORT_MIGRAPHX_FP16_ENABLE
ORT_MIGRAPHX_BF16_ENABLE
ORT_MIGRAPHX_FP8_ENABLE
ORT_MIGRAPHX_INT8_ENABLE
ORT_MIGRAPHX_MODEL_CACHE_PATH
```

这些进程级覆盖项会与 deployment 的显式精度或缓存设置冲突。

## 限制推理并发 {#limit-concurrent-inference}

对于[进程内模型](in-process.md)中的 `vela-domain`，以下配置允许两个并发调用、八个排队请求，排队超时为一秒：

```yaml
global:
  model_catalog:
    admission:
      vela-domain:
        max_concurrency: 2
        max_queue: 8
        queue_timeout_ms: 1000
        on_overflow: shed
```

`shed` 拒绝超额请求，`wait` 等待队列槽位，`fail_open` 在满载时绕过限制。`wait` 要求队列大小非零；省略 admission 表示不限制准入。共享模型的调用共享容量，请求超时包含排队时间。

更改共享模型的 admission 设置后需要重启 Router。热更新会拒绝不同的设置，当前配置继续服务。

## 更新运行中的模型 {#update-a-running-model}

将新版本放入新目录，更新 deployment，然后通过 Dashboard 或管理 API 重载。Router 准备好替代模型后才激活；加载失败时继续使用当前配置。已有请求完成后，旧模型资源才释放。

Dashboard 更新返回 `202` 时，轮询 `GET /api/router/api/v1/config/hash`，直到 `active_runtime_hash` 与更新返回的 `generated_runtime_hash` 相同。更新尚未完成时提交另一项变更会返回 `409`。详见[管理 API 参考](../../api/apiserver.md)。

嵌入空间变化后，需要[重新索引受影响文档](embeddings.md#change-a-model-without-mixing-vector-spaces)。旧知识库资产仍保留在磁盘上，只在活跃请求不再需要时清理。

`serve` 保留通过 Dashboard 或 API 保存的配置。需要用本地文件替换时，运行：

```bash
vllm-sr serve --config config.yaml --replace-active-config
```
