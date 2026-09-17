---
title: 基准测试
translation:
  source_commit: "33349fdab9ad294da19ebd11588f8adbe8771b4a"
  source_file: "docs/benchmarking/overview.md"
  outdated: false
---

# 基准测试 {#benchmarking}

按需要回答的问题选择基准。组件基准测量 Router 代码路径；评测套件测量端到端路由或模型质量；后端比较帮助评估可互换的存储和推理实现。它们的结果不能直接比较。

## 选择套件 {#choose-a-suite}

| 问题 | 套件 | 起点 |
|----------|-------|----------------|
| 路由配方、模型池或组合候选是否优于冻结基线？ | Evaluation Plane | [Evaluation Plane](evaluation-plane) |
| 代码变更是否增加了分配或组件延迟？ | `perf/` 中的 Go 微基准 | `make perf-check` |
| 路由是否在推理数据集上保持回答质量？ | `bench/` 中的推理评测 | `vllm-semantic-router-bench compare --dataset arc-challenge` |
| 会话感知路由在多轮或故障下是否保持稳定？ | 实时 Agentic 路由 | `bench/agentic_routing_live_benchmark.py` |
| 被路由的模型能否完成多轮 Agent 任务？ | 实时 Agent 任务 | `bench/agent_task_live_benchmark.py` |
| 后端是否通过 Router 报告缓存输入 token？ | 缓存 token 探测 | `bench/cache_token_probe.py` |
| 路由学习行为是否在确定性 fixture 上回归？ | 架构评测 | `make bench-router-learning` |
| 幻觉检测表现如何？ | 幻觉评测 | `make bench-hallucination` |
| 感知 grounding 的融合是否改进评分答案？ | Grounded fusion | `bench/grounded_fusion/run_ab.sh` |
| 我需要正式的 Router Flow 评测，而不是开发冒烟？ | 由 EvalScope 支撑的 Router Flow 套件 | `bench/router_flow/real_eval/` |
| 原生后端或 GPU 路径是否改变了信号提取性能？ | CPU/GPU 比较 | `bench/cpu-vs-gpu/` |
| 哪种响应缓存存储或推理绑定在这里表现更好？ | 后端 Make 目标 | [后端比较](#backend-comparisons) |

## 组件微基准 {#component-microbenchmarks}

`perf/` 包包含分类、决策评估、响应缓存操作、ExtProc 处理和 Looper 家族路径的 Go 基准。它们不需要运行中的 Router，但依赖模型的套件需要原生库和基准模型文件。

```bash
make download-models-perf
make rust
make perf-bench-quick
```

有用的目标：

- `make perf-bench` 运行完整组件集。
- `make perf-bench-classification`、`make perf-bench-decision`、`make perf-bench-cache` 和 `make perf-bench-looper` 收窄运行范围。
- `make perf-check` 记录基准输出，并在受门槛约束的分配或字节基线回归超过配置阈值时失败。
- `make perf-compare` 比较已有的 `reports/bench-output.txt`，不因结果失败。
- `make perf-profile-cpu` 和 `make perf-profile-mem` 生成 pprof 数据。

回归门槛使用 `allocs/op` 和 `B/op` 判定通过/失败。`ns/op` 作为建议报告，因为它随 runner 变化。性能 CI 针对性能域拥有的变更选择，也可在手动和夜间工作流中使用；不是每次文档或产品变更都会运行。

基线和分析细节见仓库的
[`perf/README.md`](https://github.com/vllm-project/semantic-router/blob/main/perf/README.md)。

## 端到端评测 {#end-to-end-evaluation}

当主张跨越路由、模型池组成、实时生成、Agentic 或多模态行为、偏好、安全或容量时，使用 [Evaluation Plane](evaluation-plane)。它创建一份版本化证据包，并分开组件、系统和生产证据级别。

从仓库安装基准包：

```bash
python -m pip install -e bench
```

仅在由 EvalScope 支撑的 Router Flow 套件时安装 `bench[real_eval]`。
大多数实时套件需要运行中的 OpenAI 兼容端点；有些同时需要被路由端点和直连后端基线。

### 推理数据集 {#reasoning-datasets}

打包的 CLI 支持 MMLU、ARC、GPQA、TruthfulQA、CommonsenseQA 和 HellaSwag 适配器。当默认值与你的部署不匹配时，比较运行需要显式的 Router 和直连后端端点：

```bash
vllm-semantic-router-bench compare \
  --dataset arc-challenge \
  --samples 20 \
  --router-endpoint http://localhost:8899/v1 \
  --vllm-endpoint http://localhost:8000/v1 \
  --vllm-model <served-model-name>
```

将小样本数视为冒烟测试，而不是模型质量证据。

### 会话与 Agent 工作负载 {#session-and-agent-workloads}

当被测行为依赖多轮、稳定身份、工具循环、后端故障或 Router 头时，使用实时路由脚本。每个脚本都暴露阈值标志，用于独立的本地回归判定。

```bash
python3 bench/agentic_routing_live_benchmark.py --help
python3 bench/agent_task_live_benchmark.py --help
python3 bench/cache_token_probe.py --help
```

这些脚本默认将运行产物写入 `.agent-harness/experiments/`。除非报告记录了源提交、配置、端点或模型修订、工作负载、样本数、排除项和接受阈值，否则不要把生成的报告用于公开主张。

这些脚本是诊断和回归工具。其阈值不是 Evaluation Campaign 门槛，其产物也不实现密封的 `evaluation-agent-task-ledger.v1` / `evaluation-agent-task-attempt.v1` 契约或 `live-fault-recovery` 证据契约。因此它们不能主张 Evaluation Agentic E5、G6 或 Campaign 资格。决策级重复任务证据请使用 Evaluation Plane 的 `live-agent-tasks` 源；该源在服务器校验后可以获得 Agentic E5，但有意没有 Campaign 门槛，也永远不会使 G6 合格。

### 路由学习 {#router-learning}

路由学习架构评测是确定性的，不需要实时端点：

```bash
make bench-router-learning
make bench-router-learning PROFILE=release
```

它对照所选 JSON 配置文件检查由 fixture 派生的指标。它是所表示场景的回归测试，不是生产流量基准。

### 专用套件 {#specialized-suites}

- [`bench/hallucination/`](https://github.com/vllm-project/semantic-router/tree/main/bench/hallucination)
 对照标注数据评测检测器和缓解行为。
- [`bench/grounded_fusion/`](https://github.com/vllm-project/semantic-router/tree/main/bench/grounded_fusion)
 比较 grounded-fusion 配置，可能使用评分标准打分。
- [`bench/router_flow/real_eval/`](https://github.com/vllm-project/semantic-router/tree/main/bench/router_flow/real_eval)
 是正式的 EvalScope 路径。`bench/router_flow/flow_eval.py` 是小型开发代理，不得作为可发布基准数据呈现。
- [`bench/cpu-vs-gpu/`](https://github.com/vllm-project/semantic-router/tree/main/bench/cpu-vs-gpu)
 需要文档中的加速器、驱动、容器和模型设置。

## 后端比较 {#backend-comparisons}

这些目标会启动或构建自己的依赖。请在你打算评测的硬件和容器运行时上运行：

```bash
# Response-cache stores
make benchmark-cache-comparison
make benchmark-hybrid-vs-milvus
make benchmark-redis
make benchmark-valkey

# Native inference implementations
make benchmark-openvino-classifier
make benchmark-openvino-embedding
make benchmark-openvino-vs-candle
```

不要把存储或绑定比较解释为端到端路由结果。
网络位置、预热、数据集形状、模型文件和主机争用都会改变结果。

## 报告结果 {#reporting-results}

对于任何用于指导部署或公开主张的数字，请记录：

- 仓库提交和完整 Router 配置
- 模型、数据集和依赖修订
- 硬件、驱动、运行时和后端拓扑
- 精确命令、预热、并发和样本数
- 失败和排除的样本
- 原始产物和聚合方法

在同一工作负载和环境上比较备选方案。没有这些上下文的 QPS、延迟、准确率、成本或节省数字只是本地观察，不是 Semantic Router 的预期属性。

训练期间使用的模型选择评测单独记录在[模型性能评测](../training/model-performance-eval)。
