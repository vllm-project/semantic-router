---
title: 基准测试
---

# 基准测试 {#benchmarking}

使用 [sr-bench 1.0](sr-bench) 在可复用的冻结题目上比较实际 MoM 入口与单模型。CLI 与 Dashboard 共用能力、成本、token、延迟和 wall time 证据。先用有界开发集迭代，再用不相交的保留集验收。

| 问题 | 入口 |
| --- | --- |
| Balance 能否持平最强单模型并降低成本？ | [sr-bench 1.0](sr-bench) |
| 如何预览路由并改进 Recipe？ | [调优与验证](agent-evaluation-loop) |
| 如何把已测模型证据加入路由配置？ | [自定义评测记录](custom-evaluations) |
| 代码是否导致组件性能回归？ | [组件微基准](#component-microbenchmarks) |
| 哪种缓存或推理实现更合适？ | [后端比较](#backend-comparisons) |

组件基准和 `bench/` 专用脚本是开发诊断工具，不产生完整 sr-bench 分数，也不能代替成对的真实模型比较。

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
