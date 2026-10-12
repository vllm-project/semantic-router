---
title: 准备数据和目标
translation:
  source_commit: "31fa0fdab6787b3b149894db259af13c8ce42f5a"
  source_file: "docs/benchmarking/sr-bench/tasks-and-targets.md"
  outdated: true
---

# 准备数据和目标

Parquet 来源需要在准备主机安装 `vllm-sr[bench]`。在 worker 主机或它的共享存储准备数据；本地路径不会自动上传到远端。

```bash
vllm-sr benchmark --store ./data/sr-bench dataset prepare \
  --benchmark mmlu-pro --profile quick > mmlu-quick.json
vllm-sr benchmark --store ./data/sr-bench dataset prepare \
  --benchmark gpqa-diamond --profile quick > gpqa-quick.json
vllm-sr benchmark --store ./data/sr-bench dataset combine \
  mmlu-quick.json gpqa-quick.json > quick-dataset.json
vllm-sr benchmark --store ./data/sr-bench target register --file targets.json
```

受限来源需要相应访问权限。导入本地任务需提供 `--source-path` 和 `--revision`；实际内容仍会被哈希。`--limit` 产生明确标注的子集，不要原地修改冻结文件。

目标字段包括 `id`、`kind: single|mom`、`base_url`、`model` 和可选的 `api_key_env`。计价运行需按实际模型身份提供四类 USD/百万 token 价格：`input`、`cached_input`、`cache_write`、`output`。MoM 还需固定实际配置 `config_hash`，预览使用 `preview_url`，计价时声明 `max_inference_calls`。当前直接 MoM 适配器要求完整的一次推理计量；不能用最终模型的价格代替未计量的组合调用。

运维可用目标的 `request_params` 固定原生生成参数；它们覆盖运行的 `sampling` 默认值，包括温度、推理选项以及显式设置的输出长度。对比前检查实际生效参数。运行的输出上限必须容纳目标固定的 `max_tokens`；降低上限不会改写模型 profile。需要其他原生参数时，应选择另一个由运维注册的 profile。

HLE/SimpleQA 需固定裁判和 `grader_version: sr-bench-reference-judge-v1`；τ³ 需固定模拟器和 `release: 1.0.1`。运维在 store 的 `benchmark-options.json` 配置这些依赖。外部适配器默认发现 `benchmark setup` 安装的固定环境；可用 `SR_BENCH_{LCB,SCICODE,TERMINAL,TAU3}_PYTHON` 和对应 `_ROOT` 覆盖其位置。源码和沙箱镜像仍须固定版本。预检会在付费派发前报告缺失依赖。
