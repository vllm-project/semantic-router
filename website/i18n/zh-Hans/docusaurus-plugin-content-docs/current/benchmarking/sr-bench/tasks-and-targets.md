---
is_mtpe: false
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

## DRACO 最终答案诊断

可选的 `draco-routed-answer-v1` 适配器保留 grounded-fusion 的现有 rubric，通过共享 worker 执行主体和裁判调用。它不改变已发布的基准集合，也不提供固定面板的策略比较。

在 worker 主机本地准备现有 DRACO JSON 导出：

```bash
python -m cli.sr_bench.draco \
  --source-path "$DRACO_PATH" --revision "$DRACO_REVISION" \
  --source-partition "$DRACO_PARTITION" \
  --domains Medicine,Law --seed 42 --max-samples 8 \
  --store ./data/sr-bench > draco-dataset.json
```

必须声明源版本和上游分区。输出冻结源文件内容、原始题目身份，以及旧工具的领域过滤、随机打乱和数量上限。将输出的 dataset 对象放入运行 manifest，并使用相同的 profile 和 seed。仅支持 smoke/quick 诊断；缺少输入时必须本地准备，不能自动下载或推定为保留集。

注册固定的单模型辅助裁判，并配置：

```json
{
  "draco-routed-answer-v1": {
    "judge": "rubric-judge",
    "grader_version": "draco-legacy-rubric-strict-v1",
    "batch_size": 8
  }
}
```

主体和裁判端点都必须支持 OpenAI 兼容的流式响应。旧 grounded-fusion Ollama 代理只返回完整 JSON，不能作为此适配器的裁判端点。固定生成参数，不要绕过传输层策略。每题需要一次主体调用，每批 rubric 条件需要一次裁判调用；调用预算不足时预检会拒绝执行。

Fusion 主体使用 `kind: mom` 和 `cost_policy: capability_only`，不设置 `max_inference_calls`。Router 的计价调用门禁只支持一次直接调用的上限，不能约束 Fusion。不要把组合路由标成单模型来计算价格；缺失的子调用用量或价格必须保持未知。

适配器在可信 CLI 初始化器中注册，因此复制 CLI 源码而不安装发行包元数据的 worker 镜像也能发现它。保留的 grounded-fusion 命令共享该 rubric 内核，需要 Python 3.10+ 和更新后的 CLI；reasoning/Looper 发行包要求保持不变。结果边界见[加权 rubric 诊断](./results.md#weighted-rubric-diagnostics)。
