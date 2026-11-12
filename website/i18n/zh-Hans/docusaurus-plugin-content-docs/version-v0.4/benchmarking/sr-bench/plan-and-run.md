---
title: 冻结并运行
translation:
  source_commit: "31fa0fdab6787b3b149894db259af13c8ce42f5a"
  source_file: "docs/benchmarking/sr-bench/plan-and-run.md"
  outdated: true
---

# 冻结并运行

清单使用 `version: sr-bench-1.0`，引用数据的 `path`/`sha256`，匹配其 profile/seed，并固定目标、采样和限额。完整示例见 [英文配置说明](https://vllm-sr.ai/docs/benchmarking/sr-bench/plan-and-run)。

```bash
vllm-sr benchmark plan --manifest candidate.yaml --output frozen.json
vllm-sr benchmark run --manifest frozen.json --detach --idempotency-key loop-1
vllm-sr benchmark runs
vllm-sr benchmark show RUN_ID --calls
vllm-sr benchmark report RUN_ID --output report.json
vllm-sr benchmark cancel RUN_ID
```

重复提交相同幂等键绑定相同计划，不会重发模型请求。`Ctrl-C` 仅停止 CLI 等待；停止实际工作需使用 `cancel`。异常、部分输出和派发记录全部保留，不会自动重试生成或重启停止的 worker。

默认 `cost_policy: require_priced` 要求完整价格；显式选择 `capability_only` 可以在价格未知时测试能力，但不能证明节省。Quality only 不执行美元预算停止，即使部分费用已知；仍记录已知费用并执行时间、调用次数和输出限制。绝对超时、空闲超时、输出/重复保护以及任务和运行时限约束工作。费用预留和实际费用停止不是供应商强制执行的通用硬美元上限；缺失计量必须显示未知，不能当作零。
