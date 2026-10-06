---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/algorithm/looper/ratings.md"
  outdated: false
---

# 评分

## 概览

`ratings` 调用每个候选模型，并为每个成功的模型返回一个 OpenAI 兼容的 choice。`max_concurrent` 限制并行工作量，但不限制实际执行的候选总数。

尽管名称如此，当前运行时不会对 choice 打分、投票或合成。调用方收到这些结果后，可自行比较或做下游评分。

## 主要优势

- 在同一请求上比较所有已声明的候选。
- 限制并行工作量，同时不丢弃后续候选。
- 为每个成功的模型保留一个可识别的响应 choice。

## 解决什么问题？

评测和对比客户端有时需要通过一次 Router 请求，让多个模型回答同一提示词。Ratings 提供有界扇出，且不引入裁判模型。

## 何时使用

将 Ratings 用于并排评测，或理解多个 `choices` 的应用。如果调用方期望一个合成答案，不要使用它；那种场景请用 `fusion` 或 `remom`。

## 配置

```yaml
algorithm:
  type: ratings
  ratings:
    max_concurrent: 3
    on_error: skip
```

完整示例见：
[`config/fragments/algorithm/looper/ratings.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/looper/ratings.yaml)。

## 依赖与限制

- 需要多于一个 `modelRef`，以及可访问的 `global.integrations.looper.endpoint`。
- 每个候选都会收到请求内容，因此所有候选提供商都必须被路由的数据策略允许。
- 成本随候选数量增长。并发降低墙上时钟时间，但不减少总模型调用次数。
- `on_error: skip` 返回成功的 choice；`on_error: fail` 在任一模型调用失败时使本次运行失败。若全部模型失败，运行也会失败。
- Ratings 子请求会移除工具定义；需要继续工具调用的智能体工作流请使用 Router Flow。
