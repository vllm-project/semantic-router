---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/algorithm/selection/automix.md"
  outdated: true
---

# 自动混合选择

## 概览

`automix` 是实验性选择器，按配置的质量与成本，以及内部验证和升级估计，对候选模型排序。它返回一个模型；它不是串行的 [`confidence`](../looper/confidence) Looper。

该选择器受 [Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963) 启发，但公开配置有意比研究系统更小。

## 主要优势

- 在配置的质量元数据与配置成本之间做平衡。
- 将候选集限制在匹配决策之内。
- 将实验性价值计算与路由资格分开。

## 解决什么问题？

总是选最强模型会浪费预算，总是选最便宜的模型又可能损害质量。AutoMix 根据配置元数据和内部估计，为有界候选集计算成本-质量值。

## 何时使用

在候选定价和质量元数据可用时，用 AutoMix 做实验。需要已支持、无状态且更易运维推理的策略时，优先使用 `static`、`router_dc` 或 `multi_factor`。

## 配置

```yaml
algorithm:
  type: automix
  automix:
    verification_threshold: 0.78
    max_escalations: 2
    cost_aware_routing: true
    cost_quality_tradeoff: 0.3
    discount_factor: 0.95
    use_logprob_verification: true
```

只有上面这些字段属于当前决策级 AutoMix 契约。`max_escalations` 和 `use_logprob_verification` 为兼容性而接受，但不影响 AutoMix 选择。
完整示例见：
[`config/fragments/algorithm/selection/automix.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/algorithm/selection/automix.yaml)。

## 依赖与限制

- 候选价格来自 `providers.models[].pricing`；缺少元数据会降低成本感知打分的有效性。
- 能力估计从配置的模型元数据和默认值开始。AutoMix 不会从公开 outcome 端点学习；流量或模型变化时，请显式重新调优估计。
- `verification_threshold`、配置成本、`cost_quality_tradeoff` 和 `discount_factor` 会影响单模型分数。`max_escalations` 和 `use_logprob_verification` 目前不会。
- 该决策算法本身不会发起多次后端调用。请求时生成和升级请使用 `confidence`。
- 请求内容通过配置的语义嵌入路径嵌入。
- AutoMix 是实验性的。在依赖它满足 SLO 之前，先在自己的流量上验证。
