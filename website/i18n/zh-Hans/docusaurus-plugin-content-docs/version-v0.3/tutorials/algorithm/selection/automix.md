---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/automix.md"
  outdated: false
---

# Automix

## 概览

`automix` 是一个实验性的选择算法：它基于成本与质量的价值估计，在匹配决策的候选模型中**只返回一个模型**。

对应 `config/fragments/algorithm/selection/automix.yaml`。

该选择器受 [Automatically Mixing Language Models](https://arxiv.org/abs/2310.12963) 论文启发，但其公开配置有意比研究系统更精简。

## 主要优势

- 在候选质量与配置成本之间权衡，而非总选最强模型。
- 候选集合限定在匹配的决策内。
- 将成本与质量策略保留在路由配置中，而非应用逻辑中。

## 解决什么问题？

总选最强模型会浪费预算，总选最便宜模型又可能损害质量。`automix` 对有限的候选集合按质量与成本打分，使路由在较便宜的模型预计足够好时优先选择它们。

## 何时使用

- 一条路由有多个成本与质量画像不同的候选模型
- 路由应保持成本意识，而非总选最强模型
- 处于实验阶段，并能在自己的流量上验证效果

## 核验与升级

`automix` 在请求发送前选定一个模型，不会先生成、再核验、再升级到更大的模型。

如需请求期间的生成、置信度核验（包括 AutoMix 式的蕴含核验）以及跨模型升级，请使用 [`confidence`](../looper/confidence) 算法。

## 已知限制

- 该决策算法只进行一次后端调用。多次调用的升级与核验请使用 `confidence`。
- 在 v0.3 中，AutoMix 选择器从 `global.router.model_selection.automix` 读取配置，而不是从决策中读取。`algorithm.automix` 下的六个字段会被接受，但其中只有正值的 `cost_quality_tradeoff` 会作为该决策的成本权重生效，且仅在全局配置设置了 `cost_aware_routing: true` 时生效。
- 成本感知打分的效果取决于所配置的模型价格与质量元数据。
- AutoMix 仍属实验性功能，在用于 SLO 之前请先在自己的流量上验证。

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: automix
  automix:
    verification_threshold: 0.78
    max_escalations: 2
    cost_aware_routing: true
    cost_quality_tradeoff: 0.35
    discount_factor: 0.9
    use_logprob_verification: true
```

以上数值仅为示例，并非默认值。

### 参数

| 参数 | 类型 | 说明 |
|------|------|------|
| `verification_threshold` | float | AutoMix 价值估计中的核验置信度阈值（0–1）；从全局配置读取 |
| `max_escalations` | int | 为兼容性而接受；该算法只进行一次后端调用 |
| `cost_aware_routing` | bool | 将 AutoMix 价值估计切换为成本感知打分；从全局配置读取 |
| `cost_quality_tradeoff` | float | 成本权重（0–1）；在决策级只有正值会被使用（见已知限制） |
| `discount_factor` | float | AutoMix 价值估计中的折扣因子 γ（0–1）；从全局配置读取 |
| `use_logprob_verification` | bool | 为兼容性而接受；不会启用或改变核验 |

只有上述六个字段属于 v0.3 决策级 AutoMix 的公开配置契约。其他 AutoMix 研究参数不支持在 `algorithm.automix` 下配置。
