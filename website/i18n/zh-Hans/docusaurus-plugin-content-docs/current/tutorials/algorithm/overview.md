---
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/tutorials/algorithm/overview.md"
  outdated: false
---

# 算法

## 概览

算法在决策匹配之后运行。它要么从决策的 `modelRefs` 中选择一个模型，要么通过 Looper 协调其中多个模型。它不决定路由是否有资格；信号和决策会先完成这件事。

## 主要优势

- 将路由资格与模型选择分开。
- 让每条决策的选择和编排策略可审查。
- 同时支持无状态策略和有界多模型执行。

## 解决什么问题？

匹配的路由可能有多个有效模型候选。算法让选择变得显式：固定顺序、语义契合、观测延迟、多个运行时因素、已学习的选择器，或多模型编排。

## 何时使用

当决策有多个候选，或有意运行多模型工作流时，添加算法。只有一个候选时，除非所选 Looper 支持并需要单模型执行计划，否则省略算法。

## 配置

算法是决策局部的：

```yaml
routing:
  decisions:
    - name: responsive-route
      description: Prefer the model with the best observed latency.
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: small-model
        - model: large-model
      algorithm:
        type: latency_aware
        minimum_candidates: 2
        latency_aware:
          tpot_percentile: 90
          ttft_percentile: 95
```

从下面的清单中选择算法，然后按对应指南填写必填字段和依赖。

`minimum_candidates` 是公共算法字段。无模型配方可以在绑定任何模型之前声明它；入口物化配方后，校验要求有这么多不同的 `modelRefs`。请求时的上下文资格过滤后会再次检查同一边界，因此面板、级联或选择器不会以小于策略声明的池静默运行。

## 算法清单

### 选择算法

选择算法返回一个候选模型。

| 类型 | 状态 | 目标 | 主要依赖 | 指南 |
|---|---|---|---|---|
| `static` | 已支持 | 使用声明顺序或固定领域分数 | 无 | [Static](./selection/static) |
| `router_dc` | 已支持 | 将请求语义匹配到模型描述 | 嵌入运行时和有用的模型卡片 | [Router DC](./selection/router-dc) |
| `latency_aware` | 已支持 | 优先选择观测 TTFT/TPOT 最好的候选 | 每进程延迟观测 | [Latency Aware](./selection/latency-aware) |
| `multi_factor` | 已支持 | 用可选 SLO 过滤器平衡质量、延迟、成本和负载 | 模型元数据和实时本地指标 | [Multi Factor](./selection/multi-factor) |
| `hybrid` | 已支持 | 混合多个选择器分数 | 组件选择器输入 | [Hybrid](./selection/hybrid) |
| `automix` | 实验性 | 优化估计的成本-质量值 | 候选定价和质量元数据 | [AutoMix](./selection/automix) |
| `prompt` | 实验性 | 让有界辅助模型从已声明候选中选择 | OpenAI 兼容辅助模型和 Looper 端点 | [Prompt](./selection/prompt) |
| `knn` | 实验性 | 跟随相似的已标注示例 | 已训练的选择器产物和嵌入 | [KNN](./selection/knn) |
| `kmeans` | 实验性 | 通过已学习的流量簇路由 | 已训练的选择器产物和嵌入 | [KMeans](./selection/kmeans) |
| `svm` | 实验性 | 应用已学习的决策边界 | 已训练的选择器产物和嵌入 | [SVM](./selection/svm) |
| `mlp` | 实验性 | 应用已学习的非线性分类器 | 已训练的选择器产物 | [MLP](./selection/mlp) |

### Looper 算法

Looper 算法通过 `global.integrations.looper.endpoint` 发起额外模型调用。它们会增加延迟和 token 用量，并且中间内容会发送给本次运行涉及的每个已配置 worker。

| 类型 | 状态 | 目标 | 指南 |
|---|---|---|---|
| `confidence` | 已支持 | 串行升级，直到置信度超过阈值 | [Confidence](./looper/confidence) |
| `ratings` | 已支持 | 在有界并发下从每个候选返回一个 choice | [Ratings](./looper/ratings) |
| `remom` | 已支持 | 在多轮中探索多条推理路径，然后合成 | [ReMoM](./looper/remom) |
| `fusion` | 实验性 | 运行分析面板和裁判/合成阶段 | [Fusion](./looper/fusion) |
| `workflows` | 实验性 | 执行有界的静态或规划器生成的 worker 流程 | [Router Flow](./looper/workflows) |

把实验性算法当作评测功能：在用于生产路由之前，先在自己的流量上验证。

## 运维边界

- 候选模型名必须在完整配置中通过 `routing.modelCards` 和 `providers.models` 解析。
- 已学习的选择器需要按运行时使用的同一嵌入维度和候选标签生成产物。
- 延迟和负载观测是 Router 进程本地的；它们不是集群级调度器。
- Looper 算法会与已配置的 worker 共享请求内容。选择它们之前，先做隐私和提供商边界决策。
- Looper 生成的规划器、worker、验证器、裁判和合成提示，会在派发前对照每个目标模型的已知上下文窗口检查。缺失上下文元数据为了兼容性仍视为合格。
- 用 `vllm-sr config validate --config config.yaml` 校验完整配置。
