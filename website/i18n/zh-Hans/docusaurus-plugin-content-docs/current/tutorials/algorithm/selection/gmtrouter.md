---
translation:
  source_commit: "9649d02f2582471642196bde2e086c34ccb9c88c"
  source_file: "docs/tutorials/algorithm/selection/gmtrouter.md"
  outdated: false
---

# GMT Router

## 概览 {#overview}

`gmtrouter` 是个实验性的个性化选择器。冷启动用带版本的模型智能分，之后把这个基础分和从用户过往模型反馈里学到的偏好混在一起。

## 解决什么问题 {#what-problem-does-it-solve}

全局质量排名反映不了一件事：不同用户偏好不同模型。GMT Router 从静态证据安全起步，再从反馈里学一个有界的个人偏好。

## 什么时候用 {#when-to-use}

请求带着稳定的用户身份、反馈拿得到、个性化比一个全局排名更有用——这时用它。这些输入都没有，就用无状态选择器。

## 选择行为 {#selection-behavior}

用户到达 `min_interactions_for_personalization` 之前，候选按各自能拿到的智能分排名。之后，当前的选择器用：

```text
score = 0.3 * intelligence + 0.7 * user preference
```

没学到偏好的候选，拿它的智能分再挨一点小罚。智能分缺失，用选择器的中性冷启动兜底，和实测的零分是两回事。

候选声明了 `reasoning_effort`，GMT Router 只读那个精确的证据桶，不向别的 effort 借测量。覆盖率永不乘进智能分：覆盖率高只在 GMT 分和智能分都打平时打破平局。选择诊断会报选中的分数和覆盖率，或标出智能分不可用。

## 配置 {#configuration}

```yaml
algorithm:
  type: gmtrouter
  gmtrouter:
    enable_personalization: true
    min_interactions_for_personalization: 3
    max_interactions_per_user: 100
    history_sample_size: 5
    embedding_dimension: 768
    num_gnn_layers: 2
    attention_heads: 8
    storage_path: state/gmtrouter.json
```

反馈更新有界的每人交互历史和模型偏好。配了 `storage_path`，这份状态就持久化。查询、响应和模型描述的嵌入，只在有嵌入函数可用时才参与。

## 限制 {#limitations}

- GMT Router 是实验性的；个性化要在有代表性的流量上验过再上。
- 没有用户身份的请求共用 `anonymous` 这一份偏好状态。
- 学到的偏好是模型级的，不细分 reasoning effort。
- 静态覆盖率描述不了、也加权不了学到的反馈。
