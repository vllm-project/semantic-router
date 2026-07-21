---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/selection/rl-driven.md"
  outdated: true
---

# RL Driven

## 概览

`rl_driven` 用于在线探索与个性化的选择算法。

对应 `config/algorithm/selection/rl-driven.yaml`。

## 主要优势

- 支持探索，而非总利用当前最佳模型。
- 随交互积累可个性化路由。
- 在线学习行为局部在单条决策。

## 解决什么问题？

若路由器应持续学习而非冻结当前胜者，静态选择器会成为瓶颈。`rl_driven` 为这些场景暴露基于探索的策略。

## 何时使用

- 路由应在线探索候选模型
- 个性化应随时间适应
- 可承受一定探索成本以换取长期更优选择

## 配置

在 `routing.decisions[].algorithm` 中使用：

```yaml
algorithm:
  type: rl_driven
  rl_driven:
    exploration_rate: 0.15
    exploration_decay: 0.97
    min_exploration: 0.03
    use_thompson_sampling: true
    enable_personalization: true
    personalization_blend: 0.4
    session_context_weight: 0.35
    implicit_feedback_weight: 0.5
    cost_awareness: true
    cost_weight: 0.2
    storage_path: state/rl-driven.json
    auto_save_interval: 30s
    use_router_r1_rewards: true
    cost_reward_alpha: 0.3
    format_reward_penalty: -1.0
    enable_llm_routing: false
    router_r1_server_url: ""
    llm_routing_fallback: thompson
    enable_multi_round_aggregation: false
    max_aggregation_rounds: 3
```
