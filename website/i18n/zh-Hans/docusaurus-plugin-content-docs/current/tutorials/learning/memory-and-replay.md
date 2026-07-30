---
translation:
  source_commit: "ad233487"
  source_file: "docs/tutorials/learning/memory-and-replay.md"
  outdated: false
---

# 内存与回放

## 概览

Router Learning 在热路径上使用进程内在线状态，并将 Router Replay 用作持久事件日志。请求路由不依赖同步读取外部存储。

## 主要优势

- 将热路径上的学习读取保持在本地并控制在有界范围内。
- 保持 Router Replay 作为持久审计与评估的事实来源。
- 将可变的保护状态与长期留存的回放证据分离。
- 在不拖慢请求的情况下，为离线配方学习提供所需数据。

## 解决什么问题？

学习需要历史信息，但请求路由不能在每次调用时扫描存储或回放日志。路由器会为保护和 adaptation 维护紧凑的进程内状态，随后写入持久的 replay 记录，用于审计、调试、结果和离线配方实验。

## 何时使用

- 除紧凑的响应 header 外，还需要详细的学习诊断。
- 希望评估或 agent 能在请求完成后检查路由证据。
- 希望结果在更新在线经验的同时，仍与 replay 记录保持关联。
- 计划使用生产或测试 replay 数据运行离线配方学习。

## 分层

| 层 | 热路径 | 职责 |
| --- | --- | --- |
| 保护状态 | 是 | 当前受保护模型、身份作用域、回合数、缓存/工具循环证据和切换历史。 |
| 模型经验 | 是 | 供 Adaptation 使用的质量、过度使用、可靠性、延迟、缓存和成本证据。 |
| Router Replay | 否 | 持久的路由、响应、结果和学习诊断。 |
| 离线配方学习 | 否 | 评估、发现、候选配方、配方补丁和经验 seed packs。 |

## 配置

使用现有的服务配置启用 Router Replay：

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres
```

启用 replay 后，学习诊断会写入 replay 记录：

```json
{
  "learning": {
    "protection_preflight": {
      "action": "allow_sampling",
      "scope": "conversation",
      "reason": "no_tool_or_protocol_state"
    },
    "adaptation": {
      "strategy": "routing_sampling",
      "candidate_set": "decision",
      "base_model": "small-model",
      "proposal_model": "frontier-model",
      "reason": "posterior_win"
    },
    "protection": {
      "action": "allow_switch",
      "base_model": "small-model",
      "proposal_model": "frontier-model",
      "final_model": "frontier-model",
      "switch_cost": 0.03,
      "reason": "switch_allowed"
    }
  }
}
```

不应将原始的 session、conversation、user、tenant 和 workspace 标识符存入学习诊断。应存储有界哈希值以及 source/status 字段。

## 结果

通过与 replay 关联的结果端点提交类型化反馈：

```http
POST /v1/router/outcomes
```

```json
{
  "replay_id": "replay_123",
  "source": "agent",
  "target": "model",
  "target_ref": "frontier-model",
  "verdict": "good_fit",
  "reason": "solved_complex_task",
  "score": 1.0
}
```

`target: model` 结果会更新在线模型经验。`target: route`、`target: policy`、`target: stability`、`target: provider` 和 `target: router` 结果会保留在 replay 中并用于离线配方学习，除非存在类型化的在线消费者。

## 配方学习命令

从 replay 运行离线循环：

```bash
vllm-sr eval recipe-learning \
  --replay-file replay.json \
  --recipe-file config.yaml \
  --output-dir ./router-learning-report
```

该命令会写入：

- `metrics.json`
- `findings.json`
- `experiment_results.json`
- `recipe_patch.json`
- `experience_seed_pack.json`
- 提供 `--recipe-file` 时生成的候选配方 YAML 文件
