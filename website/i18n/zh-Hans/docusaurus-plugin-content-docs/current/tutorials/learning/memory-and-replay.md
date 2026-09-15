---
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/tutorials/learning/memory-and-replay.md"
  outdated: false
---

# 记忆与回放

## 概览

路由学习在热路径上使用进程内在线状态。启用路由回放时可以记录事件；跨重启持久化需要耐用后端。请求路由不依赖同步的回放存储读取。

## 主要优势

- 让热路径学习读取保持本地且有界。
- 配置耐用后端后，可保留回放证据供审计和评估。
- 将可变防护状态与长期回放证据分开。
- 为离线配方学习提供回放数据，而不在请求路由中增加回放存储读取。

## 解决什么问题？

学习需要历史，但请求路由不能在每次调用时扫描存储或回放日志。路由器为防护和自适应保留紧凑的进程内状态。启用回放时，它还会写入审计、调试、结果和离线配方实验记录；其持久性取决于所选后端。

## 何时使用

- 需要超出紧凑响应头的详细学习诊断。
- 希望评估或智能体在请求之后检查路由证据。
- 希望结果更新在线经验，同时仍链接到回放记录。
- 计划启用回放，并从生产或测试数据运行离线配方学习。

## 分层 {#layers}

| 层 | 热路径 | 职责 |
| --- | --- | --- |
| 防护状态 | 是 | 当前受保护模型、身份范围、轮次数、缓存/工具循环证据和切换历史。 |
| 模型经验 | 是 | 供自适应使用的质量、过度使用、可靠性、延迟、缓存和成本证据。 |
| 路由回放 | 否 | 可选的路由、响应、结果和学习诊断；持久性取决于后端。 |
| 离线配方学习 | 否 | 评估、发现、候选配方、配方补丁和经验种子包。 |

## 配置

使用现有服务配置启用路由回放：

```yaml
global:
  services:
    router_replay:
      enabled: true
      store_backend: postgres
```

此示例使用 Postgres 做持久化。默认 `memory` 后端仅存在于进程内，重启后丢失记录。

启用回放时，学习诊断会写入回放记录：

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

原始 session、conversation、user、tenant 和 workspace 标识符不应存储在学习诊断中。请存储有界哈希以及 source/status 字段。

## 结果 {#outcomes}

通过与回放关联的结果端点提交类型化反馈：

```http
POST /api/v1/observability/outcomes
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

`target: model` 结果会更新在线模型经验。`target: route`、`target: policy`、`target: stability`、`target: provider` 和 `target: router` 结果会保留给回放和离线配方学习，除非存在类型化的在线消费者。

## 配方学习命令 {#recipe-learning-command}

从回放运行离线循环：

```bash
vllm-sr optimize recipe-learning \
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
- 提供 `--recipe-file` 时还会写入候选配方 YAML 文件
