---
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/tutorials/learning/protection.md"
  outdated: false
---

# 防护 {#protection}

## 概览 {#overview}

防护在不把连续性做成语义路由的情况下，保持智能体对话稳定。每个请求仍先经过正常决策路由。自适应提出模型后，防护决定是保持当前模型、允许切换，还是执行有界救援切换。

## 主要优势 {#key-advantages}

- 在智能体对话或整段会话内保持模型选择稳定。
- 保护前缀缓存、工具循环连续性和交接成本。
- 在协议敏感步骤中抑制随机探索。
- 在证据足够强时，仍允许确定性切换和有界救援。
- 让敏感决策通过决策局部控制绕过防护。

## 解决什么问题？ {#what-problem-does-it-solve}

智能体请求并非彼此独立。工具调用、提供方状态、前缀缓存和用户可见的连续性，可能让不必要的模型切换变得昂贵或令人困惑。防护为路由器提供有范围的稳定性守卫，而不把会话连续性变成语义决策规则。

## 何时使用 {#when-to-use}

- 对话应继续使用同一模型，除非切换值得付出稳定性成本。
- 整段会话应在多次用户发起的运行中保持稳定。
- 工具循环或协议状态使随机探索不安全。
- 较弱的受保护模型仍应能通过有界救援退出。

## 配置 {#configuration}

```yaml
global:
  router:
    learning:
      enabled: true
      adaptation:
        enabled: false
      protection:
        enabled: true
        scope: conversation
        identity:
          headers:
            session: x-session-id
            conversation: x-conversation-id
        tuning:
          idle_timeout_seconds: 300
          min_turns_before_switch: 1
          switch_margin: 0.05
          stability_weight: 1.0
```

这样可以独立于在线模型选择自适应来启用防护。否则，在总开关启用时，两个组件默认都会启用。

## 范围 {#scopes}

| 范围 | 保护什么 | 什么可以重新路由 |
| --- | --- | --- |
| `conversation` | 共享同一个 `x-conversation-id` 的轮次；没有该请求头时，为同一 session 中对话开头相同的轮次。 | 同一 `x-session-id` 中新的 `x-conversation-id` 或新的对话，或显式释放。 |
| `session` | 共享同一个 `x-session-id` 的轮次。 | 空闲超时、带 `adaptations.mode: bypass` 的决策，或显式释放。 |

- **使用 `conversation`**：每次 agent 运行都应单独路由。决策变化会重置最少轮次和连续性成本的偏好。
- **使用 `session`**：一个模型选择应在多次用户发起的运行中保持不变。仅仅是决策或对话发生变化，不会释放一个仍合格的当前模型；空闲超时、绕过、候选被排除、符合条件的救援或显式释放才可以。
- **两种范围**都不能把之前的模型保留在已准入的候选集合之外。

如果缺少 session 请求头，防护会失败开放，记录诊断信息，而不是让请求失败。缺少 conversation 请求头不算失败：在 `scope: conversation` 下，Router 按对话的开头（其最初的指令和用户消息）为对话命名，因此 session 中的每个对话都保有自己的键。详见[会话标识](../../api/session-identification#choose-the-identity-you-need)。

### 委派子代理的 Agent {#agents-that-delegate-to-subagents}

Claude Code 这样的 harness 会为一次运行及其子代理发送同一个 session ID。在 `scope: session` 下，这会把每个子代理都保持在其父代理获得的模型上，即使某个决策把子代理的请求路由到别处。请指定标识每个 agent 的请求头，以便分别保持它们：

```yaml
protection:
  enabled: true
  scope: session
  identity:
    headers:
      session: x-claude-code-session-id
      agent: x-claude-code-agent-id
```

Claude Code 只在子代理的请求上发送 `x-claude-code-agent-id`，因此主代理保留自己的会话键，每个子代理获得各自的键。子代理按与会话相同的规则，保持在为它的第一个请求所选的模型上。没有 `agent` 时，防护的行为与之前相同。

Codex CLI 在每个请求上都发送 `session-id` 和 `thread-id`。根线程在两者中都发送自己的 ID。Codex 在该线程上启动的评审子线程保留根线程的 `session-id`，并发送自己的 `thread-id`：

```yaml
protection:
  enabled: true
  scope: session
  identity:
    headers:
      session: session-id
      agent: thread-id
```

- **根线程的键**同时包含它的 session ID 和 thread ID，因为根线程也发送 `thread-id`。
- **两个 ID 都保持不变**，无论是工具续接，还是 `codex exec resume` 之后的轮次，因此防护在整个会话中把根线程和每个评审子线程保持在各自的模型上。
- **`codex exec review`** 会启动新的会话，因此它的评审子线程无论如何都会获得自己的键。

这些请求头已在 Codex CLI 0.167.0 上验证，配置方式见[协议兼容性指南](../../installation/protocol-compatibility)。

### 显式释放 {#explicit-release}

知道任务已进入新阶段的 harness，可以在不开启新会话的情况下请求一次新的决策。它在一个请求上发送释放请求头；防护释放该请求的键上的保持，再次运行配方，并保持配方选出的模型。

- **仍是一次运行。** session ID 不变，因此 Replay 和按运行的报告仍记为一次运行。
- **是新的决策，而不是新的模型。** 配方可以再次选出同一个模型。

```yaml
protection:
  release:
    enabled: true                        # default
    header: x-vsr-protection-release     # default
    min_user_turns_between: 2            # default; 0 removes the limit
```

任何非空值都表示请求释放。它只作用于该请求的防护键：配置了 `agent` 请求头时，释放一个子代理不会影响主代理和其他子代理的保持。

无论是否启用释放，Router 都会在请求到达后端之前删除该请求头。因此，加载配置时只接受 `x-` 开头的请求头名，并拒绝：

- 身份请求头；
- 凭据请求头，例如 `x-api-key` 或 `x-goog-api-key`；
- 网关请求头：`x-envoy-*`、`x-authz-*`、`x-forwarded-*`、`x-request-id`；
- 其他 `x-vsr-` 控制请求头。

只有同时满足以下条件，释放才会被接受：

| 条件 | 否则 |
| --- | --- |
| 该键已经保持了一个模型。新的会话或 agent 会获得正常的首次决策。 | — |
| 请求处于可迁移的轮次边界，没有进行中的工具循环，也没有不可迁移的上下文。 | `release_not_at_turn_boundary` |
| 该键上一次被接受的释放与本次之间，至少相隔 `min_user_turns_between` 个用户轮次。 | `release_interval` |
| `release.enabled` 不是 `false`。 | `release_disabled` |
| 防护对该请求以 `apply` 模式运行。 | `protection_bypassed`（带 `adaptations.mode: bypass` 的决策）、`protection_disabled`、`missing_identity`，或 `release_observe_only`（`protection.mode: observe`） |

在默认的 `min_user_turns_between: 2` 下，第 3 轮的释放允许下一次释放在第 6 轮，`0` 则取消该限制。用户轮次是工具循环之外的请求；工具跳转不计入。

释放的效果：

- **被拒绝：** 当前的保持不变，原因出现在 `x-vsr-warnings` 中，间隔不会被消耗。
- **被接受：** 跳过“保持模型”的捷径和救援。硬性的候选资格仍然适用，Replay 记录 `release_switch`，以及之前和选中的模型。
- **被接受但选中同一模型：** 当配方再次选出被保持的模型时，间隔仍会被消耗，Replay 记录 `hold_current`，原因为 `release_kept_current`。
- **启用了进度门控时：** 它的证据、冷却和振荡规则不会抑制释放；间隔取代了它们，门控追踪记录 `bypassed_by_release`。
- **在 `protection.mode: observe` 下：** 路由不变，释放被记录为未接受，且不消耗间隔。

:::warning
不要把释放请求头放进 Claude Code 的 `ANTHROPIC_CUSTOM_HEADERS`。该设置在启动时固定，并在每个请求上发送，于是每个请求都会请求释放。防护随后每隔 `min_user_turns_between + 1` 个用户轮次就重新决策一次，这实际上关闭了保持。
:::

## 守卫 {#guards}

防护有两个守卫点：

- **preflight** 在工具、协议或常规续接步骤中抑制随机采样。
- **switch guard** 使用缓存、交接、工具循环、会话和切换历史成本，接受或拒绝自适应提出的模型。

切换规则是：

```text
switch if proposal_gain >= switch_margin + stability_weight * switch_cost
```

当反复出现的后端失败或与回放关联的结果表明当前模型能力不足时，防护还可以允许确定性的 `rescue_switch`：

- **失败证据。** 存在防护身份时，后端的 `429` 和 `5xx` 响应会提供失败观测，即使自适应被关闭。这不会启用自适应模型选择或质量更新。
- **不是结果。** 仅有纠正或重试提示并不是归属于模型的结果，不过它的信号可以改变匹配的决策。
- **不是故障转移。** 救援需要另一个合格的提议和足够的证据；它不是即时的后端故障转移。

救援的边界：

- 救援仅限于可迁移轮次边界上的合格模型。它不能越过进行中的工具循环锁或不可迁移上下文锁。
- 没有这些硬性锁时，最少轮次和会话连续性的偏好可以让位于救援。
- 自适应和防护会保留决策选择器施加的候选限制，包括字典序区间。

所有权锁（防护处于 `apply` 模式且存在所配置的身份时）：

- **超出模型池：** 如果硬性锁的之前模型不在已准入的模型池中，返回 `503`，而不是转移所有权或强制使用该模型。
- **Responses 历史**在 Router 把它展开为完整的无状态请求后就变为可迁移；仅保留的 `previous_response_id` 不会产生锁。未知的 response ID 会在历史查找时失败，早于选择。
- **进行中的工具循环**在对话仍停留在准入该循环的决策上时会加锁。当分类转到另一个决策时，锁会释放，新决策从自己的候选中路由，追踪记录为 `decision_drift`。
- **不透明的提供方状态**始终绑定在创建它的模型上。

## 进度门控 {#progress-gate}

进度门控在接受防护或救援提出的模型切换之前，评估最近的结果。它不选择模型，不调用模型或工具，也不替代请求的授权和路由策略。

### 配置 {#configuration-1}

```yaml
global:
  router:
    learning:
      enabled: true
      protection:
        enabled: true
        scope: conversation
        identity:
          headers: {session: x-session-id, conversation: x-conversation-id}
        tuning:
          progress_gate:
            enabled: true
            mode: observe
            calibration_id: ""
            window_size: 8
            window_ttl_seconds: 900
            min_window_outcomes: 3
            min_consecutive_regressions: 2
            min_consecutive_recoveries: 2
            cooldown_seconds: 120
            max_switches_per_window: 2
```

- **默认关闭。** 省略该节或 `enabled` 时，门控处于关闭状态。
- **模式。** `enabled` 和 `mode` 相互独立：`observe` 只记录软性的证据决定而不应用，`enforce` 会应用它们。两者都不会放宽硬性约束，外层防护的 `apply`/`observe`/`bypass` 模式也保持独立。

`calibration_id` 以 `name@version` 的形式指明提供阈值的外部配置档：

- 启用 `enforce` 时必须设置；在 `observe` 下为空值，表示使用未校准的默认值。
- `team/session-policy@2026-09` 是一个标识，而不是校准结果：Router 既不训练校准器，也不检查该配置档的质量。
- 不要把合成的 E2E 测试夹具标识用作生产校准。

窗口与校验：

- `window_size` 接受 1–256 个样本，`window_ttl_seconds` 接受 1–86400 秒。
- 阈值之间的关系在省略的字段取得默认值之后校验：把窗口减少到两个样本，也意味着要调低默认的 `min_window_outcomes: 3`。
- 证据 TTL 与会话空闲过期以及共享存储的保留 TTL 相互独立。之后调大窗口也无法恢复已经过期的历史。

### 证据与决定 {#evidence-and-decisions}

每种结果的含义：

| 结果 | 来源 |
| --- | --- |
| `progress` | 响应采集中的正输出：一个传输层面的近似信号，而不是回答正确的证明。经过认证的归属结果接入也可以设置它。 |
| `no_progress` | 明确报告的零输出。 |
| `missing` | 用量不可用。 |
| `regression` | 经过认证的归属结果接入。 |
| `tool_error` | 一个类型化的事实；响应路径不会根据工具消息是否存在来猜测工具是否成功。 |

已知的提供方失败保持不可归因，即使之后的判定是 `failed`。

事实如何组合：

- **有界**：按数量和事件时间限定。重复的请求/模型样本会合并；不同的模型和未知的请求身份不会合并。
- **归属。** 接入拥有质量判定，采集拥有测得的用量，已知基础设施失败的来源优先于两者。
- **未知保持未知。** 缺失的置信度、成本或延迟绝不会被编造成零。
- **趋势。** 成本趋势使用配置的定价，延迟趋势使用观测到的首 token 时间。它们供回放查看，而不会被用作隐藏的选择分数。

| 原因 | 条件 |
| --- | --- |
| `hard_constraint_conflict` | 进行中的工具循环或不可迁移的上下文 |
| `cold_start` | 没有可观测的证据，包括整个窗口都已过期 |
| `insufficient_evidence` | 可归因的样本太少，或未满足连续次数/趋势阈值 |
| `cooldown` | 距上一次记录的模型变更的时间低于配置的间隔 |
| `oscillation_guard` | 按时效限定的窗口中已包含配置的最大切换次数 |

连续次数：

- **升级**需要连续的同类负面结果和非正的进度趋势；**降级**需要连续的恢复。
- **方向**由同一个质量指标和候选的确切推理强度决定。缺失或不可比较的质量证据会走保守的升级路径。
- **空缺。** 缺失和不可归因的结果既不会延长也不会打断连续次数，也不计入可归因的覆盖率。

### 硬性约束与 Replay {#hard-constraints-and-replay}

在应用提议或恢复当前模型之前，集成会重新检查：

- 现有的候选边界和配置的后端；
- 上下文上限和协议能力；
- 可迁移性锁；
- 所选算法的硬性 SLO 和质量过滤条件。

门控不能引入被策略排除的候选。如果当前模型已不再合格，软性抑制也无法恢复它；当没有合格的结果时，请求通过现有的选择错误路径失败关闭。

Replay 的 `session_policy.switch_gate` 记录：

- 证据版本、`calibration_id`、证据指标及其可用性；
- 决定和原因、切换方向、候选名称和切换历史；
- 实际发生了什么：`enforced` 描述门控模式，而 `applied`、`application_reason` 和 `final_model` 描述结果。硬性锁在两种模式下都可能生效。

被拒绝的救援保存在 `rescue_switch_gate` 中，与最终的主路径决定并列。没有发生切换时，不会编造门控判定。

维护中的 `progress-gate` E2E 配置档通过 HTTP 覆盖请求采集、经过认证的幂等反馈、抑制、允许的切换、冷却、振荡和 Replay。它的专用部署不改变 `router-replay` 的测试。

## 决策边界 {#decision-boundaries}

大多数决策不需要局部配置。对硬性的策略边界使用 `bypass`：

```yaml
recipes:
  - name: default
    decisions:
      - name: local_privacy_policy
        description: Keep privacy-sensitive traffic on the local model.
        priority: 200
        modelRefs:
          - model: local-private-model
        adaptations:
          mode: bypass
```

使用 `observe` 收集诊断信息，而不改变最终的模型：

```yaml
adaptations:
  protection:
    mode: observe
```

## 诊断 {#diagnostics}

```http
x-vsr-trace: session_phase=user_turn, learning=(protection;action=hold_current;scope=conversation;reason=cache_cost_high)
```

客户端界面应把原始动作转换为面向用户的文字：

| 动作 | 显示为 |
| --- | --- |
| `hold_current` | “保持运行模型” |
| `allow_switch` | “允许切换” |
| `rescue_switch` | “救援切换” |
| `bypass` | “已绕过学习” |

Router Replay 存储完整的防护追踪：身份来源和哈希、受保护的模型、基础模型、提议的模型、最终模型、切换成本、缓存证据、工具循环状态、模式、范围、动作和原因。
