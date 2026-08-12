---
translation:
  source_commit: "ff987c87"
  source_file: "docs/tutorials/decision/retention.md"
  outdated: false
---

# 保留指令

## 概述

当匹配的决策还必须告知响应侧运行时组件哪些状态应保留、丢弃或视为对未来轮次有价值时，请使用 `EMIT retention`。

保留指令并不是另一种检测器。信号决定路由是否匹配；保留指令描述路由匹配后产生的副作用。

## 主要优势

- 将保留策略放在产生该策略的决策旁边。
- 避免将缓存或会话保留行为隐藏在 extproc 分支中。
- 为运维人员提供一份关于保留、丢弃、TTL 和前缀保留提示的类型化契约。
- 允许运行时立即使用其中安全的部分，同时将其余部分保留为可审计的日志和追踪属性。

## 它解决什么问题？

会话感知路由需要决定的不只是下一个模型，还需要控制哪些证据可以保留到后续轮次。没有显式指令时，缓存写入、模型亲和性提示和前缀缓存偏好可能会分散到不同的运行时启发式规则中，因而难以审计。

`EMIT retention` 通过使决策的保留副作用结构化且可审查来解决这个问题。

## 运行时作用域

运行时现在会使用每个保留字段，但有一项评分偏置仍被推迟：

- `drop: true` 会跳过匹配决策的响应侧响应缓存写入。它不会阻止当前请求读取响应缓存。
- `ttl_turns`（当其值 `> 0` 时）会覆盖此条目的决策响应缓存 TTL，将其作用域限定为大约这么多个未来轮次（轮次会在缓存写入边界处映射为秒；可配置的每轮秒数调节项属于后续工作）。
- `keep_current_model: true` 会强制会话感知模型切换门控继续使用当前模型，而不受门控模式（shadow 或 enforce）影响，前提是当前模型已知且为有效候选模型。
- `prefer_prefix_retention: true` 会作为 `x-vsr-retention-prefer-prefix` 响应头发送到推理池；会话感知评分偏置以及提供商/KV 缓存淘汰集成仍属于后续工作。

每个被显式设置的字段也会作为 `x-vsr-retention-*` 响应头发送到响应中（包括显式设置的 `ttl_turns: 0`），并记录在日志/追踪中，以便推理池和运维人员审计路由器的保留意图。`drop` 与正值 `ttl_turns` 互斥（同时设置两者会导致验证拒绝）。

## 保留目标清单

除了响应缓存写入之外，会话感知路由还需要针对以下状态信号或运行时提示制定保留策略：

| 目标 | 保留为何重要 | 当前状态 |
| --- | --- | --- |
| 语义缓存响应写入 | 防止低价值、私密或不稳定的轮次成为未来的缓存命中。 | 由 `drop: true` 强制执行。 |
| 缓存写入生命周期 | 仅在有限数量的未来轮次中保留可复用响应，而不是只使用基于挂钟时间的 TTL。 | `ttl_turns` 会覆盖每个条目的响应缓存 TTL（轮次映射为秒）；可配置的每轮秒数调节项属于后续工作。 |
| 当前模型亲和性 | 避免多轮会话反复切离拥有对话上下文的模型。 | `keep_current_model` 会在任何模式下通过模型切换门控强制保留当前模型。 |
| 前缀或 KV 缓存热度 | 在后续轮次很可能出现时，保护成本高昂的提示词前缀或热态工作节点状态。 | `prefer_prefix_retention` 会作为响应头发送到推理池；评分偏置以及提供商/缓存管理器淘汰集成属于后续工作。 |
| 轮次和转换遥测 | 记录轮次索引、已选模型、token/成本总计、重试/质量趋势和模型转换，以便审计停留与切换策略。 | 由会话遥测和转换日志组件生成，而不是仅由此指令生成。 |
| 对话、工具和重放历史记录 | 保留后续分类、工具检索和离线查找表生成所需的历史记录。 | 由 Response API、工具历史记录和 Router Replay 组件负责；保留指令不应重复建立这些存储。 |

这份清单说明了为什么该指令命名为 `retention` 而不是 `semantic_cache`：跳过响应缓存写入只是更广泛的会话保留契约的第一个运行时使用方。

## DSL 往返转换作用域

为提高可读性，下面的示例使用 `DECISION_TREE`，但该语法只是一种编写便利形式。编译后的配置存储扁平的 `routing.decisions`，而基于配置的导出/反编译路径会生成扁平的 `ROUTE` 块，而不会重建原始决策树。保留字段仍会通过 DSL/配置完成往返转换，但决策树形状本身不会被往返保留。

## DSL 示例

```dsl
DECISION_TREE session_routing {
  IF pii("sensitive") {
    NAME "sensitive-turn"
    TIER 2
    MODEL "qwen3-8b" (reasoning = true)
    EMIT retention {
      drop: true
    }
  }
  ELSE IF conversation("follow_up") {
    NAME "follow-up-continuity"
    TIER 3
    MODEL "qwen3-32b" (reasoning = true)
    EMIT retention {
      keep_current_model: true
      prefer_prefix_retention: true
    }
  }
  ELSE {
    NAME "default-route"
    TIER 1
    MODEL "qwen3-8b" (reasoning = false)
  }
}
```

## 配置

编译后的配置将保留指令存储在匹配的决策下：

```yaml
routing:
  decisions:
    - name: sensitive-turn
      rules:
        operator: AND
        conditions:
          - type: pii
            name: sensitive
      modelRefs:
        - model: qwen3-8b
          use_reasoning: true
      emits:
        - kind: retention
          retention:
            drop: true
```

## 字段参考

| 字段 | 类型 | 运行时行为 |
| --- | --- | --- |
| `drop` | boolean | 当其值为 `true` 时，跳过匹配决策的响应侧响应缓存写入。 |
| `ttl_turns` | integer >= 0 | 当其值 `> 0` 时，覆盖匹配决策的响应缓存条目 TTL（轮次映射为秒）。只要被显式设置，就会作为 `x-vsr-retention-ttl-turns` 发送，包括显式设置的 `0`。 |
| `keep_current_model` | boolean | 当其值为 `true` 时，强制模型切换门控保留当前模型，而不受门控模式影响。同时作为 `x-vsr-retention-keep-current-model` 发送。 |
| `prefer_prefix_retention` | boolean | 作为 `x-vsr-retention-prefer-prefix` 发送到推理池；会话感知评分偏置和 KV 缓存驱逐集成属于后续工作。 |

验证会拒绝同一路由上重复的 `EMIT retention` 块、未知字段、无效字段类型、负值 `ttl_turns`，以及相互矛盾的 `drop: true` 与正值 `ttl_turns` 组合。

## 何时使用

在以下情况下使用保留指令：

- 路由应正常响应，但不应将响应写入语义缓存，例如涉及 PII、机密信息或一次性个性化上下文时
- 后续轮次密集的路由应为以后轮次保留连续性提示时
- 决策需要可审计的保留元数据，而所有运行时使用方尚未全部启用时

不要使用 `EMIT retention` 替代路由条件、插件配置或提供商特定的缓存 API。
