---
title: 决策排序语义
description: 把规则资格、策略次序和证据强度分开，使决策排序不随请求变化。
created: 2026-09-01
status: Proposal
translation:
  source_commit: "9649d02f2582471642196bde2e086c34ccb9c88c"
  source_file: "docs/proposals/decision-ranking-semantics.md"
  outdated: false
is_mtpe: true
---

> **状态：** 提案 · **创建于：** 2026-09-01

## 问题 {#problem}

决策选择把三个概念搅进一个数：决策是否匹配、策略把它排在哪、它的证据有多强。#3080 堵住了未打分规则冒充实效确定，但排序仍取决于某个请求恰好给哪些规则打了分，所以同一份配置能按请求排出不同的序。

## 优先级与置信度 {#priority-and-confidence}

`priority` 和 `confidence` 不是同一个候选集上的两种排序。

`priority` 每条决策都声明，永远拿得到。`confidence` 只在有东西报过一个可比的分数时才存在。这个不对称正是 #3080 需要池级回退的原因，也是 `signalConfidence` 不得不给缺失的键发明一个 `1.0` 的原因。

顺带，也没有单一的数轴可排。`SignalConfidences` 把一个分类器概率、一个嵌入相似度、一个跨轮最低相似度、一个投影输出和一个布尔常量塞进同一个 `map[string]float64`。宣布其中哪些互相可比，是 DR-06 的内容。

所以本提案把 `priority` 当次序，把 confidence 当已自宣可比的池内部的微调，把 `routing.strategy` 当同一层内两者之间的开关。

## 当前排序 {#current-ordering}

跑哪个分支，取决于有没有匹配的决策设了 tier。

| 情形 | 键，按序 |
| --- | --- |
| 有 `tier > 0` | tier 升序，catch-all 殿后，confidence 降序，priority 降序，name 升序 |
| 无 tier，`strategy: confidence` | catch-all 殿后，confidence 降序，priority 降序，name 升序 |
| 无 tier，`strategy: priority` | priority 降序，confidence 降序，name 升序 |

confidence 只作用于可比的池：任一非 catch-all 成员未打分，#3080 就把池退回 priority。分层选择下池就是 tier，否则就是整个结果集。

这张表有五个性质要逐个定夺。

1. 分层分支从不读 `routing.strategy`，于是一个设置有两个意思。
2. `priority` 分支没有 catch-all 检查，和另两个不一样，所以高优先级的 catch-all 能在那里压过一个真匹配。
3. `AND` 给匹配的子节点取平均，于是一个决策证据越多，聚合分越低：`mean(.90, .90)` 压过 `mean(.90, .90, .88)`。
4. `evalOR` 只留获胜分支的 scored 标志。于是一个多出来的匹配关键词分支，能把一个决策从 scored 降成 unscored，让整个池翻去 priority 排序——尽管那个匹配明明是给它添的证据。
5. `signalConfidence` 把缺失的键当 `1.0` 未打分，而匹配上的 `conversation` 规则会显式写 `1.0`。布尔谓词于是报出标着 scored 的最大证据，#3080 的门抓不到，因为那门只对未打分成员反应。

分数也不是同一个量。`SignalConfidences` 是个平的 `map[string]float64`，装着：

| 键 | 写入的值 |
| --- | --- |
| `domain:` | 分类器概率 |
| `embedding:` | 嵌入相似度 |
| `reask:` | 一段匹配轮次上的最低相似度 |
| `conversation:` | 谓词成立为 `1.0`，否则 `0` |
| `projection:` | 投影输出，过一道斜率校准 |
| 通用 `type: llm` | 自 #3152 起的模型报告标签分布 |

`evalAND` 把一个概率、一个相似度和一个常量加在一起。

## 提案 {#proposal}

- tier 保持硬性优先级边界。
- `routing.strategy` 在选中的 tier 内生效，于是一个设置一个意思。
- `priority` 是确定性的策略次序，不被契约未宣布可比的证据推翻。
- 缺失的分就是缺失，不是 `1.0`。#3106 已把失败的信号建成未知；缺失的需要同样待遇，原因不同而已。
- 每个叶子要么是策略要么是证据。关键词规则、`NOT`、谓词、会话谓词和投影输出是策略：它们闸住资格，不在聚合里占分量。投影复述的是运维早已做过的决定。
- confidence 只给报上来的证据排序，其种类宣布可比，其聚合不随叶子数、也不随 `OR` 哪个分支 matched 而动。
- 其余情况排序退回 priority，追踪记录排序模式、所采分数的来源、confidence 为何不适用，以及最终 tie-break。

## 兼容性 {#compatibility}

`decision.tier` 在六份出厂配置的 52 条决策上设过，但只有四个 tier 装着不止一条决策。

| 池 | 按什么排 | 为什么 |
| --- | --- | --- |
| `agent` tier 1 | priority | `local_privacy_policy` 带一个 `NOT` |
| `agent` tier 3 | priority | 每个成员都带 `keyword` 或 `NOT` |
| `privacy` tier 2 | priority | `local_privacy_policy` 带一个 `NOT` |
| `multi-objective` / `privacy-first` tier 2 | confidence | 两个成员都打了分 |

只有最后一个池受影响。`omni` 按一个布尔会话谓词匹配，报一个打了分的 `1.0`，压过 `unified_privacy_sensitive_route` 的投影分。出厂优先级恰好一致，所以今天没有误路由，但拍板的不是 priority：把那条路由提到 priority `900` 对 `omni` 的 `250`，两个策略下选中的仍是 `omni`。把会话谓词当策略，运维的次序才复位。

另三个池已经退回 priority，不受影响。

对那张表还有一句提醒。一棵树含 `keyword`、`NOT` 或谓词叶子，这里的决策就按未打分处理。这对 `AND` 是精确的——`scored` 是子节点的合取——但对 `OR` 只是近似，那里只有获胜分支的标志会传播。`agent` tier 3 的决策含 `OR` 分支，所以那里的 priority 回退是常态，不是保证。

## 交付 {#delivery}

| 切片 | 范围 | 运行时改动 |
| --- | --- | --- |
| DR-01 | 在决策教程里写清排序表和 `decision.tier` | 无 |
| DR-02 | 给每个叶子宣布策略或证据角色，把策略叶子排除出聚合 | 有 |
| DR-03 | 止住 `evalOR` 让策略分支的未打分标志盖过打了分的 | 有 |
| DR-04 | 让 `routing.strategy` 在选中的 tier 内生效 | 有 |
| DR-05 | `strategy: priority` 下把 catch-all 排最后 | 有 |
| DR-06 | 在配置校验里宣布分数种类并执行可比性检查 | 有 |
| DR-07 | 在追踪里报告排序模式、来源、回退原因和 tie-break | 有 |
| DR-08 | 跨路由冲突的配方一致性探针 | 无 |

DR-01 和 DR-08 无论采纳哪个契约都对当前行为成立。DR-02 带着出厂配置里唯一可见的排序改动。

## 开放问题 {#open-questions}

- 分数种类按信号类型宣布，还是按信号实例？
- 缺失的分数复用 #3106 的未知状态，还是用一个更弱的标记，区分从未报过与报失败？
- `strategy: priority` 下 catch-all 是否也排最后？

## 参考 {#references}

- [提示词分类路由](./prompt-classification-routing)
- [Router Learning](./router-learning-memory-and-adaptations)
- [决策概览](../tutorials/decision/overview)
