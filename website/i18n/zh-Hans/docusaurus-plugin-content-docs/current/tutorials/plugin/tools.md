---
is_mtpe: false
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/plugin/tools.md"
  outdated: false
---

# 工具

## 概述

`tools` 是处理请求工具的路由局部插件。它的 `policy` 按名称允许、阻止或移除工具；它的
`select` 块询问决策模型请求需要哪些工具，候选来自请求自带的工具（`mode: filter`）或工具目录
（`mode: add`）。

## 主要优势

- 工具策略附着在匹配的路由上。
- 每个请求只调用一次决策模型来选择工具：同时询问 `tool_need`（请求是否需要工具）和
  `tool_rank`（需要哪些工具）。
- filter 模式保持请求中工具的顺序，因此提供方的提示缓存前缀不变。

## 它解决什么问题？

工具行为是路由策略的一部分。有些路由必须完全移除工具，有些原样透传，有些只应发送请求需要的工具：
冗长的工具列表既耗费 token，也会干扰模型。`tools` 插件把这份路由局部契约写明，并由已经回答路由信号的
决策模型来选择工具。

## 何时使用

- 某条路由应禁用所有工具
- 某条路由应通过显式的允许和阻止列表限制工具访问
- 某条路由应只发送请求需要的工具，或从目录中添加相关工具
- 隐私路由应保留工具历史用于路由，但从所选模型的请求中省略先前的工具调用和结果

## 配置

在 `recipes[].decisions[].plugins` 下添加插件：

```yaml
plugins:
  - type: tools
    configuration:
      enabled: true
      policy:                         # 确定性规则，先执行
        mode: filtered                # passthrough | filtered | none
        allow_tools: [docs.search, tickets.lookup]
        block_tools: [admin.delete]
        strip_tool_history: false
      select:                         # 决策模型在策略允许的工具中选择
        mode: add                     # filter | add
        tools_db_path: config/tools_db.json
        match:
          judge: {preset: tool_rank, top_k: 3, min_ratio: 0.1, max_candidates: 255}
        need_threshold: 0.5           # P(tool_need) 低于此值时不保留也不添加工具
        preserve_count: 0             # filter 模式：请求的前几个工具始终保留
        on_error: skip                # add：skip | fail；filter：keep | strip | fail
        input: {history: false}
        timeout_ms: 1000
```

### 策略

- `passthrough`（默认）保持请求的工具不变。
- `filtered` 在 `allow_tools` 列出工具时只保留这些工具，并移除 `block_tools` 列出的工具。指名被移除工具的 `tool_choice` 也随之移除，由模型在剩下的工具中选择。
- `none` 移除所有工具以及请求的 `tool_choice`。设置 `strip_tool_history: true` 时，还会从发往
  提供方的请求中移除先前的助手工具调用和工具结果。策略在信号与决策评估之后执行，因此不会改变匹配的路由。
- `trusted_facts` 根据运维策略、网关证明和新鲜的运行时可用性（当决策的选择从目录添加工具时，
  即该目录）门控工具使用。除非启用，否则关闭。

### 选择

当请求把工具使用交给模型时（没有 `tool_choice`，或为 `auto`），选择才会执行。对回答 judge 的部署
（默认是决策模型，除非 `match.judge.deployment` 指定其他部署）的一次调用，会就当前用户轮次（设置
`input: {history: true}` 时还包括更早的用户轮次）询问两个问题：

- `tool_need`，是/否问题：回答该请求是否需要工具？
- `match` judge，默认为 `tool_rank`：选项是候选工具的选择题，每个选项由工具描述说明。它的保留规则
  `top_k`、`min_ratio` 和 `min_score` 决定保留哪些工具。

候选与结果取决于模式：

| 模式 | 候选 | 结果 |
| --- | --- | --- |
| `filter` | 策略保留的请求工具 | judge 保留的工具和前 `preserve_count` 个工具，按请求中的顺序 |
| `add` | 策略允许、且请求未携带的目录工具 | 请求自带的工具，之后是 judge 保留的工具，按得分从高到低 |

- 当 P(`tool_need`) 低于 `need_threshold` 时，judge 选出的工具既不保留也不添加。此时 filter 模式只保留前
  `preserve_count` 个工具；没有剩余工具的请求也会失去它的 `tool_choice`。
- `on_error` 决定调用失败或超时时的行为：filter 模式保留请求的工具（`keep`）、移除它们（`strip`）或拒绝请求
  （`fail`）；add 模式不添加工具（`skip`）或拒绝请求（`fail`）。被拒绝的请求得到 `503` 和代码
  [`tool_selection_failed`](../../api/router.md#routing-errors)。
- 一个问题最多有 255 个选项，所以 judge 最多读取 `match.judge.max_candidates` 个工具（默认 255）：按请求
  或目录列出的顺序取前面的工具。更多的工具从不被拒绝；也可以写 `recall: {method: bm25, top_k: ...}`，按与请求
  共有的词把工具缩小到描述最相关的那些。召回只负责缩小范围，因此没有 judge 的 `match` 块在加载时失败。
  每次截断都会在请求的 trace span 上记一个 `match.judge.truncated` 事件，并计入 `vsr_match_judge_truncated_total`。
- `timeout_ms` 限制调用时长，最多 60000。未设置时使用 systemone 部署自己的 `timeout_ms`，否则为 1000。
  judge 自己的 `on_error` 和 `timeout_ms` 不被接受：以选择的设置为准。

add 模式的目录是一个 JSON 列表，每个条目为
`{"tool": {"type": "function", "function": {...}}, "description": "..."}`；`description` 是决策模型选择该
工具时阅读的内容，缺省时使用函数自己的描述。相对的 `tools_db_path` 从 Router 的工作目录读取。加载时会检查
能读取到的目录，大小不限：名称不能重复（`duplicate_tool_name`），且策略至少允许一个工具。Router 在构建要服务的配置时会再次读取目录，并拒绝无法读取的目录。

每次选择都会记录在请求的 [router replay](./router-replay) 记录中：保留的工具、P(`tool_need`) 和结果。
它的 judge 计入 `vsr_match_judge_total{feature="tools"}`。

### 迁移

`tool_selection` 插件即 `select` 块，`global.integrations.tools` 已移除，策略字段移到了 `policy` 下。
仍使用这些形式的配置会在加载时失败并给出一行提示，`vllm-sr config migrate` 会改写它。嵌入相似度阈值
（`similarity_threshold`、`relevance_threshold`）在 judge 中没有对应项：migrate 会留下说明，需要时可以
针对决策模型调整 `min_ratio` 或 `min_score`。

工具选择控制的是哪些内容到达模型；它不授权工具执行。请在工具服务处实施权限控制，并把工具 schema 和结果视为
发往提供方的内容。完整示例：
[`config/fragments/plugin/tools/`](https://github.com/vllm-project/semantic-router/tree/main/config/fragments/plugin/tools)。
