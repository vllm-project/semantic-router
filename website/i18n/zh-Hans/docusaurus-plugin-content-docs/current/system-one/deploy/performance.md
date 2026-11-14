---
title: 性能与成本
sidebar_label: 性能与成本
sidebar_position: 2
description: System One 给一个请求带来的成本，以及限定成本的批处理、上限、预算和放置选择。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/system-one/deploy/performance.md"
  outdated: false
---

# 性能与成本 {#performance-and-cost}

一个 System One 问题的成本是一个小模型的一次前向计算，而不是聊天模型的一次生成：没有要解码的输出，也没有要解析的文本。一个请求的成本取决于它发出的调用次数、每次调用读取的内容大小，以及模型运行在哪里。

:::tip
大多数请求只发出**一次**调用：请求阶段的那一次。没有被任何内容读取的规则不会被询问，没有问题的阶段不会发出调用。
:::

## 每个请求的调用 {#calls-per-request}

| 阶段 | 调用 | 何时询问 |
| --- | --- | --- |
| 请求信号 | 每个部署一次，包含针对每个状态的所有问题 | 决策、投影或请求插件读取某条规则时 |
| `systemone` 选择器 | 一次 | 匹配的决策使用该选择器时 |
| 评审（`same_request`、`relevance`、`tool_need` + `tool_rank`） | 每个功能一次，每个被评判的候选一个条目 | 功能运行且有候选时 |
| `masking` | 一次，针对请求阶段尚未回答的文本 | 决策进行脱敏时 |
| 响应信号 | 每个部署一次 | `response_guard` 规则读取它们时 |
| `escalate` 检查 | 每个被评判的回答一次 | `accept.systemone` 或 `accept.refusal` |

## 批处理 {#batching}

- **问题共享调用。** 一个阶段向同一部署提出的所有问题一起发送，无论来自哪个信号或规则。模型运行时也会把并发请求的调用合并成批（[引擎参数](../../model-runtime/deploy/engine-args.md#batching)）。
- **问题共享答案。** 针对同一状态、向同一部署询问同一预设的规则共享一个问题。
- **只读一次状态的模型**（d3-turbo、d3-ultra）只读取一次状态，然后从这次读取中回答每个问题，因此对一个请求问第十个问题的成本比第一个多不了多少。Decision 2.0 对多问题请求的共享上下文只计算一次（`enable_shared_context`）。
- **答案会被复用。** Router 为每个本地模型保留最近的 4,096 个答案（`VLLM_SRUN_RESULT_CACHE` 改变这个数量；0 关闭它），并把远程 API 的答案保留最长 10 分钟，直到配置变化。

## 上限 {#caps}

- **评审**最多读取 `judge.max_candidates` 个候选：逐候选评审默认 64 个、最多 4096 个，choice 评审最多 255 个（也是默认值）。超出部分按召回排名截断，没有召回时按列出顺序截断，绝不失败；`vsr_match_judge_truncated_total` 会计数。
- **召回**限定评审看到的内容：缓存评审确认其召回的前 3 个命中，RAG 评审确认前 20 个文档。
- **工具**在策略过滤后超过 255 个时会被截断，或先用 `recall: {method: bm25}` 缩小范围。

## 预算与截止时间 {#budgets-and-deadlines}

| 设置 | 限定 |
| --- | --- |
| 规则或评审的 `timeout_ms` | 它的答案，最多 60,000 ms；否则使用部署的值，再否则 1,000 ms |
| 部署的 `timeout_ms` | 对附加或远程部署的一次调用尝试 |
| `global.model_catalog.signal_timeout_ms` | 一个请求的全部信号；0（默认）表示从请求自身的截止时间推导 |
| 执行器的 `budget` | `escalate` 的截止时间和调用次数，其 `answer_check` 调用计入截止时间 |

:::note
迟到的答案会让其问题成为未知，由每条规则或功能的失败策略决定。截止时间结束 Router 的等待，但不会停止已经在运行的前向计算。
:::

## 决策模型过载时 {#when-the-decision-model-is-overloaded}

决策模型没有余量时会拒绝调用：它的请求队列已满，回答 `overloaded`。这时，失败即关闭的规则会匹配。请求阶段的 `jailbreak` 或 `pii` 规则就是这样，除非其 `on_unknown` 为 `no_match`。随后，带 `fast_response` 的决策会像请求不安全一样拒绝它。负载高时，这会拒绝大多数请求，而编程代理会把这种拒绝当作最终结果。

`global.model_catalog.on_signal_overload` 让这样的请求收到客户端会重试的错误。不配置时，行为不变：

```yaml
global:
  model_catalog:
    on_signal_overload:
      action: retry          # refuse（默认）或 retry
      wait_ms: 2000          # 0 到 60,000；0 表示不等待
      retry_after_seconds: 0 # 最多 60；0 表示由 wait_ms 推导
```

`action: retry` 时：

- **等待余量。** 请求阶段每个返回 `overloaded` 的调用，都会在请求等待余量期间重新询问，无论其规则的失败策略如何：失败即关闭的规则、像 `safety` 这样失败即放行的规则，以及路由问题都一样。如果模型给出回答，请求就按没有过载时那样读取。
  - 等待从请求第一次收到过载回答起，最长 `wait_ms`，且不超过信号的截止时间。规则自己的 `timeout_ms` 限定的是其调用的每次尝试，而不是这段等待。
  - 一个请求发往同一模型的调用会按轮次一起重新询问，因此原本在一次调用中的问题会再次放在一次调用中。
- **限制负载。** 一个请求发往同一模型的调用最多分 3 轮重新询问，各轮之间至少相隔一个步长：在等待的四分之一处、一半处和结束时。它的某个调用刚占用一轮后，其他到达的调用会在同一轮中询问，因此一个只回答部分调用、拒绝其余调用的模型不会一下子耗尽请求的轮次。
  - 模型以过载拒绝调用期间，无论有多少请求在等待，每一步只有一个等待中请求的轮次询问它。步长带抖动，从 100 ms 起倍增至 1 s。
  - 在这样的一轮中，如果一个请求有超过 8 个调用在等待该模型（例如长输入的各个窗口），它只单独发送其中一个作为探测。探测不计入这 3 轮。
  - 任何调用一旦得到回答，模型就重新开放，等待它的每个请求随即进行自己的一轮。
- **不从等待中学习速度。** 等待过的防护扫描不会为其模型记录速度：这段时间不能说明模型读取得多快。它只从模型再次给出回答时起，根据此后读取的窗口判断自己是否落后于进度，因此一个已无法按配置大小的窗口读完的长输入，仍会改用更大的窗口读完。
- **返回可重试的错误。** 如果模型仍未给出回答，Router 在以下两种情况下返回客户端会重试的错误：
  - 拒绝请求的只有在过载的模型上失败即关闭的规则。这个错误代替拒绝。
  - 某条因模型过载而未读取的规则，在匹配时会拒绝该请求。这个错误代替转发。像 `safety` 这样失败即放行的规则，在模型忙得读不了它时，不再让请求通过。

  Anthropic Messages 客户端收到 `529 overloaded_error`。OpenAI Chat Completions 或 Responses 客户端收到 `429`，代码为 `overloaded`。它们的 SDK 和代理会在 `Retry-After` 之后重试 429，而有些会把 5xx 当作失败。两种错误都在流式响应的任何字节之前送达客户端，并带有：
  - `Retry-After`：取 `retry_after_seconds`，否则取 `wait_ms` 向上取整到整秒，至少 1 秒。它最多为 60：有些客户端会忽略更长的值，有些则会完整地等待。
  - `x-vsr-warnings: guard_overloaded`。
- **绝不未经检查就转发。** 只要某条可能拒绝请求的规则因过载而未读取，请求就不会被转发。
  - 匹配的规则仍会拒绝它，条件在出错时匹配的决策也一样。
  - 其他失败（例如超时，或模型尚未就绪）结果不变。
  - 超出所有上下文窗口的提示词仍会收到 `400 context_length_exceeded`。
  - 未读取的路由问题不会阻止请求继续：它能选择路由，但不能拒绝请求。

这些与拒绝以及 `signal_failed_closed` 日志分开统计：

- `vsr_signal_overload_waits_total{deployment, outcome}` 统计重新询问的调用，按最后一次的结果分类：有余量时为 `ok`，等待耗尽时为 `overloaded`，或其他失败的原因。
- `vsr_signal_overload_retryable_total{format}` 统计收到可重试错误的请求，按客户端的线路格式分类。

## 长输入 {#long-inputs}

很长的用户消息（粘贴的文档、日志、很长的工具输出）由 Router 自己按以下两种方式读取，因此本地和远程决策模型的行为一致。

**路由问题读取有界视图。** 指不读取完整输入的规则：`domains`、`complexity`、`modality`、`fact_check`、`turn_relation`、`user_feedbacks`，以及 `coverage: truncate` 的 `systemone` 规则。

- 超过部署 `routing_view` 的消息只保留开头和结尾，中间用标记 `[... 41000 tokens omitted ...]` 代替。开头占视图的四分之一，结尾占其余部分，因为请求的指令通常在消息末尾。
- 不超过视图的消息原样逐字节发送。

**防护规则分窗口读取完整输入。** 指 `jailbreak`、`safety` 和 `pii`，包括请求阶段和响应阶段。

- 超过一个 `scan_window` 的输入被切成相互重叠 `scan_window.overlap` 的窗口，跨越切点的文本总能被某个窗口完整读到。
- 读取顺序是先结尾，再开头，然后按二分读中间：先读中点处的窗口，再读四分点、八分点处的窗口，依此类推，起点每个请求都不同。任一时刻已读的窗口都分布在整个输入上，因此输入的任何部分都不会可预知地被漏读。每个窗口单独一次调用，每个部署同时最多 `scan_window.max_parallel` 个。读取同一输入的 `jailbreak`、`safety` 和 `pii` 规则共用每个窗口的调用，一个窗口读完，它的回答立即计入。
- 任何一个窗口发现攻击、危害或个人数据，整个输入即被标记。`pii` 的 span 保持其在原输入中的位置，脱敏会替换每一处。
- 使规则匹配的窗口即决定了该规则：未读的窗口不能撤销一次匹配。如果这一匹配本身就会拒绝请求（有一个带 `fast_response` 插件的决策仅凭这条规则即匹配，且没有其他决策排在它之前），该窗口会结束请求的全部防护扫描：尚未读取的窗口不再发送，请求立即被拒绝，而不必等到截止时间。
- 一个输入最多读取 256 个窗口。需要更多窗口的输入在开头和结尾之间改用更大的窗口读取，最大到 `scan_window.max_tokens`（默认是模型能接受的最大大小：验证长度为 32,768 个 token 时为 31,744 字节），使 256 个窗口覆盖整个输入。只有超过 256 个最大窗口（约 7.5 MB）的输入才会留下未读窗口，这些窗口计为未读。
- 窗口越大，其中内容被稀释得越厉害：同一段注入在 8,000 token 的窗口里得分低于在 2,000 token 的窗口里。开头和结尾，即输入的前后各 `scan_window.tokens`，始终按这一大小读取。较大窗口的回答若使某条规则的值低于其阈值，且在对数几率上相差不超过 `scan_window.reread_margin`，会在规则作出判断之前按 `scan_window.tokens` 大小的窗口重读。稀释使得分的对数几率在任何阈值附近都下降大致相同的幅度，因此默认值 3 对阈值为 0.995 的规则从约 0.91 起重读，对 0.88 的规则从约 0.27 起，对 0.5 的规则从约 0.05 起；窗口一旦匹配就不再重读。差距在一半边距以内的窗口（被稀释的匹配通常落在这里）会立即重读，排在所有尚未发送的窗口之前，仍可提前结束扫描；差距更大的窗口要等尚未读取的窗口读完后再重读，离阈值越近越先读，因此落后于进度的扫描会先读完各段合并窗口，再做这类重读。输入的窗口已经读过并得到回答的部分不会再读。
- 结尾窗口装着模型在回答前最后读到的内容，注入放在这里最有效；而紧挨输入末尾的注入要与最多一整个窗口的前文共用这个窗口。结尾窗口的回答若使某条规则的值低于其阈值，且相差不超过 `scan_window.reread_margin`，输入最后半个窗口会立即重读：只多一个窗口，匹配时仍可提前结束扫描。
- 能在信号截止时间前读完全部窗口的输入按配置的大小读取，速度与读完这些窗口所需的时间相同。落后于进度的扫描，即按已读窗口的速度在截止时间前读不完剩余窗口的扫描，只有在部署实测的速度表明能按时读完时，才把剩余窗口按连续几段合并读取，取能按时读完的最短一段；否则保持原有窗口，能读多少读多少。决策模型在任何大小的窗口里读取每个 token 的开销大致相同，因此更大的窗口节省的时间很少，很少能按时读完。

```yaml
global:
  model_catalog:
    deployments:
      primary:
        provider: model_runtime
        artifact: vllm-sr/vela3-fast
        routing_view: {max_tokens: 8192, head_share: 0.25}
        scan_window: {tokens: 8192, overlap: 512, max_parallel: 4}
      hosted-decision:
        provider: systemone
        endpoint: https://decisions.example.com/v1/systemone
        model: example-decision-model
        timeout_ms: 6000
        routing_view: {max_tokens: 4096}
        scan_window: {tokens: 4096, overlap: 256}
```

| 设置 | 默认值 | 范围 |
| --- | --- | --- |
| `routing_view.max_tokens` | 8,192；`provider: systemone` 为 4,096 | 256 到 1,048,576 |
| `routing_view.head_share` | 0.25 | 0 到 1 |
| `scan_window.tokens` | 8,192；`provider: systemone` 为 4,096 | 256 到 1,048,576 |
| `scan_window.overlap` | 512；`provider: systemone` 为 256 | 0 到小于半个窗口 |
| `scan_window.max_parallel` | 4 | 1 到 64 |
| `scan_window.max_tokens` | 模型的验证长度减去 1,024 | `scan_window.tokens` 到 1,048,576 |
| `scan_window.reread_margin` | 3（低于规则阈值的对数几率） | 0 到 20（为 0 时不重读任何窗口） |

**视图和窗口按字节计算大小。** `max_tokens` 和 `scan_window.tokens` 中的一个 token 按四个字节计算，大约是一个散文 token 所占的字节数，因此默认的 8,192 对应 32 KB 的视图。模型的验证长度（模型卡的 `validated_input_tokens`，没有时为 `max_input_tokens`）减去留给问题的 1,024 个 token，按每个 token 一个字节限制两者，因为字节级分词器不会让一个字节产生多于一个 token。JSON、base64 这类密集文本每个 token 所占的字节远少于散文，同样大小的视图可能包含约三倍的 token，但任何设置都不会让视图或窗口超出模型长度：验证长度为 32,768 个 token 时，两者都不会超过 31,744 字节。

**按截止时间设定大小。** 调用应在截止时间之前留有余量地完成：远程 API 的截止时间是规则或部署的 `timeout_ms`，本地模型的截止时间是信号的截止时间（对服务中的请求默认 45 秒，`global.model_catalog.signal_timeout_ms`）。如果一个 API 每秒读取 R 个 token，回答最小的调用需要 T 秒，那么不超过 `(timeout - T) × R` 个 token 的视图能按时得到回答：一个每秒读取 2,000 个 token、最小调用需要 2 秒的 API，在 6 秒超时内约能读取 8,000 个 token，对散文相当于 `max_tokens` 为 8,000；如果消息经常是 JSON 或 base64，则约为 2,700。一次扫描的各个窗口共用截止时间：部署每次读取 `scan_window.max_parallel` 个窗口，先读结尾和开头，因此截止时间到达时已读的窗口就是部分扫描据以判断的窗口。未能按时读到的窗口记为未读，`best_effort` 下规则依据已读窗口做出判断。落后于进度的扫描只有在部署实测的速度表明更大的窗口能按时读完时，才改用更大的窗口读取剩余部分；它们节省的只是每次调用在文本之外的固定开销，因此在单个决策模型上，约一百万 token 的输入可能只读到一部分，`scan_partial` 会如实说明。API 较慢时调低 `scan_window.tokens`；需要完整扫描的输入则调高截止时间；同时处理更多窗口只在模型还有余量合批时才有帮助。

**覆盖率与失败。** 如果截止时间在所有窗口读完之前到达，且已读窗口都没有决定该规则，由 `on_unscanned` 决定：`best_effort`（默认）根据已读窗口作出判断，`match` 直接匹配（严格模式），`no_match` 不匹配，`unknown` 交给 `on_unknown`。规则的 `min_coverage` 是 0 到 1 之间的比例，默认关闭，它规定 `best_effort` 至少要读多少窗口：低于该比例时规则视为 `unknown`，因此 `jailbreak` 或 `pii` 规则默认匹配，防护会拒绝读得太少的输入。如果一个窗口都没有读到（模型不可用，或没有窗口按时作答），`jailbreak` 和 `pii` 规则按 `on_unknown` 处理，默认匹配（失败即拦截）；`safety` 规则则为未知，由其决策的 `rules.on_unknown` 决定。

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: prompt_attack
          min_coverage: 0.5
```

每条分窗口读取输入的防护规则都会报告覆盖率：

- 请求阶段的防护规则只读了部分输入、请求仍被路由时，`x-vsr-warnings` 对每个调用方都携带 `scan_partial`；
- `vsr_signal_scan_coverage_total{signal, result}` 按 `full`、`partial` 或 `none` 统计每条防护规则的读取，`vsr_signal_scan_coverage_ratio{signal}` 给出已读窗口的比例，`vsr_signal_scan_decided_total{signal}` 统计在其余窗口读完之前就被某个窗口决定的规则；
- 路由预览（`scan_coverage`）和 Router Replay 记录（`scan_coverage`）按规则给出 `full`、`partial n/m` 或 `none`；
- 对允许查看调试头的调用方，`x-vsr-trace` 以 `scan_coverage` 成员携带该信息。

| 模型 | 每个问题的验证长度 | 最大长度 |
| --- | --- | --- |
| vela3-fast | 32,768 个 token；路由和 span 问题使用其自身 8,192 token 的窗口 | 32,768 |
| vela3-lite、-mini、-flash | 262,144 个 token；每个 span 目标 32,768 个 | 262,144 |
| vela3-turbo、d3-turbo | 32,768 个 token | 长上下文时 262,144 |
| d3-ultra | 131,072 个 token | 长上下文时 262,144 |

:::warning
在核数较少的 CPU 上，对长请求的完整输入扫描可能超过信号截止时间。先测量你的输入长度和并发量，再为部署分配专用 CPU 或 GPU。
:::

## 放置 {#placement}

- **在 CPU 上运行 vela3-fast** 是默认设置。加载后的部署约占 7 GB；给 Router 分配 12 GiB。
- **GPU** 可以运行更大的 Vela 3.0 和 Decision 规格，并在相同延迟下每秒处理多得多的请求。
- **远程 API** 会给每次调用增加一次网络往返，并且其模型没有校准阈值。
- **一个部署承担全部** 只加载一个模型。把各类型分散到多个部署，每个阶段每个部署各发出一次调用。

[选择模型、规模和硬件](../../model-runtime/get-started/choose-a-model.md)给出了各规格的实测延迟，[模型运行时记录](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)包含完整的测量数据。
