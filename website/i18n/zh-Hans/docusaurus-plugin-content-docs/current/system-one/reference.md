---
title: System One 参考
sidebar_label: 参考
description: 预设目录、支持矩阵、校准阈值、指定 System One 的配置字段，以及加载时的代码。
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/system-one/reference.md"
  outdated: false
---

# System One 参考 {#system-one-reference}

## 预设目录 {#preset-catalog}

每个应用信号、评审和检查都会提出以下问题之一。问题措辞和选项是规范的零样本定义；有训练绑定的家族会收到训练形式的问题。

| 预设 | 类型 | 选项 | 阶段与输入 | 读取 | 提问方 |
| --- | --- | --- | --- | --- | --- |
| `domain` | choice | 14 个学科 | request | 当前轮次，截断 | `domains` |
| `fact_check` | choice | `FACT_CHECK_NEEDED`、`NO_FACT_CHECK_NEEDED` | request | 当前轮次，截断 | `fact_check` |
| `feedback` | choice | `SAT`、`NEED_CLARIFICATION`、`WRONG_ANSWER`、`WANT_DIFFERENT`、`NO_FEEDBACK` | request | 当前轮次，截断，以及它所回复的答案 | `user_feedbacks` |
| `jailbreak` | choice | `benign`、`jailbreak` | request 或 response | 完整输入；响应阶段读取答案 | `jailbreak` |
| `harm` | choice | `safe`、`unsafe` | request 或 response | 完整输入；响应阶段读取答案 | 不带 `labels` 的 `safety` |
| `harm_categories` | set | 34 个[危害类别](#harm-categories) | request 或 response | 完整输入 | 带 `labels` 的 `safety` |
| `pii` | span | 11 个[个人数据标签](#personal-data-labels) | request | 完整输入，分窗口 | `pii`、`masking` |
| `hallucination` | span | `unsupported` | response | 答案及其依据上下文 | `hallucination` |
| `refusal` | choice | `compliance`、`refusal` | response | 请求和答案 | `refusal`、`escalate` 的 `accept.refusal` |
| `turn_relation` | choice | `continue`、`reask`、`shift`、`return` | request | 当前轮次和最近 4 个轮次 | `turn_relation` |
| `complexity` | choice | `easy`、`medium`、`hard` | request | 当前轮次，截断 | `complexity` |
| `modality` | choice | `text`、`image`、`text_image`、`image_edit`、`speech`、`audio`、`video`、`video_edit` | request | 当前轮次，截断 | `modality` |
| `relevance` | noul | | request | 请求和一个候选 | `context` 的源和压缩 |
| `same_request` | noul | | request | 请求和一条缓存请求 | `response_cache` |
| `tool_rank` | choice | 候选工具 | request | 请求 | `tools` |
| `tool_need` | noul | | request | 请求 | `tools` |
| `answer_check` | noul | | response | 完整的请求和答案 | `escalate` 的 `accept.systemone` |

学科为 `biology`、`business`、`chemistry`、`computer science`、`economics`、`engineering`、`health`、`history`、`law`、`math`、`other`、`philosophy`、`physics` 和 `psychology`。

### 危害类别 {#harm-categories}

`safety` 规则的 `labels` 指定以下 34 个类别中的若干个，它们是 vela3 训练过的危害问题的标签：

- **AEGIS 2.0（21 个）：** Controlled/Regulated Substances、Copyright/Trademark/Plagiarism、Criminal Planning/Confessions、Fraud/Deception、Guns and Illegal Weapons、Harassment、Hate/Identity Hate、High Risk Gov Decision Making、Illegal Activity、Immoral/Unethical、Needs Caution、Other、PII/Privacy、Political/Misinformation/Conspiracy、Profanity、Sexual、Sexual (minor)、Suicide and Self Harm、Threat、Unauthorized Advice、Violence。
- **MLCommons 危害代码（13 个）：** S1（暴力犯罪）、S2（非暴力犯罪）、S3（性相关犯罪）、S4（儿童性剥削）、S5（诽谤）、S6（专业建议）、S7（隐私）、S8（知识产权）、S9（无差别武器）、S10（仇恨）、S11（自杀与自残）、S12（性内容）和 S14（代码解释器滥用）。

vela3-fast 用训练过的 34 标签 set 问题一次性回答。在 request 阶段，Vela 3.0 的 d3 规格对每个类别各问一个 `hazard:<label>` 是/否问题，采用 vela3-lite 危害适配器训练时的形式；vela3-mini、-flash 和 -turbo 没有危害适配器，零样本回答这些问题。其他模型在同一次调用中零样本回答同样的标签。

### 个人数据标签 {#personal-data-labels}

`PERSON`、`EMAIL_ADDRESS`、`PHONE_NUMBER`、`US_SSN`、`CREDIT_CARD`、`LOCATION`、`IP_ADDRESS`、`DATE_TIME`、`US_DRIVER_LICENSE`、`US_PASSPORT` 和 `IBAN_CODE`。`pii` 规则的 `ignore_labels` 和 masking 插件的 `entity_types` 指定其中若干个。

## 支持矩阵 {#support-matrix}

**TC** 训练且校准，**T** 训练，**Z** 零样本，**—** 无法运行。

| 预设 | vela3-fast | vela3-lite | vela3-mini、-flash、-turbo | d3（本地或托管） | Decision 2.0 | Vela 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| domain、fact_check、feedback、jailbreak | TC | T | T | Z | Z | TC |
| harm | TC | T | T | Z | Z | TC |
| 危害类别 | T | request 阶段为 T，response 阶段为 Z | Z | Z | Z | Z |
| pii | TC | T | T | — | — | TC |
| hallucination | T | T | T | — | — | T |
| refusal | T | T | T | Z | Z | Z |
| modality | `text`、`image`、`text_image` 为 TC，其余为 Z | 这三个为 T，其余为 Z | 这三个为 T，其余为 Z | Z | Z | Z |
| turn_relation、complexity、relevance、same_request、tool_rank、tool_need、answer_check | Z | Z | Z | Z | Z | Z |

- Vela 3.0 的 d3 规格按问题 ID 开启训练过的适配器：`domain`、`factcheck`、`feedback`、`attack`、`p_harm` 和 `r_harm`、`r_refusal`、`modality`，以及仅在 vela3-lite 上开启的危害类别 `hazard:` 前缀；vela3-fast 和 Vela 2.0 按措辞匹配。
- d3 家族和 Vela 3.0 的 d3 规格能读取图片输入。目前没有决策模型能为路由规则读取音频或视频。

## 校准阈值 {#calibrated-thresholds}

未写 `predicate` 的规则在 router signal suite 测量过的模型上使用的默认值。其他模型使用 0.5，并给出 `uncalibrated_preset` 警告。

| 模型 | jailbreak | domain | pii | fact_check | feedback | harm | modality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vllm-sr/vela3-fast` | 0.88 | 0.41 | 0.18 | 0.9999 | 0.47 | 0.52 | 0.63 |
| `vllm-sr/Vela-2.0-0.3B` | 0.75 | 0.28 | 0.01 | 0.93 | 0.37 | 0.46 | 0.51 |
| `vllm-sr/Vela-2.0-0.8B` | 0.71 | 0.38 | 0.07 | 0.994 | 0.34 | | |
| `vllm-sr/Vela-2.0-4B` | 0.63 | 0.45 | 0.05 | 0.9984 | 0.33 | | |
| `vllm-sr/Vela-2.0-9B` | 0.42 | 0.46 | 0.14 | 0.998 | 0.35 | | |

`hallucination` 预设的 span 保留模型自己的阈值。这些值背后的记录见[模型运行时记录](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)。

## 配置字段 {#configuration-fields}

### 部署从何而来 {#where-the-deployment-comes-from}

| 字段 | 含义 |
| --- | --- |
| `global.model_catalog.system.decision_model` | `{deployment: <name>}`：回答所有未被其他设置绑定的问题。默认 `{deployment: primary}` |
| `global.model_catalog.bindings.<type>` | 某个应用信号类型规则的部署 |
| `recipes[].model_bindings.<type>` | 同上，只针对一个配方 |
| `global.model_catalog.deployments.<name>` | 一个部署：`provider: model_runtime`（本地或挂接）或 `provider: systemone`（远程 API） |
| `global.model_catalog.signal_timeout_ms` | 一个请求所有信号的截止时间 |

`systemone` 部署接受 `endpoint`、`model`、`auth`（`api_key_env` 或 `api_key_file`，或 `credential: delegated`；`header`、`prefix`）、`headers`、`timeout_ms`、`max_retries`、`tls` 和 `capabilities`（`question_types`、`presets`、`modalities`）。

### 信号 {#signals}

`recipes[].signals.systemone` 下的 `systemone` 规则：

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供条件和投影使用 |
| `question` | `type`、`instructions`，以及 `choices`、`levels` 或 `labels`；set 和 span 还有 `threshold` 和 `head` |
| `labels` | 让规则匹配的选项或标签 |
| `ignore_labels` | set 和 span：永不计入的标签 |
| `predicate` | 答案必须满足的界限：`gt`、`gte`、`lt`、`lte` |
| `deployment` | 回答问题的部署 |
| `stage` | `request`（默认）或 `response` |
| `input` | `history`、`previous_turns`（0 到 32）、`tool_results`、`answer`、`context`、`image`、`audio`、`video` |
| `coverage` | `truncate`（默认）或 `full` |
| `on_unscanned` | 使用 `coverage: full` 时：`match`、`no_match` 或 `unknown`（默认） |
| `timeout_ms` | 最长 60,000 |

应用规则接受 `name`、`description`、`deployment`、`timeout_ms`，以及其类型允许的字段：

| 类型 | 其他字段 |
| --- | --- |
| `domains` | `labels`（默认 `[name]`）、`predicate` |
| `fact_check`、`user_feedbacks` | `labels`、`predicate` |
| `jailbreak` | `predicate`、`stage`、`input.history`、`on_unknown` |
| `safety` | `labels`（危害类别）、`predicate`、`stage` |
| `pii` | `predicate`、`ignore_labels`、`input.history`、`input.tool_results`、`on_unknown` |
| `hallucination`、`refusal` | `predicate` |
| `turn_relation` | `labels`、`input.previous_turns`（默认 4） |
| `complexity` | `labels` |
| `modality` | `labels`、`predicate` |

其他字段在加载时会被拒绝。

### `match` 块 {#the-match-block}

```text
match:
  recall: {method: embedding, top_k: 20, min_similarity: 0.3}    # optional; method: embedding | bm25
  judge:  {preset: relevance, top_k: 6, min_score: 0.5, max_candidates: 64}   # optional
```

每个功能在它查找相似项的地方写这个块：`response_cache.semantic.match`、`context.sources[].match`、`context.compress.match` 和 `tools.select.match`。

| 字段 | 含义 |
| --- | --- |
| `recall.method` | `embedding`（默认；使用存储的嵌入部署）或 `bm25`（词法索引） |
| `recall.top_k`、`recall.min_similarity` | 召回返回的候选数（候选来自索引时必填），以及最低相似度：`embedding` 为余弦值，`bm25` 为 BM25 分数 |
| `judge.preset` | `relevance`、`same_request` 或 `tool_rank` |
| `judge.question` | 代替预设的自定义问题：`choice`（候选即其选项）、`noul` 或 `score` |
| `judge.deployment` | 回答问题的部署；不写时为决策模型 |
| `judge.top_k`、`judge.min_score`、`judge.min_ratio` | 保留哪些被评判的候选：最好的 `top_k` 个、至少 `min_score` 的、至少达到最佳者 `min_ratio` 的 |
| `judge.max_candidates` | 评审最多读取的候选数：choice 评审为 255（也是上限），逐个候选的评审默认 64；小于 1，或 choice 评审上大于 255，会导致加载失败 |
| `judge.on_error` | `skip`（默认；保留召回结果，没有召回时保留原样的候选）或 `fail`（该功能的匹配失败） |
| `judge.timeout_ms` | 评审调用的截止时间；未设置时使用部署的 `timeout_ms`，再没有则为 1000 ms |

块中至少要有一个阶段（否则为 `match_empty`）。

### 算法 {#algorithms}

| 字段 | 含义 |
| --- | --- |
| `algorithm.systemone` | 选择器：`instructions`、`candidates`（模型到描述的映射）、`deployment`、`timeout_ms` |
| `algorithm.escalate.accept.systemone` | `{gte, deployment, timeout_ms}`：接受 `answer_check` 概率至少为 `gte` 的答案 |
| `algorithm.escalate.accept.refusal` | `false`：接受 `refusal` 问题未标记为拒答的答案 |
| `algorithm.escalate.stages` | System One 入口：依次提问的决策模型，每个都有原生接受规则 |

## 加载时代码 {#load-time-codes}

| 代码 | 类别 | 含义 |
| --- | --- | --- |
| `span_required` | 错误 | 需要 span 的规则或插件解析到不支持 span 的部署 |
| `input_modality_unsupported` | 错误 | 规则读取其部署无法读取的媒体 |
| `match_empty` | 错误 | `match` 块既没有 `recall` 也没有 `judge` |
| `uncalibrated_preset` | 警告 | 某个预设使用中性阈值 0.5 回答 |
| `renamed_signal_type` | 警告 | `signals.decision` 或 `type: decision` 条件，按 `systemone` 加载 |
| `renamed_algorithm_type` | 警告 | `algorithm.type: decision`，按 `systemone` 加载 |
| `removed_signal_type`、`removed_field` 等 | 错误 | 已移除的形式；见[迁移指南](../installation/migrate-to-v0-5) |
