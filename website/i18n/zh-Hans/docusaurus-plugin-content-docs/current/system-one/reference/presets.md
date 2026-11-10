---
title: System One 预设
sidebar_label: 预设
sidebar_position: 1
description: Router 提出的每个内置 System One 问题，及其类型、选项、阶段和读取的输入。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/system-one/reference/presets.md"
  outdated: false
---

# System One 预设 {#system-one-presets}

预设是内置的 System One 问题：应用信号、评审和检查各自提出其中一个。下面的措辞和选项是规范的零样本定义；有训练绑定的家族会收到训练形式的问题（参见[支持矩阵](./support-matrix.md)）。

## 预设目录 {#preset-catalog}

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

## 危害类别 {#harm-categories}

`safety` 规则的 `labels` 指定以下 34 个类别中的若干个，它们是 vela3 训练过的危害问题的标签：

- **AEGIS 2.0（21 个）：** Controlled/Regulated Substances、Copyright/Trademark/Plagiarism、Criminal Planning/Confessions、Fraud/Deception、Guns and Illegal Weapons、Harassment、Hate/Identity Hate、High Risk Gov Decision Making、Illegal Activity、Immoral/Unethical、Needs Caution、Other、PII/Privacy、Political/Misinformation/Conspiracy、Profanity、Sexual、Sexual (minor)、Suicide and Self Harm、Threat、Unauthorized Advice、Violence。
- **MLCommons 危害代码（13 个）：** S1（暴力犯罪）、S2（非暴力犯罪）、S3（性相关犯罪）、S4（儿童性剥削）、S5（诽谤）、S6（专业建议）、S7（隐私）、S8（知识产权）、S9（无差别武器）、S10（仇恨）、S11（自杀与自残）、S12（性内容）和 S14（代码解释器滥用）。

各家族被问到这些类别的方式：

- vela3-fast 用训练过的 34 标签 set 问题一次性回答。
- 在 request 阶段，Vela 3.0 的 d3 规格对每个类别各问一个 `hazard:<label>` 是/否问题，采用 vela3-lite 危害适配器训练时的形式；vela3-mini、-flash 和 -turbo 没有危害适配器，零样本回答这些问题。
- 其他模型在同一次调用中零样本回答同样的标签。

## 个人数据标签 {#personal-data-labels}

`PERSON`、`EMAIL_ADDRESS`、`PHONE_NUMBER`、`US_SSN`、`CREDIT_CARD`、`LOCATION`、`IP_ADDRESS`、`DATE_TIME`、`US_DRIVER_LICENSE`、`US_PASSPORT` 和 `IBAN_CODE`。`pii` 规则的 `ignore_labels` 和 masking 插件的 `entity_types` 指定其中若干个。
