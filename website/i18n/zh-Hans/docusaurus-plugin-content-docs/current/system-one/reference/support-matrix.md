---
title: 支持矩阵与阈值
sidebar_label: 支持矩阵
sidebar_position: 2
description: 各模型家族对每个内置 System One 问题是训练回答、零样本回答还是无法回答，以及规则默认采用的校准阈值。
is_mtpe: false
translation:
  source_commit: "bc5c6bfb747ffe9c5f35d98b203427dada8c4c56"
  source_file: "docs/system-one/reference/support-matrix.md"
  outdated: false
---

# 支持矩阵与阈值 {#support-matrix-and-thresholds}

每个模型都以规范的零样本形式回答它能回答的每个预设。有些家族针对某个预设训练过，有些对它有校准阈值。本页同时列出两者，帮助你判断更换决策模型后会发生什么变化。

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

在 router signal suite 上测量过的模型上，未写 `predicate` 的规则使用的默认值：

| 模型 | jailbreak | domain | pii | fact_check | feedback | harm | modality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vllm-sr/vela3-fast` | 0.88 | 0.41 | 0.18 | 0.9999 | 0.47 | 0.52 | 0.63 |
| `vllm-sr/Vela-2.0-0.3B` | 0.75 | 0.28 | 0.01 | 0.93 | 0.37 | 0.46 | 0.51 |
| `vllm-sr/Vela-2.0-0.8B` | 0.71 | 0.38 | 0.07 | 0.994 | 0.34 | | |
| `vllm-sr/Vela-2.0-4B` | 0.63 | 0.45 | 0.05 | 0.9984 | 0.33 | | |
| `vllm-sr/Vela-2.0-9B` | 0.42 | 0.46 | 0.14 | 0.998 | 0.35 | | |

:::note
其他模型使用 0.5，并在加载时给出 `uncalibrated_preset` 警告。`hallucination` 预设的 span 保留模型自己的阈值。这些值背后的记录见[模型运行时记录](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records)；[支持的模型](../../model-runtime/supported-models/overview.md)下的每个模型页面也列出了自己的阈值。
:::
