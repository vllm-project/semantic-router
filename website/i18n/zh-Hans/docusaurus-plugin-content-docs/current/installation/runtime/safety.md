---
title: 安全模型
description: 配置提示词防护、PII 和幻觉检查，并选择失败时的处理方式。
translation:
  source_commit: "dc7f402642a8b8ecec8218e2086a4c6f186ea406"
  source_file: "docs/installation/runtime/safety.md"
  outdated: false
---

安全模型检测风险；路由决策和插件决定采取什么行动。仅启用模型不会自动阻止请求或执行脱敏。

| 检查 | 检测内容 | 配置处理方式 |
| --- | --- | --- |
| 提示词防护 | 提示词注入和越狱 | [越狱信号](/zh-Hans/docs/tutorials/signal/learned/jailbreak) |
| PII | 文本中的个人信息 | [PII 信号](/zh-Hans/docs/tutorials/signal/learned/pii) |
| 幻觉 | 回答中缺乏所提供上下文支持的说法 | [幻觉检测插件](/zh-Hans/docs/tutorials/plugin/hallucination) |
| 事实核查 | 请求是否需要事实验证 | [事实核查信号](/zh-Hans/docs/tutorials/signal/learned/fact-check) |

## 提示词防护 {#prompt-guard}

将以下片段合入配置，即可启用维护的本地防护模型：

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        enabled: true
        variant: mmbert32k
        threshold: 0.7
        on_error: block
```

然后添加越狱信号及处理匹配结果的决策。如需使用独立部署的模型，参照[外部服务](external.md)。配方中的 `prompt_guard` binding 选择模型；阈值和路由策略仍在原有位置配置。

## PII {#pii}

使用完整的 token 分类模型及匹配的 PII 标签映射。在配方中通过 `pii_classifier` binding 选择模型，并设置 `contract: token_spans.v1`。按[进程内模型](in-process.md)设置 deployment 和 adapter，再提供模型的 `mapping_path`。[PII 指南](/zh-Hans/docs/tutorials/signal/learned/pii)介绍实体阈值和脱敏配置。

外部 PII 服务必须返回带分数的实体和有效的 Unicode 文本位置。无效响应属于错误，不能视为成功且未发现实体的扫描。

## 幻觉检测 {#hallucination-detection}

本地检测器根据上下文和问题检查回答。可选的 NLI 解释器检查前提是否支持假设。使用以下配置启用维护的模型：

```yaml
global:
  model_catalog:
    modules:
      hallucination_mitigation:
        enabled: true
        detector:
          backend: candle
          model_ref: hallucination_detector
          threshold: 0.82
        explainer:
          model_ref: hallucination_explainer
          threshold: 0.9
```

这些本地模型使用 Candle。远程聊天服务可以替换检测器；NLI 仍需要受支持的本地解释器。如何提供上下文及处理检测出的文本片段，见[幻觉检测指南](/zh-Hans/docs/tutorials/plugin/hallucination)。

## 处理失败和缺失分数 {#handle-failures-and-missing-scores}

模型错误会产生未知结果。决策中的 `rules.on_unknown` 选择 `no_match`、`match` 或 `fail_request`。未设置时，提示词防护使用 `on_error`：`allow` 表示不匹配，`block` 表示按策略匹配。最终行动仍由使用该结果的决策决定。

聊天模型的判定或策略默认结果可能没有置信度。诊断会显示 `confidence: null` 和 `confidence_available: false`；这与模型分数为零不同。配置策略时，同时测试模型匹配和服务失败的情况。

自定义安全模型的准备方法见[训练与导出指南](/zh-Hans/docs/training/mmbert-safety-classifier)。

访问控制和速率限制的配置见[安全加固](/zh-Hans/docs/installation/security-hardening)。
