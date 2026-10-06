---
translation:
  source_commit: "e17d7b480284f6f1685597a696665d0394a83722"
  source_file: "docs/tutorials/plugin/hallucination.md"
  outdated: true
---

# 幻觉检测

## 概览

`hallucination` 是一个路由局部插件，用于在决策已经匹配之后进行事实核查和响应质量筛查。

## 主要优势

- 在不更改全局默认的情况下添加路由局部幻觉检查。
- 在事实置信度低时明确声明响应动作。
- 适用于检索密集或有依据回答的路由。

## 解决什么问题？

有些路由在模型回答后需要额外审查，尤其是它们承诺事实精度时。`hallucination` 让这些路由添加响应时校验，而不强迫每条路由都付出该成本。

## 何时使用

- 某条路由应事实核查或标注响应
- 有依据或由工具支撑的路由需要额外响应筛查
- 该路由应警告或标注，而不是静默放行低置信度回答

## 配置

在 `routing.decisions[].plugins` 下添加该插件：

```yaml
plugins:
  - type: hallucination
    configuration:
      enabled: true
      hallucination_action: header
      unverified_factual_action: header
      include_hallucination_details: true
```

`header` 保留模型响应并添加警告元数据。`body` 向响应正文添加警告，`none` 记录结果但不更改响应。

当在 `routing.signals.hallucination` 下声明了[幻觉信号](../signal/learned/hallucination)时，检测作为响应阶段信号运行，该插件只对其执行：规则匹配时应用 `hallucination_action`，答案没有可对照的依据上下文时应用 `unverified_factual_action`，规则不可用或不适用时不做任何事。没有规则时，插件自己分类答案，这是兼容路径，并会在加载时如此报告。

该插件依赖 `global.model_catalog.modules.hallucination_mitigation`，其检测器（默认 Vela Halu）运行在[模型运行时](model-runtime/guides/hallucination.md)中。模型响应和提供的依据上下文由该模块处理。NLI 解释（`use_nli`）已退役；`vllm-sr config migrate` 会移除该设置。长于检测器 token 窗口的依据上下文会从末尾裁剪，以便答案始终到达模型。检测可以识别无依据文本，但没有权威证据时无法确立真实性。

完整示例见：
[`config/fragments/plugin/hallucination/fact-check.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/hallucination/fact-check.yaml)。
