---
translation:
  source_commit: "6a387d587e2635de36c7ed5e4c2d513a3ec525a1"
  source_file: "docs/tutorials/signal/learned/modality.md"
  outdated: false
---

# 模态信号 {#modality-signal}

## 概览 {#overview}

`modality` 检测请求应留在文本生成、切到图像生成，还是同时支持两者。在 `routing.signals.modality` 下定义其输出标签。

已配置的 `modality_detector` 对预期输出模式分类；命名结果随后可供普通决策使用。

## 主要优势 {#key-advantages}

- 将图像生成路由与纯文本路由分开。
- 使多模态流量在 `routing.decisions` 中可见。
- 避免把模态检查混进每条决策规则。
- 可从简单请求形态路由扩展到检测器支持的多模态分类。

## 解决什么问题？ {#what-problem-does-it-solve}

文本聊天、图像生成与混合工作流常常共享同一入口，但不应共享同一模型路径。没有 modality 信号时，路由逻辑会变得脆弱且重复。

`modality` 将输出模式暴露为命名路由输入。

## 何时使用 {#when-to-use}

在以下情况使用 `modality`：

- Router 同时服务自回归与扩散风格后端
- 部分路由应只接受图像生成提示
- 多模态处理必须在路由图中保持显式
- 希望使用稳定信号名，例如 `AR`、`DIFFUSION` 或 `BOTH`

## 配置 {#configuration}

```yaml
routing:
  signals:
    modality:
      - name: AR
        description: Text-only autoregressive requests.
      - name: DIFFUSION
        description: Requests that should route into image generation flows.
      - name: BOTH
        description: Requests that need both text and image generation behavior.
```

保持规则名与决策要引用的路由行为对齐。

检测器默认关闭，因此这些规则只有在配置了 `modality_detector` 块时才会匹配。使用 `method: classifier` 且不设置 `classifier.model_path` 时，它运行 Vela 2.0 0.3B，其校准后的 `confidence_threshold` 为 0.51：

```yaml
global:
  model_catalog:
    modules:
      modality_detector:
        enabled: true
        method: classifier
        confidence_threshold: 0.51
```

没有这个块时，Router 加载配置时会记录 `modality_detector_disabled` 警告，`vllm-sr config validate` 也会通过 Router 自身的校验打印同一警告。`method: keyword` 需要 `keywords`；`method: hybrid` 还需要 `lower_threshold_ratio`，以及分类器或关键词。

基于 `BOTH` 的决策必须同时引用一个 AR 模型（在 `routing.modelCards` 中设 `modality: ar`）和一个扩散模型（`modality: diffusion`），或者引用一个全模态模型（`modality: omni`）。

## 依赖与限制 {#dependencies-and-limitations}

未声明配方模型绑定时，`classifier.max_sequence_length: 0` 保留 512-token 默认值。长提示会选取开头、中间和结尾的代表性片段用于路由，模型运行时会截断仍超出预算的输入。这项策略限制推理成本，并不对每个 token 分类。

要让模态分类接收未经路由采样的输入，请将 `modality_detector` 绑定到命名部署，并将 `input.max_tokens` 设为大于 512。设置 `input.overflow: reject` 可拒绝超出预算的输入。预算应在所选模型产物和提供方已验证的范围内；模型标称支持 32K，并不表示每种提供方都已验证 32K 执行。模块的 `classifier.max_sequence_length` 设为大于 512 的正值也会跳过路由采样，但仍保留模块的截断策略。

`method: classifier` 下，实际推理错误会使模态保持未知，由决策的 `on_unknown` 策略处理，不会产生 `AR` 匹配。`method: hybrid` 则显式允许关键词回退。

模态检测器对预期输出模式分类；它不证明后端支持请求的输入附件。请保持模型卡能力与提供方校验对齐。完整示例见：
[`config/fragments/signal/modality/multimodal.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/modality/multimodal.yaml)。
