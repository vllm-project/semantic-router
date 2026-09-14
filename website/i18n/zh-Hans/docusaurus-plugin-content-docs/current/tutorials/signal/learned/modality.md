---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
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

保持规则名与决策要引用的路由行为对齐。通过 `global.model_catalog.modules.modality_detector` 配置检测器。

## 依赖与限制 {#dependencies-and-limitations}

模态检测器对预期输出模式分类；它不证明后端支持请求的输入附件。请保持模型卡能力与提供方校验对齐。完整示例见：
[`config/fragments/signal/modality/multimodal.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/modality/multimodal.yaml)。
