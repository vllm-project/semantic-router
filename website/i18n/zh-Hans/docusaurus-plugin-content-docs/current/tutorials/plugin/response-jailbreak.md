---
translation:
  source_commit: "7c874be2"
  source_file: "docs/tutorials/plugin/response-jailbreak.md"
  outdated: false
---

# 响应越狱检测（Response Jailbreak）

## 概览

`response_jailbreak` 是一个路由局部插件，用于在模型响应返回前对其进行筛查。

## 主要优势

- 为敏感路由增加最终的响应侧越狱检查。
- 在配置中明确声明处置策略。
- 补充请求侧安全机制，而不是取代它。

## 解决什么问题？

即使请求被正确路由，生成的回答仍可能需要最终安全关卡。`response_jailbreak` 为对应路由提供明确的输出筛查步骤。

## 何时使用

- 路由需要最终的响应侧越狱筛查
- 输出应在返回前被拦截，或通过响应 header 标记
- 仅靠请求侧筛查不足以满足该工作负载的要求

## 配置

在 `routing.decisions[].plugins` 下添加该插件：

```yaml
plugins:
  - type: response_jailbreak
    configuration:
      enabled: true
      threshold: 0.85
      action: block
```

该插件使用已配置的 prompt-guard 运行时处理生成的响应文本。它会增加延迟，也可能产生误报，因此应校准阈值，并根据策略选择 `block` 或仅通过 header 处理。完整示例见：
[`config/fragments/plugin/response-jailbreak/strict.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-jailbreak/strict.yaml)。
