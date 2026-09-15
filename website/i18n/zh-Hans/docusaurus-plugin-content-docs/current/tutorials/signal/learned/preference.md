---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/signal/learned/preference.md"
  outdated: false
---

# 偏好信号 {#preference-signal}

## 概览 {#overview}

`preference` 从示例与分类器设置推断响应风格偏好。在 `routing.signals.preferences` 下定义偏好规则。

该族为学习型：使用 `global.model_catalog.modules.classifier.preference` 下的偏好分类路径。

`global.model_catalog.modules.classifier.preference.use_contrastive` 默认值为 `true`。仅在你有意使用替代分类路径时才设为 `false`。

## 主要优势 {#key-advantages}

- 个性化路由，而不把用户状态硬编码进决策。
- 偏好检测与路由结果分离。
- 支持示例驱动的风格检测，例如简短 vs 详尽回答。
- 同一偏好策略可被多条决策复用。

## 解决什么问题？ {#what-problem-does-it-solve}

用户即使问同一主题，也常常想要不同响应风格。若这些偏好只在下游处理，路由就无法选择最合适的模型或插件栈。

`preference` 将推断出的风格偏好暴露为命名路由输入。

## 何时使用 {#when-to-use}

在以下情况使用 `preference`：

- 部分用户偏好简短回答，另一些想要高细节
- 路由行为应适应稳定的风格偏好
- 希望偏好检测在多条决策间保持可复用
- 用户风格信号应影响模型选择、插件选择，或两者

## 配置 {#configuration}

```yaml
routing:
  signals:
    preferences:
      - name: terse_answers
        description: Users who prefer short, direct responses.
        examples:
          - keep it concise
          - bullet points only
          - answer in one paragraph
        threshold: 0.7
```

把示例当作偏好检测器的训练锚点，而不是字面关键词规则。

```yaml
global:
  model_catalog:
    modules:
      classifier:
        preference:
          use_contrastive: false # 可选覆盖；默认值为 true
          prototype_scoring:
            enabled: true
            cluster_similarity_threshold: 0.9
            max_prototypes: 8
            best_weight: 0.75
            top_m: 2
            margin_threshold: 0.05
```

对比模式下，Router 嵌入每条偏好规则的描述与示例，在启用 `prototype_scoring` 时将它们压缩成代表性原型，再把传入请求与这些原型比较。`margin_threshold` 让你可以拒绝模糊胜者，而不是强迫弱偏好匹配。

## 依赖与限制 {#dependencies-and-limitations}

偏好规则使用共享嵌入/分类器路径，仅从可用请求上下文推断风格。不应把它们当作持久用户同意或身份。完整示例见：
[`config/fragments/signal/preference/power-user.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/preference/power-user.yaml)。
