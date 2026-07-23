---
translation:
  source_commit: "485c74ba"
  source_file: "docs/tutorials/signal/learned/preference.md"
  outdated: true
---

# Preference 信号

## 概览

`preference` 从示例与分类器设置推断用户响应风格偏好。映射到 `config/signal/preference/`，在 `routing.signals.preferences` 中声明。

该族为学习型：使用 `global.model_catalog.modules.classifier.preference` 下的偏好分类路径。

若省略 `global.model_catalog.modules.classifier.preference.use_contrastive`，vSR 现默认为 `true`。即 `deploy/recipes/balance.yaml` 等配置可依赖偏好信号，除非显式禁用对比模式，否则无需单独全局分类器块。

## 主要优势

- 个性化路由而无需把用户状态硬编码进决策。
- 偏好检测与路由结果分离。
- 支持示例驱动的风格检测，如简洁对详尽。
- 同一偏好策略可被多条决策复用。

## 解决什么问题？

用户常在同一主题下偏好不同回答风格。若仅下游处理，路由无法选择最合适模型或插件栈。

`preference` 将推断的风格偏好暴露为命名路由输入。

## 何时使用

在以下情况使用 `preference`：

- 部分用户偏好简短回答，部分要细节
- 路由行为应适应稳定风格偏好
- 希望偏好检测在多条决策间复用
- 用户风格信号应影响模型选择、插件选择或两者

## 配置

源片段族：`config/signal/preference/`

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

将示例视为偏好检测器的训练锚点，而非字面关键词规则。

```yaml
global:
  model_catalog:
    modules:
      classifier:
        preference:
          use_contrastive: false # optional override; default is true
```
