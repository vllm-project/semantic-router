---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/signal/learned/pii.md"
  outdated: false
---

# PII 信号

## 概览

`pii` 检测请求中的敏感个人数据。映射到 `config/signal/pii/`，在 `routing.signals.pii` 中声明。

该族为学习型：使用通过 `global.model_catalog.system.pii_classifier` 配置的 PII 检测路径。

## 主要优势

- 隐私敏感路由显式化。
- 决策可在流量到达后端前拦截、降级或隔离风险流量。
- 支持低风险标识类型的允许列表。
- 隐私策略可在路由与插件间复用。

## 解决什么问题？

没有专用 PII 信号时，隐私敏感流量可能在检测前到达错误模型或插件栈。临时过滤器也使策略难审计。

`pii` 将个人数据检测变成可复用路由输入。

## 何时使用

在以下情况使用 `pii`：

- 提示可能含受监管或敏感个人数据
- 部分 PII 类型可接受，其他必须触发更安全路由
- 隐私敏感流量需要不同插件或后端
- 路由策略依赖早期 PII 检测

## 配置

源片段族：`config/signal/pii/`

```yaml
routing:
  signals:
    pii:
      - name: restricted_pii
        threshold: 0.85
        include_history: true
        pii_types_allowed:
          - EMAIL_ADDRESS
        description: Sensitive prompts where only low-risk identifiers may pass through.
```

`pii_types_allowed` 为空时，任意检测到的 PII 都可能使信号匹配。
