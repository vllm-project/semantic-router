---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/pii.md"
  outdated: false
---

# PII

## 概览

`pii` 是路由局部插件：决策匹配后应用 PII 筛查或脱敏策略。

对应 `config/plugin/pii/redact.yaml`。

## 主要优势

- PII 策略局部在需要的路由上。
- 允许的 PII 类型显式，而非隐式。
- 复用全局 PII 模型加载与路由局部 enforcement 互补。

## 解决什么问题？

部分路由可允许有限标识符，其他路由应几乎全脱敏。`pii` 显式表达路由局部策略，而不让每条路由都变成最严默认。

## 何时使用

- 一条路由需要路由专用 PII 处理
- PII 应脱敏、拦截或按允许类型收窄
- 路由应复用全局 PII 模型但不将 enforcement 全局化

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: pii
  configuration:
    enabled: true
    threshold: 0.85
    pii_types_allowed:
      - EMAIL_ADDRESS
```
