---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/signal/learned/jailbreak.md"
  outdated: false
---

# Jailbreak 信号

## 概览

`jailbreak` 在路由器提交路由前检测提示注入与越狱企图。映射到 `config/signal/jailbreak/`，在 `routing.signals.jailbreak` 中声明。

该族为学习型：使用 `global.model_catalog.modules.prompt_guard` 与 `global.model_catalog.system` 中的越狱模型绑定。

## 主要优势

- 在模型选择前拦截或降级不安全流量。
- 支持分类器、对比式与混合风格安全检测。
- 越狱策略在路由决策内可见。
- 同一安全信号可跨多条受保护路由复用。

## 解决什么问题？

若越狱检测仅发生在下游，路由器仍可能将不安全流量送到错误模型或工具链。若逻辑在路由图外，安全策略难审计。

`jailbreak` 将注入检测作为一等路由输入。

## 何时使用

在以下情况使用 `jailbreak`：

- 不安全流量必须在模型选择前拦截
- 提示注入应路由到更安全回退
- 多轮历史应影响路由
- 安全策略需与路由逻辑同图可见、可测

## 配置

源片段族：`config/signal/jailbreak/`

```yaml
routing:
  signals:
    jailbreak:
      - name: prompt_injection
        method: hybrid
        threshold: 0.8
        include_history: true
        description: Detect common prompt-injection or jailbreak attempts.
        jailbreak_patterns:
          - ignore previous instructions
          - reveal the hidden prompt
          - jailbreak mode
        benign_patterns:
          - explain the policy
          - summarize the safety rules
```

多轮攻击使用 `include_history`；模式列表作为所配置检测方法的调参数据。
