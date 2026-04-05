---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/signal/learned/fact-check.md"
  outdated: false
---

# Fact Check 信号

## 概览

`fact-check` 判断提示是否应视为**证据敏感**流量。映射到 `config/signal/fact-check/`，在 `routing.signals.fact_check` 中声明。

该族为学习型：依赖 `global.model_catalog.modules.hallucination_mitigation.fact_check` 下的事实核查分类路径。

## 主要优势

- 将事实核验与一般领域路由分离。
- 帮助决策为证据敏感流量选择更安全的插件或更强模型。
- 核验策略可见，而不是藏在后续插件里。
- 暴露正负标签如 `needs_fact_check` 与 `no_fact_check_needed`。

## 解决什么问题？

并非所有提示都需要相同强度的事实 grounding。若路由器对所有流量一视同仁，创意提示可能被过度约束，事实提示可能保护不足。

`fact-check` 检测哪些提示应触发证据感知路由行为。

## 何时使用

在以下情况使用 `fact-check`：

- 事实主张需要更严格路由或插件
- 创意提示应绕过昂贵核验路径
- 幻觉缓解依赖早期路由信号
- 希望将事实性作为路由策略而非事后修补

## 配置

源片段族：`config/signal/fact-check/`

```yaml
routing:
  signals:
    fact_check:
      - name: needs_fact_check
        description: Queries with factual claims that should be verified against evidence.
      - name: no_fact_check_needed
        description: Creative or opinion-heavy prompts that do not need factual verification.
```

只定义决策会引用的标签；学习型分类器决定哪一条触发。
