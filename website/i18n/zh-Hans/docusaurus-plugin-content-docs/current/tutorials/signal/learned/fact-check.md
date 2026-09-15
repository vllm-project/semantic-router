---
translation:
  source_commit: "e754075d1f11f729e985bf05b71114f5b807878f"
  source_file: "docs/tutorials/signal/learned/fact-check.md"
  outdated: false
---

# 事实核查信号 {#fact-check-signal}

## 概览 {#overview}

`fact-check` 判断提示是否应视为证据敏感流量。在 `routing.signals.fact_check` 下定义其标签。

该族为学习型：依赖 `global.model_catalog.modules.hallucination_mitigation.fact_check` 下的事实核查分类路径。

## 主要优势 {#key-advantages}

- 将事实核验与一般领域路由分离。
- 帮助决策为证据敏感流量选择更安全的插件或更强模型。
- 核验策略可见，而不是藏在后续插件里。
- 暴露正负标签，例如 `needs_fact_check` 与 `no_fact_check_needed`。

## 解决什么问题？ {#what-problem-does-it-solve}

并非所有提示都需要相同强度的事实依据。若 Router 对所有流量一视同仁，创意提示可能被过度约束，事实提示可能保护不足。

`fact-check` 检测哪些提示应触发证据感知路由行为。

## 何时使用 {#when-to-use}

在以下情况使用 `fact-check`：

- 事实主张需要更严格路由或插件
- 创意提示应绕过昂贵核验路径
- 幻觉缓解依赖早期路由信号
- 希望将事实性作为路由策略而非事后修补

## 配置 {#configuration}

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

## 依赖与限制 {#dependencies-and-limitations}

事实核查分类器通过 `global.model_catalog.modules.hallucination_mitigation.fact_check` 处理请求文本。它预测核验是否有用，并不核验主张。完整示例见：
[`config/fragments/signal/fact-check/needs-verification.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/fact-check/needs-verification.yaml)。

可达路由决策依赖此信号时，配置的模型必须初始化成功，否则 Router 启动失败。仅供独立诊断 API 使用的模型仍尽力初始化，不会因不可用而阻止无关路由。
