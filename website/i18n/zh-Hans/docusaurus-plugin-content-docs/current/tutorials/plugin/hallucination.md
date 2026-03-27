---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/hallucination.md"
  outdated: false
---

# Hallucination

## 概览

`hallucination` 是路由局部插件：决策已匹配后的事实核查与响应质量筛查。

对应 `config/plugin/hallucination/fact-check.yaml`。

## 主要优势

- 在不改全局默认的情况下增加路由局部幻觉检查。
- 事实置信度低时响应动作显式。
- 适合检索重或 grounded 回答路由。

## 解决什么问题？

部分路由在模型回答后需要额外审查，尤其承诺事实精度时。`hallucination` 让这些路由增加响应时核验，而不让每条路由承担成本。

## 何时使用

- 路由应对响应做事实核查或标注
- 基于工具或 grounding 的路由需要额外响应筛查
- 路由应在低置信事实时警告或标注，而非默默通过

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: hallucination
  configuration:
    enabled: true
    use_nli: true
    hallucination_action: annotate
    unverified_factual_action: warn
    include_hallucination_details: true
```
