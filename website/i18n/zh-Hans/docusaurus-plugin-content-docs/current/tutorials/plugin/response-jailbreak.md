---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/response-jailbreak.md"
  outdated: true
---

# Response Jailbreak

## 概览

`response_jailbreak` 是路由局部插件：在返回前筛查模型响应。

对应 `config/plugin/response-jailbreak/strict.yaml`。

## 主要优势

- 为敏感路由增加最终响应侧越狱检查。
- 动作策略在配置中显式。
- 补充请求侧安全而不替代请求侧。

## 解决什么问题？

即使请求路由正确，生成答案仍可能需要最终安全门。`response_jailbreak` 为路由提供显式输出筛查步骤。

## 何时使用

- 路由需要最终响应侧越狱筛查
- 输出应在返回前拦截，或通过响应头进行标记
- 仅请求侧筛查对工作负载不足

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: response_jailbreak
  configuration:
    enabled: true
    threshold: 0.85
    action: block
```
