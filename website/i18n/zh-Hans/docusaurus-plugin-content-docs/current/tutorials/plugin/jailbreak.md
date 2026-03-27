---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/jailbreak.md"
  outdated: false
---

# Jailbreak

## 概览

`jailbreak` 是路由局部插件：对已匹配请求或输出用越狱阈值筛查。

对应 `config/plugin/jailbreak/guard.yaml`。

## 主要优势

- 决策匹配后为路由增加专用越狱 enforcement。
- 阈值局部在需要的流量上。
- 补充全局模型加载而不强制全局路由行为。

## 解决什么问题？

部分路由比其他路由需要更严的越狱 enforcement。`jailbreak` 为需要的路由显式表达匹配后安全策略。

## 何时使用

- 一条路由比路由器其余部分需要更严的越狱控制
- 已匹配路由应拦截或标注可疑流量
- 除全局模型可用性外还需要路由局部安全

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: jailbreak
  configuration:
    enabled: true
    threshold: 0.85
```
