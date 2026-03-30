---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/content-safety.md"
  outdated: false
---

# Content Safety

## 概览

`content-safety` 是可复用的路由局部安全包：在一个片段中组合多个安全插件。

对应 `config/plugin/content-safety/hybrid.yaml`。

## 主要优势

- 在多条路由间复用一致的多插件安全链。
- 需要多个插件时路由局部安全仍可读。
- 将包显式化，而非手工散落多个插件片段。

## 解决什么问题？

部分路由需要同时多种安全控制。不必反复手写 jailbreak、PII 与响应筛查插件，`content-safety` 将该链打包为可复用片段。

## 何时使用

- 一条路由需要多个安全插件一起生效
- 希望多条路由共用同一条可复用审核链
- 路由应同时应用请求侧与响应侧检查

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugins:
  - type: jailbreak
    configuration:
      enabled: true
      threshold: 0.6
  - type: pii
    configuration:
      enabled: true
      pii_types_allowed: []
  - type: response_jailbreak
    configuration:
      enabled: true
      threshold: 0.8
      action: annotate
```
