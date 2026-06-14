---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/content-safety.md"
  outdated: true
---

# Content Safety

## 概览

`content-safety` 是可复用的路由局部安全包：在一个片段中组合多个已支持的安全类插件。

对应 `config/plugin/content-safety/hybrid.yaml`。

## 主要优势

- 在多条路由间复用一致的多插件安全链。
- 需要多个插件时路由局部安全仍可读。
- 将包显式化，而非手工散落多个插件片段。

## 解决什么问题？

部分路由需要同时多种安全控制。不必反复手写响应筛查、路由局部护栏提示与审计头，`content-safety` 将该链打包为可复用片段。

## 何时使用

- 一条路由需要多个安全插件一起生效
- 希望多条路由共用同一条可复用审核链
- 路由应同时应用路由局部护栏与响应侧筛查

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugins:
  - type: system_prompt
    configuration:
      enabled: true
      mode: insert
      system_prompt: Apply the platform safety policy before answering and clearly note when a request needs additional review.
  - type: header_mutation
    configuration:
      add:
        - name: X-Safety-Profile
          value: standard
  - type: response_jailbreak
    configuration:
      enabled: true
      threshold: 0.8
      action: header
```
