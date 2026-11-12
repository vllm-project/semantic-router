---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/plugin/content-safety.md"
  outdated: false
---

# 内容安全

## 概览

内容安全把受支持的路由局部安全插件组合成一份可复用策略。它是配置包，不是单独的插件类型。

## 主要优势

- 在各路由间复用一致的多插件安全链。
- 即使需要多个插件，也能让路由局部安全保持可读。
- 明确声明该包，而不是手工散落各个插件片段。

## 解决什么问题？

有些路由需要同时使用不止一种安全控制。该包让这些路由的响应筛查、路由局部守卫提示词和审计请求头保持一致。

## 何时使用

- 某条路由需要同时使用多个安全插件
- 希望为多条路由使用一条可复用的审核链
- 该路由应同时应用路由局部指引和响应侧筛查

## 配置

在 `routing.decisions[].plugins` 下添加打包的插件条目：

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

这是组合示例，不是 `content_safety` 插件类型。`system_prompt` 添加请求侧指引，`header_mutation` 添加策略标签，`response_jailbreak` 评估生成的响应。该包不会运行请求侧内容分类器，请求头也不是内容安全的证明。请校准响应筛查，并判断仅处理请求头是否足够。

完整包见：
[`config/fragments/plugin/content-safety/hybrid.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/content-safety/hybrid.yaml)。
