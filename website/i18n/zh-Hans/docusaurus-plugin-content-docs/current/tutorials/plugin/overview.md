---
translation:
  source_commit: "baa07413"
  source_file: "docs/tutorials/plugin/overview.md"
  outdated: false
---

# 插件（Plugin）

## 概览

在已匹配决策在模型选择之后仍需要**额外路由局部行为**时使用插件。

在规范 v0.3 YAML 中，插件位于 `routing.decisions[].plugins`。

## 主要优势

- 将路由局部行为挂在需要它的路由上。
- 避免把所有行为推入 `global:` 默认。
- 单条路由可选用缓存、变更、检索或安全控制而不影响其他路由。
- 直接映射 `config/plugin/` 片段树，每个插件或插件包一页教程。

## 解决什么问题？

并非每条路由都需要相同的后选行为。有的需要语义缓存，有的需要 system prompt 变更，有的需要路由局部安全执行。

插件使路由局部处理显式，而非过载全局运行时设置。

## 何时使用

在以下情况使用 `plugin/`：

- 仅一条路由或路由族需要额外处理
- 行为应在路由匹配之后发生
- 共享后端在 `global:`，但每路由行为必须保持局部
- 希望在 `config/plugin/` 下复用路由局部片段

## 配置

规范位置：

```yaml
routing:
  decisions:
    - name: cached_support
      plugins:
        - type: semantic-cache
          configuration:
            enabled: true
```

插件文档与 `config/plugin/` 一一对应。

### 响应与变更

- [Fast Response](./fast-response)
- [Header Mutation](./header-mutation)
- [Image Generation](./image-gen)
- [Request Parameters](./request-params)
- [System Prompt](./system-prompt)
- [Tools](./tools)

### 检索与记忆

- [Memory](./memory)
- [RAG](./rag)
- [Router Replay](./router-replay)
- [Semantic Cache](./semantic-cache)

### 安全与生成

- [Content Safety](./content-safety)
- [Hallucination](./hallucination)
- [Response Jailbreak](./response-jailbreak)
