---
translation:
  source_commit: "b2db276cf1b5057c31f2ab2bddbd181e5692dbb6"
  source_file: "docs/tutorials/plugin/overview.md"
  outdated: false
---

# 插件

## 概览

插件在决策匹配后添加路由局部行为。它们可以改写请求、检索上下文、短路生成、检查响应，或控制保留哪些运维数据。

共享服务和存储属于 `global:`；决策插件只为一条路由启用并调优该行为。

## 主要优势

- 把行为附着在需要它的路由上。
- 复用共享存储和服务，而不重复其配置。
- 让请求改写、检索和响应检查可审计。

## 解决什么问题？

即使共享同一 Router，各路由也常常需要不同行为。插件把这些差异放在决策旁边，而不是藏进应用中间件或全局启用。

## 何时使用

当行为应仅在特定路由匹配后生效时，使用插件。当每条路由共享同一服务或后台存储时，改用 `global:`。插件条目位于 `routing.decisions[].plugins` 下。

## 配置

```yaml
routing:
  decisions:
    - name: cached-support
      description: Reuse cached responses for support requests.
      priority: 100
      rules:
        operator: AND
        conditions: []
      modelRefs:
        - model: support-model
      plugins:
        - type: response_cache
          configuration:
            enabled: true
            ttl_seconds: 3600
```

## 插件清单 {#plugin-inventory}

| 类型 | 目标 | 共享依赖 | 指南 |
|---|---|---|---|
| `fast_response` | 不调用模型即返回配置的响应 | 无 | [快速响应](./fast-response) |
| `system_prompt` | 插入、替换或追加路由专用指令 | 无 | [系统提示词](./system-prompt) |
| `header_mutation` | 添加、更新或删除下游请求头 | 无 | [请求头修改](./header-mutation) |
| `request_params` | 强制请求参数限制 | 无 | [请求参数](./request-params) |
| `tools` | 允许、阻止、过滤或移除工具和工具历史 | 可选的全局工具目录 | [工具](./tools) |
| `tool_selection` | 从目录添加工具，或按语义过滤调用方工具 | 嵌入运行时；`add` 模式需要工具数据库 | [工具选择](./tool-selection) |
| `context_compression` | 缩减绑定提供商的大型工具输出或历史 | 可选的嵌入运行时和恢复存储 | [上下文压缩](./context-compression) |
| `response_cache` | 复用兼容的先前响应 | `global.stores.response_cache` | [响应缓存](./response-cache) |
| `memory` | 检索并可选存储对话记忆 | `global.stores.memory` | [记忆](./memory) |
| `rag` | 在生成前检索文档 | 已配置的 RAG/向量后端 | [RAG](./rag) |
| `router_replay` | 覆盖单条路由的回放采集 | `global.services.router_replay` | [路由回放](./router-replay) |
| `shadow_dispatch` | 在采样流量上观察次要模型，且不触碰线上响应 | 用于结果采集的回放记录（`router_replay`） | [Shadow Dispatch](./shadow-dispatch) |
| `hallucination` | 检查响应中的事实依据 | 按配置的幻觉/NLI 模块 | [幻觉检测](./hallucination) |
| `response_jailbreak` | 筛查生成响应中的越狱内容 | Prompt-guard 运行时 | [响应越狱](./response-jailbreak) |

[内容安全](./content-safety) 打包了三个受支持插件，而不是额外的插件类型。

## 运维边界 {#operational-boundaries}

- 当多个插件改写绑定提供商的请求或响应时，它们会相互作用。Router 管道固定其执行顺序；在 YAML 中重排条目不会改变该顺序。
- 检索、memory、缓存和回放可能持久化从请求派生的内容。为所选后端配置保留策略、租户/用户范围、认证和加密。
- 请求头和提示词改写可能跨越信任边界。不要把不受信任的调用方元数据复制到特权请求头或系统指令中。
- 部署前校验完整配方，以便不支持的插件名或不兼容设置在流量到达 Router 之前失败。
