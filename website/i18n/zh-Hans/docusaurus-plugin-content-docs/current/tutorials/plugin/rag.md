---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/rag.md"
  outdated: true
---

# RAG

## 概览

`rag` 是路由局部插件：检索增强生成。

对应 `config/plugin/rag/milvus.yaml`。

## 主要优势

- 检索局部在真正需要的路由上。
- 后端专用检索设置集中在一处。
- 避免强制每条路由注入文档或工具上下文。

## 解决什么问题？

部分路由在回答前需要外部文档检索，多数不需要。`rag` 让已匹配路由执行检索与注入，而不全局化该行为。

## 何时使用

- 路由应在最终模型调用前拉取文档或事实
- 检索应使用 Milvus 或其他显式后端
- 不同路由需要不同检索设置

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: rag
  configuration:
    enabled: true
    backend: milvus
    top_k: 5
    similarity_threshold: 0.78
    injection_mode: tool_role
    on_failure: warn
    backend_config:
      collection: docs
      reuse_cache_connection: true
      content_field: content
      metadata_field: metadata
```
