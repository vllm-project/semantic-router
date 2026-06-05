---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/plugin/header-mutation.md"
  outdated: false
---

# Header Mutation

## 概览

`header_mutation` 是路由局部插件：添加、更新或删除下游请求头。

对应 `config/plugin/header-mutation/tenant-routing.yaml`。

## 主要优势

- 下游头策略局部在已匹配路由。
- 在一个插件中支持增、改、删。
- 适合租户路由、调试与下游策略提示。

## 解决什么问题？

部分路由需要与路由器其余部分不同的下游头。`header_mutation` 显式表达该变换，而非藏在代理或应用代码中。

## 何时使用

- 一条路由应将租户或套餐元数据写入头
- 下游服务期望路由专用头
- 仅对选中流量添加调试或溯源头

## 配置

在 `routing.decisions[].plugins` 下使用：

```yaml
plugin:
  type: header_mutation
  configuration:
    add:
      - name: X-Tenant-Tier
        value: premium
    update:
      - name: X-Route-Source
        value: semantic-router
    delete:
      - X-Debug-Trace
```
