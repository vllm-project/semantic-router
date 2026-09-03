---
translation:
  source_commit: "7c874be2"
  source_file: "docs/tutorials/plugin/header-mutation.md"
  outdated: false
---

# Header 修改（Header Mutation）

## 概览

`header_mutation` 是一个路由局部插件，用于添加、更新或删除下游 header。

## 主要优势

- 将下游 header 策略限制在匹配的路由内。
- 在一个插件中支持 add、update 和 delete 操作。
- 适用于租户路由、调试和下游策略提示。

## 解决什么问题？

有些路由需要不同于路由器其他部分的下游 header。`header_mutation` 明确声明这种转换，而不是将其隐藏在代理或应用代码中。

## 何时使用

- 某个路由应在 header 中标记租户或套餐元数据
- 下游服务需要路由专用的 header
- 调试或来源 header 只应添加到选定流量

## 配置

在 `routing.decisions[].plugins` 下添加该插件：

```yaml
plugins:
  - type: header_mutation
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

Header 值是静态配置，而不是模板。不得使用不受信任的请求元数据来构造身份或授权 header。完整示例见：
[`config/fragments/plugin/header-mutation/tenant-routing.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/header-mutation/tenant-routing.yaml)。
