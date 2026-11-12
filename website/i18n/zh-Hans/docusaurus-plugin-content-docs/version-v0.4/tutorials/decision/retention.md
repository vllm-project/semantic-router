---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/decision/retention.md"
  outdated: false
---

# 保留指令

## 概览

保留指令为匹配的决策添加有界的状态处理说明。它可以跳过响应缓存写入、缩短该写入的存活时间、在会话中保持当前模型，或发出前缀保留提示。

保留是 `emits` 指令，不是信号、算法或插件。

## 主要优势

- 把状态处理意图放在产生它的路由旁边。
- 使用类型化、有界字段，而不是临时头。
- 让缓存和会话副作用在路由诊断中可见。

## 解决什么问题？

有些响应是私密的、短时的，或对持续会话有价值。把该策略放在决策旁边，使其可审查，并防止宽泛的缓存或亲和默认值被盲目应用。

## 何时使用

当匹配路由需要以下副作用之一时使用保留：

- 不要把该响应写入响应缓存
- 为该响应应用更短的缓存存活时间
- 在已知会话中保持当前模型
- 告诉推理池优先做提示词前缀保留

不要用保留来匹配流量、授权调用方、脱敏内容或配置存储后端。

## 配置

```yaml
routing:
  decisions:
    - name: sensitive-turn
      description: Drop retained state for sensitive turns.
      priority: 200
      rules:
        operator: AND
        conditions:
          - type: pii
            name: restricted_pii
      modelRefs:
        - model: private-model
      emits:
        - kind: retention
          retention:
            drop: true
```

等价 DSL 是：

```dsl
ROUTE sensitive-turn {
  PRIORITY 200
  WHEN pii("restricted_pii")
  MODEL "private-model"
  EMIT retention {
    drop: true
  }
}
```

## 字段与运行时效果

| 字段 | 效果 | 重要限制 |
|---|---|---|
| `drop: true` | 跳过匹配决策的响应缓存写入 | 它不会阻止同一请求更早的缓存读取 |
| `ttl_turns` | 用 Router 的回合到时间映射覆盖响应缓存条目存活时间 | 它是缓存 TTL 提示，不是持久会话保留 |
| `keep_current_model: true` | 强制模型切换门保持在已知当前模型上 | 当会话/当前模型身份不可用时没有效果 |
| `prefer_prefix_retention: true` | 为推理池发出 `x-vsr-retention-prefer-prefix` | Router 本身不管理提供商的 KV-cache 驱逐 |

`drop: true` 和正数 `ttl_turns` 不能同时设置。一条决策只能发出一条保留指令。显式值通过有界的 `x-vsr-retention-*` 响应头和路由诊断暴露。

## 数据与安全

该指令携带策略元数据，而不是提示词或响应内容。它影响的缓存、会话、回放和提供商表面仍需要各自的认证、加密、租户隔离和保留设置。

部署前校验完整配方，让不兼容的保留设置在流量到达 Router 之前失败。
