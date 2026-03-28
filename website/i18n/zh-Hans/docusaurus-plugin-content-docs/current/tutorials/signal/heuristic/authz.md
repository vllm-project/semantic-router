---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/signal/heuristic/authz.md"
  outdated: false
---

# Authz 信号

## 概览

`authz` 将身份与策略绑定转为可复用的路由输入。映射到 `config/signal/authz/`，在 `routing.signals.role_bindings` 中声明。

该族为启发式：用显式角色与主体匹配请求身份，而非分类器输出。

## 主要优势

- 无需额外模型推理即可为高端、内部或租户流量路由。
- 访问策略在 `routing.decisions` 内可见。
- 同一身份规则可被多条路由复用。
- RBAC 驱动路由可在 YAML 中审计。

## 解决什么问题？

没有 `authz` 信号时，路由决策无法直接看到用户层级或角色成员，访问敏感路由被推入零散中间件，策略难审查。

`authz` 将成员关系暴露为命名信号，可与领域、安全或插件逻辑组合。

## 何时使用

在以下情况使用 `authz`：

- 管理员与终端用户流量需不同路由
- 高端层级解锁更强模型或插件
- 租户或组成员改变路由资格
- 路由策略应与其余路由图处于同一图内

## 配置

源片段族：`config/signal/authz/`

```yaml
routing:
  signals:
    role_bindings:
      - name: admin
        description: Requests from platform administrators.
        role: admin
        subjects:
          - kind: Group
            name: platform-admins
      - name: premium_user
        description: Requests from paid end users.
        role: premium_user
        subjects:
          - kind: Group
            name: premium-tier
```

当信号应来自已认证身份与策略元数据，而非提示内容时，使用 `role_bindings`。
