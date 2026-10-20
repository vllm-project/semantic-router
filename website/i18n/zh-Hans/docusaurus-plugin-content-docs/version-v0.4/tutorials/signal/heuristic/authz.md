---
translation:
  source_commit: "7c874be29871f6d00b36b2e21b3e549e846b98c5"
  source_file: "docs/tutorials/signal/heuristic/authz.md"
  outdated: false
---

# 授权信号 {#authz-signal}

## 概览 {#overview}

`authz` 将身份与策略绑定转为 `routing.signals.role_bindings` 下可复用的路由输入。

该族为启发式：用显式角色与主体匹配请求身份，而非分类器输出。

## 主要优势 {#key-advantages}

- 无需额外模型推理即可为高端、内部或租户范围流量路由。
- 访问策略在 `routing.decisions` 内可见。
- 同一身份规则可被多条路由复用。
- RBAC 驱动路由可在 YAML 中审计。

## 解决什么问题？ {#what-problem-does-it-solve}

没有 `authz` 信号时，路由决策无法直接看到用户层级或角色成员。这会把访问敏感路由推入零散中间件，策略更难审查。

`authz` 将角色成员关系暴露为命名信号，决策可将其与领域、安全或插件逻辑组合。

## 何时使用 {#when-to-use}

在以下情况使用 `authz`：

- 管理员与终端用户流量需不同路由
- 高端层级解锁更强模型或插件
- 租户或组成员改变路由资格
- 路由策略应与其余路由逻辑处于同一图内

## 配置 {#configuration}

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

## 依赖与限制 {#dependencies-and-limitations}

身份来自 `global.services.authz`；信号本身不认证请求。只信任由认证层设置或清洗过的请求头。完整示例见：
[`config/fragments/signal/authz/rbac.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/authz/rbac.yaml)。
