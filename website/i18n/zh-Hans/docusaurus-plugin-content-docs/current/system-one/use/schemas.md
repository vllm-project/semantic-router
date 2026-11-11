---
title: System One schema
sidebar_label: Schema
sidebar_position: 2
description: listener 对客户端使用哪个 System One schema、对上游使用哪个，Router 如何在两者之间转换，以及如何升级。
is_mtpe: false
translation:
  source_commit: "6b782822edead98397acc6401112e6ec1e45af96"
  source_file: "docs/system-one/use/schemas.md"
  outdated: false
---

# System One schema {#system-one-schemas}

schema ID 写明 System One API 遵循的是谁的规范、哪个版本。Router 支持三种：

| Schema | 含义 |
| --- | --- |
| `systemone/0.2` | 已发布的 System One 标准 0.2 版。仓库在 `config/schemas/systemone` 中保存我们自己对该接口的描述，附带来源和核对时的 sha256。 |
| `vllm-sr/5.0` | 标准加上 vllm-sr 扩展集 5.0：`roles`、`options`、`preset` 和 `over` 等问题提示、`set` 与 `span` 问题类型、视频资源、批量端点、平铺的错误字段和模型卡。见[扩展参考](../reference/extension-reference.md)。 |
| `vllm-sr/4.0` | 旧的运行时契约 4.0.0，保留下来让尚未升级的服务器和客户端继续工作。 |

Router 的两侧各自写明 schema，互不影响：listener 可以对客户端保持 `vllm-sr/4.0`，而所有后端都使用 `vllm-sr/5.0`。

## listener 使用的 schema {#the-schema-a-listener-speaks}

listener 的 `apis` 列出 `systemone` 时，它提供 System One API：

```yaml
listeners:
  - name: decision-api
    port: 8899
    apis:
      systemone:
        schema: vllm-sr/4.0      # 默认 vllm-sr/5.0
        models: [d3, vela3-fast]
        max_deadline_ms: 600000
```

| Schema | 请求 | 回答 |
| --- | --- | --- |
| `vllm-sr/5.0`（默认） | 严格解析：5.0 未定义的任何成员都以 400 拒绝，写明其名称，若是 4.0 形式还写明替代形式。没有 `roles` 的对象 state 中没有任何部分有角色。放在 state 前面的媒体在那里读取。 | 标准的 `detail`（422 时为列表）加平铺的 `code`、`question`、`state`、`reason` 和 `limit`；过载时返回 529；批量结果在 `responses` 下。 |
| `vllm-sr/4.0` | 4.0.0 接受的所有形式（`choices`、`levels`、`labels`、`images`、`videos`、`states` 批量、只有 preset 的问题、`options.long_context`）以及所有 5.0 形式，路径相同，无需请求头。没有 `roles` 的对象 state 保留 4.0 的隐式角色，资源在其占位符所在处读取。 | 现有的 4.0 形式；错误同时带有 4.0 的 `error` 和标准的 `detail` 及平铺字段，两类客户端都能读取。 |

`/v1/decisions` 在每个 listener 上都是原生的：它接受 Router 的内部请求形式（见下文），并在 200 中让每个失败的问题带自己的错误，与 schema 无关。

### listener 何时可以切换到 5.0 {#when-a-listener-can-switch-to-50}

Router 在 `vsr_systemone_listener_diagnostics_total{listener, schema, field, action}` 中统计每个 listener 的客户端发送了什么：

- `action="legacy_form"`，按 `field`：请求使用的 4.0 形式（`states`、`images`、`videos`、`choices`、`levels`、`labels`、`preset_only`、`long_context`）以及 `implicit_roles`，即没有 `roles` 的对象 state，5.0 listener 会以不同方式读取它；
- `field="request"` 且 `action="passed_through"` 或 `"translated"`：每个请求如何继续转发。

当没有客户端再使用旧形式时，先通知客户端，再把 listener 切换到 `vllm-sr/5.0`。listener 也可以一直保持 `vllm-sr/4.0`。

## 上游使用的 schema {#the-schema-an-upstream-speaks}

`provider: systemone` 部署以及 `api_format: systemone` 的 provider 模型写明其后端使用的 schema。默认是已发布的标准：

```yaml
global:
  model_catalog:
    deployments:
      serve-d3:                    # vllm-sr serve
        provider: systemone
        endpoint: https://decisions.example.com/v1/systemone
        model: vllm-sr/d3
        schema: vllm-sr/5.0
      gateway-d3:                  # 企业 LLM 网关后面的决策模型
        provider: systemone
        endpoint: https://gateway.example.com/v1/systemone
        model: d3
        schema: systemone/0.2      # 默认
        deviations: [state_text]
        max_concurrency: 8         # 默认
        timeout_ms: 6000
        auth: {header: X-Gateway-Key, prefix: "", api_key_env: GATEWAY_KEY}
providers:
  models:
    - name: pool-d3                # vllm-sr/auto 池成员
      api_format: systemone
      schema: vllm-sr/4.0          # 尚未升级的后端
      provider_model_id: vllm-sr/d3
      backend_refs:
        - provider: systemone-compatible
          endpoint: https://old-decisions.example.com/v1
```

| Schema | Router 发送的内容 |
| --- | --- |
| `vllm-sr/5.0` | 完整的 5.0 请求：显式 `roles`（Router 的部分名称一直以来的角色）、带 `preset` 提示的类型化 preset 问题、放在 state 前面的媒体，以及一次调用 `<endpoint>/batch` 发送一个阶段的所有 state。 |
| `vllm-sr/4.0` | 现有的 4.0 请求体：一个阶段作为一个 `states` 批量。与 4.0 服务器的 state 键所隐含的角色不同的 roles 会被拒绝。 |
| `systemone/0.2` | 只发送标准请求：一个阶段拆成每个 state 一次调用，最多 `max_concurrency` 个同时进行（滑动窗口，每次调用都在调用方的截止时间内）；preset 展开为其规范的 `instructions` 和 `criteria`；去掉 `roles`、`options` 和其他提示；不发送截止时间字段（使用客户端超时）。span、set、视频和完整覆盖规则在加载时失败，并写明规则和部署。标准没有模型卡，所以使用部署声明的 `capabilities`。 |

错过截止时间的调用只让它自己那个 state 的问题变为未知，于是它们的 `on_error` 生效。Router 读取标准的 400 `detail`、422 `detail` 列表（不带其中写明的问题再问一次）以及 429 或 529 的 `Retry-After`（在截止时间内允许时等待一次）。

### 偏差（deviations） {#deviations}

`deviations` 声明后端偏离其 schema 的方式，每一项都单独记录并测试：

| 偏差 | Schema | 作用 |
| --- | --- | --- |
| `state_text` | `systemone/0.2` | 把对象或数组 state 作为其精确的 `json.dumps(state, ensure_ascii=False)` 文本发送：`", "` 和 `": "` 分隔符，键保持顺序，非 ASCII 字符不做 `\u` 转义。Decision 3 把对象 state 渲染为完全相同的文本，因此两者读取结果相同；Vela 把 JSON 对象字符串读作该对象。用于只接受字符串 state 的后端。 |

## Router 如何转换 {#how-the-router-converts}

每次转换都经过同一个与 schema 无关的表示，即 IR（`systemoneprotocol`）：请求、批量、问题、答案、错误和模型列表，JSON 保持顺序。listener 的 codec 把客户端请求读入 IR；上游的 codec 把 IR 写成上游的 schema。

Router 在部署内部的跳转，即发往模型运行时 `/v1/bundle` 和 `/v1/decisions` 的请求，以**内部形式**携带 IR：4.0 请求形式加上 `roles`，其中没有 `roles` 表示 4.0 的隐式角色，`roles: null` 表示没有角色。内部 codec 不丢失任何信息：它读入的请求写回时保持不变，往返测试在各模型族自己的自检请求和模糊测试语料上检查这一点。因此，Router 构建的每个请求，以及 4.0 客户端发给 4.0 listener 的每个请求，其模型输入逐字节保持不变。

listener 决定每个请求如何继续转发，即它的**计划（plan）**：

- **透传。** 已是内部形式的请求（所有 4.0 请求，以及 4.0 listener 的 `/v1/systemone` 上的所有 5.0 请求）先解码并检查，再以客户端自己的字节转发。
- **转换。** 其他请求（5.0 listener 的请求、批量端点、带长上下文请求头的请求）由 IR 写成内部形式。

两种计划都计入上述 listener 诊断指标。

## 升级 {#upgrading}

`vllm-sr config migrate` 把已移除的 listener 字段移入 `apis`：`models` 变成每个 LLM API 的 `models`，`systemone` 块变成带 `schema: vllm-sr/4.0`（其客户端使用的契约）的 `apis.systemone`。对于 `vllm-sr/...` 模型的 systemone 部署或 provider 模型，它会写入 `schema: vllm-sr/5.0`。本版本的运行时在 `/v1/systemone` 上使用 `vllm-sr/5.0`；如需直接继续服务 4.0 客户端，请用 `vllm-srun serve --systemone-schema vllm-sr/4.0` 启动它。

基于 Router 或 Engine 构建的托管 Decision API 分两步升级：

1. 升级所有组件（包括后端），Router 或 Engine 与运行时之间使用 `vllm-sr/5.0`。公开 listener 保持 `apis.systemone.schema: vllm-sr/4.0`，客户端感觉不到变化；新客户端已经可以在那里使用 5.0 形式。直接调用运行时的工具在同一版本中改用 5.0 形式。没有前端、由运行时直接提供的公开端点，需要在前面加 Engine，或在运行时上使用相同的 listener schema。
2. 之后（可选），把内部调用方改为标准或 5.0 形式，并在 listener 的旧形式计数显示外部 4.0 流量已经消失后，把它切换到 `vllm-sr/5.0`。
