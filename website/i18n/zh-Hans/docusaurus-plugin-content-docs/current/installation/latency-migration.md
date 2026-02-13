---
translation:
  source_commit: "dd5c06f"
  source_file: "docs/installation/latency-migration.md"
  outdated: false
is_mtpe: true
sidebar_position: 5
---

# 延迟路由迁移指南

延迟路由已从 signal 层迁移到 decision algorithm 层。

- 推荐配置：`decision.algorithm.type: latency_aware`
- 推荐算法块：`decision.algorithm.latency_aware`
- 旧配置（已弃用）：`signals.latency` 或 `signals.latency_rules` + `conditions.type: latency`

请在旧配置兼容窗口结束前完成迁移。

## 变更内容

### 旧模式（已弃用）

- 在 `signals.latency`（CLI 配置）或 `signals.latency_rules`（router 配置）中定义延迟阈值
- 在规则里使用 `rules.conditions[].type: latency` 进行匹配

### 新模式（推荐）

- 将延迟策略放在 decision algorithm：
  - `algorithm.type: latency_aware`
  - `algorithm.latency_aware.tpot_percentile`
  - `algorithm.latency_aware.ttft_percentile`
- 请求理解仍放在 signal 层（keyword/domain/embedding/...）。

## 向后兼容行为

在弃用窗口期内，旧配置可自动迁移，但仅限“无损迁移”场景。

自动迁移必须同时满足：

1. 该 decision 只有一个旧版 latency condition。
2. `rules.operator` 为 `AND`。
3. latency condition 引用了有效的旧版 latency rule。
4. `decision.algorithm` 不存在，或 `type: static`。
5. 去掉 latency condition 后，仍至少保留一个非 latency condition。

满足以上条件时，router/CLI 会：

- 将 decision 重写为 `algorithm.type: latency_aware`
- 将百分位参数复制到 `algorithm.latency_aware`
- 删除旧版 latency signal rule
- 打印 deprecation 警告

## 新旧混用会报错

不要在同一个配置文件里混用：

- 旧版 `signals.latency` + `conditions.type: latency`
- 与任意 `algorithm.type: latency_aware` decision

## 迁移步骤

### 1. 识别旧版延迟配置

搜索以下字段：

- `signals.latency`（CLI 风格）
- `signals.latency_rules`（router 风格）
- `conditions.type: latency`

### 2. 把每个旧 decision 转成算法配置

把延迟阈值迁移到 `algorithm.latency_aware`。

迁移前（旧）：

```yaml
signals:
  latency:
    - name: "low_latency"
      tpot_percentile: 10
      ttft_percentile: 10

decisions:
  - name: "fast_route"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "other"
        - type: "latency"
          name: "low_latency"
    modelRefs:
      - model: "openai/gpt-oss-120b"
      - model: "gpt-5.2"
```

迁移后（推荐）：

```yaml
decisions:
  - name: "fast_route"
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "other"
    modelRefs:
      - model: "openai/gpt-oss-120b"
      - model: "gpt-5.2"
    algorithm:
      type: "latency_aware"
      latency_aware:
        tpot_percentile: 10
        ttft_percentile: 10
```

### 3. 删除旧版延迟字段

所有相关 decision 都迁移后，删除：

- `signals.latency`
- `signals.latency_rules`
- 所有 `conditions.type: latency`

### 4. 用新配置启动并验证

使用更新后的配置启动 router/CLI，确认没有旧版 latency 警告或迁移错误。

## 常见迁移错误

- `legacy latency config ... cannot be used with decision.algorithm.type=latency_aware`
  - 原因：同一配置里混用了新旧方案。
- `only static can be auto-migrated to latency_aware`
  - 原因：旧 latency condition 所在 decision 使用了非 `static` 算法。
- `multiple legacy latency conditions are not supported for auto-migration`
  - 原因：同一个 decision 里有多个 `type: latency` condition。
- `... rules.operator=... cannot be auto-migrated; only AND is supported`
  - 原因：旧 latency condition 使用了 `OR`。
- `... no non-latency conditions remain`
  - 原因：去掉 latency condition 后没有剩余条件。

## 建议

完成迁移后，仅在 `algorithm.type: latency_aware` 配置延迟路由，把请求信号层和模型选择层明确分离。
